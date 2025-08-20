from __future__ import annotations

import math
import gc
from dataclasses import dataclass
from typing import Callable, Iterable, List, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
import bitsandbytes as bnb


# ---------------------------------------------------------------------------
# Config & Utilities
# ---------------------------------------------------------------------------

@dataclass
class QLoRAConfig:
    r: int = 64  # low-rank dimension
    alpha: int = 128  # scaling factor
    dropout: float = 0.05  # dropout on input features
    fan_in_fan_out: bool = False  # set True if base weight is transposed

    quant_type: str = 'fp4'
    compute_dtype: torch.dtype = torch.bfloat16
    compress_statistics: bool = False
    quant_storage: torch.dtype = torch.uint8


    @property
    def scale(self) -> float:
        return self.alpha / self.r


def _match_name(name: str, keywords: Iterable[str]) -> bool:
    return any(k in name for k in keywords)


def _get_parent(root: nn.Module, full_name: str) -> Tuple[nn.Module, str]:
    parts = full_name.split(".")
    for p in parts[:-1]:
        root = getattr(root, p)
    return root, parts[-1]

# ---------------------------------------------------------------------------
# QLoRA Linear
# ---------------------------------------------------------------------------

class QLoRALinear(nn.Module):
    """A `nn.Linear` wrapped with a LoRA adapter (single expert).

    Args:
        base (nn.Linear): The frozen base projection.
        cfg (LoRAConfig): Hyper-parameters for the adapter.
    """

    def __init__(self, base: nn.Linear, cfg: QLoRAConfig):
        super().__init__()
        if not isinstance(base, nn.Linear):
            raise TypeError("QLoRALinear expects an nn.Linear to wrap")

        self.cfg = cfg
        for p in base.parameters():
            p.requires_grad_(False)

        in_f, out_f = base.in_features, base.out_features
        if cfg.fan_in_fan_out:
            in_f, out_f = out_f, in_f

        self.base = bnb.nn.Linear4bit(
                    input_features=base.in_features,
                    output_features=base.out_features,
                    bias=base.bias is not None,
                    quant_type=self.cfg.quant_type,
                    compute_dtype=self.cfg.compute_dtype,
                    compress_statistics=self.cfg.compress_statistics,
                    quant_storage=self.cfg.quant_storage,
                )
        self.base.load_state_dict(base.state_dict())
        self.base.weight.requires_grad = False

        # LoRA parameters (rank-r decomposition)
        self.A = nn.Parameter(torch.zeros(cfg.r, in_f, dtype=base.weight.dtype))  # (r, in)
        self.B = nn.Parameter(torch.zeros(out_f, cfg.r, dtype=base.weight.dtype))  # (out, r)

        # Initialization per LoRA paper
        nn.init.kaiming_uniform_(self.A, a=math.sqrt(5))
        nn.init.zeros_(self.B)

        self.dropout = nn.Dropout(cfg.dropout) if cfg.dropout > 0.0 else nn.Identity()

        # Merge state flag
        self._merged: bool = False

    # ---------------------------------------------------------
    # Utility helpers
    # ---------------------------------------------------------

    def extra_repr(self) -> str:  # shows up with print(module)
        return (
            f"in_features={self.base.in_features}, out_features={self.base.out_features}, "
            f"r={self.cfg.r}, alpha={self.cfg.alpha}, dtype={self.base.weight.dtype}, "
            f"merged={self._merged}"
        )

    def _lora_delta(self) -> torch.Tensor:
        """Compute LoRA weight delta = B @ A (returns same dtype as base weight)."""
        delta = (self.B @ self.A) * self.cfg.scale  # (out,in)
        if self.cfg.fan_in_fan_out:
            delta = delta.T  # match original layout
        return delta.to(dtype=self.base.weight.dtype)

    @torch.no_grad()
    def merge(self) -> None:
        """Manually merge LoRA weights into the frozen base layer for inference."""
        if self._merged or self.cfg.r == 0:
            return
        self.base.weight.data += self._lora_delta()
        self._merged = True

    @torch.no_grad()
    def unmerge(self) -> None:
        """Undo :py:meth:`merge`. Rarely needed (e.g., to resume training after merging)."""
        if not self._merged or self.cfg.r == 0:
            return
        self.base.weight.data -= self._lora_delta()
        self._merged = False

    # ---------------------------------------------------------
    # Expose base weight for compatibility
    # ---------------------------------------------------------

    @property
    def weight(self):  # type: ignore
        """Alias to underlying base layer's weight parameter (read-only)."""
        return self.base.weight

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # type: ignore
        base_out = self.base(x)
        x_dp = self.dropout(x)

        proj_r   = F.linear(x_dp, self.A)          # (..., r)
        lora_out = F.linear(proj_r, self.B)        # (..., out)

        lora_out = lora_out * self.cfg.scale
        return base_out + lora_out


# ---------------------------------------------------------------------------
# Injection Utility
# ---------------------------------------------------------------------------

def inject_qlora(
    model: nn.Module,
    cfg: QLoRAConfig | None = None,
    *,
    target_keywords: Iterable[str] | None = None,
    filter_fn: Callable[[str, nn.Module], bool] | None = None,
) -> Tuple[nn.Module, List[str]]:
    """Replace matching `nn.Linear` layers with `LoRALinear` (in-place).

    Args:
        model: The model to modify in-place.
        cfg: QLoRA configuration. If `None`, defaults will be used.
        target_keywords: If provided, only layers whose names contain any of the
            keywords will be adapted. Useful to restrict LoRA to e.g. "q_proj".
        filter_fn: Custom callable `(name, module) -> bool` to decide whether to
            adapt a given layer.

    Returns:
        The modified model (same object) and the list of wrapped layer names.
    """

    cfg = cfg or QLoRAConfig()
    wrapped: List[str] = []
    device = model.device

    for name, module in model.named_modules():
        if not isinstance(module, nn.Linear):
            continue
        if target_keywords and not _match_name(name, target_keywords):
            continue
        if filter_fn and not filter_fn(name, module):
            continue

        parent, attr = _get_parent(model, name)
        qlora_layer = QLoRALinear(module, cfg)
        setattr(parent, attr, qlora_layer)
        wrapped.append(name)

    # Keep track of adapter parameter names for lightweight checkpointing
    if wrapped:
        adapter_names = [
            f"{w}.A" for w in wrapped
        ] + [
            f"{w}.B" for w in wrapped
        ]
        existing = getattr(model, "_adapter_param_names", set())
        model._adapter_param_names = set(existing).union(adapter_names)

    if not wrapped:
        raise RuntimeError("No linear layers matched for LoRA injection.")

    gc.collect()
    if device != torch.device('cpu'):
        torch.cuda.empty_cache()

    return model, wrapped