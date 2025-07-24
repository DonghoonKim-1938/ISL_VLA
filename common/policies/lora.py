from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Callable, Iterable, List, Tuple

import torch
import torch.nn as nn

# ---------------------------------------------------------------------------
# Config & Utilities
# ---------------------------------------------------------------------------

@dataclass
class LoRAConfig:
    r: int = 8  # low-rank dimension
    alpha: int = 16  # scaling factor
    dropout: float = 0.05  # dropout on input features
    fan_in_fan_out: bool = False  # set True if base weight is transposed

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
# LoRA Linear
# ---------------------------------------------------------------------------

class LoRALinear(nn.Module):
    """A `nn.Linear` wrapped with a LoRA adapter (single expert).

    Args:
        base (nn.Linear): The frozen base projection.
        cfg (LoRAConfig): Hyper-parameters for the adapter.
    """

    def __init__(self, base: nn.Linear, cfg: LoRAConfig):
        super().__init__()
        if not isinstance(base, nn.Linear):
            raise TypeError("LoRALinear expects an nn.Linear to wrap")

        self.base = base
        self.cfg = cfg
        for p in self.base.parameters():
            p.requires_grad_(False)

        in_f, out_f = self.base.in_features, self.base.out_features
        if cfg.fan_in_fan_out:
            in_f, out_f = out_f, in_f

        # LoRA parameters (rank-r decomposition)
        self.A = nn.Parameter(torch.zeros(cfg.r, in_f))  # (r, in)
        self.B = nn.Parameter(torch.zeros(out_f, cfg.r))  # (out, r)

        # Initialization per LoRA paper
        nn.init.kaiming_uniform_(self.A, a=math.sqrt(5))
        nn.init.zeros_(self.B)

        self.dropout = nn.Dropout(cfg.dropout) if cfg.dropout > 0.0 else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # type: ignore
        base_out = self.base(x)
        x_dp = self.dropout(x)

        # Efficient LoRA projection with einsum
        #   step1:  (r, in)  – project to rank
        projected_r = torch.einsum("...i,ri->...r", x_dp, self.A)
        #   step2:  (out, r) – back to out features
        lora_out = torch.einsum("...r,or->...o", projected_r, self.B)

        return base_out + lora_out * self.cfg.scale


# ---------------------------------------------------------------------------
# Injection Utility
# ---------------------------------------------------------------------------

def inject_lora(
    model: nn.Module,
    cfg: LoRAConfig | None = None,
    *,
    target_keywords: Iterable[str] | None = None,
    filter_fn: Callable[[str, nn.Module], bool] | None = None,
) -> Tuple[nn.Module, List[str]]:
    """Replace matching `nn.Linear` layers with `LoRALinear` (in-place).

    Args:
        model: The model to modify in-place.
        cfg: LoRA configuration. If `None`, defaults will be used.
        target_keywords: If provided, only layers whose names contain any of the
            keywords will be adapted. Useful to restrict LoRA to e.g. "q_proj".
        filter_fn: Custom callable `(name, module) -> bool` to decide whether to
            adapt a given layer.

    Returns:
        The modified model (same object) and the list of wrapped layer names.
    """

    cfg = cfg or LoRAConfig()
    wrapped: List[str] = []

    for name, module in model.named_modules():
        if not isinstance(module, nn.Linear):
            continue
        if target_keywords and not _match_name(name, target_keywords):
            continue
        if filter_fn and not filter_fn(name, module):
            continue

        parent, attr = _get_parent(model, name)
        lora_layer = LoRALinear(module, cfg)
        setattr(parent, attr, lora_layer)
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
    return model, wrapped 