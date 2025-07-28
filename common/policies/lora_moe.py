from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Callable, Iterable, List, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

# ---------------------------------------------------------------------------
# Config & Utilities
# ---------------------------------------------------------------------------

@dataclass
class LoRAMoEConfig:
    r: int = 8  # low‑rank dim per expert
    alpha: int = 16  # scaling factor
    num_experts: int = 4  # number of LoRA experts
    dropout: float = 0.05  # applied to input of adapters
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
# Mixture‑of‑LoRA Linear
# ---------------------------------------------------------------------------

class MoELoRALinear(nn.Module):
    """A `nn.Linear` wrapped with *multiple* LoRA experts and a router.

    Args:
        base (nn.Linear): The frozen base projection.
        cfg (LoRAMoEConfig): Hyper‑parameters.
    """

    def __init__(self, base: nn.Linear, cfg: LoRAMoEConfig):
        super().__init__()
        if not isinstance(base, nn.Linear):
            raise TypeError("MoELoRALinear expects an nn.Linear to wrap")

        self.base = base
        self.cfg = cfg
        for p in self.base.parameters():
            p.requires_grad_(False)

        in_f, out_f = self.base.in_features, self.base.out_features
        if cfg.fan_in_fan_out:
            in_f, out_f = out_f, in_f

        # LoRA expert parameters – grouped tensors for efficiency
        self.A = nn.Parameter(torch.zeros(cfg.num_experts, cfg.r, in_f))  # (E, r, in)
        self.B = nn.Parameter(torch.zeros(cfg.num_experts, out_f, cfg.r))  # (E, out, r)
        # Init per LoRA paper
        nn.init.kaiming_uniform_(self.A, a=math.sqrt(5))
        nn.init.zeros_(self.B)

        # Router (token‑wise gating)
        self.router = nn.Linear(in_f, cfg.num_experts, bias=False)
        nn.init.zeros_(self.router.weight)

        self.dropout = nn.Dropout(cfg.dropout) if cfg.dropout > 0.0 else nn.Identity()

        # expose merge flag similar to LoRA
        self._merged: bool = False

    # ---------------------------------------------------------
    # Expose base weight param
    # ---------------------------------------------------------

    @property
    def weight(self):  # type: ignore
        return self.base.weight

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # type: ignore
        """Assumes input shape (..., in_features).  Router acts on last dim."""
        base_out = self.base(x)
        x_dp = self.dropout(x)

        # Compute gates
        gates = torch.softmax(self.router(x_dp), dim=-1)  # (..., E)

        # Flatten leading dims for batched matmul
        leading_shape = x_dp.shape[:-1]              # (...)
        in_f = self.base.in_features
        x_flat = x_dp.reshape(-1, in_f)              # (N, in)

        A_t = self.A.transpose(2, 1)           # (E, in, r)
        B_t = self.B.transpose(2, 1)           # (E, r, out)

        # (N, 1, in) × (E, in, r) -> (N, E, r)
        proj_r = torch.matmul(x_flat.unsqueeze(1), A_t)  # (N, E, r)

        # (N, E, r) × (E, r, out) -> (N, E, out)
        lora_out = torch.matmul(proj_r, B_t)             # (N, E, out)

        # Restore leading shape
        lora_out = lora_out.reshape(*leading_shape, self.cfg.num_experts, out_f)

        # Weighted sum over experts without einsum: (..., E, out)
        weighted = (gates * self.cfg.scale).unsqueeze(-1) * lora_out  # (..., E, out)
        lora_mix = weighted.sum(dim=-2)                                # (..., out)
        
        return base_out + lora_mix


def inject_lora_moe(
    model: nn.Module,
    cfg: LoRAMoEConfig | None = None,
    *,
    target_keywords: Iterable[str] | None = None,
    filter_fn: Callable[[str, nn.Module], bool] | None = None,
) -> Tuple[nn.Module, List[str]]:
    """Replace matching `nn.Linear` layers with `MoELoRALinear` (in-place)."""

    cfg = cfg or LoRAMoEConfig()
    wrapped: List[str] = []

    for name, module in model.named_modules():
        if not isinstance(module, nn.Linear):
            continue
        if target_keywords and not _match_name(name, target_keywords):
            continue
        if filter_fn and not filter_fn(name, module):
            continue

        parent, attr = _get_parent(model, name)
        moe_layer = MoELoRALinear(module, cfg)
        setattr(parent, attr, moe_layer)
        wrapped.append(name)

    # Track adapter parameter names
    if wrapped:
        adapter_names = []
        for w in wrapped:
            adapter_names.extend([f"{w}.A", f"{w}.B", f"{w}.router.weight"])
        existing = getattr(model, "_adapter_param_names", set())
        model._adapter_param_names = set(existing).union(adapter_names)

    if not wrapped:
        raise RuntimeError("No linear layers matched for LoRA-MoE injection.")
    return model, wrapped