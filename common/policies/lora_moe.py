from __future__ import annotations

import math
import gc
from dataclasses import dataclass
from typing import Callable, Iterable, List, Tuple

import torch
import torch.nn as nn

from common.policies.lora import LoraLinear, LoraConfig

# ---------------------------------------------------------------------------
# Config & Utilities
# ---------------------------------------------------------------------------

@dataclass
class LoraMoEConfig(LoraConfig):
    layer_type: str = "lora_moe"

    r: int = 8  # low‑rank dim per expert
    alpha: int = 16  # scaling factor
    num_experts: int = 4  # number of LoRA experts
    dropout: float = 0.05  # applied to input of adapters
    fan_in_fan_out: bool = False  # set True if base weight is transposed
    routing: str = "weighted"   # "weighted", "top1", "top2"
    quantize: bool = False

    quant_type: str = 'fp4'
    compute_dtype: torch.dtype = torch.bfloat16
    compress_statistics: bool = False
    quant_storage: torch.dtype = torch.uint8

    @property
    def scale(self) -> float:
        return self.alpha / self.r

# ---------------------------------------------------------------------------
# Mixture‑of‑LoRA Linear
# ---------------------------------------------------------------------------

class LoraMoELinear(LoraLinear):
    """A `nn.Linear` wrapped with *multiple* LoRA experts and a router.

    Args:
        base (nn.Linear): The frozen base projection.
        cfg (LoraMoEConfig): Hyper‑parameters.
    """

    def __init__(self, base: nn.Linear, cfg: LoraMoEConfig):
        super().__init__(base, cfg)
        if not isinstance(base, nn.Linear):
            raise TypeError("MoELoRALinear expects an nn.Linear to wrap")

        self.cfg = cfg
        self._load_base(base, cfg.quantize)
        for p in self.base.parameters():
            p.requires_grad_(False)

        in_f, out_f = self.base.in_features, self.base.out_features
        if cfg.fan_in_fan_out:
            in_f, out_f = out_f, in_f

        # LoRA expert parameters – grouped tensors for efficiency
        self.A = nn.Parameter(torch.zeros(cfg.num_experts, cfg.r, in_f, dtype=base.weight.dtype))  # (E, r, in)
        self.B = nn.Parameter(torch.zeros(cfg.num_experts, out_f, cfg.r, dtype=base.weight.dtype))  # (E, out, r)
        # Init per LoRA paper
        nn.init.kaiming_uniform_(self.A, a=math.sqrt(5))
        nn.init.zeros_(self.B)

        # Router (token‑wise gating)
        self.track_router_stats = False
        self.router = nn.Linear(in_f, cfg.num_experts, bias=False, dtype=base.weight.dtype)
        nn.init.kaiming_uniform_(self.router.weight, a=math.sqrt(5))

        self.dropout = nn.Dropout(cfg.dropout) if cfg.dropout > 0.0 else nn.Identity()

        # expose merge flag similar to LoRA
        self._merged: bool = False

        self._last_router_logits = None
        self._last_gates = None

    # ---------------------------------------------------------
    # Expose base weight param
    # ---------------------------------------------------------
    def _fill_cache(self, logits: torch.Tensor, gates: torch.Tensor, detach: bool = False):
        """DDP/ckpt 안전하게 저장: graph와 분리된 텐서만 보관."""
        if detach:
            self._last_router_logits = logits.detach().float()
            self._last_gates = gates.detach().float()
        else:
            self._last_router_logits = logits
            self._last_gates = gates

    def get_router_tensor(self) -> Tuple[torch.Tensor, torch.Tensor]:
        return self._last_router_logits, self._last_gates

    def clear_cache(self):
        self._last_router_logits = None
        self._last_gates = None

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # type: ignore
        """Assumes input shape (..., in_features).  Router acts on last dim."""
        base_out = self.base(x)
        x_dp = self.dropout(x)

        # Router logits + gates
        router_logits = self.router(x_dp)  # (..., E)
        gates = torch.softmax(router_logits, dim=-1)  # (..., E)

        self._fill_cache(router_logits, gates, detach=self.track_router_stats)

        gates = self._mask_gates(gates)
        lora_mix = self._compute_lora_mix(x_dp, gates)

        return base_out + lora_mix

    def _mask_gates(self, gates: torch.Tensor)-> torch.Tensor:
        if self.cfg.routing == "top1":
            _, top_idx = torch.topk(gates, k=1, dim=-1)  # (..., 1)
            mask = torch.zeros_like(gates).scatter_(-1, top_idx, 1.0)
            gates = mask

        elif self.cfg.routing == "top2":
            top_vals, top_idx = torch.topk(gates, k=2, dim=-1)  # (..., 2)
            mask = torch.zeros_like(gates).scatter_(-1, top_idx, top_vals)
            gates = mask / (mask.sum(dim=-1, keepdim=True) + 1e-9)

        return gates

    def _compute_lora_mix(self, x: torch.Tensor, gates: torch.Tensor) -> torch.Tensor:
        # Flatten leading dims for batched matmul
        leading_shape = x.shape[:-1]  # (...)
        in_f = self.base.in_features

        A_t = self.A.transpose(-1, 1)  # (E, in, r)
        B_t = self.B.transpose(-1, 1)  # (E, r, out)

        # (N, 1, in) × (E, in, r) -> (N, E, r)
        proj_r = torch.matmul(x.unsqueeze(1), A_t.unsqueeze(0))  # (B, E, S, r)

        # (N, E, r) × (E, r, out) -> (N, E, out)
        lora_out = torch.matmul(proj_r, B_t.unsqueeze(0))  # (B, E, S, out)

        # Restore leading shape
        out_f = self.base.out_features
        lora_out = lora_out.transpose(1, 2).reshape(*leading_shape, self.cfg.num_experts, out_f)

        # Weighted sum over experts without einsum: (..., E, out)
        weighted = (gates * self.cfg.scale).unsqueeze(-1) * lora_out  # (..., E, out)
        lora_mix = weighted.sum(dim=-2)

        return lora_mix
