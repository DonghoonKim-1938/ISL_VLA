from __future__ import annotations

import math
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    import bitsandbytes as bnb  # optional for qadalora
except Exception:
    bnb = None


def _dtype_map(dtype: str) -> torch.dtype:
    return {
        "torch.float16": torch.float16,
        "torch.bfloat16": torch.bfloat16,
        "torch.float32": torch.float32,
        "torch.uint8": torch.uint8,
    }[dtype]


@dataclass
class AdaLoraConfig:
    layer_type: str = "adalora"

    r_max: int = 16
    r_min: int = 1
    init_r: int = 8
    target_rank: int = 8  # average target rank across layers

    alpha: int = 16
    dropout: float = 0.05
    fan_in_fan_out: bool = False
    quantize: bool = False

    # allocation schedule
    warmup_steps: int = 1000
    alloc_start_step: int = 1000
    alloc_end_step: int = 5000
    alloc_interval: int = 100

    # importance estimation
    importance_ema_decay: float = 0.95
    importance_weight: float = 0.1
    beta1: float = 0.85
    beta2: float = 0.85

    # regularization
    orth_reg_weight: float = 0.0

    # quantization
    quant_type: str = 'fp4'
    compute_dtype_: str = 'torch.bfloat16'
    compress_statistics: bool = False
    quant_storage_: str = 'torch.uint8'

    @property
    def compute_dtype(self) -> torch.dtype:
        return _dtype_map(self.compute_dtype_)

    @property
    def quant_storage(self) -> torch.dtype:
        return _dtype_map(self.quant_storage_)


class AdaLoraLinear(nn.Module):
    def __init__(self, base: nn.Linear, cfg: AdaLoraConfig):
        super().__init__()
        if not isinstance(base, nn.Linear):
            raise TypeError("AdaLoraLinear expects an nn.Linear to wrap")

        self.cfg = cfg
        self._load_base(base, cfg.quantize)

        in_f, out_f = self.base.in_features, self.base.out_features
        if cfg.fan_in_fan_out:
            in_f, out_f = out_f, in_f

        self.U = nn.Parameter(torch.zeros(out_f, cfg.r_max, dtype=self.base.weight.dtype))
        self.V = nn.Parameter(torch.zeros(cfg.r_max, in_f, dtype=self.base.weight.dtype))
        self.s = nn.Parameter(torch.zeros(cfg.r_max, dtype=self.base.weight.dtype))

        self.register_buffer('mask', torch.zeros(cfg.r_max, dtype=torch.bool))
        self.mask[: max(cfg.r_min, min(cfg.init_r, cfg.r_max))] = True

        nn.init.kaiming_uniform_(self.U, a=math.sqrt(5))
        nn.init.kaiming_uniform_(self.V, a=math.sqrt(5))
        nn.init.zeros_(self.s)

        self.register_buffer('fisher_U', torch.zeros_like(self.U))
        self.register_buffer('fisher_V', torch.zeros_like(self.V))
        self.register_buffer('fisher_s', torch.zeros_like(self.s))

        self.dropout = nn.Dropout(cfg.dropout) if cfg.dropout > 0.0 else nn.Identity()
        self._merged: bool = False
        self._step_count = 0

        self.A = None
        self.B = None

    def extra_repr(self) -> str:
        return (
            f"in_features={self.base.in_features}, out_features={self.base.out_features}, "
            f"r_max={self.cfg.r_max}, r_eff={self.effective_rank}, alpha={self.cfg.alpha}"
        )

    @property
    def effective_rank(self) -> int:
        return int(self.mask.sum().item())

    @property
    def weight(self):
        """Alias to underlying base layer's weight parameter (read-only)."""
        return self.base.weight

    def _load_base(self, base: nn.Linear, quantize: bool):
        if quantize:
            assert bnb is not None, "bitsandbytes required for quantized AdaLoRA"
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
        else:
            self.base = base
        self.base.weight.requires_grad = False

    def compute_importance_scores(self) -> torch.Tensor:
        fisher_U_mean = self.fisher_U.mean(dim=0)
        fisher_V_mean = self.fisher_V.mean(dim=1)
        return (self.s ** 2) * (fisher_U_mean + fisher_V_mean + self.cfg.importance_weight * self.fisher_s)

    @torch.no_grad()
    def update_importance(self) -> None:
        if not self.training:
            return
        d = self.cfg.importance_ema_decay
        if self.U.grad is not None:
            self.fisher_U.mul_(d).add_((1 - d) * (self.U.grad ** 2))
        if self.V.grad is not None:
            self.fisher_V.mul_(d).add_((1 - d) * (self.V.grad ** 2))
        if self.s.grad is not None:
            self.fisher_s.mul_(d).add_((1 - d) * (self.s.grad ** 2))

    @torch.no_grad()
    def reallocate_rank(self, num_activate: int, num_prune: int) -> None:
        # prune lowest among active
        if num_prune > 0 and self.effective_rank > self.cfg.r_min:
            scores = self.compute_importance_scores()
            active_idx = torch.where(self.mask)[0]
            if len(active_idx) > self.cfg.r_min:
                k = min(num_prune, len(active_idx) - self.cfg.r_min)
                _, local = scores[active_idx].topk(k, largest=False)
                self.mask[active_idx[local]] = False
        # activate highest among inactive
        if num_activate > 0:
            scores = self.compute_importance_scores()
            inactive_idx = torch.where(~self.mask)[0]
            if len(inactive_idx) > 0:
                k = min(num_activate, len(inactive_idx))
                _, local = scores[inactive_idx].topk(k, largest=True)
                sel = inactive_idx[local]
                nn.init.kaiming_uniform_(self.U[:, sel], a=math.sqrt(5))
                nn.init.kaiming_uniform_(self.V[sel, :], a=math.sqrt(5))
                nn.init.zeros_(self.s[sel])
                self.fisher_U[:, sel] = 0.0
                self.fisher_V[sel, :] = 0.0
                self.fisher_s[sel] = 0.0
                self.mask[sel] = True

    def step(self) -> None:
        self._step_count += 1

    def should_reallocate(self, step: int) -> bool:
        return (
            step >= self.cfg.alloc_start_step and step <= self.cfg.alloc_end_step and step % self.cfg.alloc_interval == 0
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        base_out = self.base(x)
        x_dp = self.dropout(x)
        active_s = self.s * self.mask.float()
        proj = F.linear(x_dp, self.V)  # (..., r_max)
        proj = proj * active_s.unsqueeze(0)
        # Ensure dtype consistency
        proj = proj.to(dtype=self.U.dtype)
        out = F.linear(proj, self.U)
        scale = self.cfg.alpha / max(1, self.effective_rank)
        return base_out + out * scale

    def finalize(self) -> None:
        if self.effective_rank == 0:
            return
        active = self.mask
        self.U = nn.Parameter(self.U[:, active])
        self.V = nn.Parameter(self.V[active, :])
        self.s = nn.Parameter(self.s[active])
        self.A = nn.Parameter(torch.diag(self.s) @ self.V)
        self.B = nn.Parameter(self.U)
        del self.mask, self.fisher_U, self.fisher_V, self.fisher_s
