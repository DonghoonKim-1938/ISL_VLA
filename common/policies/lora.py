from __future__ import annotations

import math
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F

import bitsandbytes as bnb

# ---------------------------------------------------------------------------
# Config & Utilities
# ---------------------------------------------------------------------------

def _dtype_map(dtype: str) -> torch.dtype:
    dtype_map = {
        "torch.float16": torch.float16,
        "torch.bfloat16": torch.bfloat16,
        "torch.float32": torch.float32,
        "torch.uint8": torch.uint8,
    }
    return dtype_map[dtype]


@dataclass
class LoraConfig:
    layer_type: str = "lora"

    r: int = 8  # low-rank dimension
    alpha: int = 16  # scaling factor
    dropout: float = 0.05  # dropout on input features
    fan_in_fan_out: bool = False  # set True if base weight is transposed
    quantize : bool = False

    quant_type: str = 'fp4'
    compute_dtype_: str = 'torch.bfloat16'
    compress_statistics: bool = False
    quant_storage_: str = 'torch.uint8'

    @property
    def scale(self) -> float:
        return self.alpha / self.r

    @property
    def compute_dtype(self) -> torch.dtype:
        return _dtype_map(self.compute_dtype_)

    @property
    def quant_storage(self) -> torch.dtype:
        return _dtype_map(self.quant_storage_)

# ---------------------------------------------------------------------------
# LoRA Linear
# ---------------------------------------------------------------------------

class LoraLinear(nn.Module):
    """A `nn.Linear` wrapped with a LoRA adapter (single expert).

    Args:
        base (nn.Linear): The frozen base projection.
        cfg (LoraConfig): Hyper-parameters for the adapter.
    """

    def __init__(self, base: nn.Linear, cfg: LoraConfig):
        super().__init__()
        if not isinstance(base, nn.Linear):
            raise TypeError("LoRALinear expects an nn.Linear to wrap")

        self.cfg = cfg
        self._load_base(base, cfg.quantize)

        in_f, out_f = self.base.in_features, self.base.out_features
        if cfg.fan_in_fan_out:
            in_f, out_f = out_f, in_f

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

    def _load_base(self, base: nn.Linear, quantize: bool):
        if quantize:
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