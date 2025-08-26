from dataclasses import dataclass
from typing import Optional, Dict, Callable, Iterable, Tuple, List

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from common.policies.lora import LoraConfig, LoraLinear


@dataclass
class LoraMSPConfig(LoraConfig):
    num_experts: int = 4
    layer_type: str = "lora_msp"


class LoraMSPLinear(LoraLinear):
    def __init__(self, base: nn.Linear, cfg: LoraMSPConfig):
        super().__init__(base, cfg)
        if not isinstance(base, nn.Linear):
            raise TypeError("MoELoRALinear expects an nn.Linear to wrap")

        self.cfg = cfg
        self.r = cfg.r
        self._load_base(base, cfg.quantize)
        for p in self.base.parameters():
            p.requires_grad_(False)

        in_f, out_f = self.base.in_features, self.base.out_features
        if cfg.fan_in_fan_out:
            in_f, out_f = out_f, in_f

        # LoRA expert parameters – grouped tensors for efficiency
        self.A = nn.Parameter(torch.zeros(cfg.num_experts * cfg.r, in_f, dtype=base.weight.dtype))  # (E, r, in)
        self.B = nn.Parameter(torch.zeros(out_f, cfg.num_experts * cfg.r, dtype=base.weight.dtype))  # (E, out, r)
        # Init per LoRA paper
        nn.init.kaiming_uniform_(self.A, a=math.sqrt(5))
        nn.init.zeros_(self.B)

        # Router (token‑wise gating)
        self.track_router_stats = False
        self.router = nn.Linear(in_f, cfg.num_experts * cfg.r, bias=False, dtype=base.weight.dtype)
        nn.init.kaiming_uniform_(self.router.weight, a=math.sqrt(5))

        self.dropout = nn.Dropout(cfg.dropout) if cfg.dropout > 0.0 else nn.Identity()

        # expose merge flag similar to LoRA
        self._merged: bool = False

        self._last_router_logits = None
        self._last_gates = None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.to(dtype=self.A.dtype)
        base_out = self.base(x)
        x_dp = self.dropout(x)

        # Router logits + gates
        router_logits = self.router(x_dp)  # (..., E)
        router_logits = F.relu(router_logits)  # (..., E)

        Sigma = torch.diag_embed(router_logits)

        leading_shape = x.shape[:-1]  # (...)
        A_t = self.A.transpose(-1, 0)
        B_t = self.B.transpose(-1, 0)

        proj_r = torch.matmul(x, A_t)  # (B, S, r)
        proj_r = torch.matmul(proj_r.unsqueeze(-2), Sigma)  # (B, S, r)
        lora_out = torch.matmul(proj_r.squeeze(-2), B_t)  # (B, S, out)

        lora_out = lora_out * self.cfg.scale

        return base_out + lora_out

    def load_adapter_as_expert(
            self,
            name: str,
            state: dict[str, torch.Tensor],
            expert_id: int,
            train_experts: bool = True,
    ) -> Tuple[List[str | None], bool]:
        expert_bank = range(self.r * expert_id, self.r * (expert_id + 1))

        A_key = f"{name}.A"
        B_key = f"{name}.B"

        found = True
        missing = []

        if A_key not in state:
            missing.append(A_key)
            found = False
        if B_key not in state:
            missing.append(B_key)
            found = False

        if found:
            self.A[expert_bank, :].copy_(state[A_key])
            self.B[:, expert_bank].copy_(state[B_key])

        self.A.requires_grad_(train_experts)
        self.B.requires_grad_(train_experts)
        self.router.requires_grad_(True)

        return missing, found
