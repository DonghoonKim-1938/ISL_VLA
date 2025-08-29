from dataclasses import dataclass
from typing import Optional, Dict, Callable, Iterable, Tuple, List

import math

import torch
import torch.distributed as dist
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

        self.register_buffer("router_logit_ma", torch.zeros(self.router.weight.shape[0], dtype=self.router.weight.dtype, device=self.router.weight.device))
        self.momentum = 0.99
        self.temperature = 0.07

        self.id_Sigma = nn.Parameter(
            torch.eye(cfg.num_experts * cfg.r, dtype=base.weight.dtype),
            requires_grad=False
        )

        self.dropout = nn.Dropout(cfg.dropout) if cfg.dropout > 0.0 else nn.Identity()

        # expose merge flag similar to LoRA
        self._merged: bool = False

        self._last_router_logits = None
        self._last_gates = None
        self._last_res = None
        self._last_id_reg = None

    def _fill_cache(self, logits: torch.Tensor, gates: torch.Tensor, res:torch.Tensor, id_reg: torch.Tensor, detach: bool = False):
        """DDP/ckpt 안전하게 저장: graph와 분리된 텐서만 보관."""
        if detach:
            self._last_router_logits = logits.detach().float()
            self._last_gates = gates.detach().float()
            self._last_res = res.detach().float()
            self._last_id_reg = id_reg.detach().float()
        else:
            self._last_router_logits = logits
            self._last_gates = gates
            self._last_res = res
            self._last_id_reg = id_reg

    def clear_cache(self):
        self._last_router_logits = None
        self._last_gates = None
        self._last_res = None
        self._last_id_reg = None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.to(dtype=self.A.dtype)
        base_out = self.base(x)
        x_dp = self.dropout(x)

        # Router logits + gates
        router_logits = self.router(x_dp) - self.router_logit_ma
        router_logits = F.gelu(router_logits / self.temperature)  # (..., E)
        gates = torch.softmax(router_logits, dim=-1)

        Sigma = torch.diag_embed(gates)

        leading_shape = x.shape[:-1]  # (...)
        A_t = self.A.transpose(-1, 0)
        B_t = self.B.transpose(-1, 0)

        proj_r = torch.matmul(x, A_t)  # (B, S, r)
        proj_r = torch.matmul(proj_r.unsqueeze(-2), Sigma)  # (B, S, r)
        lora_out = torch.matmul(proj_r.squeeze(-2), B_t)  # (B, S, out)

        proj_r_teacher = torch.matmul(x, A_t)
        lora_out_teacher = torch.matmul(proj_r_teacher.squeeze(-2), B_t)

        lora_out = lora_out * self.cfg.scale

        residual = lora_out - lora_out_teacher
        id_reg = torch.norm(torch.matmul(self.A, A_t) - self.id_Sigma) + torch.norm(torch.matmul(B_t, self.B) - self.id_Sigma)
        id_reg = torch.clamp(id_reg, max=10.0)

        self._fill_cache(router_logits, gates, residual, id_reg, detach=self.track_router_stats)
        self.update_router_ema(router_logits)
        return base_out + lora_out

    def compute_balance_loss(self) -> torch.Tensor:
        if self._last_gates is not None:
            E = self._last_gates.shape[-1]
        else:
            return

        # p_j : 확률 평균
        p = self._last_gates.mean(dim=tuple(range(self._last_gates.dim() - 1)))  # (E,)

        # f_j : 실제 토큰 분포 (hard one‑hot)
        hard = F.one_hot(self._last_gates.argmax(-1), E)
        f = hard.float().mean(dim=tuple(range(hard.dim() - 1)))

        loss = (f * p).sum() * E  # N·(f·p)

        return loss

    def compute_z_loss(self) -> torch.Tensor:
        if self._last_router_logits is not None:
            logits = self._last_router_logits
        elif self._last_gates is not None:
            logits = self._last_gates.clamp_min(1e-9).log()
        else:
            return

        z = torch.logsumexp(logits, dim=-1)  # (...)
        z = torch.clamp(z, max=10.0, min=-1.0)
        # loss = (z ** 2).mean()
        loss = torch.log1p(z).mean()

        return loss

    def compute_spec_loss(self, ground_rank: int, target_rank: int) -> torch.Tensor:
        if self._last_router_logits is None:
            return torch.tensor(0.0, dtype=self.A.dtype, device=self.A.device)

        ground_vals, _ = torch.topk(self._last_router_logits, ground_rank)
        target_vals, _ = torch.topk(self._last_router_logits, target_rank)

        denom = (target_vals ** 2).sum()
        num = (ground_vals ** 2).sum()

        E = num / (denom+1e-9)
        return 1 - E

    def compute_mod_loss(self, weight: torch.Tensor = 1) -> torch.Tensor:
        return torch.norm(self._last_res) if self._last_res is not None else torch.tensor(0.0, dtype=self.A.dtype, device=self.A.device)

    def compute_id_loss(self) -> torch.Tensor:
        return self._last_id_reg if self._last_id_reg is not None else torch.tensor(0.0, dtype=self.A.dtype, device=self.A.device)

    # ---------------------------------------------------------
    # External update of router logits moving average
    # ---------------------------------------------------------
    @torch.no_grad()
    def update_router_ema(self, router_logits: torch.Tensor):

        batch_sum = router_logits.sum(dim=(0,1))

        if dist.is_available() and dist.is_initialized():
            dist.all_reduce(batch_sum)
            world_size = dist.get_world_size()
        else:
            world_size = 1

        batch_mean = batch_sum / (router_logits.size(0) * world_size)

        self.router_logit_ma.mul_(self.momentum).add_(batch_mean * (1 - self.momentum))

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
        self.router.weight.requires_grad_(True)

        return missing, found
