from dataclasses import dataclass
from typing import Optional, Dict, Callable, Iterable, Tuple, List

import math

import matplotlib.pyplot as plt

import torch
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F

from common.policies.lora import LoraConfig, LoraLinear

@dataclass
class LoraMSPConfig(LoraConfig):
    num_experts: int = 4
    layer_type: str = "lora_msp"
    target_threshold: float = 0.9

    use_spec_loss: bool = False
    use_modular_loss: bool = False
    use_id_loss: bool = False

    router_projection: bool = False
    routing: str = "weighted"   # "weighted", "top1", "top2"

    router_weight_update: str = "vanilla"


class LoraMSPLinear(LoraLinear):
    def __init__(self, base: nn.Linear, cfg: LoraMSPConfig):
        super().__init__(base, cfg)
        if not isinstance(base, nn.Linear):
            raise TypeError("MoELoRALinear expects an nn.Linear to wrap")

        self.cfg = cfg
        self.r = cfg.r
        self._load_base(base, cfg.quantize)
        self.dtype = base.weight.dtype
        for p in self.base.parameters():
            p.requires_grad_(False)

        in_f, out_f = self.base.in_features, self.base.out_features
        if cfg.fan_in_fan_out:
            in_f, out_f = out_f, in_f

        # LoRA expert parameters – grouped tensors for efficiency
        self.A = nn.Parameter(torch.zeros(cfg.num_experts * cfg.r, in_f, dtype=self.dtype))  # (E, r, in)
        self.B = nn.Parameter(torch.zeros(out_f, cfg.num_experts * cfg.r, dtype=self.dtype))  # (E, out, r)

        # Init per LoRA paper
        nn.init.kaiming_uniform_(self.A, a=math.sqrt(5))
        nn.init.zeros_(self.B)

        # Router (token‑wise gating)
        self.track_router_stats = False
        if self.cfg.router_projection:
            self.router = nn.Linear(in_f, cfg.num_experts, bias=False, dtype=self.dtype)
            self.router_proj = nn.Parameter(torch.zeros(cfg.num_experts, cfg.num_experts * cfg.r, dtype=self.dtype))
            self.initialize_router_proj()
        else:
            self.router = nn.Linear(in_f, cfg.num_experts * cfg.r, bias=False, dtype=self.dtype)
        nn.init.kaiming_uniform_(self.router.weight, a=math.sqrt(5))

        self.register_buffer("router_weight_ma", torch.zeros((cfg.num_experts * cfg.r, in_f), dtype=self.dtype, device=self.router.weight.device))
        self.register_buffer("router_logit_ma", torch.zeros(self.router.weight.shape[0], dtype=self.dtype, device=self.router.weight.device))
        self.momentum = 0.95
        self.temperature = 0.07

        self.id_Sigma = nn.Parameter(
            torch.eye(cfg.num_experts * cfg.r, dtype=self.dtype),
            requires_grad=False
        )

        self.dropout = nn.Dropout(cfg.dropout) if cfg.dropout > 0.0 else nn.Identity()

        # expose merge flag similar to LoRA
        self._merged: bool = False

        self._last_router_logits = None
        self._last_gates = None
        self._last_res = None
        self._last_id_reg = None
        self._last_top_counts = None

        self.update_router_weight_ema_flag = True

    def _threshold_mask(self, logits: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        threshold = self.cfg.target_threshold
        sorted_vals, sorted_idx = torch.sort(logits, dim=-1, descending=True)

        total_sq_sum = (sorted_vals ** 2).sum(dim=-1, keepdim=True)  # (batch, seq, 1)
        cumsum_sq = torch.cumsum(sorted_vals ** 2, dim=-1)

        mask_sorted = cumsum_sq <= (threshold * total_sq_sum)
        mask_sorted[..., 0] = True

        mask = torch.zeros_like(mask_sorted, dtype=torch.bool)
        mask.scatter_(-1, sorted_idx, mask_sorted)

        top_counts = mask.sum(dim=-1)

        return mask, top_counts

    def _fill_cache(
            self,
            logits: torch.Tensor,
            gates: torch.Tensor,
            res:torch.Tensor,
            id_reg: torch.Tensor,
            top_counts: torch.Tensor,
            detach: bool = False
    ):
        """DDP/ckpt 안전하게 저장: graph와 분리된 텐서만 보관."""
        if detach:
            self._last_router_logits = logits.detach().float()
            self._last_gates = gates.detach().float()
            self._last_res = res.detach().float()
            self._last_id_reg = id_reg.detach().float()
            self._last_top_counts = top_counts.detach().float()
        else:
            self._last_router_logits = logits
            self._last_gates = gates
            self._last_res = res
            self._last_id_reg = id_reg
            self._last_top_counts = top_counts

    def clear_cache(self):
        self._last_router_logits = None
        self._last_gates = None
        self._last_res = None
        self._last_id_reg = None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.to(dtype=self.dtype)
        base_out = self.base(x)
        x_dp = self.dropout(x)

        # Router logits from MoE router
        router_logits = self.router_(x_dp)

        # Router logits + gates
        # router_logits = self.router_(x_dp) - self.router_logit_ma
        # router_logits_sig = torch.sigmoid(router_logits / self.temperature)  # (..., E)
        # router_logits_sig = F.relu(router_logits / self.temperature)  # (..., E)

        mask, top_counts = self._threshold_mask(router_logits)
        # masked_router_logits = router_logits_sig.masked_fill(~mask, float('-inf'))
        masked_router_logits = router_logits.masked_fill(~mask, 0.0)
        gates = torch.softmax(masked_router_logits, dim=-1)

        self._topk = top_counts

        A_t = self.A.transpose(-1, 0)
        B_t = self.B.transpose(-1, 0)

        proj_r = torch.einsum('bsi,ir,bsr->bsr', x, A_t, masked_router_logits)  # (B, S, r)
        lora_out = torch.einsum('bsr,ro->bso', proj_r, B_t)  # (B, S, out)

        scaled_lora_out = lora_out * self.cfg.scale

        if self.cfg.use_modular_loss:
            proj_r_teacher = torch.matmul(x, A_t)
            lora_out_teacher = torch.matmul(proj_r_teacher.squeeze(-2), B_t)

            residual = lora_out - lora_out_teacher
        else:
            residual = None

        if self.cfg.use_id_loss:
            id_reg = torch.norm(torch.matmul(self.A, A_t) - self.id_Sigma) + torch.norm(torch.matmul(B_t, self.B) - self.id_Sigma)
        else:
            id_reg = None

        self._fill_cache(router_logits, gates, residual, id_reg, top_counts, detach=self.track_router_stats)

        # self.update_router_logit_ema(router_logits)
        if self.update_router_weight_ema_flag:
            self.update_router_weight_ema(self.router.weight)

        return base_out + scaled_lora_out

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

    def _compute_spec_loss(self) -> bool:
        return (not self.cfg.use_spec_loss) or (self._last_router_logits is None)

    def compute_spec_loss(self) -> torch.Tensor:
        if self._compute_spec_loss():
            return torch.tensor(0.0, dtype=self.dtype, device=self.A.device)

        ground_rank = self.cfg.num_experts * self.cfg.r
        target_rank = self._last_top_counts

        ground_vals, target_vals = self._topk_vals(ground_rank, target_rank)

        denom = (target_vals ** 2).sum()
        num = (ground_vals ** 2).sum()

        E = torch.clamp(num / (denom+1e-9), min=0.0, max=1.0)
        return 1 - E

    def _topk_vals(
            self,
            ground_rank: int | torch.Tensor,
            target_rank: int | torch.Tensor,
    ) -> List[torch.Tensor]:
        logits = self._last_router_logits
        B, S, R = logits.shape

        masked_vals = []
        for r in [ground_rank, target_rank]:
            if isinstance(r, int):
                r= torch.full((B, S), r, device=logits.device, dtype=torch.long)

            max_k = r.max().item()  # 전체 중 가장 큰 k

            topk_vals, topk_idx = torch.topk(logits, k=max_k, dim=-1)  # (B, S, max_k)

            arange = torch.arange(max_k, device=logits.device).view(1, 1, -1)  # (1,1,max_k)
            valid_mask = arange < r.unsqueeze(-1)  # (B,S,max_k)

            masked_logits = torch.zeros_like(logits)
            masked_logits.scatter_(-1, topk_idx, topk_vals * valid_mask)

            masked_vals.append(masked_logits)

        return masked_vals

    def compute_mod_loss(self, weight: torch.Tensor = 1) -> torch.Tensor:
        if self.cfg.use_modular_loss:
            return torch.norm(self._last_res) if self._last_res is not None else torch.tensor(0.0, dtype=self.dtype, device=self.A.device)
        else:
            return torch.tensor(0.0, dtype=self.dtype, device=self.A.device)

    def compute_id_loss(self) -> torch.Tensor:
        return self._last_id_reg if self._last_id_reg is not None else torch.tensor(0.0, dtype=self.dtype, device=self.A.device)

    # ---------------------------------------------------------
    # External update of router logits moving average
    # ---------------------------------------------------------
    @torch.no_grad()
    def update_router_logit_ema(self, router_logits: torch.Tensor):

        batch_sum = router_logits.sum(dim=(0,1))

        if dist.is_available() and dist.is_initialized():
            dist.all_reduce(batch_sum)
            world_size = dist.get_world_size()
        else:
            world_size = 1

        batch_mean = batch_sum / (router_logits.size(0) * router_logits.size(1) * world_size)

        self.router_logit_ma.mul_(self.momentum).add_(batch_mean * (1 - self.momentum))

    @torch.no_grad()
    def update_router_weight_ema(self, weight: torch.Tensor):

        batch_mean = self._get_batch_mean(weight)
        batch_sum = weight.sum(dim=(0,1))

        if dist.is_available() and dist.is_initialized():
            dist.all_reduce(batch_sum)
            world_size = dist.get_world_size()
        else:
            world_size = 1

        batch_mean = batch_sum / (weight.size(0) * world_size)

        self.router_weight_ma.mul_(self.momentum).add_(batch_mean * (1 - self.momentum))

    def _get_batch_mean(self, tensor: torch.Tensor) -> torch.Tensor:
        batch_sum = tensor.sum(dim=(0,1))

        if dist.is_available() and dist.is_initialized():
            dist.all_reduce(batch_sum)
            world_size = dist.get_world_size()
        else:
            world_size = 1

        batch_mean = batch_sum / (tensor.size(0) * world_size)

        return batch_mean

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
            with torch.no_grad():
                self.A[expert_bank, :] = state[A_key].to(self.A.device)
                self.B[:, expert_bank] = state[B_key].to(self.A.device)

        self.A.requires_grad_(train_experts)
        self.B.requires_grad_(train_experts)
        self.router.weight.requires_grad_(True)

        return missing, found

    @property
    def top_k(self) -> torch.Tensor:
        return self._topk.to('cpu')

    def initialize_router_proj(self):
        base = torch.eye(self.cfg.num_experts, dtype=self.dtype)
        weight = base.repeat_interleave(self.cfg.r, dim=0)
        self.router_proj = nn.Parameter(weight, requires_grad=True)

    def router_(self, x:torch.Tensor) -> torch.Tensor:
        if self.cfg.router_projection:
            router_logits = self.router(x)
            gates = torch.softmax(router_logits, dim=-1)
            gates = self._mask_gates(gates)
            router_logits = torch.matmul(gates, self.router_proj.transpose(0, 1))
        else:
            router_logits = self.router(x)
        return router_logits

    def load_adapter_as_moe(
            self,
            name: str,
            state: dict[str, torch.Tensor],
            train_experts: bool = True,
            train_routers: bool = True,
    ) -> Tuple[List[str | None], bool]:
        A_key = f"{name}.A"
        B_key = f"{name}.B"
        router_key = f"{name}.router.weight"

        found = True
        missing = []

        if A_key not in state:
            missing.append(A_key)
            found = False
        if B_key not in state:
            missing.append(B_key)
            found = False
        if router_key not in state:
            missing.append(router_key)
            found = False
        if found:
            with torch.no_grad():
                self.A = nn.Parameter(state[A_key].view(-1, state[A_key].size(-1)).to(self.A.device))
                self.B = nn.Parameter(state[B_key].permute(1,0,2).contiguous().view(state[B_key].size(-2), -1).to(self.A.device))
                self.router.weight = nn.Parameter(state[router_key].to(self.A.device))

        self.A.requires_grad_(train_experts)
        self.B.requires_grad_(train_experts)
        self.router.weight.requires_grad_(train_routers)

        return missing, found

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