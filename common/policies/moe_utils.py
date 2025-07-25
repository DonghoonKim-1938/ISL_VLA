from __future__ import annotations

from typing import Tuple

import torch
import torch.nn.functional as F
import torch.nn as nn

from common.policies.lora_moe import MoELoRALinear

__all__ = ["router_balance_loss"]

def router_balance_loss(model: nn.Module, *, coeff: float = 0.01):
    """Switch‑style load‑balancing loss."""
    losses = []
    for module in model.modules():
        if isinstance(module, MoELoRALinear) and hasattr(module, "_last_gates"):
            gates = module._last_gates         # shape: (..., E), softmax 확률
            E = gates.shape[-1]

            # p_j : 확률 평균
            p = gates.mean(dim=tuple(range(gates.dim() - 1)))          # (E)

            # f_j : 실제 토큰 분포 (one‑hot 근사)
            # hard assignment가 저장돼 있지 않다면, soft 선택 합으로 근사해도 OK
            hard = torch.nn.functional.one_hot(gates.argmax(-1), E)    # (..., E)
            f = hard.float().mean(dim=tuple(range(hard.dim() - 1)))    # (E)
            # --- load‑balancing ---
            loss = (f * p).sum() * E          # N · (f·p)
            losses.append(loss)

    if not losses:
        device = next(model.parameters()).device
        return torch.tensor(0.0, device=device), 0

    return coeff * torch.stack(losses).mean(), len(losses)
