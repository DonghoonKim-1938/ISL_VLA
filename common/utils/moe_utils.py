import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple

from common.policies.lora_moe import LoraMoELinear

# ---------- ① Load‑Balancing Loss (Switch 스타일) ----------
def compute_balance_loss(
    gates: torch.Tensor,
) -> torch.Tensor | None:
    """Switch‑style load‑balancing loss = N · (f·p)."""
    if gates is not None:
        E = gates.shape[-1]
    else:
        return

    # p_j : 확률 평균
    p = gates.mean(dim=tuple(range(gates.dim() - 1)))     # (E,)

    # f_j : 실제 토큰 분포 (hard one‑hot)
    hard = F.one_hot(gates.argmax(-1), E)
    f = hard.float().mean(dim=tuple(range(hard.dim() - 1)))

    loss = (f * p).sum() * E                           # N·(f·p)

    return loss


# ---------- ② Router Z‑Loss ----------
def compute_z_loss(
    logits: torch.Tensor,
    gates: torch.Tensor,
) -> torch.Tensor | None:
    r"""Penalize large log‑sum‑exp of router logits for numerical stability.
    \mathcal L_z = coeff · mean( logsumexp(logits, dim=-1)^2 )
    """
    if logits is not None:
        pass
    elif gates is not None:
        logits = gates.clamp_min(1e-9).log()
    else:
        return

    z = torch.logsumexp(logits, dim=-1)                # (...)
    loss = (z ** 2).mean()

    return loss


def compute_router_loss(
    model: nn.Module,
) -> Tuple[torch.Tensor, torch.Tensor]:
    lb_losses, z_losses = [], []

    for module in model.modules():
        if isinstance(module, LoraMoELinear):
            logits, gates = module.get_router_tensor()            # (..., E)

            lb_loss = compute_balance_loss(gates)
            z_loss = compute_z_loss(logits, gates)

            lb_losses.append(lb_loss) if lb_loss is not None else None
            z_losses.append(z_loss) if z_loss is not None else None

            module.clear_cache()

    lb_loss = torch.stack(lb_losses).mean()
    z_loss = torch.stack(z_losses).mean()

    return lb_loss, z_loss