import torch
import torch.nn as nn
from typing import Tuple

# ---------- ① Load‑Balancing Loss (Switch 스타일) ----------
def router_balance_loss(
    model: nn.Module,
    *,
    coeff: float = 0.01,
) -> Tuple[torch.Tensor, int]:
    """Switch‑style load‑balancing loss = N · (f·p)."""
    losses = []
    for module in model.modules():
        if isinstance(module, MoELoRALinear) and hasattr(module, "_last_gates"):
            gates = module._last_gates                      # (..., E) softmax 확률
            E = gates.shape[-1]

            # p_j : 확률 평균
            p = gates.mean(dim=tuple(range(gates.dim() - 1)))     # (E,)

            # f_j : 실제 토큰 분포 (hard one‑hot)
            hard = torch.nn.functional.one_hot(gates.argmax(-1), E)
            f = hard.float().mean(dim=tuple(range(hard.dim() - 1)))

            loss = (f * p).sum() * E                            # N·(f·p)
            losses.append(loss)

    if not losses:
        device = next(model.parameters()).device
        return torch.tensor(0.0, device=device), 0

    return coeff * torch.stack(losses).mean(), len(losses)


# ---------- ② Router Z‑Loss ----------
def router_z_loss(
    model: nn.Module,
    *,
    coeff: float = 1e-3,
) -> Tuple[torch.Tensor, int]:
    r"""Penalize large log‑sum‑exp of router logits for numerical stability.

    \mathcal L_z = coeff · mean( logsumexp(logits, dim=-1)^2 )
    """
    losses = []
    for module in model.modules():
        # 라우터가 최근 step의 'pre‑softmax logits'를 따로 저장해 두었다고 가정
        if isinstance(module, MoELoRALinear):
            if hasattr(module, "_last_router_logits"):
                logits = module._last_router_logits            # (..., E)
            elif hasattr(module, "_last_gates"):
                # fallback: log(prob) 로 근사 (정확하지 않지만 없는 것보단 낫다)
                logits = (module._last_gates.clamp_min(1e-9)).log()
            else:
                continue

            z = torch.logsumexp(logits, dim=-1)                # (...)
            loss = (z ** 2).mean()
            losses.append(loss)

    if not losses:
        device = next(model.parameters()).device
        return torch.tensor(0.0, device=device), 0

    return coeff * torch.stack(losses).mean(), len(losses)


# ---------- ③ 두 보조 손실을 합치는 헬퍼 ----------
def moe_aux_loss(
    model: nn.Module,
    *,
    lb_coeff: float = 0.01,
    z_coeff: float = 1e-3,
) -> torch.Tensor:
    lb_loss, _ = router_balance_loss(model, coeff=lb_coeff)
    z_loss, _  = router_z_loss(model,      coeff=z_coeff)
    return lb_loss + z_loss