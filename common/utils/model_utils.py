from __future__ import annotations

import torch.nn as nn

__all__ = ["compute_param_norm"]

def compute_param_norm(model: nn.Module, only_trainable: bool = True) -> float:
    """Return global L2 norm of model parameters.

    Args:
        model: PyTorch module.
        only_trainable: If True, consider parameters with ``requires_grad`` only.
    """
    total = 0.0
    for p in model.parameters():
        if (not only_trainable) or p.requires_grad:
            total += p.detach().pow(2).sum().item()
    return total ** 0.5 