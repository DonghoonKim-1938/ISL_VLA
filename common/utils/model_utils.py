from __future__ import annotations

import torch.nn as nn

__all__ = ["compute_param_norm", "compute_grad_norm"]

# ---------------------------------------------------------
# Freezing helpers
# ---------------------------------------------------------

def freeze_non_adapters(model: nn.Module) -> None:
    """Freeze all parameters except those recorded in ``_adapter_param_names``."""
    keep: set[str] = set(getattr(model, "_adapter_param_names", set()))
    for n, p in model.named_parameters():
        p.requires_grad = n in keep

# ---------------------------------------------------------
#  Parameter, Gradient norm helper
# ---------------------------------------------------------

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


def compute_grad_norm(model: nn.Module, only_trainable: bool = True) -> float:
    """Return global L2 norm of gradients for given model parameters.

    Args:
        model: PyTorch module.
        only_trainable: If True, consider parameters with ``requires_grad`` only.
    """
    total = 0.0
    for p in model.parameters():
        if ((not only_trainable) or p.requires_grad) and (p.grad is not None):
            total += p.grad.detach().pow(2).sum().item()
    return total ** 0.5 