from __future__ import annotations

from pathlib import Path
from typing import Tuple, Set

import torch
import safetensors.torch as sft

__all__ = [
    "get_adapter_param_names",
    "collect_adapter_state_dict",
    "save_adapters",
    "load_adapters",
]


def get_adapter_param_names(model: torch.nn.Module) -> Set[str]:
    """Return the set of parameter names that belong to adapters (recorded during injection)."""
    return set(getattr(model, "_adapter_param_names", set()))


def collect_adapter_state_dict(model: torch.nn.Module) -> dict[str, torch.Tensor]:
    """Gather a state_dict containing only adapter parameters."""
    names = get_adapter_param_names(model)
    if not names:
        raise ValueError("Model has no registered adapter parameters. Did you inject adapters?")
    full_state = model.state_dict()
    return {k: v.detach().cpu() for k, v in full_state.items() if k in names}


def save_adapters(model: torch.nn.Module, save_path: str | Path) -> None:
    """Save only adapter weights to *save_path* (.safetensors)."""
    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    state = collect_adapter_state_dict(model)
    sft.save_file(state, str(save_path))


def load_adapters(
    model: torch.nn.Module,
    adapters_file: str | Path,
    *,
    device: str | torch.device = "cpu",
) -> Tuple[list[str], list[str]]:
    """Load adapter weights from *adapters_file* into *model*.

    Returns missing_keys, unexpected_keys from `load_state_dict` for inspection.
    """
    adapters_file = Path(adapters_file)
    if not adapters_file.exists():
        raise FileNotFoundError(adapters_file)
    state = sft.load_file(str(adapters_file), device=str(device))
    res = model.load_state_dict(state, strict=False)
    return res, model