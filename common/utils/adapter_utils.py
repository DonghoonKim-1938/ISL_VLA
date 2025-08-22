from __future__ import annotations

from pathlib import Path
from typing import Tuple, Set

import torch
import torch.nn as nn
import safetensors.torch as sft

from common.policies.lora_moe import MoELoRALinear

__all__ = [
    "get_adapter_param_names",
    "collect_adapter_state_dict",
    "save_adapters",
    "load_adapters",
    "load_adapters_as_expert"
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


def load_adapters_as_expert(
        model: nn.Module,
        adapters_file: str | Path,
        expert_id: int,
        device: str | torch.device = "cpu",
) -> Tuple[list[str], list[str]]:
    """
    Load a pretrained LoRA adapter (saved as state_dict) into an existing MoE-LoRA model
    by inserting it as one expert (expert_id).

    Args:
        model (nn.Module): Model with injected MoELoRALinear modules.
        adapters_file (str | Path): Path to LoRA adapter weights (.pt or .bin).
        expert_id (int): Index of expert slot to overwrite with LoRA adapter.
    """
    # Load pretrained LoRA adapter state
    state = sft.load_file(str(adapters_file), device=str(device))

    replaced = 0
    missing_keys = []
    unexpected_keys = []

    for name, module in model.named_modules():
        if isinstance(module, MoELoRALinear):
            # Expected keys in LoRA adapter (e.g., from PEFT): 'lora_A.weight', 'lora_B.weight'
            # Match shape to MoE expert slot
            A_key = f"{name}.A"
            B_key = f"{name}.B"

            found = True
            if A_key not in state:
                missing_keys.append(A_key)
                found = False
            if B_key not in state:
                missing_keys.append(B_key)
                found = False

            if found:
                with torch.no_grad():
                    module.A[expert_id].copy_(state[A_key])
                    module.B[expert_id].copy_(state[B_key])
                replaced += 1

                # Freeze expert weights
                module.A.requires_grad_(False)
                module.B.requires_grad_(False)
                module.router.requires_grad_(True)

    # unexpected_keys = state.keys() - actually used keys
    used_keys = {f"{name}.A" for name, m in model.named_modules() if isinstance(m, MoELoRALinear)} | \
                {f"{name}.B" for name, m in model.named_modules() if isinstance(m, MoELoRALinear)}
    unexpected_keys = [k for k in state.keys() if k not in used_keys]

    if missing_keys:
        print(f"[WARN] Missing keys: {missing_keys}")
    if unexpected_keys:
        print(f"[WARN] Unexpected keys: {unexpected_keys}")

    if replaced == 0:
        raise Exception("No matching LoRA modules found in state_dict!")
    else:
        res = f"[INFO] Successfully injected LoRA into {replaced} MoELoRALinear layers."

    return missing_keys, unexpected_keys, res