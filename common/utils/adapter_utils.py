from __future__ import annotations

import gc
from pathlib import Path
from typing import Tuple, Set, Iterable, Callable, List, Optional, Type

import torch
import torch.nn as nn
import safetensors.torch as sft

from common.policies.lora_ada import LoraADALinear
from common.policies.lora import LoraConfig, LoraLinear
from common.policies.lora_moe import LoraMoELinear

__all__ = [
    "match_name",
    "get_parent",
    "get_adapter_param_names",
    "collect_adapter_state_dict",
    "save_adapters",
    "load_adapters",
    "load_adapters_as_expert"
]

from common.policies.lora_msp import LoraMSPLinear
from common.policies.pretrained import PreTrainedPolicy


def match_name(name: str, keywords: Iterable[str]) -> bool:
    return any(k in name for k in keywords)


def get_parent(root: nn.Module, full_name: str) -> Tuple[nn.Module, str]:
    parts = full_name.split(".")
    for p in parts[:-1]:
        root = getattr(root, p)
    return root, parts[-1]


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
) -> Tuple[list[str], nn.Module]:
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
        adapter_files: List[str | Path],
        device: str | torch.device = "cpu",
) -> Optional[List[str]]:
    res = []
    for expert_id, adapter_file in enumerate(adapter_files):
        # Load pretrained LoRA adapter state
        state = sft.load_file(str(adapter_file), device=str(device))

        replaced = 0
        missing_keys = []
        used_keys = []

        for name, module in model.named_modules():
           if isinstance(module, LoraLinear):
               missing, found = module.load_adapter_as_expert(name, state, expert_id)
               missing_keys += missing
               if found:
                   used_keys += [f"{name}.A", f"{name}.B"]
                   replaced += 1

        unexpected_keys = [k for k in state.keys() if k not in used_keys]

        if missing_keys:
            print(f"[WARN] Missing keys: {missing_keys}")
        if unexpected_keys:
            print(f"[WARN] Unexpected keys: {unexpected_keys}")

        if replaced == 0:
            raise Exception("No matching LoRA modules found in state_dict!")
        else:
            res += [f"[INFO] Successfully injected LoRA into {replaced} LoraLinear layers in {adapter_file}."]

    gc.collect()
    if device != torch.device('cpu'):
        torch.cuda.empty_cache()

    return res


def inject_adapters(
    model: PreTrainedPolicy,
    cfg: LoraConfig | None = None,
    target_keywords: Iterable[str] | None = None,
    filter_fn: Callable[[str, nn.Module], bool] | None = None,
) -> Tuple[nn.Module, List[str]]:
    assert isinstance(cfg, LoraConfig)
    device = model.device

    def _map_lora(layer_type: str) -> Type[LoraLinear]:
        if layer_type == "lora":
            return LoraLinear
        elif layer_type == "lora_moe":
            return LoraMoELinear
        elif layer_type == "lora_msp":
            return LoraMSPLinear
        elif layer_type == "lora_ada":
            return LoraADALinear
        else:
            raise ValueError(f"Unknown adapter type: {layer_type}")

    layer_cls = _map_lora(cfg.layer_type)

    # Special-case keyword to adapt every linear layer regardless of name
    if target_keywords and (
            (isinstance(target_keywords, (list, tuple, set)) and "all-linear" in target_keywords)
            or (isinstance(target_keywords, str) and target_keywords == "all-linear")
    ):
        target_keywords = None

    wrapped = _inject_layers(
        model=model,
        cfg=cfg,
        layer_cls=layer_cls,
        target_keywords=target_keywords,
        filter_fn=filter_fn,
    )

    adapter_names = get_adapter_names(layer_cls)

    # Keep track of adapter parameter names for lightweight checkpointing
    if wrapped:
        adapter_param_names = []
        for w in wrapped:
            for adapter_name in adapter_names:
                adapter_param_names.append(f"{w}.{adapter_name}")
        existing = getattr(model, "_adapter_param_names", set())
        model._adapter_param_names = set(existing).union(adapter_param_names)

    if not wrapped:
        raise RuntimeError("No linear layers matched for LoRA injection.")

    gc.collect()
    if device != torch.device('cpu'):
        torch.cuda.empty_cache()

    return model, wrapped

def _inject_layers(
    model: PreTrainedPolicy,
    cfg: LoraConfig,
    layer_cls: Type[LoraLinear],
    target_keywords: Iterable[str] | None = None,
    filter_fn: Callable[[str, nn.Module], bool] | None = None,
):
    wrapped = []

    for name, module in model.named_modules():
        if not isinstance(module, nn.Linear):
            continue
        if target_keywords and not match_name(name, target_keywords):
            continue
        if filter_fn and not filter_fn(name, module):
            continue

        parent, attr = get_parent(model, name)
        lora_layer = layer_cls(module, cfg)
        setattr(parent, attr, lora_layer)
        wrapped.append(name)

    return wrapped

def load_adapters_as_moe(
    model: PreTrainedPolicy,
    adapter_file_path: List[str],
    device: str | torch.device = "cpu",
):
    adapter_file = adapter_file_path[0]
    state = sft.load_file(str(adapter_file), device=str(device))

    res = []

    replaced = 0
    missing_keys = []
    used_keys = []

    for name, module in model.named_modules():
       if isinstance(module, LoraLinear):
           missing, found = module.load_adapter_as_moe(name, state)
           missing_keys += missing
           if found:
               used_keys += [f"{name}.A", f"{name}.B", f"{name}.router.weight"]
               replaced += 1

    unexpected_keys = [k for k in state.keys() if k not in used_keys]

    if missing_keys:
        print(f"[WARN] Missing keys: {missing_keys}")
    if unexpected_keys:
        print(f"[WARN] Unexpected keys: {unexpected_keys}")

    if replaced == 0:
        raise Exception("No matching LoRA modules found in state_dict!")
    else:
        res += [f"[INFO] Successfully injected LoRA into {replaced} LoraLinear layers in {adapter_file}."]

    gc.collect()
    if device != torch.device('cpu'):
        torch.cuda.empty_cache()

    return res

def get_adapter_names(layer_cls: Type[LoraLinear]) -> List[str]:
    if layer_cls is LoraLinear:
        return ["A", "B"]
    elif layer_cls is LoraMSPLinear:
        return ["A", "B", "router.weight","router_proj"]
    elif layer_cls is LoraMoELinear:
        return ["A", "B", "router.weight"]
    elif layer_cls is LoraADALinear:
        return ["A", "B", "E", "rank_num"]
    else:
        raise TypeError(f"Unknown layer class {layer_cls}")