from __future__ import annotations

"""Prefix-Tuning via past_key_values for HuggingFace Transformer models.

This implementation prepends learnable key/value vectors to each transformer's
`past_key_value` cache, making it compatible with rotary or other custom
attention mechanisms without wrapping individual attention modules.
"""

from dataclasses import dataclass
from typing import Iterable, List, Tuple

import torch
import torch.nn as nn

__all__ = ["PrefixTuningConfig", "inject_prefix_tuning"]


@dataclass
class PrefixTuningConfig:
    num_virtual_tokens: int = 16  # prefix length
    init_std: float = 0.02        # normal init std


class _PrefixEncoder(nn.Module):
    """Layer-wise learnable prefix key/value tensors."""

    def __init__(self, n_layers: int, n_heads: int, head_dim: int, cfg: PrefixTuningConfig):
        super().__init__()
        self.n_tokens = cfg.num_virtual_tokens
        shape = (n_layers, self.n_tokens, n_heads, head_dim)  # (L, T, H, D)
        self.key = nn.Parameter(torch.randn(shape) * cfg.init_std)
        self.value = nn.Parameter(torch.randn_like(self.key))

    def forward(self, layer_idx: int, bsz: int) -> Tuple[torch.Tensor, torch.Tensor]:
        # (T, H, D) → (B, H, T, D)
        k = self.key[layer_idx].unsqueeze(0).expand(bsz, -1, -1, -1)  # (B,T,H,D)
        v = self.value[layer_idx].unsqueeze(0).expand_as(k)
        k = k.transpose(1, 2)  # (B,H,T,D)
        v = v.transpose(1, 2)
        return k.contiguous(), v.contiguous()


# ---------------------------------------------------------
# helpers
# ---------------------------------------------------------

def _find_attention_modules(model: nn.Module) -> List[nn.Module]:
    attn_layers = []
    for m in model.modules():
        if hasattr(m, "num_heads") and hasattr(m, "bias_k"):
            attn_layers.append(m)
    return attn_layers


# ---------------------------------------------------------
# injection
# ---------------------------------------------------------

def inject_prefix_tuning(
    model: nn.Module,
    cfg: PrefixTuningConfig | None = None,
    *,
    target_keywords: Iterable[str] | None = None,  # kept for API compat, unused
) -> Tuple[nn.Module, List[str]]:
    cfg = cfg or PrefixTuningConfig()

    attn_layers = _find_attention_modules(model)
    if not attn_layers:
        raise RuntimeError("No attention layers found for prefix-tuning injection.")

    sample = attn_layers[0]
    n_layers = len(attn_layers)
    n_heads = sample.num_heads
    head_dim = sample.head_dim if hasattr(sample, "head_dim") else sample.head_dim_size

    prefix_enc = _PrefixEncoder(n_layers, n_heads, head_dim, cfg)
    model.add_module("prefix_encoder", prefix_enc)

    # register parameters for freezing / saving
    adapter_names = {f"prefix_encoder.{n}" for n, _ in prefix_enc.named_parameters()}
    model._adapter_param_names = set(getattr(model, "_adapter_param_names", set())).union(adapter_names)

    injected = []
    for idx, attn in enumerate(attn_layers):

        def _pre_hook(module, args, kwargs, layer_idx=idx):  # type: ignore
            hidden_states = args[0] if len(args) else kwargs["hidden_states"]
            bsz = hidden_states.size(0)
            k, v = prefix_enc(layer_idx, bsz)

            past_key = past_val = None
            key_name = "past_key_value" if "past_key_value" in kwargs else "layer_past" if "layer_past" in kwargs else None
            if key_name and kwargs[key_name] is not None:
                past_key, past_val = kwargs[key_name]

            if past_key is None:
                new_past = (k, v)
            else:
                new_past = (torch.cat([k, past_key], dim=2), torch.cat([v, past_val], dim=2))

            if key_name:
                kwargs[key_name] = new_past  # type: ignore
            else:
                kwargs["past_key_value"] = new_past  # default
            return args, kwargs

        attn.register_forward_pre_hook(_pre_hook, with_kwargs=True)
        injected.append(f"{attn.__class__.__name__}_{idx}")

    return model, injected 