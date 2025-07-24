from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Iterable, List, Tuple

import torch
import torch.nn as nn

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------


@dataclass
class PrefixTuningConfig:
    num_virtual_tokens: int = 16  # length of the virtual prefix
    init_std: float = 0.02  # std for normal initialization
    dropout: float = 0.0  # dropout applied to prefix embeddings during training


# ---------------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------------


def _match_name(name: str, keywords: Iterable[str]) -> bool:
    return any(k in name for k in keywords)


def _get_parent(root: nn.Module, full_name: str) -> Tuple[nn.Module, str]:
    parts = full_name.split(".")
    for p in parts[:-1]:
        root = getattr(root, p)
    return root, parts[-1]


# ---------------------------------------------------------------------------
# Prefix-tuned Multi-Head Attention
# ---------------------------------------------------------------------------


class PrefixTunedMHA(nn.Module):
    """A wrapper around `nn.MultiheadAttention` that prepends learnable key/value prefixes.

    The query remains unchanged. Prefixes are concatenated to `key` and `value` along the
    sequence dimension. Works for both `(seq, batch, dim)` and `(batch, seq, dim)` layouts
    depending on the underlying `batch_first` setting of the base module.
    """

    def __init__(self, base: nn.MultiheadAttention, cfg: PrefixTuningConfig):
        super().__init__()
        if not isinstance(base, nn.MultiheadAttention):
            raise TypeError("PrefixTunedMHA expects an nn.MultiheadAttention to wrap")

        self.base = base
        self.cfg = cfg

        embed_dim = base.embed_dim
        num_tokens = cfg.num_virtual_tokens

        # Learnable prefixes (sequence_len, embed_dim)
        # We keep them in seq_len major format; will transpose when `batch_first`.
        self.prefix_key = nn.Parameter(torch.empty(num_tokens, embed_dim))
        self.prefix_value = nn.Parameter(torch.empty(num_tokens, embed_dim))

        nn.init.normal_(self.prefix_key, std=cfg.init_std)
        nn.init.normal_(self.prefix_value, std=cfg.init_std)

        self.dropout = nn.Dropout(cfg.dropout) if cfg.dropout > 0.0 else nn.Identity()

    # NOTE: replicates signature from torch 2.0 MHA but only keeps common args.
    def forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor | None = None,
        value: torch.Tensor | None = None,
        **kwargs,
    ):
        base = self.base
        batch_first = getattr(base, "batch_first", False)

        # Determine key/value default to query if not provided
        if key is None:
            key = query
        if value is None:
            value = query

        if batch_first:
            # (B, S, D)
            batch = query.shape[0]
            # Expand prefix over batch
            p_k = self.prefix_key[None, :, :].expand(batch, -1, -1)
            p_v = self.prefix_value[None, :, :].expand(batch, -1, -1)

            key = torch.cat([p_k, key], dim=1)
            value = torch.cat([p_v, value], dim=1)
        else:
            # (S, B, D)
            batch = query.shape[1]
            p_k = self.prefix_key[:, None, :].expand(-1, batch, -1)
            p_v = self.prefix_value[:, None, :].expand(-1, batch, -1)

            key = torch.cat([p_k, key], dim=0)
            value = torch.cat([p_v, value], dim=0)

        # Optional dropout on prefixes (applied after concat so masking unaffected)
        key = self.dropout(key)
        value = self.dropout(value)

        return base(query, key, value, **kwargs)


# ---------------------------------------------------------------------------
# Injection utility
# ---------------------------------------------------------------------------


def inject_prefix_tuning(
    model: nn.Module,
    cfg: PrefixTuningConfig | None = None,
    *,
    target_keywords: Iterable[str] | None = None,
    filter_fn: Callable[[str, nn.Module], bool] | None = None,
) -> Tuple[nn.Module, List[str]]:
    """Replace matching `nn.MultiheadAttention` layers with `PrefixTunedMHA` (in-place)."""

    cfg = cfg or PrefixTuningConfig()
    wrapped: List[str] = []

    for name, module in model.named_modules():
        if not isinstance(module, nn.MultiheadAttention):
            continue
        if target_keywords and not _match_name(name, target_keywords):
            continue
        if filter_fn and not filter_fn(name, module):
            continue

        parent, attr = _get_parent(model, name)
        prefix_layer = PrefixTunedMHA(module, cfg)
        setattr(parent, attr, prefix_layer)
        wrapped.append(name)

    # Track adapter params for lightweight checkpointing
    if wrapped:
        adapter_names = [f"{w}.prefix_key" for w in wrapped] + [f"{w}.prefix_value" for w in wrapped]
        existing = getattr(model, "_adapter_param_names", set())
        model._adapter_param_names = set(existing).union(adapter_names)

    if not wrapped:
        raise RuntimeError("No attention layers matched for Prefix-Tuning injection.")

    return model, wrapped 