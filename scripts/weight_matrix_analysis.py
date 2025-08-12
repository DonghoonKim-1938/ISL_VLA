#!/usr/bin/env python
"""
Weight-matrix analysis utility

Usage
-----
python scripts/weight_matrix_analysis.py \
    --pretrained /path/to/pretrained.safetensors \
    --finetuned  /path/to/finetuned.safetensors

The script will:
1. Load both safetensor files (bfloat16 / fp16 / fp32 supported).
2. Identify 2-D tensors whose names end with ".weight" (typical linear layers).
3. For each shared weight name:
   • compute singular values of the pretrained matrix
   • compute singular values of the finetuned matrix
   • compute singular values of the difference (finetuned ‑ pretrained)
4. Print the top-k singular values (default 10) for quick inspection.

Singular-value calculation uses `torch.linalg.svdvals`, which supports
CUDA if tensors are already on GPU.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict

import torch
from safetensors import safe_open
from safetensors.torch import load_file as load_safetensor


def load_weights(path: str | Path, device: str = "cpu") -> Dict[str, torch.Tensor]:
    """Load tensors from a safetensors file into Torch tensors on the selected device."""
    path = Path(path)
    if path.is_dir():
        # Automatically pick first *.safetensors file inside the directory
        candidates = sorted(path.glob("*.safetensors"))
        if not candidates:
            raise FileNotFoundError(f"{path} 는 디렉터리이며 .safetensors 파일을 찾을 수 없습니다.")
        path = candidates[0]
        print(f"→ {path.name} 파일을 사용합니다.")
    if not path.is_file():
        raise FileNotFoundError(f"{path} 파일이 존재하지 않습니다.")
    return load_safetensor(str(path), device=device)


def filter_linear_weights(tensors: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
    """Return only 2-D matrices whose names end with `.weight`."""
    return {
        name: tensor
        for name, tensor in tensors.items()
        if tensor.ndim == 2 and name.endswith(".weight")
    }


def compute_singular_values(matrix: torch.Tensor, k: int | None = None) -> torch.Tensor:
    """Compute singular values; optionally return top-k (descending)."""
    # torch.linalg.svdvals returns descending order
    svals = torch.linalg.svdvals(matrix.float())  # promote for numerical stability
    if k is not None:
        svals = svals[:k]
    return svals


def main():
    parser = argparse.ArgumentParser(description="Weight-matrix SVD analysis")
    parser.add_argument("--pretrained", type=Path, required=True, help="Path to pretrained .safetensors")
    parser.add_argument("--finetuned", type=Path, required=True, help="Path to finetuned .safetensors")
    parser.add_argument("--device", type=str, default="cpu", help="Device to run SVD on (cpu/cuda:0 ...)")
    parser.add_argument("--plot", action="store_true", help="Save scatter plots of 90% SV counts per layer.")
    args = parser.parse_args()

    print("Loading tensors …")
    pretrained_tensors = load_weights(args.pretrained, device=args.device)
    finetuned_tensors = load_weights(args.finetuned, device=args.device)

    pt_linear = filter_linear_weights(pretrained_tensors)
    ft_linear = filter_linear_weights(finetuned_tensors)

    shared_keys = sorted(set(pt_linear) & set(ft_linear))
    if not shared_keys:
        print("No shared linear weight matrices found between checkpoints.")
        return

    device = torch.device(args.device)

    # Containers per category
    categories = {
        "vision_tower": {"layers": [], "pt": [], "ft": [], "diff": []},
        "language_model": {"layers": [], "pt": [], "ft": [], "diff": []},
        "gemma_expert": {"layers": [], "pt": [], "ft": [], "diff": []},
        "others": {"layers": [], "pt": [], "ft": [], "diff": []},
    }

    import re

    def extract_layer_idx(tensor_name: str) -> int | None:
        """Extract the last numeric value after '.layers.' pattern."""
        matches = re.findall(r"\.layers\.([0-9]+)", tensor_name)
        return int(matches[-1]) if matches else None

    print(f"Found {len(shared_keys)} shared linear layers. Computing SVD + 90% energy counts…")
    for idx, name in enumerate(shared_keys):
        pt_w = pt_linear[name].to(device)
        ft_w = ft_linear[name].to(device)
        diff_w = ft_w - pt_w
        shape_str = f"{pt_w.shape[0]}×{pt_w.shape[1]}"

        # full singular values
        sv_pt = torch.linalg.svdvals(pt_w.float()).cpu()
        sv_ft = torch.linalg.svdvals(ft_w.float()).cpu()
        sv_diff = torch.linalg.svdvals(diff_w.float()).cpu()

        def count_until_90(s: torch.Tensor) -> int:
            """Return the smallest k such that first k singular values account for ≥90% energy.
            Robust to zero-valued (all-zero) spectra.
            """
            if torch.all(s == 0):
                return 0
            cs = torch.cumsum(s, dim=0)
            total = cs[-1]
            if total == 0:
                return 0
            mask = (cs / total) >= 0.9
            idx_arr = mask.nonzero(as_tuple=False)
            return int(idx_arr[0].item() + 1) if idx_arr.numel() else len(s)

        c_pt = count_until_90(sv_pt)
        c_ft = count_until_90(sv_ft)
        c_diff = count_until_90(sv_diff)

        # classify category
        if "vision_tower" in name:
            cat = "vision_tower"
        elif "language_model" in name:
            cat = "language_model"
        elif "expert" in name or "gemma_expert" in name:
            cat = "gemma_expert"
        else:
            cat = "others"

        layer_idx = extract_layer_idx(name)
        if layer_idx is None:
            # Fallback to enumeration index if pattern not found
            layer_idx = idx

        categories[cat]["layers"].append(layer_idx)
        categories[cat]["pt"].append(c_pt)
        categories[cat]["ft"].append(c_ft)
        categories[cat]["diff"].append(c_diff)

        print(f"[{idx:03d}] {name} (shape: {shape_str})")
        print(f"  90% SV count – pretrained: {c_pt}, finetuned: {c_ft}, delta: {c_diff}")
        print("-" * 60)

    if args.plot:
        import matplotlib.pyplot as plt

        for cat, data in categories.items():
            if not data["layers"]:
                continue
            plt.figure(figsize=(10, 6))
            plt.scatter(data["layers"], data["pt"], label="pretrained")
            plt.scatter(data["layers"], data["ft"], label="finetuned")
            plt.scatter(data["layers"], data["diff"], label="delta")
            plt.xlabel(f"Layer index ({cat})")
            plt.ylabel("# SVs to reach 90% energy")
            plt.title(f"Singular Value 90% energy counts – {cat}")
            plt.legend()
            plt.tight_layout()
            out_path = Path(f"svd_90p_counts_{cat}.png")
            plt.savefig(out_path, dpi=150)
            print(f"Scatter plot saved to {out_path.resolve()}")


if __name__ == "__main__":
    main() 