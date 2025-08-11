#!/usr/bin/env bash
# -------------------------------------------------------------
# Helper script to run weight-matrix SVD analysis
#
# Usage:
#   bash scripts/weight_matrix_analysis.sh \
#       /path/to/pretrained.safetensors \
#       /path/to/finetuned.safetensors 
#
# Optional ENV variables:
#   DEVICE  – torch device string (default: cuda:0 if available else cpu)
# -------------------------------------------------------------

set -euo pipefail

if [[ $# -ne 2 ]]; then
  echo "[ERROR] Expected 2 arguments: PRETRAINED_PATH FINETUNED_PATH" >&2
  exit 1
fi

PRETRAINED=$1
FINETUNED=$2

# Choose default device
if [[ -z "${DEVICE:-}" ]]; then
  if python - <<<'import torch, sys; sys.exit(0 if torch.cuda.is_available() else 1)'; then
    DEVICE="cuda:0"
  else
    DEVICE="cpu"
  fi
fi

echo "Running SVD analysis on device: $DEVICE"

python scripts/weight_matrix_analysis.py \
  --pretrained "$PRETRAINED" \
  --finetuned  "$FINETUNED" \
  --device "$DEVICE" \
  --plot
