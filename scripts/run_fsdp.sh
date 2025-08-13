#!/usr/bin/env bash

# GPU 목록 (쉼표로 구분)
DEVICES=0,1

# 환경 변수 -------------------------------------------------------------
export PYTHONPATH=$(pwd)
export TOKENIZERS_PARALLELISM=false
# FSDP 통신 버킷 크기 (MB) ‑ 필요시 조정
export TORCH_DISTRIBUTED_BUCKET_CAP_MB=10

# 실험 설정 ------------------------------------------------------------
BASELINE="pi0"
DATA_ROOT_DIR="piper_multitask"

# ----------------------------------------------------------------------
# FSDP 분산 학습 실행
# ----------------------------------------------------------------------
CUDA_VISIBLE_DEVICES=${DEVICES} \
  torchrun \
    --master-port=29400 \
    --nproc_per_node=2 \
    scripts/train.py \
    --policy.path=/ckpt/pi0 \
    --dist_mode=fsdp \
    --lora_cfg='{"r":32,"alpha":64}' \
    --target_keywords='["all-linear"]' \
    --train_dataset.repo_id="/datasets/${DATA_ROOT_DIR}/lerobot_5hz" \
    --train_dataset.root="/datasets/${DATA_ROOT_DIR}/lerobot_5hz" \
    --test_dataset.repo_id="/datasets/${DATA_ROOT_DIR}/lerobot_5hz" \
    --test_dataset.root="/datasets/${DATA_ROOT_DIR}/lerobot_5hz" \
    --wandb.project=ISL_VLA \
    --wandb.enable=false \
    --wandb.disable_artifact=true \
    --output_dir=/result/${BASELINE}_fullFT_multitask_fsdp \
    --job_name=${BASELINE}_fullFT_multitask_fsdp \
    --batch_size=1 \
    --num_workers=1 \
    --log_freq=1 \
    --save_freq=1 \
    --test_freq=1 \
    --steps=30000 