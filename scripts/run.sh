DEVICES=2,3,4\
TORCH_DISTRIBUTED_BUCKET_CAP_MB=10
export PYTHONPATH=$(pwd)

BASELINE="pi0"
DATA_ROOT_DIR="piper_corn_grape_0717_1.2k"

CUDA_VISIBLE_DEVICES=${DEVICES} \
  torchrun \
    --master-port=29300 \
    --nproc_per_node=3 \
    scripts/train.py \
    --policy.path=/ckpt/pi0 \
    --use_ddp=true \
    --use_lora_moe=true \
    --lora_moe_cfg='{"r":16,"alpha":32}' \
    --target_keywords='["q_proj","k_proj","v_proj"]' \
    --train_dataset.repo_id=/datasets/${DATA_ROOT_DIR}/lerobot_5hz/train \
    --train_dataset.root=/datasets/${DATA_ROOT_DIR}/lerobot_5hz/train \
    --test_dataset.repo_id=/datasets/${DATA_ROOT_DIR}/lerobot_5hz/test \
    --test_dataset.root=/datasets/${DATA_ROOT_DIR}/lerobot_5hz/test \
    --wandb.project=ISL_VLA \
    --wandb.enable=true \
    --wandb.disable_artifact=true \
    --output_dir=/result/${BASELINE}_loraMoE_0731_${DATA_ROOT_DIR} \
    --job_name=${BASELINE}_loraMoE_0731_${DATA_ROOT_DIR} \
    --batch_size=8 \
    --num_workers=16 \
    --log_freq=10 \
    --save_freq=5000 \
    --test_freq=100 \
    --steps=40000