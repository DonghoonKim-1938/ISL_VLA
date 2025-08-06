DEVICES=0,1,2,3,4,5
TORCH_DISTRIBUTED_BUCKET_CAP_MB=10
export PYTHONPATH=$(pwd)

BASELINE="pi0"
DATA_ROOT_DIR="piper_open_the_pot"

CUDA_VISIBLE_DEVICES=${DEVICES} \
  torchrun \
    --master-port=29300 \
    --nproc_per_node=6 \
    scripts/train.py \
    --policy.path=/ckpt/pi0 \
    --use_ddp=true \
    --use_lora=true \
    --lora_cfg='{"r":16,"alpha":32}' \
    --target_keywords='["all-linear"]' \
    --train_dataset.repo_id="/datasets/${DATA_ROOT_DIR}/lerobot_5hz" \
    --train_dataset.root="/datasets/${DATA_ROOT_DIR}/lerobot_5hz" \
    --test_dataset.repo_id="/datasets/${DATA_ROOT_DIR}/lerobot_5hz" \
    --test_dataset.root="/datasets/${DATA_ROOT_DIR}/lerobot_5hz" \
    --wandb.project=ISL_VLA \
    --wandb.enable=true \
    --wandb.disable_artifact=true \
    --output_dir=/result/${BASELINE}_lora_openthepot \
    --job_name=${BASELINE}_lora_openthepot \
    --batch_size=8 \
    --num_workers=16 \
    --log_freq=10 \
    --save_freq=5000 \
    --test_freq=100 \
    --steps=30000