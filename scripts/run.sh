DEVICES=0,1,2,3,4,5,6
TORCH_DISTRIBUTED_BUCKET_CAP_MB=10
export PYTHONPATH=$(pwd)
export TOKENIZERS_PARALLELISM=false

BASELINE="pi0"
DATA_ROOT_DIR="piper_multitask"

CUDA_VISIBLE_DEVICES=${DEVICES} \
  torchrun \
    --master-port=29300 \
    --nproc_per_node=7 \
    scripts/train.py \
    --policy.path=/ckpt/pi0 \
    --dist_mode=ddp \
    --lora_cfg='{"r":32,"alpha":64}' \
    --target_keywords='["all-linear"]' \
    --train_dataset.repo_id="/datasets/${DATA_ROOT_DIR}/lerobot_5hz" \
    --train_dataset.root="/datasets/${DATA_ROOT_DIR}/lerobot_5hz" \
    --test_dataset.repo_id="/datasets/${DATA_ROOT_DIR}/lerobot_5hz" \
    --test_dataset.root="/datasets/${DATA_ROOT_DIR}/lerobot_5hz" \
    --wandb.project=ISL_VLA \
    --wandb.enable=true \
    --wandb.disable_artifact=true \
    --output_dir=/result/${BASELINE}_fullFT_multitask \
    --job_name=${BASELINE}_fullFT_multitask \
    --batch_size=6 \
    --num_workers=16 \
    --log_freq=10 \
    --save_freq=5000 \
    --test_freq=100 \
    --steps=30000