DEVICES=0
TORCH_DISTRIBUTED_BUCKET_CAP_MB=10
export PYTHONPATH=$(pwd)
export TOKENIZERS_PARALLELISM=false

BASELINE="pi0"
DATA_ROOT_DIR="piper_multitask"

METHOD="lora_moe_r128_top1_preEx+FT"
DATASET="multitask"

CUDA_VISIBLE_DEVICES=${DEVICES} \
  python \
    scripts/train.py \
    --policy.path=/ckpt/pi0 \
    --dist_mode="none" \
    --method.core="lora_msp" \
    --gradient_checkpointing=true \
    --train_dataset.repo_id="/datasets/${DATA_ROOT_DIR}/lerobot_5hz" \
    --train_dataset.root="/datasets/${DATA_ROOT_DIR}/lerobot_5hz" \
    --test_dataset.repo_id="/datasets/${DATA_ROOT_DIR}/lerobot_5hz" \
    --test_dataset.root="/datasets/${DATA_ROOT_DIR}/lerobot_5hz" \
    --wandb.project=TEST \
    --wandb.enable=true \
    --wandb.disable_artifact=true \
    --output_dir=/result/pi0_TEST \
    --job_name=lora_msp_test \
    --batch_size=4 \
    --num_workers=16 \
    --log_freq=10 \
    --save_freq=5000 \
    --test_freq=100 \
    --steps=30000