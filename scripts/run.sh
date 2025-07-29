DEVICES=3,4
TORCH_DISTRIBUTED_BUCKET_CAP_MB=10
export PYTHONPATH=$(pwd)

CUDA_VISIBLE_DEVICES=${DEVICES} \
  torchrun \
    --master-port=29400 \
    --nproc_per_node=2 \
    scripts/train.py \
    --policy.path=/ckpt/pi0 \
    --use_ddp=true \
    --use_lora=true \
    --lora_cfg='{"r":16,"alpha":32}' \
    --target_keywords='["q_proj","k_proj","v_proj"]' \
    --train_dataset.repo_id=/datasets/piper_grape0724/lerobot_5hz/train \
    --train_dataset.root=/datasets/piper_grape0724/lerobot_5hz/train \
    --test_dataset.repo_id=/datasets/piper_grape0724/lerobot_5hz/test \
    --test_dataset.root=/datasets/piper_grape0724/lerobot_5hz/test \
    --wandb.project=ISL_VLA \
    --wandb.enable=true \
    --wandb.disable_artifact=true \
    --output_dir=/result/pi0_lora_piper_grape_0724 \
    --job_name=pi0_lora_piper_grape_0724 \
    --batch_size=14 \
    --num_workers=16 \
    --log_freq=10 \
    --save_freq=5000 \
    --test_freq=100 \
    --steps=40000