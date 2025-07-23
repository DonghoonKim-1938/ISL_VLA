DEVICES=0,1,2,3,4,5
TORCH_DISTRIBUTED_BUCKET_CAP_MB=10

CUDA_VISIBLE_DEVICES=${DEVICES} \
  torchrun \
    --nproc_per_node=6 \
    ./train.py \
    --policy.path=/ckpt/pi0 \
    --use_ddp=true \
    --use_lora=true \
    --lora_cfg='{"r":16,"alpha":32}' \
    --target_keywords='["q_proj","k_proj","v_proj"]' \
    --dataset.repo_id=/data/piper_lerobot/lerobot_aligncups_5hz/train \
    --dataset.root=/data/piper_lerobot/lerobot_aligncups_5hz/train \
    --wandb.enable=true \
    --output_dir=/result/pi0_ddp_20250616_piper_5hz_aligncups \
    --job_name=pi0_ddp_piper_aligncups \
    --wandb.disable_artifact=true \
    --batch_size=6 \
    --num_workers=16 \
    --log_freq=10 \
    --save_freq=10000 \
    --steps=40000

    --master-port 29400
--nproc_per_node=1
./train.py
--policy.path=/ckpt/pi0
--use_ddp=true
--use_lora=true
--train_dataset.repo_id=/data/piper_lerobot/lerobot_aligncups_5hz/train
--train_dataset.root=/data/piper_lerobot/lerobot_aligncups_5hz/train
--test_dataset.repo_id=/data/piper_lerobot/lerobot_aligncups_5hz/test
--test_dataset.root=/data/piper_lerobot/lerobot_aligncups_5hz/test
--wandb.enable=true
--output_dir=/result/pi0_20250714_train_test
--job_name=train_test_test
--wandb.disable_artifact=true
--batch_size=1
--num_workers=8
--log_freq=1
--save_freq=10
--test_freq=5
--steps=20