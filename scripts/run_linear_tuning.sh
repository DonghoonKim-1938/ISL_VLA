DEVICES=0,1,2,3,4
TORCH_DISTRIBUTED_BUCKET_CAP_MB=10
export PYTHONPATH=$(pwd)

BASELINE="pi0"
DATA_ROOT_DIR="piper_open_the_pot"

CUDA_VISIBLE_DEVICES=${DEVICES} \
  torchrun \
    --master-port=29300 \
    --nproc_per_node=5 \
    scripts/train.py \
    --policy.path=/ckpt/pi0 \
    --use_ddp=true \
    --train_linear_only=true \
    --train_dataset.repo_id="/datasets/${DATA_ROOT_DIR}/lerobot_5hz" \
    --train_dataset.root="/datasets/${DATA_ROOT_DIR}/lerobot_5hz" \
    --test_dataset.repo_id="/datasets/${DATA_ROOT_DIR}/lerobot_5hz" \
    --test_dataset.root="/datasets/${DATA_ROOT_DIR}/lerobot_5hz" \
    --wandb.project=ISL_VLA \
    --wandb.enable=true \
    --wandb.disable_artifact=true \
    --output_dir=/result/${BASELINE}_full_linear_openthepot \
    --job_name=${BASELINE}_full_linear_openthepot \
    --batch_size=6 \
    --num_workers=16 \
    --log_freq=10 \
    --save_freq=5000 \
    --test_freq=100 \
    --steps=30000 