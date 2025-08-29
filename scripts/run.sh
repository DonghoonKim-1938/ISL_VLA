DEVICES=0,1,2,3
TORCH_DISTRIBUTED_BUCKET_CAP_MB=10
export PYTHONPATH=$(pwd)
export TOKENIZERS_PARALLELISM=false

BASELINE="pi0"
DATA_ROOT_DIR="piper_multitask"

METHOD="lora_moe_r128_top1_preEx+FT"
DATASET="multitask"

CUDA_VISIBLE_DEVICES=${DEVICES} \
  torchrun \
    --master-port=29300 \
    --nproc_per_node=4 \
    scripts/train.py \
    --policy.path=/ckpt/pi0 \
    --dist_mode=ddp \
    --method.target_keywords=["all-linear"] \
    --method.core="lora_msp" \
    --method.aux_loss_cfg='{
        "lb_coeff": 1e-3,
        "z_coeff": 0.01,
        "spec_coeff": 1e-3,
        "mod_coeff": 1e-3,
        "id_coeff": 1e-3,
    }' \
    --gradient_checkpointing=true \
    --train_dataset.repo_id="/datasets/${DATA_ROOT_DIR}/lerobot_5hz" \
    --train_dataset.root="/datasets/${DATA_ROOT_DIR}/lerobot_5hz" \
    --test_dataset.repo_id="/datasets/${DATA_ROOT_DIR}/lerobot_5hz" \
    --test_dataset.root="/datasets/${DATA_ROOT_DIR}/lerobot_5hz" \
    --wandb.project=ISL_VLA \
    --wandb.enable=true \
    --wandb.disable_artifact=true \
    --output_dir=/result/${BASELINE}_${METHOD}_${DATASET} \
    --job_name=${BASELINE}_${METHOD}_${DATASET} \
    --batch_size=4 \
    --num_workers=16 \
    --log_freq=10 \
    --save_freq=5000 \
    --test_freq=100 \
    --steps=30000