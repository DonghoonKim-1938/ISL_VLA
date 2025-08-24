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
    --use_lora_moe=true \
    --use_pretrained_lora=true \
    --lora_moe_cfg='{"r":128,"alpha":256,"routing":"top1"}' \
    --adapter_file_paths="[
        '/result/pi0_lora_r128_all-linear_openthepot/020000/pretrained_model/adapters.safetensors',
        '/result/pi0_lora_r128_all-linear_pickplace/030000/pretrained_model/adapters.safetensors',
        '/result/pi0_lora_r128_all-linear_pourtheblock/030000/pretrained_model/adapters.safetensors',
        '/result/pi0_lora_r128_all-linear_pressthebutton/030000/pretrained_model/adapters.safetensors'
    ]" \
    --gradient_checkpointing=true \
    --target_keywords='["all-linear"]' \
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