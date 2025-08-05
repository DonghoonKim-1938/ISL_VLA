DEVICES=0
TORCH_DISTRIBUTED_BUCKET_CAP_MB=10
export PYTHONPATH=$(pwd)

CONDA_ENV_PATH=/home/uhnam/miniconda3/bin/conda
CONDA_ENV_NAME="lerobot"

BASELINE="pi0"
DATA_ROOT_DIR="piper_open_the_pot"

BASELINE_PATH=/ckpt/pi0
ADAPTER_PATH=/result/pi0_qlora_openthepot/checkpoints/005000/pretrained_model

CUDA_VISIBLE_DEVICES=${DEVICES} \
  ${CONDA_ENV_PATH} run -n ${CONDA_ENV_NAME} \
    --policy.path=${BASELINE_PATH} \
    --adapter_path=${ADAPTER_PATH} \
    --use_qlora=true \
    --train_dataset.repo_id="/datasets/${DATA_ROOT_DIR}/lerobot_5hz" \
    --train_dataset.root="/datasets/${DATA_ROOT_DIR}/lerobot_5hz" \
    --use_devices=true \
    --target_keywords='["q_proj","k_proj","v_proj"]' \
