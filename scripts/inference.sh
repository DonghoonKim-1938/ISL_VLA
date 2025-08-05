DEVICES=0
TORCH_DISTRIBUTED_BUCKET_CAP_MB=10
export PYTHONPATH=$(pwd)

CONDA_ENV_PATH=/home/minji/miniconda3/bin/conda
CONDA_ENV_NAME="lerobot"

BASELINE="pi0"
DATA_ROOT_DIR="ep1200_openthepot"

BASELINE_PATH=/home/minji/Desktop/data/pi0
ADAPTER_PATH=/home/minji/Desktop/data/finetuned_model/pi0_multitask/pi0_20250805_openthepot_qlora/pretrained_model

CUDA_VISIBLE_DEVICES=${DEVICES} \
  ${CONDA_ENV_PATH} run -n ${CONDA_ENV_NAME} python ./scripts/eval_real_time.py \
    --policy.path=${BASELINE_PATH} \
    --adapter_path=${ADAPTER_PATH} \
    --use_qlora=true \
    --train_dataset.repo_id="/home/minji/Desktop/data/data_config/${DATA_ROOT_DIR}" \
    --train_dataset.root="/home/minji/Desktop/data/data_config/${DATA_ROOT_DIR}" \
    --use_devices=true \
    --target_keywords='["q_proj","k_proj","v_proj"]' \
