DEVICES=0,1,2,3,4
TORCH_DISTRIBUTED_BUCKET_CAP_MB=10
CUDA_VISIBLE_DEVICES=${DEVICES}

for w_decay in 1.0 1.5
do
    for grad_clip_norm in 1.0 10.0
    do
        for lr in 2.5e-5 2.5e-6
        do
          torchrun \
            --master-port 29300 \
            --nproc_per_node=5 \
            ./train.py \
            --policy.path=/ckpt/pi0 \
            --use_lora=true \
            --use_ddp=true \
            --train_dataset.repo_id=/data/piper_lerobot/lerobot_aligncups_5hz/train \
            --train_dataset.root=/data/piper_lerobot/lerobot_aligncups_5hz/train \
            --test_dataset.repo_id=/data/piper_lerobot/lerobot_aligncups_5hz/test \
            --test_dataset.root=/data/piper_lerobot/lerobot_aligncups_5hz/test \
            --wandb.project=isl \
            --wandb.enable=true \
            --wandb.disable_artifact=true \
            --output_dir=/result/pi0_20250721_lr_${lr}_grad_clip_norm_${grad_clip_norm}_w_decay_${w_decay} \
            --job_name=pi0_20250721_lr_${lr}_grad_clip_norm_${grad_clip_norm}_w_decay_${w_decay} \
            --use_policy_training_preset=false \
            --optimizer.type=adamw \
            --optimizer.lr=${lr} \
            --optimizer.weight_decay=${w_decay} \
            --optimizer.grad_clip_norm=${grad_clip_norm} \
            --optimizer.betas="[0.9,0.95]" \
            --optimizer.eps=1e-08 \
            --scheduler.type=cosine_decay_with_warmup \
            --scheduler.peak_lr=${lr} \
            --scheduler.decay_lr=2.5e-6 \
            --scheduler.num_warmup_steps=1000 \
            --scheduler.num_decay_steps=30000 \
            --batch_size=6 \
            --num_workers=16 \
            --log_freq=10 \
            --save_freq=5000 \
            --test_freq=100 \
            --steps=10000
        done
    done
done