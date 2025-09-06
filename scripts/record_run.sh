for i in {20..59}
do

  python scripts/record.py \
  --dataset_path="/home/minji/Desktop/codes/lerobot/data" \
  --episode_num=$i \
  --episode_len=10 \
  --task="push the button" \
  --fps=30 \
  --recorded_by="ms" \

done