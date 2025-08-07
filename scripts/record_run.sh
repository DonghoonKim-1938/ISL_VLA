for i in {500..500}
do

  python record.py \
  --dataset_path="/home/minji/Desktop/codes/lerobot/data" \
  --episode_num=$i \
  --episode_len=10 \
  --task="open the pot" \
  --fps=30 \
  --recorded_by="gh" \

done