for i in {300..319}
do

  python record.py \
  --dataset_path="/home/minji/Desktop/codes/lerobot/data" \
  --episode_num=$i \
  --episode_len=10 \
  --task="pick the grape and put it to the basket" \
  --fps=30 \
  --recorded_by="dh" \

done