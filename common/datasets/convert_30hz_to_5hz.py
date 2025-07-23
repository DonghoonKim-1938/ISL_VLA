import os
from tqdm import tqdm
import cv2
from datasets import load_dataset, Dataset, Features, Sequence, Value

def cvt_vid(origin, output):
    cap = cv2.VideoCapture(origin)
    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    # 출력 비디오 설정 (5Hz로 저장)
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out = cv2.VideoWriter(output, fourcc, 5, (width, height))  # fps=5

    frame_idx = 0
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # 6 프레임마다 하나 저장
        if frame_idx % 6 == 0:
            out.write(frame)

        frame_idx += 1

    cap.release()
    out.release()


def convert_30hz_to_5hz(index):
    os.makedirs(f"/data/piper_lerobot/lerobot_aligncups_5hz/train/data/chunk-{index//50:03d}", exist_ok=True)
    os.makedirs(f"/data/piper_lerobot/lerobot_aligncups_5hz/train/videos/chunk-{index//50:03d}/observation.images.exo", exist_ok=True)
    os.makedirs(f"/data/piper_lerobot/lerobot_aligncups_5hz/train/videos/chunk-{index//50:03d}/observation.images.wrist", exist_ok=True)
    os.makedirs(f"/data/piper_lerobot/lerobot_aligncups_5hz/train/videos/chunk-{index//50:03d}/observation.images.table", exist_ok=True)

    parquet_file_path = f"/data/piper_lerobot/lerobot_aligncups/data/chunk-{index//50:03d}/episode_{index:06d}.parquet"
    parquet_file_path_des = f"/data/piper_lerobot/lerobot_aligncups_5hz/train/data/chunk-{index//50:03d}/episode_{index:06d}.parquet"

    exo_video_file_path = f"/data/piper_lerobot/lerobot_aligncups/videos/chunk-{index//50:03d}/observation.images.exo/episode_{index:06d}.mp4"
    exo_video_file_path_des = f"/data/piper_lerobot/lerobot_aligncups_5hz/train/videos/chunk-{index//50:03d}/observation.images.exo/episode_{index:06d}.mp4"

    wrist_video_file_path = f"/data/piper_lerobot/lerobot_aligncups/videos/chunk-{index//50:03d}/observation.images.wrist/episode_{index:06d}.mp4"
    wrist_video_file_path_des = f"/data/piper_lerobot/lerobot_aligncups_5hz/train/videos/chunk-{index//50:03d}/observation.images.wrist/episode_{index:06d}.mp4"

    table_video_file_path = f"/data/piper_lerobot/lerobot_aligncups/videos/chunk-{index//50:03d}/observation.images.table/episode_{index:06d}.mp4"
    table_video_file_path_des = f"/data/piper_lerobot/lerobot_aligncups_5hz/train/videos/chunk-{index//50:03d}/observation.images.table/episode_{index:06d}.mp4"

    dataset = load_dataset("parquet", data_files=parquet_file_path)['train']
    features = Features({
        "action": Sequence(Value("int64"), length=7),
        "observation.state": Sequence(Value("int64"), length=7),
        "timestamp": Value("float32"),
        "frame_index": Value("int64"),
        "episode_index": Value("int64"),
        "index": Value("int64"),
        "task_index": Value("int64")
    })

    dataset = dataset.to_dict()

    dataset['action'] = [elem[0] for elem in dataset['action']]
    dataset['observation.state'] = [elem[0] for elem in dataset['observation.state']]
    dataset['task_index'] = [index%4 for _ in dataset['task_index']]

    dataset = Dataset.from_dict(dataset, features=features)
    sampled_dataset = dataset.select(range(0, len(dataset), 6))
    sampled_dataset.to_parquet(parquet_file_path_des)

    cvt_vid(exo_video_file_path, exo_video_file_path_des)
    cvt_vid(wrist_video_file_path, wrist_video_file_path_des)
    cvt_vid(table_video_file_path, table_video_file_path_des)


if __name__ == "__main__":
    for i in tqdm(range(120,200)):
        convert_30hz_to_5hz(i)