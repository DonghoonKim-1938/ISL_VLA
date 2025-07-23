import os
import pandas as pd

def change_episode_task_index(index: int):
    parquet_file_path = f"/data/piper_lerobot/lerobot_aligncups/train/data/chunk-{index//50:03d}/episode_{index:06d}.parquet"
    df = pd.read_parquet(parquet_file_path)
    df['task_index'] = index%4
    df.to_parquet(parquet_file_path, index=False)


if __name__ == "__main__":
    for i in range(120):
        change_episode_task_index(i)