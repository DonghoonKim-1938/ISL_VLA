import json
import os

from tqdm import tqdm
from datasets import load_dataset
import cv2
import numpy as np

from common.constants import META_INFO_TEMPLATE, META_STATS_TEMPLATE, TASK_LIST
from common.datasets.compute_stats import sample_indices


def compute_online_stats(data_iter, reduce_axes=None, reshape=None):
    count = 0
    mean = None
    M2 = None
    max_val = None
    min_val = None

    for x in tqdm(data_iter):
        x = np.asarray(x, dtype=np.float64)

        # Reduce early if specified (e.g., mean over H,W)
        if reduce_axes is not None:
            x = np.mean(x, axis=reduce_axes, keepdims=False)

        if mean is None:
            mean = np.zeros_like(x)
            M2 = np.zeros_like(x)
            max_val = np.copy(x)
            min_val = np.copy(x)

        count += 1
        delta = x - mean
        mean += delta / count
        delta2 = x - mean
        M2 += delta * delta2

        max_val = np.maximum(max_val, x)
        min_val = np.minimum(min_val, x)

    std = np.sqrt(M2 / count) if count > 1 else np.zeros_like(mean)

    if reshape:
        mean = mean.reshape(reshape)
        std = std.reshape(reshape)
        max_val = max_val.reshape(reshape)
        min_val = min_val.reshape(reshape)

    return {
        "mean": mean.tolist(),
        "std": std.tolist(),
        "max": max_val.tolist(),
        "min": min_val.tolist(),
    }


def create_meta(root_dir, episodes):
    os.makedirs(os.path.join(root_dir,"meta"), exist_ok=True)

    ep = []
    tasks = []
    info = META_INFO_TEMPLATE
    stats = META_STATS_TEMPLATE

    tasks_dict = {}
    for k,v in enumerate(TASK_LIST):
        tasks_dict[str(k)]=v

    total_frames = 0
    sample_ep_indices = sample_indices(episodes)

    action_df = []
    observation_state_df = []
    timestamp_df = []
    frame_index_df = []
    episode_index_df = []
    task_index_df = []
    index_df = []
    observation_images_exo_df = []
    observation_images_wrist_df = []
    observation_images_table_df = []

    # fetch data of each episode
    for i in tqdm(range(episodes)):
        parquet_file = os.path.join(root_dir, f"data/chunk-{i//50:03d}/episode_{i:06d}.parquet")
        exo_file = os.path.join(root_dir, f"videos/chunk-{i//50:03d}/observation.images.exo/episode_{i:06d}.mp4")
        wrist_file = os.path.join(root_dir, f"videos/chunk-{i//50:03d}/observation.images.wrist/episode_{i:06d}.mp4")
        table_file = os.path.join(root_dir, f"videos/chunk-{i//50:03d}/observation.images.table/episode_{i:06d}.mp4")

        parquet_data = load_dataset("parquet", data_files=parquet_file)['train']
        ep.append({
            "episode_index": i,
            "tasks": tasks_dict[str(parquet_data[0]['task_index'])],
            "length": len(parquet_data),
        })

        total_frames += len(parquet_data)
        exo_cap = cv2.VideoCapture(exo_file)
        wrist_cap = cv2.VideoCapture(wrist_file)
        table_cap = cv2.VideoCapture(table_file)

        if i in sample_ep_indices:
            for frame_idx in tqdm(range(len(parquet_data))):
                frame_data = parquet_data[frame_idx]

                action_df.append(frame_data['action'])
                observation_state_df.append(frame_data['observation.state'])
                timestamp_df.append(frame_data['timestamp'])
                frame_index_df.append(frame_data['frame_index'])
                episode_index_df.append(frame_data['episode_index'])
                task_index_df.append(frame_data['task_index'])
                index_df.append(frame_data['index'])

                _, exo_frame = exo_cap.read()
                _, wrist_frame = wrist_cap.read()
                _, table_frame = table_cap.read()

                observation_images_exo_df.append(exo_frame)
                observation_images_wrist_df.append(wrist_frame)
                observation_images_table_df.append(table_frame)

    # save meta/episodes.jsonl
    with open(os.path.join(root_dir, 'meta/episodes.jsonl'), "w") as f:
        for entry in ep:
            json.dump(entry, f)
            f.write("\n")

    # save meta/info.json
    info['total_episodes'] = episodes
    info['total_frames'] = total_frames
    info['total_videos'] = episodes
    info['total_chunks'] = (episodes // info['chunks_size'])+1
    info['splits']['train'] = f"0:{episodes}"

    with open(os.path.join(root_dir, 'meta/info.json'), "w") as f:
        json.dump(info, f, indent=4)

    # save meta/tasks.jsonl
    for k,v in tasks_dict.items():
        tasks.append({
            "task_index": int(k),
            "task": v
        })

    with open(os.path.join(root_dir, 'meta/tasks.jsonl'), "w") as f:
        for entry in tasks:
            json.dump(entry, f)
            f.write("\n")

    # save meta/stats.json
    # stats["action"]=compute_online_stats(action_df)
    # stats["observation.state"]=compute_online_stats(observation_state_df)
    # stats["timestamp"]=compute_online_stats(timestamp_df)
    # stats["frame_index"]=compute_online_stats(frame_index_df)
    # stats["episode_index"]=compute_online_stats(episode_index_df)
    # stats["task_index"]=compute_online_stats(task_index_df)
    # stats["index"]=compute_online_stats(index_df)
    #
    # stats["observation.images.exo"] = compute_online_stats(
    #     observation_images_exo_df,
    #     reduce_axes=(0, 1),
    #     reshape=(3, 1, 1)
    # )
    # stats["observation.images.wrist"] = compute_online_stats(
    #     observation_images_wrist_df,
    #     reduce_axes=(2, 3),
    #     reshape=(3, 1, 1)
    # )
    # stats["observation.images.table"] = compute_online_stats(
    #     observation_images_table_df,
    #     reduce_axes=(2, 3),
    #     reshape=(3, 1, 1)
    # )
    stacked_action = np.stack(action_df, axis=0)
    stacked_observation_state = np.stack(observation_state_df, axis=0)
    stacked_timestamp = np.stack(timestamp_df, axis=0)
    stacked_frame_index = np.stack(frame_index_df, axis=0)
    stacked_episode_index = np.stack(episode_index_df, axis=0)
    stacked_task_index = np.stack(task_index_df, axis=0)
    stacked_index = np.stack(index_df, axis=0)
    stats["action"] = {
        "mean": stacked_action.mean(axis=0).tolist(),
        "std": stacked_action.std(axis=0).tolist(),
        "max": stacked_action.max(axis=0).tolist(),
        "min": stacked_action.min(axis=0).tolist(),
    }
    stats["observation.state"] = {
        "mean": stacked_observation_state.mean(axis=0).tolist(),
        "std": stacked_observation_state.std(axis=0).tolist(),
        "max": stacked_observation_state.max(axis=0).tolist(),
        "min": stacked_observation_state.min(axis=0).tolist(),
    }
    stats["timestamp"] = {
        "mean": stacked_timestamp.mean(axis=0).reshape(1).tolist(),
        "std": stacked_timestamp.std(axis=0).reshape(1).tolist(),
        "max": stacked_timestamp.max(axis=0).reshape(1).tolist(),
        "min": stacked_timestamp.min(axis=0).reshape(1).tolist(),
    }
    stats["frame_index"] = {
        "mean": stacked_frame_index.mean(axis=0).reshape(1).tolist(),
        "std": stacked_frame_index.std(axis=0).reshape(1).tolist(),
        "max": stacked_frame_index.max(axis=0).reshape(1).tolist(),
        "min": stacked_frame_index.min(axis=0).reshape(1).tolist(),
    }
    stats["episode_index"] = {
        "mean": stacked_episode_index.mean(axis=0).reshape(1).tolist(),
        "std": stacked_episode_index.std(axis=0).reshape(1).tolist(),
        "max": stacked_episode_index.max(axis=0).reshape(1).tolist(),
        "min": stacked_episode_index.min(axis=0).reshape(1).tolist(),
    }
    stats["task_index"] = {
        "mean": stacked_task_index.mean(axis=0).reshape(1).tolist(),
        "std": stacked_task_index.std(axis=0).reshape(1).tolist(),
        "max": stacked_task_index.max(axis=0).reshape(1).tolist(),
        "min": stacked_task_index.min(axis=0).reshape(1).tolist(),
    }
    stats["index"] = {
        "mean": stacked_index.mean(axis=0).reshape(1).tolist(),
        "std": stacked_index.std(axis=0).reshape(1).tolist(),
        "max": stacked_index.max(axis=0).reshape(1).tolist(),
        "min": stacked_index.min(axis=0).reshape(1).tolist(),
    }
    del stacked_action, stacked_observation_state, stacked_timestamp, stacked_frame_index, stacked_episode_index, stacked_task_index, stacked_index

    stacked_exo = np.stack(observation_images_exo_df, axis=0)
    stacked_wrist = np.stack(observation_images_wrist_df, axis=0)
    stacked_table = np.stack(observation_images_table_df, axis=0)

    stats["observation.images.exo"]={
        "mean":stacked_exo.mean(axis=(0,1,2)).reshape(3,1,1).tolist(),
        "std":stacked_exo.std(axis=(0,1,2)).reshape(3,1,1).tolist(),
        "max":stacked_exo.max(axis=(0,1,2)).reshape(3,1,1).tolist(),
        "min":stacked_exo.min(axis=(0,1,2)).reshape(3,1,1).tolist(),
    }
    stats["observation.images.wrist"]={
        "mean":stacked_wrist.mean(axis=(0,1,2)).reshape(3,1,1).tolist(),
        "std":stacked_wrist.std(axis=(0,1,2)).reshape(3,1,1).tolist(),
        "max":stacked_wrist.max(axis=(0,1,2)).reshape(3,1,1).tolist(),
        "min":stacked_wrist.min(axis=(0,1,2)).reshape(3,1,1).tolist(),
    }
    stats["observation.images.table"]={
        "mean":stacked_table.mean(axis=(0,1,2)).reshape(3,1,1).tolist(),
        "std":stacked_table.std(axis=(0,1,2)).reshape(3,1,1).tolist(),
        "max":stacked_table.max(axis=(0,1,2)).reshape(3,1,1).tolist(),
        "min":stacked_table.min(axis=(0,1,2)).reshape(3,1,1).tolist(),
    }
    del stacked_exo, stacked_wrist, stacked_table

    with open(os.path.join(root_dir,"meta/stats.json"),"w") as f:
        json.dump(stats,f,indent=4)


if __name__ == "__main__":
    root_dir = "/datasets/open_the_pot/lerobot"
    episodes = 120
    create_meta(root_dir, episodes)

    # root_dir = "/datasets/piper_grape0711/lerobot_5hz/test"
    # episodes = 216
    # create_meta(root_dir, episodes)