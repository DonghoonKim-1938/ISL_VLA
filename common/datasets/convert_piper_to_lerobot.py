import json
import pickle
import pandas as pd
import numpy as np
import re
import os
import cv2
from tqdm import tqdm
import pyarrow

FPS = 5

def write_data(target_path, data_path, episode_num):
    with open(data_path+f'/aligncups_episode{episode_num}.pickle', 'rb') as f:
        data = pickle.load(f)

    action_end_pose_data = np.array(data['action']['end_pose_data'])
    action_gripper_data = np.array(data['action']['gripper_data'])[:,0]
    action_gripper_data = np.expand_dims(action_gripper_data, -1)
    # print(action_end_pose_data.shape, action_gripper_data.shape)
    action_data = np.hstack((action_end_pose_data, action_gripper_data))
    # print(action_data.shape)

    state_end_pose_data = np.array(data['state']['end_pose_data'])
    state_gripper_data = np.array(data['state']['gripper_data'])[:,0]
    state_gripper_data = np.expand_dims(state_gripper_data, -1)
    # print(state_end_pose_data.shape, state_gripper_data.shape)
    state_data = np.hstack((state_end_pose_data, state_gripper_data))

    index = np.arange(action_data.shape[0])

    index_ = np.arange(state_data.shape[0]) + 150 * episode_num
    # print(index_)

    timestamp = index * 0.2

    frame_index = index

    episode_index = np.ones_like(index) * (episode_num%20)

    task_index = np.zeros_like(index)

    df = pd.DataFrame({
        'action': [row for row in action_data],
        'observation.state': [row for row in state_data],
        'timestamp': timestamp,
        'frame_index': frame_index,
        'episode_index': episode_index,
        'task_index': task_index,
        'index': index_,
    })

    df.to_parquet(target_path+f'/data/chunk-000/episode_{episode_num}.parquet', engine='pyarrow')


def write_video(target_path, data_path, episode_num, cam='exo'):
    fps = FPS

    folder_path = data_path+f'/{cam}'
    output_path = target_path+f'/videos/chunk-000/observation.images.{cam}/episode_{episode_num:06d}.mp4'
    pattern = re.compile(r"color_img_(\d+)\.jpeg")

    files = os.listdir(folder_path)
    image_files = sorted(
        [f for f in files if pattern.match(f)],
        key=lambda x: int(pattern.search(x).group(1)),
        reverse=False
    )

    first_image_path = os.path.join(folder_path, image_files[0])
    first_image = cv2.imread(first_image_path)
    height, width, _ = first_image.shape

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # mp4 코덱
    video_writer = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    for image_file in image_files:
        image_path = os.path.join(folder_path, image_file)
        image = cv2.imread(image_path)
        video_writer.write(image)

    video_writer.release()
    # print(f'released {episode_num} {cam} video!')


def load_data_to_total(total, data):
    if total is None:
        total = data
    else:
        total = np.vstack((total, data))
    return total


def convert_piper_to_lerobot(target_path, data_path, episode_num):
    write_data(target_path, data_path, episode_num)
    # write_video(target_path, data_path, episode_num,cam='exo')
    # write_video(target_path, data_path, episode_num,cam='wrist')


def create_action_stats(piper_path):
    action_data_total = None
    state_data_total = None
    index_total = None
    timestamp_total = None
    frame_index_total = None
    episode_index_total = None
    task_index_total = None

    for episode_num in range(20):
        with open(piper_path + f'/train/aligncups_episode{episode_num}/aligncups_episode{episode_num}.pickle', 'rb') as f:
            data = pickle.load(f)

        action_end_pose_data = np.array(data['action']['end_pose_data'])
        action_gripper_data = np.array(data['action']['gripper_data'])[:, 0]
        action_gripper_data = np.expand_dims(action_gripper_data, -1)
        action_data = np.hstack((action_end_pose_data, action_gripper_data))

        state_end_pose_data = np.array(data['state']['end_pose_data'])
        state_gripper_data = np.array(data['state']['gripper_data'])[:, 0]
        state_gripper_data = np.expand_dims(state_gripper_data, -1)
        state_data = np.hstack((state_end_pose_data, state_gripper_data))

        index = np.arange(action_data.shape[0]).reshape(-1, 1)
        index_ = index + 150 * episode_num
        timestamp = index * 0.2
        frame_index = index
        episode_index = np.ones_like(index)
        task_index = np.zeros_like(index)

        action_data_total = load_data_to_total(action_data_total, action_data)
        state_data_total = load_data_to_total(state_data_total, state_data)
        index_total = load_data_to_total(index_total, index_)
        timestamp_total = load_data_to_total(timestamp_total, timestamp)
        frame_index_total = load_data_to_total(frame_index_total, frame_index)
        episode_index_total = load_data_to_total(episode_index_total, episode_index)
        task_index_total = load_data_to_total(task_index_total, task_index)
    return {
        'action':{
            'mean': np.mean(action_data_total, axis=0).tolist(),
            'std': np.std(action_data_total, axis=0).tolist(),
            'max': np.max(action_data_total, axis=0).tolist(),
            'min': np.min(action_data_total, axis=0).tolist(),
        },
        'observation.state':{
            'mean': np.mean(state_data_total, axis=0).tolist(),
            'std': np.std(state_data_total, axis=0).tolist(),
            'max': np.max(state_data_total, axis=0).tolist(),
            'min': np.min(state_data_total, axis=0).tolist(),
        },
        'timestamp':{
            'mean': np.mean(timestamp_total, axis=0).tolist(),
            'std': np.std(timestamp_total, axis=0).tolist(),
            'max': np.max(timestamp_total, axis=0).tolist(),
            'min': np.min(timestamp_total, axis=0).tolist(),
        },
        'frame_index':{
            'mean': np.mean(frame_index_total, axis=0).tolist(),
            'std': np.std(frame_index_total, axis=0).tolist(),
            'max': np.max(frame_index_total, axis=0).tolist(),
            'min': np.min(frame_index_total, axis=0).tolist(),
        },
        'episode_index':{
            'mean': np.mean(episode_index_total, axis=0).tolist(),
            'std': np.std(episode_index_total, axis=0).tolist(),
            'max': np.max(episode_index_total, axis=0).tolist(),
            'min': np.min(episode_index_total, axis=0).tolist(),
        },
        'task_index':{
            'mean': np.mean(task_index_total, axis=0).tolist(),
            'std': np.std(task_index_total, axis=0).tolist(),
            'max': np.max(task_index_total, axis=0).tolist(),
            'min': np.min(task_index_total, axis=0).tolist(),
        },
        'index':{
            'mean': np.mean(index_total, axis=0).tolist(),
            'std': np.std(index_total, axis=0).tolist(),
            'max': np.max(index_total, axis=0).tolist(),
            'min': np.min(index_total, axis=0).tolist(),
        },
    }


def create_image_stats(piper_path):
    exo_data_total = None
    wrist_data_total = None

    for cam in ['exo', 'wrist']:
        for episode_num in tqdm(range(20)):
            folder_path = piper_path + f'/train/aligncups_episode{episode_num}/{cam}'
            pattern = re.compile(r"color_img_(\d+)\.jpeg")

            files = os.listdir(folder_path)
            image_files = sorted(
                [f for f in files if pattern.match(f)],
                key=lambda x: int(pattern.search(x).group(1)),
                reverse=False
            )

            for image_file in image_files:
                image_path = os.path.join(folder_path, image_file)
                image = cv2.imread(image_path)
                if cam == 'exo':
                    exo_data_total = load_data_to_total(exo_data_total, np.mean(image, axis=(0,1)).reshape(1, -1))
                else:
                    wrist_data_total = load_data_to_total(wrist_data_total, np.mean(image, axis=(0,1)).reshape(1, -1))

    return {
        'observation.images.exo':{
            'mean': np.mean(exo_data_total, axis=0).reshape(-1,1,1).tolist(),
            'std': np.std(exo_data_total, axis=0).reshape(-1,1,1).tolist(),
            'max': np.max(exo_data_total, axis=0).reshape(-1,1,1).tolist(),
            'min': np.min(exo_data_total, axis=0).reshape(-1,1,1).tolist(),
        },
        'observation.images.wrist':{
            'mean': np.mean(wrist_data_total, axis=0).reshape(-1,1,1).tolist(),
            'std': np.std(wrist_data_total, axis=0).reshape(-1,1,1).tolist(),
            'max': np.max(wrist_data_total, axis=0).reshape(-1,1,1).tolist(),
            'min': np.min(wrist_data_total, axis=0).reshape(-1,1,1).tolist(),
        },
    }


def create_stats(piper_path, lerobot_path):
    action_stats_dict = create_action_stats(piper_path)
    image_stats_dict = create_image_stats(piper_path)
    stats_dict = action_stats_dict | image_stats_dict
    with open(lerobot_path + '/train/meta/stats.json', 'w') as f:
        json.dump(stats_dict, f, indent=4)

    with open(lerobot_path + '/test/meta/stats.json', 'w') as f:
        json.dump(stats_dict, f, indent=4)


if __name__ == '__main__':
    piper_path = '/data/piper_5hz'
    lerobot_path = '/data/piper_lerobot/lerobot_aligncups_5hz'

    for episode_num in tqdm(range(20,22)):
        if episode_num < 20:
            data_path = piper_path + f'/train/aligncups_episode{episode_num}'
            target_path = lerobot_path + '/train'
        else:
            data_path = piper_path + f'/validation/aligncups_episode{episode_num}'
            target_path = lerobot_path + '/test'

        convert_piper_to_lerobot(target_path, data_path, episode_num)
    # create_stats(piper_path, lerobot_path)