import cv2
import torch
import os
import pandas as pd
import json
from datetime import datetime

class VideoWriter():
    def __init__(
            self,
            output_path,
            fourcc,
            fps,
            shape
    ):
        self.writer = cv2.VideoWriter(output_path, fourcc, fps, shape)
        self.type = None

    def write(self, frame):
        self.writer.write(frame)

    def release(self):
        self.writer.release()


class PiperDataset(torch.utils.data.Dataset):
    def __init__(
            self,
            root: str | None = None,
            episode_num: int = None,
            episode_len: int = None,
            recorded_by: str = None,
            create_video: bool = True,
            fps: int = 30,
            save_as_lerobot: bool = True,
            save_as_rlds: bool = False,
    ):
        self.root = root
        self.episode_num = episode_num
        self.episode_len = episode_len
        self.create_video = create_video
        self.fps = fps

        self.save_as_lerobot = save_as_lerobot
        self.save_as_rlds = save_as_rlds

        self.recorded_by = recorded_by

        if not os.path.isdir(self.root):
            self.create_root_dir()

        self.episode_file = "episode.pickle"
        self.episode_path = os.path.join(self.root, str(self.episode_num))
        self.episode = []

        self.frames = 0

        self.lerobot_chunk_num = self.episode_num // 50

    def save_episode(self):
        if not os.path.isdir(self.episode_path):
            self.create_episode_dir()

        self.save_metadata()

        for item in self.episode:
            for key in item:
                if isinstance(item[key], torch.Tensor):
                    item[key] = item[key].tolist()

        df = pd.DataFrame(self.episode)
        df.to_pickle(os.path.join(self.episode_path,self.episode_file))

        if self.save_as_lerobot:
            self.save_episode_lerobot(df)

        if self.save_as_rlds:
            self.save_episode_rlds(df)

        if self.create_video:
            self.save_video()

    def add_frame(self, frame):
        self.episode.append(frame)

    def create_root_dir(self):
        os.makedirs(self.root, exist_ok=True)

    def create_episode_dir(self):
        os.makedirs(self.episode_path, exist_ok=True)

    def save_metadata(self):
        with open(os.path.join(self.episode_path, "meta.json"), "w") as f:
            json.dump(
                {
                    "task_name": self.root.split('/')[-1],
                    "episode_num": self.episode_num,
                    "episode_len": self.episode_len,
                    "frame_num": self.frames,
                    "date": datetime.now().strftime("%Y/%m/%d, %H:%M"),
                    "creator": self.recorded_by
                },
                f,
                indent=4,
            )

    def save_video(self, data_type='default'):
        height, width = cv2.imdecode(self.episode[0]["observation.images.wrist"], cv2.IMREAD_COLOR).shape[:2]
        video_writers = []
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")

        if self.episode[0]["observation.images.wrist"] is not None:
            if data_type == 'default':
                output_path_wrist = os.path.join(self.episode_path, 'wrist.mp4')
            if data_type == 'lerobot':
                output_path_dir = os.path.join(self.lerobot_dataset_path, f'videos/chunk-{self.lerobot_chunk_num:03d}/observation.images.wrist')
                os.makedirs(output_path_dir, exist_ok=True)
                output_path_wrist = os.path.join(output_path_dir, f'episode_{self.episode_num:06d}.mp4')
            if data_type == 'rlds':
                raise NotImplementedError

            video_writer_wrist = VideoWriter(output_path_wrist, fourcc, self.fps, (width, height))
            video_writer_wrist.type = 'wrist'
            video_writers.append(video_writer_wrist)

        if self.episode[0]["observation.images.exo"] is not None:
            if data_type == 'default':
                output_path_exo = os.path.join(self.episode_path, 'exo.mp4')
            if data_type == 'lerobot':
                output_path_dir = os.path.join(self.lerobot_dataset_path, f'videos/chunk-{self.lerobot_chunk_num:03d}/observation.images.exo')
                os.makedirs(output_path_dir, exist_ok=True)
                output_path_exo = os.path.join(output_path_dir, f'episode_{self.episode_num:06d}.mp4')
            if data_type == 'rlds':
                raise NotImplementedError

            video_writer_exo = VideoWriter(output_path_exo, fourcc, self.fps, (width, height))
            video_writer_exo.type = 'exo'
            video_writers.append(video_writer_exo)

        if self.episode[0]["observation.images.table"] is not None:
            if data_type == 'default':
                output_path_table = os.path.join(self.episode_path, 'table.mp4')
            if data_type == 'lerobot':
                output_path_dir = os.path.join(self.lerobot_dataset_path, f'videos/chunk-{self.lerobot_chunk_num:03d}/observation.images.table')
                os.makedirs(output_path_dir, exist_ok=True)
                output_path_table = os.path.join(output_path_dir, f'episode_{self.episode_num:06d}.mp4')
            if data_type == 'rlds':
                raise NotImplementedError

            video_writer_table = VideoWriter(output_path_table, fourcc, self.fps, (width, height))
            video_writer_table.type = 'table'
            video_writers.append(video_writer_table)

        for item in self.episode:
            for video_writer in video_writers:
                enc_img = item[f'observation.images.{video_writer.type}']
                dec_img = cv2.imdecode(enc_img, cv2.IMREAD_COLOR)
                image = dec_img
                video_writer.write(image)

        for video_writer in video_writers:
            video_writer.release()

    def save_episode_lerobot(self, df):
        self.lerobot_dataset_path = os.path.join(self.root, 'lerobot')

        self.lerobot_episode_path = os.path.join(self.lerobot_dataset_path, f'data/chunk-{self.lerobot_chunk_num:03d}')
        self.lerobot_videos_path = os.path.join(self.lerobot_dataset_path, f'videos/chunk-{self.lerobot_chunk_num:03d}')

        os.makedirs(self.lerobot_episode_path, exist_ok=True)
        os.makedirs(self.lerobot_videos_path, exist_ok=True)

        self.lerobot_parquet_path = os.path.join(self.lerobot_episode_path, f'episode_{self.episode_num:06d}.parquet')
        columns_to_save = ['action','observation.state', 'timestamp' ,'frame_index', 'episode_index','task_index']
        df[columns_to_save].to_parquet(self.lerobot_parquet_path, index=False)

        self.save_video('lerobot')

    def save_episode_rlds(self, df):
        raise NotImplementedError()