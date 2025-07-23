import logging
import time
import os

from piper_sdk import C_PiperForwardKinematics
from pprint import pformat
from dataclasses import asdict

from configs.record_ours import RecordOursPipelineConfig
from common.utils.utils import init_devices, get_task_index, init_keyboard_listener
from common.datasets.piper_dataset import PiperDataset
from common.robot_devices.robot_utils import read_end_pose_ctrl, read_end_pose_msg

from configs import parser
from common.utils.utils import init_logging

@parser.wrap()
def record_episodes(cfg: RecordOursPipelineConfig):
    logging.info(pformat(asdict(cfg)))
    if cfg.use_devices:
        piper, cam = init_devices(cfg, is_recording=True)

        wrist_rs_cam = cam['wrist_rs_cam']
        exo_rs_cam = cam['exo_rs_cam']
        table_rs_cam = cam['table_rs_cam']
    else:
        piper = None
        wrist_rs_cam = None
        exo_rs_cam = None
        table_rs_cam = None

    if cfg.use_devices:
        wrist_rs_cam.start_recording()
        exo_rs_cam.start_recording()
        table_rs_cam.start_recording()
        time.sleep(0.1)
        logging.info("Devices started recording")

    task = cfg.task
    task_index = get_task_index(task)
    fps = cfg.fps

    dataset_path = os.path.join(cfg.dataset_path, task)
    piper_dataset = PiperDataset(
        root=dataset_path,
        episode_num=cfg.episode_num,
        episode_len=cfg.episode_len,
        create_video=cfg.create_video,
        fps=fps,
        recorded_by=cfg.recorded_by
    )
    logging.info(f"Dataset path: {dataset_path}")

    fk = C_PiperForwardKinematics()
    listener, event = init_keyboard_listener()

    frame_index = 0
    while True:
        logging.info("RECORDING.....")
        if event["stop recording"]:
            logging.info("Exit episode")
            break

        t0 = time.time()

        end_pose_msg = read_end_pose_msg(piper)
        end_pose_ctrl = read_end_pose_ctrl(piper, fk)

        frame = {
            'timestamp': 1 / fps * frame_index,
            'frame_index': frame_index,
            'episode_index': cfg.episode_num,
            # 'index': cfg.episode_num * piper_dataset.num_frames + frame_index,
            'task_index': task_index,
            'action': end_pose_msg,
            'action_fk': end_pose_ctrl,
            'observation.state': end_pose_msg,
            'observation.images.wrist': wrist_rs_cam.image_for_record_enc(),
            'observation.images.exo': exo_rs_cam.image_for_record_enc(),
            'observation.images.table': table_rs_cam.image_for_record_enc(),
        }
        piper_dataset.add_frame(frame)
        t_act = time.time() - t0
        time.sleep(max(0.0, 1 / fps - t_act))

        frame_index += 1

    listener.stop()
    logging.info(f"Episode finished with frame index {frame_index}")
    piper_dataset.episode_len = frame_index / 30
    piper_dataset.frames = frame_index

    if cfg.use_devices:
        wrist_rs_cam.stop_recording()
        exo_rs_cam.stop_recording()
        table_rs_cam.stop_recording()
        logging.info("Devices stopped recording")

    return piper_dataset

if __name__ == "__main__":
    init_logging()
    dataset = record_episodes()
    dataset.save_episode()