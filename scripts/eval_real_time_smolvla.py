import time
import logging
from pprint import pformat
from dataclasses import asdict
from pathlib import Path

import matplotlib.pyplot as plt
from termcolor import colored
import torch
import safetensors.torch as sft
import copy

from common.constants import GRIPPER_EFFORT
from common.robot_devices.robot_utils import read_end_pose_msg, ctrl_end_pose, read_joint_msg, set_zero_configuration
from common.utils.utils import (
    load_buffer,
    get_current_action,
    random_piper_action,
    random_piper_image,
    plot_trajectory,
    pretty_plot,
    log_time,
    init_devices
)
from common.utils.wandb_utils import WandBLogger
from configs.eval_real_time_ours import EvalRealTimeOursPipelineConfig

from common.utils.logging_utils import AverageMeter, MetricsTracker
from common.utils.random_utils import set_seed
from common.utils.utils import (
    get_safe_torch_device,
    init_logging,
    format_big_number,
    init_keyboard_listener
)
from configs import parser

from common.policies.factory import make_policy

# Adapter injection utilities
from common.policies.qlora import inject_qlora, QLoRAConfig as InjectQLoRAConfig
from common.policies.lora import inject_lora, LoRAConfig as InjectLoRAConfig
from common.policies.prefix_tuning import inject_prefix_tuning, PrefixTuningConfig
from common.policies.lora_moe import inject_lora_moe
from common.policies.lora_moe import LoRAMoEConfig
from common.datasets.lerobot_dataset import LeRobotDatasetMetadata
from common.utils.adapter_utils import load_adapters

from dataclasses import asdict, is_dataclass
from enum import Enum
from pathlib import Path
from typing import Any

from collections import deque




def create_batch(piper, table_rs_cam, wrist_rs_cam, use_devices, task, use_end_pose: bool = True):
    if use_devices:
        return {
            'observation.state': read_end_pose_msg(piper) if use_end_pose else read_joint_msg(piper),
            'observation.images.table': table_rs_cam.image_for_inference(),
            'observation.images.wrist': wrist_rs_cam.image_for_inference(),
            'task': [task],
        }
    else:
        return {
            'observation.state': random_piper_action(),
            'observation.images.table': random_piper_image(),
            'observation.images.wrist': random_piper_image(),
            'task': [task],
        }

def safe_serialize(x: Any):
    if x is None:
        return None
    if isinstance(x, Enum):
        return x.value
    if is_dataclass(x):
        return {k: safe_serialize(v) for k, v in asdict(x).items()}
    if isinstance(x, dict):
        return {k: safe_serialize(v) for k, v in x.items()}
    if isinstance(x, (list, tuple, set)):
        return [safe_serialize(v) for v in x]
    if isinstance(x, Path):
        return str(x)
    try:
        import numpy as np
        if isinstance(x, np.generic):
            return x.item()
    except Exception:
        pass
    return x if isinstance(x, (str, int, float, bool)) else str(x)

@parser.wrap()
def eval_real_time(cfg: EvalRealTimeOursPipelineConfig):
    cfg.qlora_cfg = {'r':cfg.rank_size,'alpha': cfg.rank_size * 2}
    cfg.lora_cfg = {'r':cfg.rank_size,'alpha': cfg.rank_size * 2}
    print(cfg.lora_cfg)
    cfg.target_keywords=["all-linear"]

    ###############
    # INIT DEVICES
    ###############
    if cfg.use_devices:
        piper, cam = init_devices(cfg)
        wrist_rs_cam = cam['wrist_rs_cam']
        exo_rs_cam = cam['exo_rs_cam']
        table_rs_cam = cam['table_rs_cam']

        listener, event, task = init_keyboard_listener()

    else:
        piper = None
        wrist_rs_cam = None
        exo_rs_cam = None
        table_rs_cam = None

        listener, event, task = None, None, None

    # logging.info(pformat(cfg.to_dict()))
    try:
        logging.info(pformat(cfg.to_dict()))
    except Exception:
        logging.warning("cfg.to_dict() failed; falling back to safe_serialize for logging.")
        logging.info(pformat(safe_serialize(cfg)))

    device = get_safe_torch_device(cfg.policy.device, log=True)
    torch.backends.cudnn.benchmark = True

    if cfg.seed is not None:
        set_seed(cfg.seed)

    logging.info(colored("Output dir:", "yellow", attrs=["bold"]) + f" {cfg.output_dir}")

    ###############
    # LOAD DATASET
    ###############
    logging.info("Creating dataset")
    train_dataset_meta = LeRobotDatasetMetadata(
        cfg.train_dataset.repo_id, cfg.train_dataset.root, revision=cfg.train_dataset.revision
    )

    logging.info("Making policy.")

    pretrained_path = Path(cfg.policy.pretrained_path) if cfg.policy and cfg.policy.pretrained_path else None
    adapter_path = Path(cfg.adapter_path) if cfg.adapter_path else None

    ###############
    # MAKE POLICY
    ###############
    policy = make_policy(
        cfg=cfg.policy,
        ds_meta=train_dataset_meta,
    )

    # Adapter injection
    if getattr(cfg, "use_qlora", False):
        qlora_cfg_obj = InjectQLoRAConfig(**(cfg.qlora_cfg or {}))
        policy, _ = inject_qlora(policy, qlora_cfg_obj, target_keywords=cfg.target_keywords)
        logging.info("Injected QLoRA modules")

    elif getattr(cfg, "use_lora", False):
        lora_cfg_obj = InjectLoRAConfig(**(cfg.lora_cfg or {}))
        policy, _ = inject_lora(policy, lora_cfg_obj, target_keywords=cfg.target_keywords)
        logging.info("Injected LoRA modules")

    elif getattr(cfg, "use_prefix_tuning", False):
        pt_cfg_obj = PrefixTuningConfig(**(cfg.prefix_tuning_cfg or {}))
        policy, _ = inject_prefix_tuning(policy, pt_cfg_obj, target_keywords=cfg.target_keywords)
        logging.info("Injected Prefix-Tuning modules")

    elif getattr(cfg, "use_lora_moe", False):
        lora_moe_cfg_obj = LoRAMoEConfig(**(cfg.lora_moe_cfg or {}))
        policy, _ = inject_lora_moe(policy, lora_moe_cfg_obj, target_keywords=cfg.target_keywords)
        logging.info("Injected LoRA-MoE modules")

    policy.to(device)

    if pretrained_path and pretrained_path.is_dir():
        if cfg.use_qlora or cfg.use_lora or cfg.use_prefix_tuning:
            adapters_file = cfg.adapter_path / "adapters.safetensors"
        else:
            adapters_file = None

        if adapters_file and adapters_file.exists():
            res, policy = load_adapters(policy, adapters_file, device=device)

    ###############
    # LOG BEFORE EVAL
    ###############
    step = 0
    num_total_params = sum(p.numel() for p in policy.parameters())

    logging.info(colored("Output dir:", "yellow", attrs=["bold"]) + f" {cfg.output_dir}")
    if cfg.env is not None:
        logging.info(f"{cfg.env.task=}")
    logging.info(f"{num_total_params=} ({format_big_number(num_total_params)})")

    if cfg.use_devices:
        wrist_rs_cam.start_recording()
        exo_rs_cam.start_recording()
        table_rs_cam.start_recording()
        logging.info("Devices started recording")

    policy.eval()

    logging.info("Start offline evaluation on a fixed dataset")

    buffer = [[] for _ in range(policy.config.n_action_steps)]
    action_pred_list = []

    fig_2d, ax_2d = plt.subplots(4, 2, figsize=[25, 15])
    fig_3d, ax_3d = plt.subplots(subplot_kw={'projection': '3d'}, figsize=[25, 15])


    ###############
    # EVAL LOOP
    ###############
    while True:
        t_start = log_time()

        # emergency stop
        if cfg.use_devices and event["stop recording"]:

            set_zero_configuration(piper)
            time.sleep(1)
            logging.info('EMERGENCY STOP... RESTARTING...')
            policy.reset()
            event['stop recording'] = False
            continue

        if cfg.use_devices and task['task1 : open the pot']:
            set_zero_configuration(piper)
            time.sleep(3)
            cfg.task = "open the pot"
            logging.info(cfg.task)
            policy.reset()
            task['task1 : open the pot'] = False
            continue
        if cfg.use_devices and task['task2 : pour the block']:
            set_zero_configuration(piper)
            time.sleep(3)
            cfg.task = "pour the block into the basket"
            logging.info(cfg.task)
            policy.reset()
            task['task2 : pour the block'] = False
            continue
        if cfg.use_devices and task['task3 : press the button']:
            set_zero_configuration(piper)
            time.sleep(3)
            cfg.task = "press the button"
            logging.info(cfg.task)
            policy.reset()
            task['task3 : press the button'] = False
            continue
        if cfg.use_devices and task['task4 : pick and place']:
            set_zero_configuration(piper)
            time.sleep(3)
            cfg.task = "pick and place the grape in the basket"
            logging.info(cfg.task)
            policy.reset()
            task['task4 : pick and place'] = False
            continue


        # create batch
        print(cfg.task)
        batch = create_batch(piper, table_rs_cam, wrist_rs_cam, cfg.use_devices, cfg.task)

        for key in batch:
            if isinstance(batch[key], torch.Tensor):
                batch[key] = batch[key].to(device, non_blocking=True)

        # infer data
        action_pred = policy.select_action(batch).squeeze()
        # if len(policy._action_queue) < cfg.infer_chunk:
        #     policy.reset()


        # actuate robot
        end_pose_data = action_pred[:6].cpu().to(dtype=int).tolist()
        gripper_data = [action_pred[6].cpu().to(dtype=int), GRIPPER_EFFORT]
     #   print(action_pred)
        ctrl_end_pose(piper, end_pose_data, gripper_data) if piper is not None else None

        # log data
        action_pred_list.append(action_pred.cpu() if isinstance(action_pred, torch.Tensor) else action_pred)

        step += 1
        time.sleep(0.2)

        if step > cfg.max_steps:
            break
        pass

    plot_trajectory(ax_2d, action_pred_list)
    pretty_plot(fig_2d, ax_2d, title='2d traj')

    plot_trajectory(ax_3d, action_pred_list, projection='3d')
    pretty_plot(fig_3d, ax_3d, title='3d traj')

    fig_2d.show()
    fig_3d.show()


if __name__ == "__main__":
    init_logging()
    eval_real_time()