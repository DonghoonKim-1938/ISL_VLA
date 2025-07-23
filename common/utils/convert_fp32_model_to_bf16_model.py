import time
import logging
from pprint import pformat, pp
from dataclasses import asdict

import matplotlib.pyplot as plt
from termcolor import colored
import torch
import numpy as np
from huggingface_hub import login
from piper_sdk import C_PiperInterface

from common.constants import GRIPPER_EFFORT
from common.robot_devices.cam_utils import RealSenseCamera
from common.robot_devices.robot_utils import read_end_pose_msg, set_zero_configuration, ctrl_end_pose
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
from configs.eval_real_time_ours import EvalRealTimeOursPipelineConfig

from configs import parser

from common.policies.factory import make_policy
from common.datasets.lerobot_dataset import LeRobotDatasetMetadata

from common.utils.random_utils import set_seed
from common.utils.utils import (
    get_safe_torch_device,
    init_logging,
    format_big_number,
)

@parser.wrap()
def eval_main(cfg: EvalRealTimeOursPipelineConfig):
    logging.info(pformat(asdict(cfg)))
    if cfg.use_devices:
        piper, cam = init_devices(cfg)

        wrist_rs_cam = cam['wrist_rs_cam']
        exo_rs_cam = cam['exo_rs_cam']
        table_rs_cam = cam['table_rs_cam']
    else:
        piper = None
        wrist_rs_cam = None
        exo_rs_cam = None
        table_rs_cam = None

    device = get_safe_torch_device(cfg.policy.device, log=True)

    torch.backends.cudnn.benchmark = True
    torch.backends.cuda.matmul.allow_tf32 = True
    set_seed(cfg.seed)

    logging.info(colored("Output dir:", "yellow", attrs=["bold"]) + f" {cfg.output_dir}")

    logging.info("Creating dataset")
    train_dataset_meta = LeRobotDatasetMetadata(
        cfg.train_dataset.repo_id, cfg.train_dataset.root, revision=cfg.train_dataset.revision
    )

    logging.info("Making policy.")

    policy_fp32 = make_policy(
        cfg=cfg.policy,
        ds_meta=train_dataset_meta,
    )
    # policy_bf16 = make_policy(
    #     cfg=cfg.policy,
    #     ds_meta=train_dataset_meta,
    #     bf16 = True
    # )

    model = policy_fp32.model


if __name__ == "__main__":
    init_logging()
    eval_main()