import time
import logging
from pprint import pformat
from pathlib import Path

import matplotlib.pyplot as plt
from termcolor import colored
import torch

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
from configs.eval_real_time_ours import EvalRealTimeOursPipelineConfig

from common.utils.random_utils import set_seed
from common.utils.utils import (
    get_safe_torch_device,
    init_logging,
    format_big_number,
    init_keyboard_listener
)
from configs import parser

from common.policies.factory import make_policy, wrap_policy

from common.datasets.lerobot_dataset import LeRobotDatasetMetadata
from common.policies.extensions import ExtendedConfig
from common.policies.lora_msp import LoraMSPConfig
from common.policies.lora_moe import LoraMoELinear


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


@parser.wrap()
def eval_real_time(cfg: EvalRealTimeOursPipelineConfig):
    # ---------------------------------------------------------
    # HYPERPARAMETERS FOR DEBUGGING
    # ---------------------------------------------------------
    cfg.method = ExtendedConfig()
    cfg.method.core = 'lora_msp'
    cfg.method.lora_cfg = LoraMSPConfig(
        r=32,
        alpha=64,
        quantize=False,
        num_experts=4,
        target_threshold=0.99,
        use_spec_loss=False,
        use_modular_loss=False,
        use_id_loss=False,
        router_projection=True,
        routing='top1'
    )
    cfg.method.target_keywords = ["all-linear"]
    cfg.method.adapter_file_path = [
        '/home/minji/Desktop/data/finetuned_model/MoE_LoRa_paper/multi/pi0/lora_msp/pi0_lora_msp_r32_moe_start_0.99__multitask/025000/pretrained_model/adapters.safetensors'
    ]
    cfg.method.is_train = False

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

    logging.info(pformat(cfg.to_dict()))
    #cfg.target_keywords = ['q_proj', 'k_proj', 'v_proj']
    cfg.target_keywords =["linear", "mlp", "proj"]
    cfg.task = 'open the pot'

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

    policy, res = wrap_policy(
        policy = policy,
        cfg = cfg.method,
        is_master = True,
        device = device,
    )
    logging.info(res)


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

            stt = read_end_pose_msg(piper)
            end_pose_data = stt[0][:6].tolist()
            gripper_data = [torch.tensor(60000), GRIPPER_EFFORT]
            ctrl_end_pose(piper, end_pose_data, gripper_data) if piper is not None else None
            print('gripper open')

            time.sleep(3)
            cfg.task = "open the pot"
            logging.info(cfg.task)
            policy.reset()
            task['task1 : open the pot'] = False
            continue
        if cfg.use_devices and task['task2 : pour the block']:
            set_zero_configuration(piper)

            stt = read_end_pose_msg(piper)
            end_pose_data = stt[0][:6].tolist()
            gripper_data = [torch.tensor(60000), GRIPPER_EFFORT]
            ctrl_end_pose(piper, end_pose_data, gripper_data) if piper is not None else None
            print('gripper open')

            time.sleep(3)
            cfg.task = "pour the block into the basket"
            logging.info(cfg.task)
            policy.reset()
            task['task2 : pour the block'] = False
            continue
        if cfg.use_devices and task['task3 : push the button']:
            set_zero_configuration(piper)
            time.sleep(3)
            cfg.task = "push the button"
            logging.info(cfg.task)
            policy.reset()
            task['task3 : push the button'] = False
            continue
        if cfg.use_devices and task['task4 : pick and place']:
            set_zero_configuration(piper)

            stt = read_end_pose_msg(piper)
            end_pose_data = stt[0][:6].tolist()
            gripper_data = [torch.tensor(60000), GRIPPER_EFFORT]
            ctrl_end_pose(piper, end_pose_data, gripper_data) if piper is not None else None
            print('gripper open')

            time.sleep(3)
            cfg.task = "pick and place the grape in the basket"
            logging.info(cfg.task)
            policy.reset()
            task['task4 : pick and place'] = False
            continue

            #
        # create batch
        print(cfg.task)
        batch = create_batch(piper, table_rs_cam, wrist_rs_cam, cfg.use_devices, cfg.task)
        t_create_batch = log_time()

        for key in batch:
            if isinstance(batch[key], torch.Tensor):
                batch[key] = batch[key].to(device, non_blocking=True)
        t_batch_to_gpu = log_time()

        # infer data
        with torch.no_grad():
            action_pred = policy.select_action(batch).squeeze()
        if len(policy._action_queue) < cfg.infer_chunk:
            policy.reset()
        logged_time = policy.logged_time
        t_action_pred = log_time()
        if cfg.temporal_ensemble:
            action_pred_queue = policy._action_queue.copy()
            action_pred_queue.extendleft(action_pred.unsqueeze(0))
            policy.reset()

            buffer = load_buffer(buffer, action_pred_queue)
            buffer, action_pred = get_current_action(buffer)
            buffer.append([])

        # actuate robot
        end_pose_data = action_pred[:6].cpu().to(dtype=int).tolist()
        gripper_data = [action_pred[6].cpu().to(dtype=int), GRIPPER_EFFORT]
        ctrl_end_pose(piper, end_pose_data, gripper_data) if piper is not None else None
        t_action_publish = log_time()

        ###############
        # === Router 통계 출력 ===
        expert_counts = None
        for name, module in policy.named_modules():
            if isinstance(module, LoraMoELinear):
                _, gates = module.get_router_tensor()
                if gates is not None:
                    # (batch, seq, experts) → experts 합산
                    layer_sum = gates.sum(dim=(0, 1))  # shape [E]1
                    if expert_counts is None:
                        expert_counts = layer_sum
                    else:
                        expert_counts += layer_sum

        if expert_counts is not None:
            print("*Expert usage (all layers combined):", expert_counts.cpu().tolist())
        # ========================

        ###############

        # log data
        action_pred_list.append(action_pred.cpu() if isinstance(action_pred, torch.Tensor) else action_pred)

        step += 1
        time.sleep(0.2)

        t_total = log_time()
        logged_time = logged_time | {
            "action_pred": action_pred,
            "t_create_batch": t_create_batch - t_start,
            "t_batch_to_gpu": t_batch_to_gpu - t_create_batch,
            "t_action_pred": t_action_pred - t_batch_to_gpu,
            "t_action_publish": t_action_publish - t_action_pred,
            "t_total": t_total - t_start,
        }
        logging.info(colored(pformat(logged_time), "yellow", attrs=["bold"]))

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