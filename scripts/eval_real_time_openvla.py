import time
import logging
from pprint import pformat, pp
from dataclasses import asdict
from transformers import AutoModelForVision2Seq, AutoProcessor, BitsAndBytesConfig
import matplotlib.pyplot as plt
from termcolor import colored
import torch
import numpy as np
from huggingface_hub import login
from piper_sdk import C_PiperInterface
import PIL.Image as Image
from common.constants import GRIPPER_EFFORT
from common.robot_devices.cam_utils import RealSenseCamera
from common.robot_devices.robot_utils import read_end_pose_msg, set_zero_configuration, ctrl_end_pose
from common.utils.utils import (
    random_piper_image_openvla,
    plot_trajectory,
    pretty_plot,
    log_time,
    init_devices
)
from configs.eval_real_time_ours import EvalRealTimeOursPipelineConfig
from configs import parser
#
# from common.policies.factory import make_policy
# from common.datasets.lerobot_dataset import LeRobotDatasetMetadata
from common.utils.random_utils import set_seed
from common.utils.wandb_utils import WandBLogger
from common.utils.utils import (
    get_safe_torch_device,
    init_logging,
    format_big_number,
)
def create_batch(piper, table_rs_cam, use_devices, task):
    if use_devices:
        return {
            # 'observation.state': read_end_pose_msg(piper),
            # 'observation.images.exo': exo_rs_cam.image_for_inference(),
            # 'observation.images.wrist': wrist_rs_cam.image_for_inference(),
            'observation.images.table': table_rs_cam.image_for_inference_openvla(),
            'task': [task],
        }
    else:
        return {
            # 'observation.state': random_piper_action(),
            # 'observation.images.exo': random_piper_image(),
            # 'observation.images.wrist': random_piper_image(),
            'observation.images.table': random_piper_image_openvla(),
            'task': [task],
        }
@parser.wrap()
def eval_main(cfg: EvalRealTimeOursPipelineConfig):
    logging.info(pformat(asdict(cfg)))
    if cfg.use_devices:
        piper, cam = init_devices(cfg)
        # wrist_rs_cam = cam['wrist_rs_cam']
        # exo_rs_cam = cam['exo_rs_cam']
        table_rs_cam = cam['table_rs_cam']
    else:
        piper = None
        # wrist_rs_cam = None
        # exo_rs_cam = None
        table_rs_cam = None
    #
    if cfg.wandb.enable and cfg.wandb.project:
        wandb_logger = WandBLogger(cfg)
    else:
        wandb_logger = None
        logging.info(colored("Logs will be saved locally.", "yellow", attrs=["bold"]))
    device = get_safe_torch_device(cfg.policy.device, log=True)
    torch.backends.cudnn.benchmark = True
    torch.backends.cuda.matmul.allow_tf32 = True
    set_seed(cfg.seed)
    logging.info(colored("Output dir:", "yellow", attrs=["bold"]) + f" {cfg.output_dir}")
    logging.info("Creating dataset")
    dtype = torch.bfloat16

    processor = AutoProcessor.from_pretrained(
        "/home/minji/Desktop/codes/ckpt/step_2000",
                                    trust_remote_code=True)
    dtype = torch.bfloat16
    policy = AutoModelForVision2Seq.from_pretrained(
        "/home/minji/Desktop/codes/ckpt/step_2000",
        # attn_implementation="flash_attention_2",  # [Optional] Requires `flash_attn`
        torch_dtype=dtype,
        quantization_config=BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4", bnb_4bit_use_double_quant=True, bnb_4bit_compute_dtype=dtype),
        # low_cpu_mem_usage=True,
        trust_remote_code=True
    ).to("cuda")

    import json
    with open("/home/minji/Desktop/codes/ckpt/step_2000/config.json","rb") as f:
        config = json.load(f)
    policy.norm_stats = config['norm_stats']

    step = 0
    num_total_params = sum(p.numel() for p in policy.parameters())
    logging.info(colored("Output dir:", "yellow", attrs=["bold"]) + f" {cfg.output_dir}")
    if cfg.env is not None:
        logging.info(f"{cfg.env.task=}")
    logging.info(f"{num_total_params=} ({format_big_number(num_total_params)})")
    if cfg.use_devices:
        table_rs_cam.start_recording()
        logging.info("Devices started recording")
    policy.eval()
    logging.info("Start offline evaluation on a fixed dataset")
    # buffer = [[] for _ in range(policy.config.n_action_steps)]
    action_pred_list = []
    fig_2d, ax_2d = plt.subplots(4, 2, figsize=[25, 15])
    fig_3d, ax_3d = plt.subplots(subplot_kw={'projection': '3d'}, figsize=[25, 15])
    l = np.load("/home/minji/Desktop/codes/lerobot/lerobot/custom_scripts/scripts/predictions_111_latest.npy")
    for i in range(10):
        end_pose_data = l[i][:6].astype(int).tolist()
        gripper_data = [l[i][6].astype(int), GRIPPER_EFFORT]
        ctrl_end_pose(piper, end_pose_data, gripper_data) if piper is not None else None
        print(l[i])
        time.sleep(0.2)
    while True:
        t_start = log_time()
        # create batch
        batch = create_batch(piper, table_rs_cam, cfg.use_devices, cfg.task)
        t_create_batch = log_time()
        for key in batch:
            if isinstance(batch[key], torch.Tensor):
                batch[key] = batch[key].to(device, non_blocking=True)
        t_batch_to_gpu = log_time()
        # infer data
        prompt = f"In: What action should the robot take to {batch['task']}?\nOut:"
        inputs = processor(prompt, batch['observation.images.table']).to("cuda", dtype=torch.bfloat16)
        action_pred = policy.predict_action(**inputs, unnorm_key="piper5_hz_subtask", do_sample=False)
        # logged_time = policy.logged_time
        t_action_pred = log_time()
        # t_action_pred = log_time()
        # if cfg.temporal_ensemble:
        #     action_pred_queue = policy._action_queue.copy()
        #     action_pred_queue.extendleft(action_pred.unsqueeze(0))
        #     policy.reset()
        #
        #     buffer = load_buffer(buffer, action_pred_queue)
        #     buffer, action_pred = get_current_action(buffer)
        #     buffer.append([])
        # actuate robot
        end_pose_data = action_pred[:6].astype(int).tolist()
        gripper_data = [action_pred[6].astype(int), GRIPPER_EFFORT]
        ctrl_end_pose(piper, end_pose_data, gripper_data) if piper is not None else None
        t_action_publish = log_time()
        # log data
        action_pred_list.append(action_pred.cpu() if isinstance(action_pred, torch.Tensor) else action_pred)
        step += 1
        time.sleep(0.2)
        t_total = log_time()
        logged_time = {
            "t_create_batch": t_create_batch - t_start,
            "t_batch_to_gpu": t_batch_to_gpu - t_create_batch,
            "t_action_pred": t_action_pred - t_batch_to_gpu,
            "t_action_publish": t_action_publish - t_action_pred,
            "t_total": t_total - t_start,
            "action_pred": action_pred
        }
        logging.info(colored(pformat(logged_time), "yellow", attrs=["bold"]))
        if step > cfg.max_steps:
            break
        pass
    plot_trajectory(ax_2d, action_pred_list)
    pretty_plot(ax_2d)
    plot_trajectory(ax_3d, action_pred_list, projection='3d')
    pretty_plot(ax_3d)
    fig_2d.show()
    fig_3d.show()

if __name__ == "__main__":
    init_logging()
    eval_main()
