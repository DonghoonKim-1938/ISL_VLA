#!/usr/bin/env python

# Copyright 2024 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
import logging
import os
import os.path as osp
import platform
import subprocess
import time
from copy import copy
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch

from PIL import Image

from common.robot_devices.robot_utils import init_robot
from common.robot_devices.cam_utils import RealSenseCamera
from common.constants import TASK_LIST

def none_or_int(value):
    if value == "None":
        return None
    return int(value)


def inside_slurm():
    """Check whether the python process was launched through slurm"""
    # TODO(rcadene): return False for interactive mode `--pty bash`
    return "SLURM_JOB_ID" in os.environ


def auto_select_torch_device() -> torch.device:
    """Tries to select automatically a torch device."""
    if torch.cuda.is_available():
        logging.info("Cuda backend detected, using cuda.")
        return torch.device("cuda")
    elif torch.backends.mps.is_available():
        logging.info("Metal backend detected, using cuda.")
        return torch.device("mps")
    else:
        logging.warning("No accelerated backend detected. Using default cpu, this will be slow.")
        return torch.device("cpu")


# TODO(Steven): Remove log. log shouldn't be an argument, this should be handled by the logger level
def get_safe_torch_device(try_device: str, log: bool = False) -> torch.device:
    """Given a string, return a torch.device with checks on whether the device is available."""
    try_device = str(try_device)
    match try_device:
        case "cuda":
            assert torch.cuda.is_available()
            device = torch.device("cuda")
        case "mps":
            assert torch.backends.mps.is_available()
            device = torch.device("mps")
        case "cpu":
            device = torch.device("cpu")
            if log:
                logging.warning("Using CPU, this will be slow.")
        case _:
            device = torch.device(try_device)
            if log:
                logging.warning(f"Using custom {try_device} device.")

    return device


def get_safe_dtype(dtype: torch.dtype, device: str | torch.device):
    """
    mps is currently not compatible with float64
    """
    if isinstance(device, torch.device):
        device = device.type
    if device == "mps" and dtype == torch.float64:
        return torch.float32
    else:
        return dtype


def is_torch_device_available(try_device: str) -> bool:
    try_device = str(try_device)  # Ensure try_device is a string
    if try_device == "cuda":
        return torch.cuda.is_available()
    elif try_device == "mps":
        return torch.backends.mps.is_available()
    elif try_device == "cpu":
        return True
    else:
        raise ValueError(f"Unknown device {try_device}. Supported devices are: cuda, mps or cpu.")


def is_amp_available(device: str):
    if device in ["cuda", "cpu"]:
        return True
    elif device == "mps":
        return False
    else:
        raise ValueError(f"Unknown device '{device}.")


def init_logging():
    def custom_format(record):
        dt = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        fnameline = f"{record.pathname}:{record.lineno}"
        message = f"{record.levelname} {dt} {fnameline[-15:]:>15} {record.msg}"
        return message

    logging.basicConfig(level=logging.INFO)

    for handler in logging.root.handlers[:]:
        logging.root.removeHandler(handler)

    formatter = logging.Formatter()
    formatter.format = custom_format
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)
    logging.getLogger().addHandler(console_handler)


def format_big_number(num, precision=0):
    suffixes = ["", "K", "M", "B", "T", "Q"]
    divisor = 1000.0

    for suffix in suffixes:
        if abs(num) < divisor:
            return f"{num:.{precision}f}{suffix}"
        num /= divisor

    return num


def _relative_path_between(path1: Path, path2: Path) -> Path:
    """Returns path1 relative to path2."""
    path1 = path1.absolute()
    path2 = path2.absolute()
    try:
        return path1.relative_to(path2)
    except ValueError:  # most likely because path1 is not a subpath of path2
        common_parts = Path(osp.commonpath([path1, path2])).parts
        return Path(
            "/".join([".."] * (len(path2.parts) - len(common_parts)) + list(path1.parts[len(common_parts) :]))
        )


def print_cuda_memory_usage():
    """Use this function to locate and debug memory leak."""
    import gc

    gc.collect()
    # Also clear the cache if you want to fully release the memory
    torch.cuda.empty_cache()
    print("Current GPU Memory Allocated: {:.2f} MB".format(torch.cuda.memory_allocated(0) / 1024**2))
    print("Maximum GPU Memory Allocated: {:.2f} MB".format(torch.cuda.max_memory_allocated(0) / 1024**2))
    print("Current GPU Memory Reserved: {:.2f} MB".format(torch.cuda.memory_reserved(0) / 1024**2))
    print("Maximum GPU Memory Reserved: {:.2f} MB".format(torch.cuda.max_memory_reserved(0) / 1024**2))


def capture_timestamp_utc():
    return datetime.now(timezone.utc)


def say(text, blocking=False):
    system = platform.system()

    if system == "Darwin":
        cmd = ["say", text]

    elif system == "Linux":
        cmd = ["spd-say", text]
        if blocking:
            cmd.append("--wait")

    elif system == "Windows":
        cmd = [
            "PowerShell",
            "-Command",
            "Add-Type -AssemblyName System.Speech; "
            f"(New-Object System.Speech.Synthesis.SpeechSynthesizer).Speak('{text}')",
        ]

    else:
        raise RuntimeError("Unsupported operating system for text-to-speech.")

    if blocking:
        subprocess.run(cmd, check=True)
    else:
        subprocess.Popen(cmd, creationflags=subprocess.CREATE_NO_WINDOW if system == "Windows" else 0)


def log_say(text, play_sounds, blocking=False):
    logging.info(text)

    if play_sounds:
        say(text, blocking)


def get_channel_first_image_shape(image_shape: tuple) -> tuple:
    shape = copy(image_shape)
    if shape[2] < shape[0] and shape[2] < shape[1]:  # (h, w, c) -> (c, h, w)
        shape = (shape[2], shape[0], shape[1])
    elif not (shape[0] < shape[1] and shape[0] < shape[2]):
        raise ValueError(image_shape)

    return shape


def has_method(cls: object, method_name: str) -> bool:
    return hasattr(cls, method_name) and callable(getattr(cls, method_name))


def is_valid_numpy_dtype_string(dtype_str: str) -> bool:
    """
    Return True if a given string can be converted to a numpy dtype.
    """
    try:
        # Attempt to convert the string to a numpy dtype
        np.dtype(dtype_str)
        return True
    except TypeError:
        # If a TypeError is raised, the string is not a valid dtype
        return False


def load_buffer(buffer, action_pred_queue):
    for item in buffer:
        item.append(action_pred_queue.popleft().squeeze())
    return buffer

def get_current_action(buffer, m=1.0):
    current_action_stack = torch.stack(buffer.pop(0), dim=0)
    indices = torch.arange(current_action_stack.shape[0])
    weights = torch.exp(-m*indices).cuda()
    weighted_actions = current_action_stack * weights[:, None]  # 가중치 적용
    current_action = weighted_actions.sum(dim=0) / weights.sum()
    return buffer, current_action

def random_piper_action():
    (x, y, z) = torch.rand(3, dtype=torch.float32) * 600000
    (rx, ry, rz) = torch.rand(3, dtype=torch.float32) * 180000
    gripper = torch.rand(1, dtype=torch.float32) * 100000
    return torch.tensor([x,y,z,rx,ry,rz,gripper]).reshape(1,7)

def random_piper_image():
    return torch.rand(1, 3, 480, 640, dtype=torch.float32)

def random_piper_image_openvla(width=224, height=224):
    random_array = np.random.randint(0, 256, size=(height, width, 3), dtype=np.uint8)
    random_image = Image.fromarray(random_array, mode="RGB")
    return random_image

def plot_trajectory(ax, action_list, projection='2d', mode='pred'):
    # action_list = np.array(action_list)
    action_list = torch.stack(action_list, dim=0)
    color = 'r' if mode == 'pred' else 'b'
    if projection == '2d':
        ax[0,0].plot(action_list[:,0], color=color)
        ax[1,0].plot(action_list[:,1], color=color)
        ax[2,0].plot(action_list[:,2], color=color)
        ax[0,1].plot(action_list[:,3], color=color)
        ax[1,1].plot(action_list[:,4], color=color)
        ax[2,1].plot(action_list[:,5], color=color)
        ax[3,0].plot(action_list[:,6], color=color)
    elif projection == '3d':
        ax.plot(action_list[:,0], action_list[:,1], action_list[:,2], color=color)
    else:
        raise ValueError('projection must be \"2d\" or \"3d\"')


def pretty_plot(fig, ax, title):
    fig.suptitle(title)

    ax[0, 0].legend(['prediction', 'ground truth'])
    ax[1, 0].legend(['prediction', 'ground truth'])
    ax[2, 0].legend(['prediction', 'ground truth'])
    ax[0, 1].legend(['prediction', 'ground truth'])
    ax[1, 1].legend(['prediction', 'ground truth'])
    ax[2, 1].legend(['prediction', 'ground truth'])
    ax[3, 0].legend(['prediction', 'ground truth'])

    ax[0, 0].set_xlabel('x')
    ax[1, 0].set_xlabel('y')
    ax[2, 0].set_xlabel('z')
    ax[0, 1].set_xlabel('rot_x')
    ax[1, 1].set_xlabel('rot_y')
    ax[2, 1].set_xlabel('rot_z')
    ax[3, 0].set_xlabel('gripper')

    ax[0, 0].grid()
    ax[1, 0].grid()
    ax[2, 0].grid()
    ax[0, 1].grid()
    ax[1, 1].grid()
    ax[2, 1].grid()
    ax[3, 0].grid()

def log_time():
    return time.perf_counter()

def init_devices(cfg, is_recording=False):
    fps = cfg.fps
    cam_list = cfg.cam_list
    cam = {
        'wrist_rs_cam': None,
        'exo_rs_cam': None,
        'table_rs_cam': None,
    }
    piper = init_robot(is_recording=is_recording)
    if 'wrist' in cam_list:
        cam['wrist_rs_cam'] = RealSenseCamera('wrist', fps)
    if 'exo' in cam_list:
        cam['exo_rs_cam'] = RealSenseCamera('exo', fps)
    if 'table' in cam_list:
        cam['table_rs_cam'] = RealSenseCamera('table', fps)
    return piper, cam

def get_task_index(task):
    task_list = TASK_LIST

    if task not in task_list:
        logging.info("TASK NOT IN TASK LIST")
        raise NotImplementedError
    return task_list.index(task)

def init_keyboard_listener():
    from pynput import keyboard

    event = {}
    event['stop recording'] = False
    task = {}
    task['task1 : open the pot'] = False
    task['task2 : pour the block'] = False
    task['task3 : push the button'] = False
    task['task4 : pick and place'] = False

    def on_press(key):
        if key == keyboard.Key.esc:
            event['stop recording'] = True
            print("esc is pressed")
        elif key == keyboard.KeyCode.from_char('1'):
            task['task1 : open the pot'] = True
            print("task1 : open the pot")
        elif key == keyboard.KeyCode.from_char('2'):
            task['task2 : pour the block'] = True
            print("task2 : pour the block")
        elif key == keyboard.KeyCode.from_char('3'):
            task['task3 : push the button'] = True
            print("task3 : push the button")
        elif key == keyboard.KeyCode.from_char('4'):
            task['task4 : pick and place'] = True
            print("task4 : pick and place")

    listener = keyboard.Listener(on_press=on_press)
    listener.start()

    return listener, event, task

def is_ddp_master(use_ddp: bool, rank: int):
    return (not use_ddp) or (use_ddp and (rank == 0))
