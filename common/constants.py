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
# keys
import os
from pathlib import Path

import torch
from huggingface_hub.constants import HF_HOME

OBS_ENV = "observation.environment_state"
OBS_ROBOT = "observation.state"
OBS_IMAGE = "observation.image"
OBS_IMAGES = "observation.images"
ACTION = "action"

# files & directories
CHECKPOINTS_DIR = "checkpoints"
LAST_CHECKPOINT_LINK = "last"
PRETRAINED_MODEL_DIR = "pretrained_model"
TRAINING_STATE_DIR = "training_state"
RNG_STATE = "rng_state.safetensors"
TRAINING_STEP = "training_step.json"
OPTIMIZER_STATE = "optimizer_state.safetensors"
OPTIMIZER_PARAM_GROUPS = "optimizer_param_groups.json"
SCHEDULER_STATE = "scheduler_state.json"

# cache dir
default_cache_path = Path(HF_HOME) / "lerobot"
HF_LEROBOT_HOME = Path(os.getenv("HF_LEROBOT_HOME", default_cache_path)).expanduser()

if "LEROBOT_HOME" in os.environ:
    raise ValueError(
        f"You have a 'LEROBOT_HOME' environment variable set to '{os.getenv('LEROBOT_HOME')}'.\n"
        "'LEROBOT_HOME' is deprecated, please use 'HF_LEROBOT_HOME' instead."
    )

WRIST_CAM_SN = "339522300614"
EXO_CAM_SN = "f1371608"
TABLE_CAM_SN = "f1371426"

GRIPPER_EFFORT = 500

TASK_LIST = [
    "test",
    "align the cups",
    "pick the grape and put it to the basket",
    "pick the corn and put it to the basket",
    "open the pot",
    "pour the block into the basket",
    "push the button",
]

META_INFO_TEMPLATE = {
    "codebase_version": "v2.0",
    "robot_type": "piper",
    "total_episodes": 600,
    "total_frames": 35200,
    "total_tasks": 3,
    "total_videos": 600,
    "total_chunks": 12,
    "chunks_size": 50,
    "fps": 5,
    "splits": {
        "train": "0:600"
    },
    "data_path": "data/chunk-{episode_chunk:03d}/episode_{episode_index:06d}.parquet",
    "video_path": "videos/chunk-{episode_chunk:03d}/{video_key}/episode_{episode_index:06d}.mp4",
    "features": {
        "action": {
            "dtype": "float32",
            "shape": [
                7
            ],
            "names": [
                "x",
                "y",
                "z",
                "rx",
                "ry",
                "rz",
                "gripper"
            ]
        },
        "observation.state": {
            "dtype": "float32",
            "shape": [
                7
            ],
            "names": [
                "x",
                "y",
                "z",
                "rx",
                "ry",
                "rz",
                "gripper"
            ]
        },
        "observation.images.table": {
            "dtype": "video",
            "shape": [
                480,
                640,
                3
            ],
            "names": [
                "height",
                "width",
                "channels"
            ],
            "info": {
                "video.fps": 5.0,
                "video.height": 480,
                "video.width": 640,
                "video.channels": 3,
                "video.codec": "av1",
                "video.pix_fmt": "yuv420p",
                "video.is_depth_map": False,
                "has_audio": False
            }
        },
        "observation.images.wrist": {
            "dtype": "video",
            "shape": [
                480,
                640,
                3
            ],
            "names": [
                "height",
                "width",
                "channels"
            ],
            "info": {
                "video.fps": 5.0,
                "video.height": 480,
                "video.width": 640,
                "video.channels": 3,
                "video.codec": "av1",
                "video.pix_fmt": "yuv420p",
                "video.is_depth_map": False,
                "has_audio": False
            }
        },
        "timestamp": {
            "dtype": "float32",
            "shape": [
                1
            ],
            "names": None
        },
        "frame_index": {
            "dtype": "int64",
            "shape": [
                1
            ],
            "names": None
        },
        "episode_index": {
            "dtype": "int64",
            "shape": [
                1
            ],
            "names": None
        },
        "index": {
            "dtype": "int64",
            "shape": [
                1
            ],
            "names": None
        },
        "task_index": {
            "dtype": "int64",
            "shape": [
                1
            ],
            "names": None
        }
    }
}
META_STATS_TEMPLATE = {
    "action": {
        "mean": [
            184750.00428977,
	        62906.23678977,
	        263574.93028409,
	        -56545.81463068,
            43992.02488636,
	        -80643.40144886,
	        60145.86931818
        ],
        "std": [
            102962.65069191,
	        113760.93033841,
	        92895.24751794,
	        148413.88694479,
	        29076.630872,
	        116143.05429282,
            13249.54133897
        ],
        "max": [
            448213,
            321698,
            574275,
            180000,
            90000,
            180000,
            72240
        ],
        "min": [
            -56501,
            -211263,
            0,
            -179969,
            -11012,
            -179968,
            -1470
        ]
    },
    "observation.state": {
        "mean": [
            184747.916875,
            62905.64642045,
            263577.69204545,
            -56504.12042614,
            43991.99284091,
            -80637.97610795,
            60145.88522727
        ],
        "std": [
            102962.28290167,
            113760.24414793,
            92895.45537357,
            148429.34473241,
            29076.48597257,
            116142.33207768,
            13249.43522619
        ],
        "max": [
            448213,
            321698,
            574275,
            180000,
            90000,
            180000,
            72240
        ],
        "min": [
            -56501,
            -211263,
            0,
            -179969,
            -11012,
            -179968,
            -1470
        ]
    },
    "timestamp": {
        "mean": [
            5.912466049194336
        ],
        "std": [
            3.6292903423309326
        ],
        "max": [
            16.799999237060547
        ],
        "min": [
            0.0
        ]
    },
    "frame_index": {
        "mean": [
            177.37397727272727
        ],
        "std": [
            108.87871008404305
        ],
        "max": [
            504
        ],
        "min": [
            0
        ]
    },
    "episode_index": {
        "mean": [
            288.24105113636364
        ],
        "std": [
            174.30801099730152
        ],
        "max": [
            599
        ],
        "min": [
            0
        ]
    },
    "task_index": {
        "mean": [
            2.0
        ],
        "std": [
            0.0
        ],
        "max": [
            2
        ],
        "min": [
            2
        ]
    },
    "index": {
        "mean": [
            17599.5
        ],
        "std": [
            10161.5090742796
        ],
        "max": [
            35199
        ],
        "min": [
            0
        ]
    },
    "observation.images.table": {
        "mean": [
            [
                [
                    91.68148143
                ]
            ],
            [
                [
                    95.98407115
                ]
            ],
            [
                [
                    99.49233975
                ]
            ]
        ],
        "std": [
            [
                [
                    42.90627931
                ]
            ],
            [
                [
                    46.07452878
                ]
            ],
            [
                [
                    47.25431926
                ]
            ]
        ],
        "max": [
            [
                [
                    255.0
                ]
            ],
            [
                [
                    255.0
                ]
            ],
            [
                [
                    255.0
                ]
            ]
        ],
        "min": [
            [
                [
                    0.0
                ]
            ],
            [
                [
                    0.0
                ]
            ],
            [
                [
                    0.0
                ]
            ]
        ]
    },
    "observation.images.wrist": {
        "mean": [
            [
                [
                    93.06738208
                ]
            ],
            [
                [
                    98.86193299
                ]
            ],
            [
                [
                    128.35764402
                ]
            ]
        ],
        "std": [
            [
                [
                    70.559473
                ]
            ],
            [
                [
                    75.98714874
                ]
            ],
            [
                [
                    68.42766891
                ]
            ]
        ],
        "max": [
            [
                [
                    255.0
                ]
            ],
            [
                [
                    255.0
                ]
            ],
            [
                [
                    255.0
                ]
            ]
        ],
        "min": [
            [
                [
                    0.0
                ]
            ],
            [
                [
                    0.0
                ]
            ],
            [
                [
                    0.0
                ]
            ]
        ]
    }
}

def deg2rad(deg):
    return deg * torch.pi / 180


def rad2deg(rad):
    return rad * 180 / torch.pi