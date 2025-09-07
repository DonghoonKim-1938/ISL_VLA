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
import re
from glob import glob
from pathlib import Path
from typing import Optional, Tuple

import numpy as np
from huggingface_hub.constants import SAFETENSORS_SINGLE_FILE
from termcolor import colored

from common.constants import PRETRAINED_MODEL_DIR
from configs.train import TrainPipelineConfig


def cfg_to_group(cfg: TrainPipelineConfig, return_list: bool = False) -> list[str] | str:
    """Return a group name for logging. Optionally returns group name as list."""
    lst = [
        f"policy:{cfg.policy.type}",
        f"train_dataset:{cfg.train_dataset.repo_id}",
        f"test_dataset:{cfg.test_dataset.repo_id}",
        f"seed:{cfg.seed}",
    ]
    return lst if return_list else "-".join(lst)


def get_wandb_run_id_from_filesystem(log_dir: Path) -> str:
    # Get the WandB run ID.
    paths = glob(str(log_dir / "wandb/latest-run/run-*"))
    if len(paths) != 1:
        raise RuntimeError("Couldn't get the previous WandB run ID for run resumption.")
    match = re.search(r"run-([^\.]+).wandb", paths[0].split("/")[-1])
    if match is None:
        raise RuntimeError("Couldn't get the previous WandB run ID for run resumption.")
    wandb_run_id = match.groups(0)[0]
    return wandb_run_id


def get_safe_wandb_artifact_name(name: str):
    """WandB artifacts don't accept ":" or "/" in their name."""
    return name.replace(":", "_").replace("/", "_")


class WandBLogger:
    """A helper class to log object using wandb."""

    def __init__(self, cfg: TrainPipelineConfig):
        self.cfg = cfg.wandb
        self.log_dir = cfg.output_dir
        self.job_name = cfg.job_name
        self.env_fps = None
        self._group = cfg_to_group(cfg)

        # Set up WandB.
        os.environ["WANDB_SILENT"] = "True"
        import wandb

        wandb_run_id = (
            cfg.wandb.run_id
            if cfg.wandb.run_id
            else get_wandb_run_id_from_filesystem(self.log_dir)
            if cfg.resume
            else None
        )
        wandb.init(
            id=wandb_run_id,
            project=self.cfg.project,
            entity=self.cfg.entity,
            name=self.job_name,
            notes=self.cfg.notes,
            tags=cfg_to_group(cfg, return_list=True),
            dir=self.log_dir,
            config=cfg.to_dict(),
            # TODO(rcadene): try set to True
            save_code=False,
            # TODO(rcadene): split train and eval, and run async eval with job_type="eval"
            job_type="train_eval",
            resume="must" if cfg.resume else None,
            mode=self.cfg.mode if self.cfg.mode in ["online", "offline", "disabled"] else "online",
        )
        print(colored("Logs will be synced with wandb.", "blue", attrs=["bold"]))
        logging.info(f"Track this run --> {colored(wandb.run.get_url(), 'yellow', attrs=['bold'])}")
        self._wandb = wandb

    def log_policy(self, checkpoint_dir: Path):
        """Checkpoints the policy to wandb."""
        if self.cfg.disable_artifact:
            return

        step_id = checkpoint_dir.name
        artifact_name = f"{self._group}-{step_id}"
        artifact_name = get_safe_wandb_artifact_name(artifact_name)
        artifact = self._wandb.Artifact(artifact_name, type="model")
        artifact.add_file(checkpoint_dir / PRETRAINED_MODEL_DIR / SAFETENSORS_SINGLE_FILE)
        self._wandb.log_artifact(artifact)

    def log_dict(self, d: dict, step: int, mode: str = "train"):
        if mode not in {"train", "eval"}:
            raise ValueError(mode)

        processed: dict[str, int | float | str] = {}
        for k, v in d.items():
            # Fast path for scalars/strings
            if isinstance(v, (int, float, str)):
                processed[k] = v
                continue

            # Try to handle tensors/arrays/lists by logging summary stats
            arr = None
            try:
                import torch  # type: ignore
                if isinstance(v, torch.Tensor):
                    if v.numel() == 1:
                        processed[k] = v.item()
                        continue
                    arr = v.detach().float().cpu().numpy()
            except Exception:
                pass

            if arr is None:
                try:
                    import numpy as np  # type: ignore
                    if isinstance(v, np.ndarray):
                        arr = v
                except Exception:
                    pass

            if arr is None and isinstance(v, (list, tuple)) and len(v) > 0 and all(
                isinstance(x, (int, float)) for x in v
            ):
                try:
                    import numpy as np  # type: ignore
                    arr = np.asarray(v)
                except Exception:
                    arr = None

            if arr is not None:
                try:
                    import numpy as np  # type: ignore
                    processed[f"{k}_mean"] = float(np.mean(arr))
                    processed[f"{k}_std"] = float(np.std(arr))
                    processed[f"{k}_min"] = float(np.min(arr))
                    processed[f"{k}_max"] = float(np.max(arr))
                except Exception:
                    logging.debug(
                        f'WandB logging of key "{k}" failed during stats computation.'
                    )
                continue

            logging.debug(
                f'WandB logging of key "{k}" was ignored as its type is not handled by this wrapper.'
            )

        for k, v in processed.items():
            self._wandb.log({f"{mode}/{k}": v}, step=step)

    def log_hist(
        self,
        values: np.ndarray,
        step: int,
        mode: str = "k_dist",
        title: Optional[str] = None,
    ):
        self._wandb.log({f"{mode}/{title}": self._wandb.Histogram(values)}, step=step)

    def log_bar(
        self,
        values: np.ndarray,
        column_name: Tuple[str, str],
        step: int,
        mode: str = "k_dist",
        title: Optional[str] = None,
    ):
        x_axis, y_axis = column_name
        table = self._wandb.Table(columns=[x_axis, y_axis])
        for idx, val in enumerate(values):
            table.add_data(int(idx), float(val))
        line_plot = self._wandb.plot.bar(table, x_axis, y_axis, title=title)
        self._wandb.log({f"{mode}/{title}": line_plot}, step=step)


    def log_video(self, video_path: str, step: int, mode: str = "train"):
        if mode not in {"train", "eval"}:
            raise ValueError(mode)

        wandb_video = self._wandb.Video(video_path, fps=self.env_fps, format="mp4")
        self._wandb.log({f"{mode}/video": wandb_video}, step=step)