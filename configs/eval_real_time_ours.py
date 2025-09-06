import datetime as dt
import logging
from dataclasses import dataclass, field
from typing import Any
from pathlib import Path
import draccus

from common import envs, policies  # noqa: F401
from configs import parser
from configs.default import DatasetConfig, EvalConfig, WandBConfig
from configs.policies import PreTrainedConfig
from common.policies.extensions import ExtendedConfig


@dataclass
class EvalRealTimeOursPipelineConfig:
    # Either the repo ID of a model hosted on the Hub or a path to a directory containing weights
    # saved using `Policy.save_pretrained`. If not provided, the policy is initialized from scratch
    # (useful for debugging). This argument is mutually exclusive with `--config`.
    train_dataset: DatasetConfig
    eval: EvalConfig = field(default_factory=EvalConfig)
    policy: PreTrainedConfig | None = None
    method: ExtendedConfig | None = None
    output_dir: Path | None = None
    job_name: str | None = None
    seed: int | None = 1000
    wandb: WandBConfig = field(default_factory=WandBConfig)
    num_workers: int = 4
    resume: bool = False
    log_freq: int = 100
    temporal_ensemble: bool = False
    fps: int = 5
    use_devices: bool = True
    task: str = 'test'
    max_steps: int = 1000000
    cam_list: list[str] = field(default_factory=lambda: ['wrist', 'exo', 'table'])

    # Adapter configs
    adapter_path: Path | None = None

    # Adapter options
    use_qlora: bool | None = False
    use_lora: bool | None = False
    use_prefix_tuning: bool | None = False
    use_lora_moe: bool | None = False
    rank_size: int = 16
    infer_chunk: int = 40

    # Adapter injection filtering: only layers whose names contain any of these keywords will be wrapped.
    # If None or empty, all matching layers are wrapped.
    target_keywords: str | None = None

    # Adapter specific hyper-parameters (overrides defaults).
    qlora_cfg: dict[str, Any] | None = None
    lora_cfg: dict[str, Any] | None = None
    prefix_tuning_cfg: dict[str, Any] | None = None
    lora_moe_cfg: dict[str, Any] | None = None


    def __post_init__(self):
        # HACK: We parse again the cli args here to get the pretrained path if there was one.
        policy_path = parser.get_path_arg("policy")
        if policy_path:
            cli_overrides = parser.get_cli_overrides("policy")
            self.policy = PreTrainedConfig.from_pretrained(policy_path, cli_overrides=cli_overrides)
            self.policy.pretrained_path = policy_path

        else:
            logging.warning(
                "No pretrained path was provided, evaluated policy will be built from scratch (random weights)."
            )

        if not self.job_name:
            self.job_name = f"{self.policy.type}"

        if not self.output_dir:
            now = dt.datetime.now()
            eval_dir = f"{now:%Y-%m-%d}/{now:%H-%M-%S}"
            self.output_dir = Path("outputs/eval") / eval_dir

    @classmethod
    def __get_path_fields__(cls) -> list[str]:
        """This enables the parser to load config from the policy using `--policy.path=local/dir`"""
        return ["policy"]

    def to_dict(self) -> dict:
        return draccus.encode(self)
