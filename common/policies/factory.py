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
from typing import Optional, Tuple, List

import torch
from torch import nn
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP, MixedPrecision, StateDictType

from common.datasets.lerobot_dataset import LeRobotDatasetMetadata
from common.datasets.utils import dataset_to_policy_features
from common.policies.pi0.configuration_pi0 import PI0Config
from common.policies.pretrained import PreTrainedPolicy
from common.policies.smolvla.configuration_smolvla import SmolVLAConfig
from common.utils.adapter_utils import inject_adapters, load_adapters_as_expert, load_adapters_as_moe, load_adapters
from configs.policies import PreTrainedConfig
from configs.types import FeatureType
from common.policies.extensions import ExtendedConfig

from common.policies.lora import LoraConfig
from common.policies.lora_moe import LoraMoEConfig
from common.policies.lora_msp import LoraMSPConfig
from common.utils.model_utils import freeze_non_adapters


def get_policy_class(name: str) -> PreTrainedPolicy:
    """Get the policy's class and config class given a name (matching the policy class' `name` attribute)."""

    if name == "pi0":
        from common.policies.pi0.modeling_pi0 import PI0Policy

        return PI0Policy

    elif name == "smolvla":
        from common.policies.smolvla.modeling_smolvla import SmolVLAPolicy

        return SmolVLAPolicy
    else:
        raise NotImplementedError(f"Policy with name {name} is not implemented.")


def make_policy_config(policy_type: str, **kwargs) -> PreTrainedConfig:
    if policy_type == "pi0":
        return PI0Config(**kwargs)
    elif policy_type == "smolvla":
        return SmolVLAConfig(**kwargs)
    else:
        raise ValueError(f"Policy type '{policy_type}' is not available.")


def make_policy(
    cfg: PreTrainedConfig,
    ds_meta: LeRobotDatasetMetadata | None = None,
) -> nn.Module:
    """Make an instance of a policy class.

    This function exists because (for now) we need to parse features from either a dataset or an environment
    in order to properly dimension and instantiate a policy for that dataset or environment.

    Args:
        cfg (PreTrainedConfig): The config of the policy to make. If `pretrained_path` is set, the policy will
            be loaded with the weights from that path.
        ds_meta (LeRobotDatasetMetadata | None, optional): Dataset metadata to take input/output shapes and
            statistics to use for (un)normalization of inputs/outputs in the policy. Defaults to None.
        env_cfg (EnvConfig | None, optional): The config of a gym environment to parse features from. Must be
            provided if ds_meta is not. Defaults to None.

    Raises:
        ValueError: Either ds_meta or env and env_cfg must be provided.
        NotImplementedError: if the policy.type is 'vqbet' and the policy device 'mps' (due to an incompatibility)

    Returns:
        PreTrainedPolicy: _description_
    """
    # NOTE: Currently, if you try to run vqbet with mps backend, you'll get this error.
    # NotImplementedError: The operator 'aten::unique_dim' is not currently implemented for the MPS device. If
    # you want this op to be added in priority during the prototype phase of this feature, please comment on
    # https://github.com/pytorch/pytorch/issues/77764. As a temporary fix, you can set the environment
    # variable `PYTORCH_ENABLE_MPS_FALLBACK=1` to use the CPU as a fallback for this op. WARNING: this will be
    # slower than running natively on MPS.
    policy_cls = get_policy_class(cfg.type)

    kwargs = {}
    if ds_meta is not None:
        features = dataset_to_policy_features(ds_meta.features)
        kwargs["dataset_stats"] = ds_meta.stats
    else:
        if not cfg.pretrained_path:
            logging.warning(
                "You are instantiating a policy from scratch and its features are parsed from an environment "
                "rather than a dataset. Normalization modules inside the policy will have infinite values "
                "by default without stats from a dataset."
            )
        raise Exception("NO ENVIRONMENT")

    cfg.output_features = {key: ft for key, ft in features.items() if ft.type is FeatureType.ACTION}
    cfg.input_features = {key: ft for key, ft in features.items() if key not in cfg.output_features}
    kwargs["config"] = cfg

    if cfg.pretrained_path:
        # Load a pretrained policy and override the config if needed (for example, if there are inference-time
        # hyperparameters that we want to vary).
        kwargs["pretrained_name_or_path"] = cfg.pretrained_path
        policy = policy_cls.from_pretrained(**kwargs)
    else:
        # Make a fresh policy.
        policy = policy_cls(**kwargs)

    policy.to(cfg.device)
    assert isinstance(policy, nn.Module)

    # policy = torch.compile(policy, mode="reduce-overhead")

    return policy

def wrap_policy(
    policy: PreTrainedPolicy,
    cfg: ExtendedConfig,
    is_master: bool = True,
    device: str | torch.device = "cpu",
) -> Tuple[nn.Module, List[str] | str]:
    method = cfg.core
    policy, train_router_loss, lora_cfg_obj = get_lora_cfg_obj(policy, cfg, method, is_master, device)

    if lora_cfg_obj is not None:
        policy, _ = inject_adapters(policy, lora_cfg_obj, target_keywords=cfg.target_keywords)
        policy = policy.to(device=device)
        freeze_non_adapters(policy)
        policy.train_aux_loss = True

    if cfg.is_train:
        if cfg.adapter_file_path:
            if cfg.expert_source == 'lora':
                assert lora_cfg_obj.num_experts == len(cfg.adapter_file_path)
                res = load_adapters_as_expert(policy, cfg.adapter_file_path)
            elif cfg.expert_source == 'lora_moe':
                assert len(cfg.adapter_file_path) == 1
                res = load_adapters_as_moe(policy, cfg.adapter_file_path)
        else:
            res = f"Not Injecting Adapters"

        if train_router_loss:
            policy.enable_router_loss()

    else:
        if cfg.adapter_file_path:
            res, policy = load_adapters(policy, cfg.adapter_file_path[0])
        else:
            raise Exception(f"No adapter_file_path provided")


    return policy, res


def dist_policy(
    policy: nn.Module,
    dist_mode: Optional[str] = None,
    is_distributed: bool = False,
    local_rank: int = 0,
    device: str | torch.device = "cpu",
) -> Tuple[nn.Module, str]:
    if dist_mode == 'none':
        res = f"Not Wrapped for torch.distributed"
        return policy, res

    elif dist_mode == "fsdp":
        assert is_distributed

        mp_policy = MixedPrecision(
            param_dtype=torch.bfloat16,
            reduce_dtype=torch.float32,
            buffer_dtype=torch.bfloat16,
        )

        # Utilities to normalize dtypes across the model prior to FSDP wrapping
        def _force_cast_all_float_parameters(module: torch.nn.Module, target_dtype: torch.dtype):
            for _, param in module.named_parameters(recurse=True):
                if isinstance(param.data,
                              torch.Tensor) and param.data.is_floating_point() and param.data.dtype != target_dtype:
                    param.data = param.data.to(target_dtype)

        # Ensure all trainable parameters are bfloat16
        _force_cast_all_float_parameters(policy, target_dtype=torch.bfloat16)

        # Robust construction across torch versions: prefer use_orig_params, else fall back
        policy = FSDP(
            policy,
            device_id=device,
            mixed_precision=mp_policy,
            use_orig_params=True,
        )
        policy = policy.to(device=device)
        res = f"Wrapped FSDP module for local rank {local_rank}"

        return policy, res

    elif dist_mode == "ddp":
        assert is_distributed

        policy = DDP(
            policy,
            device_ids=[local_rank],
            output_device=device,
            gradient_as_bucket_view=True,
            find_unused_parameters=True,
        )
        res = f"Wrapped DDP module for local rank {local_rank}"

        return policy.module, res

    else:
        raise NotImplementedError(f"{dist_mode} not implemented")

def get_lora_cfg_obj(
    policy: nn.Module,
    cfg: ExtendedConfig,
    method: str,
    is_master: bool,
    device: str | torch.device = "cpu",
) -> Tuple[PreTrainedPolicy, bool, Optional[LoraConfig]]:
    train_router_loss = False
    lora_cfg_obj = None

    if method == "train_linear_only":
        policy.unfreeze_linear_layers()
        policy = policy.to(device=device)
        if is_master:
            logging.info("Unfreezed Linear only")

    elif method == "linear_probing":
        policy.unfreeze_action_out_proj()
        policy = policy.to(device=device)
        if is_master:
            logging.info("Unfreezed action output projection")

    elif method == "lora":
        lora_cfg_obj = cfg.lora_cfg if hasattr(cfg, "lora_moe_cfg") else LoraConfig()
        if is_master:
            logging.info("Injected LoRA modules")

    elif method == "qlora":
        lora_cfg_obj = cfg.lora_cfg if hasattr(cfg, "lora_moe_cfg") else LoraConfig()
        lora_cfg_obj.quantize = True
        if is_master:
            logging.info("Injected QLoRA modules")

    elif method == "lora_moe":
        lora_cfg_obj = cfg.lora_cfg if hasattr(cfg, "lora_cfg") else LoraMoEConfig()
        train_router_loss = True

        if is_master:
            logging.info("Injected LoRA-MoE modules")

    elif method == "qlora_moe":
        lora_cfg_obj = cfg.lora_cfg if hasattr(cfg, "lora_cfg") else LoraMoEConfig()
        lora_cfg_obj.quantize = True
        train_router_loss = True

        if is_master:
            logging.info("Injected QLoRA-MoE modules")

    elif method =="lora_msp":
        lora_cfg_obj = cfg.lora_cfg if hasattr(cfg, "lora_cfg") else LoraMSPConfig()
        train_router_loss = True

        if is_master:
            logging.info("Injected LoRA-MSP modules")

    elif method == "vanilla":
        if is_master:
            logging.info("Using Vanilla model")

    else:
        raise NotImplementedError(f"{method} not implemented")

    return policy, train_router_loss, lora_cfg_obj