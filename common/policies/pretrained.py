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
import abc
import logging
import os
from pathlib import Path
from typing import Type, TypeVar, Optional, List

import packaging
import safetensors
from huggingface_hub import hf_hub_download
from huggingface_hub.constants import SAFETENSORS_SINGLE_FILE
from huggingface_hub.errors import HfHubHTTPError
from safetensors.torch import load_model as load_model_as_safetensor
from safetensors.torch import save_model as save_model_as_safetensor
import torch
from torch import Tensor, nn
from transformers import PreTrainedModel as HFPreTrainedModel

from common.constants import OBS_ROBOT, ACTION
from common.policies.extensions import ExtendedConfig
from common.utils.hub import HubMixin
from common.utils.model_utils import *
from common.utils.model_utils import resize_with_pad
from common.utils.moe_utils import compute_router_loss
from configs.policies import PreTrainedConfig

T = TypeVar("T", bound="PreTrainedPolicy")

class PreTrainedPolicy(HubMixin, HFPreTrainedModel, abc.ABC):
    """
    Base class for policy models.
    """
    config_class: None
    name: None

    def __init__(self, config: PreTrainedConfig, *inputs, **kwargs):
        super().__init__(config, *inputs, **kwargs)
        if not isinstance(config, PreTrainedConfig):
            raise ValueError(
                f"Parameter config in `{self.__class__.__name__}(config)` should be an instance of class "
                "`PreTrainedConfig`. To create a model from a pretrained model use "
                f"`model = {self.__class__.__name__}.from_pretrained(PRETRAINED_MODEL_NAME)`"
            )
        self.config = config
        self._compute_router_loss = False

    def __init_subclass__(cls, **kwargs):
        super().__init_subclass__(**kwargs)
        if not getattr(cls, "config_class", None):
            raise TypeError(f"Class {cls.__name__} must define 'config_class'")
        if not getattr(cls, "name", None):
            raise TypeError(f"Class {cls.__name__} must define 'name'")

    def _save_pretrained(self, save_directory: Path) -> None:
        # Save config first
        self.config._save_pretrained(save_directory)

        model_to_save = self.module if hasattr(self, "module") else self

        # Always save the full model for safety / exact resume.
        save_model_as_safetensor(model_to_save, str(save_directory / SAFETENSORS_SINGLE_FILE))

    @classmethod
    def from_pretrained(
        cls: Type[T],
        pretrained_name_or_path: str | Path,
        *,
        config: PreTrainedConfig | None = None,
        force_download: bool = False,
        resume_download: bool | None = None,
        proxies: dict | None = None,
        token: str | bool | None = None,
        cache_dir: str | Path | None = None,
        local_files_only: bool = False,
        revision: str | None = None,
        strict: bool = False,
        **kwargs,
    ) -> T:
        """
        The policy is set in evaluation mode by default using `policy.eval()` (dropout modules are
        deactivated). To train it, you should first set it back in training mode with `policy.train()`.
        """
        if config is None:
            config = PreTrainedConfig.from_pretrained(
                pretrained_name_or_path=pretrained_name_or_path,
                force_download=force_download,
                resume_download=resume_download,
                proxies=proxies,
                token=token,
                cache_dir=cache_dir,
                local_files_only=local_files_only,
                revision=revision,
                **kwargs,
            )
        model_id = str(pretrained_name_or_path)
        instance = cls(config, **kwargs)
        if os.path.isdir(model_id):
            print("Loading weights from local directory")
            model_file = os.path.join(model_id, SAFETENSORS_SINGLE_FILE)
            model_is_safetensor = True
            if not os.path.isfile(model_file):
                model_file = os.path.join(model_id, "model.pt")
                assert os.path.isfile(model_file)
                model_is_safetensor = False

            policy = cls._load_as_safetensor(instance, model_file, config.device, strict, model_is_safetensor=model_is_safetensor)
        else:
            try:
                model_file = hf_hub_download(
                    repo_id=model_id,
                    filename=SAFETENSORS_SINGLE_FILE,
                    revision=revision,
                    cache_dir=cache_dir,
                    force_download=force_download,
                    proxies=proxies,
                    resume_download=resume_download,
                    token=token,
                    local_files_only=local_files_only,
                )
                policy = cls._load_as_safetensor(instance, model_file, config.device, strict)
            except HfHubHTTPError as e:
                raise FileNotFoundError(
                    f"{SAFETENSORS_SINGLE_FILE} not found on the HuggingFace Hub in {model_id}"
                ) from e

        policy.to(config.device)
        policy.eval()
        return policy

    @classmethod
    def _load_as_safetensor(cls, model: T, model_file: str, map_location: str, strict: bool, model_is_safetensor: bool = True) -> T:
        if model_is_safetensor:
            if packaging.version.parse(safetensors.__version__) < packaging.version.parse("0.4.3"):
                load_model_as_safetensor(model, model_file, strict=strict)
                if map_location != "cpu":
                    logging.warning(
                        "Loading model weights on other devices than 'cpu' is not supported natively in your version of safetensors."
                        " This means that the model is loaded on 'cpu' first and then copied to the device."
                        " This leads to a slower loading time."
                        " Please update safetensors to version 0.4.3 or above for improved performance."
                    )
                    model.to(map_location)
            else:
                safetensors.torch.load_model(model, model_file, strict=strict, device=map_location)
        else:
            state_dict = torch.load(model_file, map_location=map_location)
            model.load_state_dict(state_dict)
            model.to(map_location)
        return model

    @abc.abstractmethod
    def get_optim_params(self) -> dict:
        """
        Returns the policy-specific parameters dict to be passed on to the optimizer.
        """
        raise NotImplementedError

    @abc.abstractmethod
    def reset(self):
        raise NotImplementedError

    def forward(
            self,
            batch: dict[str, Tensor],
            noise=None,
            time=None,
            method: Optional[ExtendedConfig] = None,
            ranks: Optional[List[int]] = None,
    ) -> tuple[Tensor, dict[str, Tensor]]:
        if self.config.adapt_to_pi_aloha:
            batch[OBS_ROBOT] = self._pi_aloha_decode_state(batch[OBS_ROBOT])
            batch[ACTION] = self._pi_aloha_encode_actions_inv(batch[ACTION])

        batch = self.normalize_inputs(batch)
        batch = self.normalize_targets(batch)

        images, img_masks = self.prepare_images(batch)
        state = self.prepare_state(batch)
        lang_tokens, lang_masks = self.prepare_language(batch)
        actions = self.prepare_action(batch)
        actions_is_pad = batch.get("action_is_pad")

        loss_dict = {}
        losses = self.model.forward(images, img_masks, lang_tokens, lang_masks, state, actions, noise, time)
        loss_dict["losses_after_forward"] = losses.clone()

        if actions_is_pad is not None:
            in_episode_bound = ~actions_is_pad
            losses = losses * in_episode_bound.unsqueeze(-1)
            loss_dict["losses_after_in_ep_bound"] = losses.clone()

        # Remove padding
        losses = losses[:, :, : self.config.max_action_dim]
        loss_dict["losses_after_rm_padding"] = losses.clone()

        # For backward pass
        loss = losses.mean()
        # For logging
        loss_dict["l2_loss"] = loss.item()

        if self._compute_router_loss:
            assert self.train_aux_loss
            aux_loss, loss_dict = self._router_forward(method.aux_loss_cfg, loss_dict, ranks=ranks)
        else:
            aux_loss = torch.tensor(0.0, device=loss.device, dtype=loss.dtype)

        total_loss = loss+aux_loss

        return total_loss, loss_dict

    def _router_forward(
            self,
            aux_loss_cfg: dict,
            loss_dict: dict,
            ranks=None
    ):
        if ranks is None:
            ranks = [64, 32]

        lb_coeff = aux_loss_cfg.get("lb_coeff", 0.01)
        z_coeff = aux_loss_cfg.get("z_coeff", 1e-3)
        spec_coeff = aux_loss_cfg.get("spec_coeff", 0.0)
        mod_coeff = aux_loss_cfg.get("mod_coeff", 0.0)
        id_coeff = aux_loss_cfg.get("id_coeff", 0.0)

        lb_loss, z_loss, spec_loss, mod_loss, id_loss = compute_router_loss(self, ground_rank=ranks[0], target_rank=ranks[1])

        aux_loss = lb_coeff * lb_loss + z_coeff * z_loss + spec_coeff * spec_loss + mod_coeff * mod_loss + id_coeff * id_loss

        loss_dict["router_balance_loss"] = lb_loss.item()
        loss_dict["router_z_loss"] = z_loss.item()
        loss_dict["router_spec_loss"] = spec_loss.item()
        loss_dict["router_mod_loss"] = mod_loss.item()
        loss_dict["router_id_loss"] = id_loss.item()
        loss_dict["moe_aux_loss"] = aux_loss.item()

        return aux_loss, loss_dict

    def clear_cache(self):
        for module in self.children():
            if hasattr(module, "clear_cache"):
                module.clear_cache()

    @abc.abstractmethod
    def select_action(self, batch: dict[str, Tensor]) -> Tensor:
        """Return one action to run in the environment (potentially in batch mode).

        When the model uses a history of observations, or outputs a sequence of actions, this method deals
        with caching.
        """
        raise NotImplementedError

    def enable_router_loss(self):
        self._compute_router_loss = True

    def get_component_norms(self) -> dict[str, float]:
        """Return L2 norms of parameters and gradients for main Pi0 components.

        Returns a dictionary with keys:
            vision_param_norm, vision_grad_norm,
            lang_param_norm, lang_grad_norm,
            action_param_norm, action_grad_norm

        Gradient norms require that backward() has already been executed in the
        current iteration.
        """

        # Vision tower (frozen or trainable)
        vision_modules = self.vision_modules()

        # Language model (PaliGemma text encoder)
        lang_modules = self.language_modules()

        # Action generator (Gemma expert + projection / MLP blocks)
        action_modules = self.action_modules()

        # Compute L2 norms
        vision_param_norm_sq = 0.0
        vision_grad_norm_sq = 0.0
        for m in vision_modules:
            vision_param_norm_sq += compute_param_norm(m, only_trainable=True) ** 2
            vision_grad_norm_sq += compute_grad_norm(m, only_trainable=True) ** 2

        lang_param_norm_sq = 0.0
        lang_grad_norm_sq = 0.0
        for m in lang_modules:
            lang_param_norm_sq += compute_param_norm(m, only_trainable=True) ** 2
            lang_grad_norm_sq += compute_grad_norm(m, only_trainable=True) ** 2

        action_param_norm_sq = 0.0
        action_grad_norm_sq = 0.0
        for m in action_modules:
            action_param_norm_sq += compute_param_norm(m, only_trainable=True) ** 2
            action_grad_norm_sq += compute_grad_norm(m, only_trainable=True) ** 2

        metrics = {
            "vision_param_norm": vision_param_norm_sq ** 0.5,
            "vision_grad_norm": vision_grad_norm_sq ** 0.5,
            "lang_param_norm": lang_param_norm_sq ** 0.5,
            "lang_grad_norm": lang_grad_norm_sq ** 0.5,
            "action_param_norm": action_param_norm_sq ** 0.5,
            "action_grad_norm": action_grad_norm_sq ** 0.5,
        }

        return metrics

    @abc.abstractmethod
    def vision_modules(self) -> list[str]:
        raise NotImplementedError

    @abc.abstractmethod
    def language_modules(self) -> list[str]:
        raise NotImplementedError

    @abc.abstractmethod
    def action_modules(self) -> list[str]:
        raise NotImplementedError

    def prepare_images(self, batch):
        """Apply Pi0 preprocessing to the images, like resizing to 224x224 and padding to keep aspect ratio, and
        convert pixel range from [0.0, 1.0] to [-1.0, 1.0] as requested by SigLIP.
        """
        images = []
        img_masks = []

        present_img_keys = [key for key in self.config.image_features if key in batch]
        missing_img_keys = [key for key in self.config.image_features if key not in batch]

        if len(present_img_keys) == 0:
            raise ValueError(
                f"All image features are missing from the batch. At least one expected. (batch: {batch.keys()}) (image_features:{self.config.image_features})"
            )

        # Preprocess image features present in the batch
        for key in present_img_keys:
            img = batch[key]

            if self.config.resize_imgs_with_padding is not None:
                img = resize_with_pad(img, *self.config.resize_imgs_with_padding, pad_value=0)

            # Normalize from range [0,1] to [-1,1] as expacted by siglip
            img = img * 2.0 - 1.0

            bsize = img.shape[0]
            device = img.device
            mask = torch.ones(bsize, dtype=torch.bool, device=device)
            images.append(img)
            img_masks.append(mask)

        # Create image features not present in the batch
        # as fully 0 padded images.
        for num_empty_cameras in range(len(missing_img_keys)):
            if num_empty_cameras >= self.config.empty_cameras:
                break
            img = torch.ones_like(img) * -1
            mask = torch.zeros_like(mask)
            images.append(img)
            img_masks.append(mask)

        return images, img_masks

class PreTrainedFlowMatching(nn.Module):
    def __init__(self, config: PreTrainedConfig):
        super().__init__()

    def set_requires_grad(self):
        for params in self.state_proj.parameters():
            params.requires_grad = self.config.train_state_proj

    def sample_noise(self, shape: torch.Tensor, device: torch.device) -> torch.Tensor:
        noise = torch.normal(
            mean=0.0,
            std=1.0,
            size=shape,
            dtype=self.sample_dtype if self.sample_dtype.is_floating_point else torch.bfloat16,
            device=device,
        )
        return noise

    @abc.abstractmethod
    def sample_time(self, bsize: int, device: torch.device) -> torch.Tensor:
        raise NotImplementedError
