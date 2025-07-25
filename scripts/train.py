import logging
import time
import os
import datetime
from contextlib import nullcontext
from pprint import pformat
from typing import Any

# LoRA / Prefix / LoRA-MoE injection utilities
from common.policies.lora import inject_lora, LoRAConfig as InjectLoRAConfig
from common.policies.prefix_tuning import inject_prefix_tuning
from common.policies.lora_moe import inject_lora_moe
from common.policies.prefix_tuning import PrefixTuningConfig as PTConfig
from common.policies.lora_moe import LoRAMoEConfig

import torch
from termcolor import colored
from torch.amp import GradScaler
from torch.nn.parallel import DistributedDataParallel
from torch.optim import Optimizer
import torch.distributed as dist

from common.datasets.make_dataloader import make_dataloader
from common.utils.train_utils import batch_to_device
from common.utils.logging_utils import log_wandb_tracker, AverageMeter, MetricsTracker
from common.utils.random_utils import set_seed
from common.utils.train_utils import (
    get_step_checkpoint_dir,
    load_training_state,
    save_checkpoint,
    update_last_checkpoint,
    save_training_state,
)
from common.utils.model_utils import compute_param_norm
from common.policies.moe_utils import moe_aux_loss
from common.utils.utils import (
    format_big_number,
    get_safe_torch_device,
    has_method,
    init_logging,
)
# adapter utils
from common.utils.adapter_utils import save_adapters
from common.utils.wandb_utils import WandBLogger
from configs import parser
from configs.train import TrainPipelineConfig
from common.policies.factory import make_policy
from common.policies.pretrained import PreTrainedPolicy
from common.policies.utils import get_device_from_parameters
from common.datasets.factory import make_dataset
from common.datasets.utils import cycle
from common.optim.factory import make_optimizer_and_scheduler


def update_policy(
    train_metrics: MetricsTracker,
    policy: PreTrainedPolicy,
    batch: Any,
    optimizer: Optimizer,
    grad_clip_norm: float,
    grad_scaler: GradScaler,
    lr_scheduler=None,
    use_amp: bool = False,
    lock=None,
    moe_aux_cfg: dict | None = None,
) -> tuple[MetricsTracker, dict]:
    start_time = time.perf_counter()
    device = get_device_from_parameters(policy)
    policy.train()
    with torch.autocast(device_type=device.type) if use_amp else nullcontext():
        loss, output_dict = policy.forward(batch)

        # Auxiliary MoE losses (balance + z-loss)
        if moe_aux_cfg is not None:
            aux_loss = moe_aux_loss(
                policy,
                lb_coeff=moe_aux_cfg.get("lb_coeff", 0.01),
                z_coeff=moe_aux_cfg.get("z_coeff", 1e-3),
            )
            loss = loss + aux_loss
            output_dict = output_dict or {}
            output_dict["moe_aux_loss"] = aux_loss.item()
        # TODO(rcadene): policy.unnormalize_outputs(out_dict)
    grad_scaler.scale(loss).backward()

    # Unscale the gradient of the optimizer's assigned params in-place **prior to gradient clipping**.
    grad_scaler.unscale_(optimizer)

    grad_norm = torch.nn.utils.clip_grad_norm_(
        policy.parameters(),
        grad_clip_norm,
        error_if_nonfinite=False,
    )

    # Optimizer's gradients are already unscaled, so scaler.step does not unscale them,
    # although it still skips optimizer.step() if the gradients contain infs or NaNs.
    with lock if lock is not None else nullcontext():
        grad_scaler.step(optimizer)
    # Updates the scale for next iteration.
    grad_scaler.update()

    param_norm = compute_param_norm(policy, only_trainable=True)

    optimizer.zero_grad()

    # Step through pytorch scheduler at every batch instead of epoch
    if lr_scheduler is not None:
        lr_scheduler.step()

    if has_method(policy, "update"):
        # To possibly update an internal buffer (for instance an Exponential Moving Average like in TDMPC).
        policy.update()

    train_metrics.loss = loss.item()
    train_metrics.grad_norm = grad_norm.item()
    train_metrics.param_norm = param_norm
    train_metrics.lr = optimizer.param_groups[0]["lr"]
    train_metrics.update_s = time.perf_counter() - start_time
    return train_metrics, output_dict


def test_policy(
    test_metrics: MetricsTracker,
    policy: PreTrainedPolicy,
    batch: Any,
    use_amp: bool = False,
) -> tuple[MetricsTracker, dict]:
    start_time = time.perf_counter()
    device = get_device_from_parameters(policy)
    policy.eval()
    with torch.autocast(device_type=device.type) if use_amp else nullcontext():
        loss, output_dict = policy.forward(batch)
        # TODO(rcadene): policy.unnormalize_outputs(out_dict)

    test_metrics.loss = loss.item()
    return test_metrics, output_dict


@parser.wrap()
def train(cfg: TrainPipelineConfig):
    cfg.validate()

    device = get_safe_torch_device(cfg.policy.device, log=True)
    torch.backends.cuda.matmul.allow_tf32 = True

    if cfg.use_ddp:
        if os.environ.get("LOCAL_RANK", -1) == -1:  # not called by torchrun, do not initialize dist.
            device, local_rank = torch.device("cuda"), 0  # single GPU

        if not dist.is_initialized():
            dist.init_process_group(backend="nccl", timeout=datetime.timedelta(minutes=30))
        local_rank = dist.get_rank()
        device = torch.device("cuda", local_rank)
        torch.cuda.set_device(device)  # needed!
        print(f"Local Rank ({local_rank}) Initialized for DDP")
    cfg.policy.device = str(device)

    logging.info(pformat(cfg.to_dict()))

    if cfg.seed is not None:
        set_seed(cfg.seed)

    if cfg.wandb.enable and cfg.wandb.project:
        if cfg.use_ddp and (dist.get_rank() != 0):
            pass
        else:
            wandb_logger = WandBLogger(cfg)
    else:
        wandb_logger = None
        logging.info(colored("Logs will be saved locally.", "yellow", attrs=["bold"]))

    logging.info("Creating dataset")
    train_dataset = make_dataset(cfg, split="train")
    test_dataset = make_dataset(cfg, split="test")

    logging.info("Creating policy")
    policy = make_policy(
        cfg=cfg.policy,
        ds_meta = train_dataset.meta
    )

    # Adapter tuning options -------------------------------------------------
    if getattr(cfg, "use_lora", False):
        # Standard LoRA (rank=16 by default)
        lora_cfg_obj = InjectLoRAConfig(**(cfg.lora_cfg or {})) if hasattr(cfg, "lora_cfg") else InjectLoRAConfig()
        policy, _ = inject_lora(policy, lora_cfg_obj, target_keywords=cfg.target_keywords)
        policy = policy.to(device=device)
        logging.info("Injected LoRA modules")

    elif getattr(cfg, "use_prefix_tuning", False):
        # Prefix tuning (custom implementation)
        pt_cfg_obj = PTConfig(**(cfg.prefix_tuning_cfg or {})) if hasattr(cfg, "prefix_tuning_cfg") else PTConfig()
        policy, _ = inject_prefix_tuning(policy, pt_cfg_obj, target_keywords=cfg.target_keywords)
        policy = policy.to(device=device)
        logging.info("Injected Prefix-Tuning modules")

    elif getattr(cfg, "use_lora_moe", False):
        # Mixture-of-LoRA experts
        lora_moe_cfg_obj = LoRAMoEConfig(**(cfg.lora_moe_cfg or {})) if hasattr(cfg, "lora_moe_cfg") else LoRAMoEConfig()
        policy, _ = inject_lora_moe(policy, lora_moe_cfg_obj, target_keywords=cfg.target_keywords)
        policy = policy.to(device=device)
        logging.info("Injected LoRA-MoE modules")

    if cfg.use_ddp:
        if dist.is_initialized() and dist.is_available():
            policy = DistributedDataParallel(
                policy,
                device_ids=[local_rank],
                output_device=device,
                gradient_as_bucket_view=True,
                find_unused_parameters=True,
            )
            policy_m = policy.module
        logging.info("Wrapped DDP module")
    else:
        policy_m = policy

    logging.info("Creating optimizer and scheduler")
    optimizer, lr_scheduler = make_optimizer_and_scheduler(cfg, policy_m)
    grad_scaler = GradScaler(device.type, enabled=cfg.policy.use_amp)

    step = 0  # number of policy updates (forward + backward + optim)

    if cfg.resume:
        step, optimizer, lr_scheduler = load_training_state(cfg.checkpoint_path, optimizer, lr_scheduler)

    num_learnable_params = sum(p.numel() for p in policy.parameters() if p.requires_grad)
    num_total_params = sum(p.numel() for p in policy.parameters())

    if not cfg.use_ddp or (cfg.use_ddp and dist.get_rank() == 0):
        logging.info(colored("Output dir:", "yellow", attrs=["bold"]) + f" {cfg.output_dir}")
        logging.info(f"{cfg.steps=} ({format_big_number(cfg.steps)})")
        logging.info(f"{train_dataset.num_frames=} ({format_big_number(train_dataset.num_frames)})")
        logging.info(f"{train_dataset.num_episodes=}")
        logging.info(f"{num_learnable_params=} ({format_big_number(num_learnable_params)})")
        logging.info(f"{num_total_params=} ({format_big_number(num_total_params)})")

    # create dataloader for offline training
    train_dataloader = make_dataloader(cfg, train_dataset, device)
    train_dl_iter = cycle(train_dataloader)

    cfg.batch_size = cfg.batch_size // 2
    test_dataloader = make_dataloader(cfg, test_dataset, device)
    test_dl_iter = cycle(test_dataloader)
    cfg.batch_size = 2*cfg.batch_size

    # Determine MoE balance coeff if needed
    moe_aux_cfg = None
    if getattr(cfg, "use_lora_moe", False):
        moe_aux_cfg = {
            "lb_coeff": (cfg.lora_moe_cfg or {}).get("lb_coeff", 0.01),
            "z_coeff": (cfg.lora_moe_cfg or {}).get("z_coeff", 1e-3),
        }

    policy.train()

    train_metrics = {
        "loss": AverageMeter("loss", ":.3f"),
        "grad_norm": AverageMeter("grdn", ":.3f"),
        "param_norm": AverageMeter("pnorm", ":.3f"),
        "lr": AverageMeter("lr", ":0.1e"),
        "update_s": AverageMeter("updt_s", ":.3f"),
        "dataloading_s": AverageMeter("data_s", ":.3f"),
    }

    test_metrics = {
        "loss": AverageMeter("loss", ":.3f"),
    }

    train_tracker = MetricsTracker(
        cfg.batch_size,
        train_dataset.num_frames,
        train_dataset.num_episodes,
        train_metrics,
        initial_step=step
    )

    test_tracker = MetricsTracker(
        cfg.batch_size,
        test_dataset.num_frames,
        test_dataset.num_episodes,
        test_metrics,
        initial_step=step,
    )

    logging.info("Start offline training on a fixed dataset")
    for _ in range(step, cfg.steps):
        if cfg.use_ddp:
            dist.barrier()

        start_time = time.perf_counter()
        batch = next(train_dl_iter)
        train_tracker.dataloading_s = time.perf_counter() - start_time

        for key in batch:
            if isinstance(batch[key], torch.Tensor):
                batch[key] = batch[key].to(device, non_blocking=True)

        train_tracker, output_dict = update_policy(
            train_tracker,
            policy,
            batch,
            optimizer,
            cfg.optimizer.grad_clip_norm,
            grad_scaler=grad_scaler,
            lr_scheduler=lr_scheduler,
            use_amp=cfg.policy.use_amp,
            moe_aux_cfg=moe_aux_cfg,
        )

        # Note: eval and checkpoint happens *after* the `step`th training update has completed, so we
        # increment `step` here.
        step += 1
        train_tracker.step()
        test_tracker.step()
        is_log_step = cfg.log_freq > 0 and step % cfg.log_freq == 0
        is_saving_step = step % cfg.save_freq == 0 or step == cfg.steps
        is_test_step = cfg.test_freq > 0 and step % cfg.test_freq == 0

        if is_log_step:
            if cfg.use_ddp and (dist.get_rank() != 0):
                pass
            else:
                logging.info(train_tracker)
                log_wandb_tracker(wandb_logger, train_tracker, output_dict, step)

        if cfg.save_checkpoint and is_saving_step:
            if cfg.use_ddp and (dist.get_rank() != 0):
                pass
            else:
                logging.info(f"Checkpoint policy after step {step}")
                checkpoint_dir = get_step_checkpoint_dir(cfg.output_dir, cfg.steps, step)

                if any([getattr(cfg, "use_lora", False), getattr(cfg, "use_prefix_tuning", False), getattr(cfg, "use_lora_moe", False)]):
                    # Save only adapter weights + training state to keep checkpoint light
                    pretrained_dir = checkpoint_dir / "pretrained_model"
                    cfg.save_pretrained(pretrained_dir)
                    save_adapters(policy_m, pretrained_dir / "adapters.safetensors")
                    save_training_state(checkpoint_dir, step, optimizer, lr_scheduler)
                else:
                    # Full model checkpoint
                    save_checkpoint(checkpoint_dir, step, cfg, policy_m, optimizer, lr_scheduler)

                update_last_checkpoint(checkpoint_dir)
                if wandb_logger:
                    wandb_logger.log_policy(checkpoint_dir)

        if is_test_step:
            test_batch = next(test_dl_iter)
            test_batch = batch_to_device(test_batch, device)

            test_tracker, output_dict = test_policy(
                test_tracker,
                policy,
                test_batch,
                use_amp=cfg.policy.use_amp,
            )

            if cfg.use_ddp and (dist.get_rank() != 0):
                pass
            else:
                logging.info(test_tracker)
                log_wandb_tracker(wandb_logger, test_tracker, output_dict, step, mode='eval')

    logging.info("End of training")
    dist.destroy_process_group()


if __name__ == "__main__":
    init_logging()
    train()
