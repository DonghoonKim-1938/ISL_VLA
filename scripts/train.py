import logging
import time
import os
import datetime
from contextlib import nullcontext
from pprint import pformat
from typing import Any, List
import gc

# LoRA / Prefix / LoRA-MoE injection utilities

import torch
from termcolor import colored
from torch.amp import GradScaler
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP, MixedPrecision, StateDictType

from torch.optim import Optimizer
import torch.distributed as dist

from common.datasets.make_dataloader import make_dataloader
from common.policies.extensions import ExtendedConfig
from common.policies.lora import LoraConfig
from common.policies.lora_moe import LoraMoEConfig
from common.policies.lora_msp import LoraMSPConfig, LoraMSPLinear
from common.utils.train_utils import batch_to_device
from common.utils.logging_utils import log_wandb_tracker, AverageMeter, MetricsTracker, log_wandb_k_dist, log_csv_k_dist
from common.utils.random_utils import set_seed
from common.utils.train_utils import (
    get_step_checkpoint_dir,
    load_training_state,
    save_checkpoint,
    update_last_checkpoint,
    save_training_state,
)
from common.utils.model_utils import compute_param_norm
from common.utils.utils import (
    format_big_number,
    get_safe_torch_device,
    has_method,
    init_logging,
    is_ddp_master
)
# adapter utils
from common.utils.adapter_utils import save_adapters
from common.utils.wandb_utils import WandBLogger
from configs import parser
from configs.train import TrainPipelineConfig
from common.policies.factory import make_policy, wrap_policy, dist_policy
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
    method: ExtendedConfig | None = None,
) -> tuple[MetricsTracker, dict]:
    start_time = time.perf_counter()
    device = get_device_from_parameters(policy)
    policy.train()
    with torch.autocast(device_type=device.type) if use_amp else nullcontext():
        loss, output_dict = policy.forward(batch, method = method)
    grad_scaler.scale(loss).backward()
    # policy.clear_cache()

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

    # Component-wise norms (Pi0-specific) -- computed inside the model for minimal trainer changes
    vision_param_norm = vision_grad_norm = 0.0
    lang_param_norm = lang_grad_norm = 0.0
    action_param_norm = action_grad_norm = 0.0

    _policy_obj = policy.module if isinstance(policy, DDP) else policy

    if hasattr(_policy_obj, "get_component_norms"):
        comp_norms = _policy_obj.get_component_norms()
        vision_param_norm = comp_norms.get("vision_param_norm", 0.0)
        vision_grad_norm = comp_norms.get("vision_grad_norm", 0.0)
        lang_param_norm = comp_norms.get("lang_param_norm", 0.0)
        lang_grad_norm = comp_norms.get("lang_grad_norm", 0.0)
        action_param_norm = comp_norms.get("action_param_norm", 0.0)
        action_grad_norm = comp_norms.get("action_grad_norm", 0.0)

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
    train_metrics.vision_param_norm = vision_param_norm
    train_metrics.vision_grad_norm = vision_grad_norm
    train_metrics.lang_param_norm = lang_param_norm
    train_metrics.lang_grad_norm = lang_grad_norm
    train_metrics.action_param_norm = action_param_norm
    train_metrics.action_grad_norm = action_grad_norm
    train_metrics.lr = optimizer.param_groups[0]["lr"]
    train_metrics.update_s = time.perf_counter() - start_time
    return train_metrics, output_dict


def test_policy(
    test_metrics: MetricsTracker,
    policy: PreTrainedPolicy,
    batch: Any,
    use_amp: bool = False,
    method: ExtendedConfig | None = None,
) -> tuple[MetricsTracker, dict]:
    start_time = time.perf_counter()
    device = get_device_from_parameters(policy)
    policy.eval()
    with torch.no_grad():
        with torch.autocast(device_type=device.type) if use_amp else nullcontext():
            loss, output_dict = policy.forward(batch, method=method)
            # TODO(rcadene): policy.unnormalize_outputs(out_dict)

    test_metrics.loss = loss.item()
    return test_metrics, output_dict


@parser.wrap()
def train(cfg: TrainPipelineConfig):
    cfg.validate()

    # ---------------------------------------------------------
    # HYPERPARAMETERS FOR DEBUGGING
    # ---------------------------------------------------------
    cfg.method = ExtendedConfig()
    cfg.method.core = 'lora_msp'
    # cfg.method.lora_cfg = LoraMoEConfig(
    #     r=32,
    #     alpha=64,
    #     quantize=False,
    #     num_experts=4,
    #     routing='top1'
    # )
    cfg.method.lora_cfg = LoraMSPConfig(
        r=32,
        alpha=64,
        quantize=False,
        num_experts=4,
        target_threshold_init=0.9,
        target_threshold_end=0.5,
        threshold_scheduling='linear',
        use_spec_loss=True,
        use_modular_loss=False,
        use_id_loss=False,
        router_projection=True,
        routing='weighted'
    )
    cfg.method.target_keywords = ["all-linear"]
    # cfg.method.adapter_file_path = [
    #     '/result/pi0_lora_r32_openthepot/checkpoints/030000/pretrained_model/adapters.safetensors',
    #     '/result/pi0_lora_r32_pickplace/030000/pretrained_model/adapters.safetensors',
    #     '/result/pi0_lora_r32_pourtheblock/030000/pretrained_model/adapters.safetensors',
    #     '/result/pi0_lora_r32_pushthebutton/030000/pretrained_model/adapters.safetensors'
    # ]
    # cfg.method.adapter_file_path = [
    #     '/result/SmolVLA_lora_moe_r32_weighted_multitask/030000/pretrained_model/adapters.safetensors'
    # ]
    cfg.gradient_checkpointing = True
    cfg.method.aux_loss_cfg = {
        "lb_coeff": 1e-3,
        "z_coeff": 0.01,
        "spec_coeff": 0.01,
        "mod_coeff": 1e-3,
        "id_coeff": 1e-3,
    }
    cfg.method.expert_source = 'lora_moe'

    # ---------------------------------------------------------
    # distributed mode flags
    # ---------------------------------------------------------
    dist_mode = getattr(cfg, "dist_mode", "none")  # 'ddp', 'fsdp', 'none'
    use_ddp = dist_mode == "ddp"
    use_fsdp = dist_mode == "fsdp"
    is_distributed = use_ddp or use_fsdp

    device = get_safe_torch_device(cfg.policy.device, log=True)
    torch.backends.cuda.matmul.allow_tf32 = True

    if is_distributed:
        if os.environ.get("LOCAL_RANK", -1) == -1:  # not called by torchrun, do not initialize dist.
            device, local_rank = torch.device("cuda"), 0  # single GPU

        if not dist.is_initialized():
            dist.init_process_group(backend="nccl", timeout=datetime.timedelta(minutes=5))
        local_rank = dist.get_rank()
        device = torch.device("cuda", local_rank)
        torch.cuda.set_device(device)  # needed!
        logging.info(f"Local Rank ({local_rank}) Initialized for {dist_mode.upper()}")
    else:
        local_rank = 0
    cfg.policy.device = str(device)

    if is_ddp_master(is_distributed, local_rank):
        logging.info(pformat(cfg.to_dict()))

    if cfg.seed is not None:
        set_seed(cfg.seed)

    if cfg.wandb.enable and cfg.wandb.project:
        if is_ddp_master(is_distributed, local_rank):
            wandb_logger = WandBLogger(cfg)
    else:
        wandb_logger = None
        logging.info(colored("Logs will be saved locally.", "yellow", attrs=["bold"]))

    if is_ddp_master(is_distributed, local_rank):
        logging.info("Creating dataset")

    train_dataset = make_dataset(cfg, split="train")
    test_dataset = make_dataset(cfg, split="test")

    if is_ddp_master(is_distributed, local_rank):
        logging.info("Creating policy")
        
    policy = make_policy(
        cfg=cfg.policy,
        ds_meta = train_dataset.meta,
    )

    policy, res = wrap_policy(
        policy = policy,
        cfg = cfg.method,
        is_master = is_ddp_master(is_distributed, local_rank),
        device = device,
    )
    if is_ddp_master(is_distributed, local_rank):
        logging.info(res)

    policy_m, res = dist_policy(
        policy = policy,
        dist_mode = dist_mode,
        is_distributed = dist.is_initialized() and dist.is_available(),
        local_rank = local_rank,
        device = device,
    )
    if is_ddp_master(is_distributed, local_rank):
        logging.info(res)

    if is_ddp_master(is_distributed, local_rank):
        logging.info("Creating optimizer and scheduler")
    optimizer, lr_scheduler = make_optimizer_and_scheduler(cfg, policy_m)
    grad_scaler = GradScaler(device.type, enabled=cfg.policy.use_amp)

    step = 0  # number of policy updates (forward + backward + optim)

    if cfg.resume:
        logging.info(f"Resuming training from checkpoint: {cfg.checkpoint_path}")
        step, optimizer, lr_scheduler = load_training_state(cfg.checkpoint_path, optimizer, lr_scheduler)

    num_learnable_params = sum(p.numel() for p in policy.parameters() if p.requires_grad)
    num_total_params = sum(p.numel() for p in policy.parameters())
    learnable_params_proportion = 100.0 * (num_learnable_params / num_total_params) if num_total_params > 0 else 0.0

    if is_ddp_master(is_distributed, local_rank):
        logging.info(colored("Output dir:", "yellow", attrs=["bold"]) + f" {cfg.output_dir}")
        logging.info(f"{cfg.steps=} ({format_big_number(cfg.steps)})")
        logging.info(f"{train_dataset.num_frames=} ({format_big_number(train_dataset.num_frames)})")
        logging.info(f"{train_dataset.num_episodes=}")
        logging.info(f"{num_learnable_params=} ({format_big_number(num_learnable_params)})")
        logging.info(f"{num_total_params=} ({format_big_number(num_total_params)})")
        logging.info(f"learnable_param_proportion={learnable_params_proportion:.2f}%")
        if wandb_logger:
            wandb_logger.log_dict(
                {
                    "num_total_params": int(num_total_params),
                    "num_trainable_params": int(num_learnable_params),
                    "learnable_params_proportion": float(learnable_params_proportion),
                },
                step=step,
                mode="train",
            )

    # create dataloader for offline training
    train_dataloader = make_dataloader(cfg, train_dataset, device)
    train_dl_iter = cycle(train_dataloader)

    test_dataloader = make_dataloader(cfg, test_dataset, device)
    test_dl_iter = cycle(test_dataloader)

    policy.train()

    train_metrics = {
        "loss": AverageMeter("loss", ":.3f"),
        "grad_norm": AverageMeter("grdn", ":.3f"),
        "param_norm": AverageMeter("pnorm", ":.3f"),
        "lr": AverageMeter("lr", ":0.1e"),
        "update_s": AverageMeter("updt_s", ":.3f"),
        "dataloading_s": AverageMeter("data_s", ":.3f"),
        "vision_param_norm": AverageMeter("v_pn", ":0.1e"),
        "vision_grad_norm": AverageMeter("v_gn", ":0.1e"),
        "lang_param_norm": AverageMeter("l_pn", ":0.1e"),
        "lang_grad_norm": AverageMeter("l_gn", ":0.1e"),
        "action_param_norm": AverageMeter("a_pn", ":0.1e"),
        "action_grad_norm": AverageMeter("a_gn", ":0.1e"),
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

    if is_ddp_master(is_distributed, local_rank):
        logging.info("Start offline training on a fixed dataset")

    if cfg.gradient_checkpointing:
        # assert not (cfg.use_lora_moe or cfg.use_qlora_moe)
        policy_m.supports_gradient_checkpointing = True
        policy_m.gradient_checkpointing_enable()

    for _ in range(step, cfg.steps):
        if is_distributed:
            dist.barrier(device_ids=[local_rank])

        start_time = time.perf_counter()
        batch = next(train_dl_iter)
        train_tracker.dataloading_s = time.perf_counter() - start_time

        for key in batch:
            if isinstance(batch[key], torch.Tensor):
                batch[key] = batch[key].to(device, non_blocking=True)

        if (cfg.method.core == 'lora_msp') and (cfg.method.lora_cfg.threshold_scheduling is not None):
            for m in policy.modules():
                if isinstance(m, LoraMSPLinear):
                    m.set_threshold(step/cfg.steps)

        train_tracker, output_dict = update_policy(
            train_tracker,
            policy,
            batch,
            optimizer,
            cfg.optimizer.grad_clip_norm,
            grad_scaler=grad_scaler,
            lr_scheduler=lr_scheduler,
            use_amp=cfg.policy.use_amp,
            method=cfg.method,
        )
        if is_distributed:
            dist.barrier(device_ids=[local_rank])

        # Note: eval and checkpoint happens *after* the `step`th training update has completed, so we
        # increment `step` here.
        step += 1
        train_tracker.step()
        test_tracker.step()
        is_log_step = cfg.log_freq > 0 and step % cfg.log_freq == 0
        is_saving_step = step % cfg.save_freq == 0 or step == cfg.steps
        is_test_step = cfg.test_freq > 0 and step % cfg.test_freq == 0
        is_k_plot_step = cfg.k_plot_freq > 0 and step % cfg.k_plot_freq == 0 and cfg.method.core == 'lora_msp'

        if is_log_step:
            if is_distributed and (dist.get_rank() != 0):
                pass
            else:
                logging.info(train_tracker)
                log_wandb_tracker(wandb_logger, train_tracker, output_dict, step)

        if is_k_plot_step:
            if is_distributed and (dist.get_rank() != 0):
                pass
            else:
                k_dist = policy_m.get_k_distribution()
                log_csv_path = cfg.output_dir / "k_dist_log"
                log_wandb_k_dist(wandb_logger, k_dist, step)
                log_csv_k_dist(log_csv_path, k_dist, step) if is_log_step else None

        if cfg.save_checkpoint and is_saving_step:
            if isinstance(policy, FSDP):
                with FSDP.state_dict_type(policy, StateDictType.FULL_STATE_DICT):
                    full_state_dict = {k: v.cpu() for k, v in policy.state_dict().items()}

            if is_ddp_master(is_distributed, local_rank):
                logging.info(f"Checkpoint policy after step {step}")
                checkpoint_dir = get_step_checkpoint_dir(cfg.output_dir, cfg.steps, step)

                if cfg.use_adapters:
                    # Save only adapter weights + training state to keep checkpoint light
                    pretrained_dir = checkpoint_dir / "pretrained_model"
                    cfg.save_pretrained(pretrained_dir)
                    save_adapters(policy_m, pretrained_dir / "adapters.safetensors")
                    save_training_state(checkpoint_dir, step, optimizer, lr_scheduler)
                elif not isinstance(policy, FSDP):
                    # Full model checkpoint
                    save_checkpoint(checkpoint_dir, step, cfg, policy_m, optimizer, lr_scheduler)
                else:
                    assert isinstance(policy, FSDP)
                    save_checkpoint(checkpoint_dir, step, cfg, policy_m, optimizer, lr_scheduler, is_sharded=True, state_dict=full_state_dict)

                    del full_state_dict
                    gc.collect()
                    torch.cuda.empty_cache()

                update_last_checkpoint(checkpoint_dir)
                if wandb_logger:
                    wandb_logger.log_policy(checkpoint_dir)

            if is_distributed:
                dist.barrier()

        if is_test_step:
            test_batch = next(test_dl_iter)
            test_batch = batch_to_device(test_batch, device)

            test_tracker, output_dict = test_policy(
                test_tracker,
                policy,
                test_batch,
                use_amp=cfg.policy.use_amp,
                method=cfg.method,
            )

            if is_ddp_master(is_distributed, local_rank):
                logging.info(test_tracker)
                log_wandb_tracker(wandb_logger, test_tracker, output_dict, step, mode='eval')

    logging.info("End of training")
    if is_distributed and dist.is_initialized():
        dist.destroy_process_group()


if __name__ == "__main__":
    init_logging()
    train()
