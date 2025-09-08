import torch
import torch.nn as nn
from typing import Tuple

from common.policies.lora_moe import LoraMoELinear
from common.policies.lora_msp import LoraMSPLinear


def compute_router_loss(
    model: nn.Module,
    ground_rank: int = 64,
    target_rank: int = 64,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    lb_losses, z_losses, spec_losses, mod_losses, id_losses = [], [], [], [], []

    for module in model.modules():
        if isinstance(module, LoraMoELinear) or isinstance(module, LoraMSPLinear):
            lb_loss = module.compute_balance_loss() if hasattr(module, "compute_balance_loss") else torch.Tensor([0.0], device=model.device, dtype=module.dtype)
            z_loss = module.compute_z_loss() if hasattr(module, "compute_z_loss") else torch.Tensor([0.0], device=model.device, dtype=module.dtype)
            spec_loss = module.compute_spec_loss() if hasattr(module, "compute_spec_loss") else torch.Tensor(0.0, device=model.device, dtype=module.dtype)
            mod_loss = module.compute_mod_loss() if hasattr(module, "compute_mod_loss") else torch.Tensor(0.0, device=model.device, dtype=module.dtype)
            id_loss = module.compute_id_loss() if hasattr(module, "compute_id_loss") else torch.Tensor(0.0, device=model.device, dtype=module.dtype)

            lb_losses.append(lb_loss) if lb_loss is not None else None
            z_losses.append(z_loss) if z_loss is not None else None
            spec_losses.append(spec_loss) if spec_loss is not None else None
            mod_losses.append(mod_loss) if mod_loss is not None else None
            id_losses.append(id_loss) if id_loss is not None else None

    lb_loss = torch.stack(lb_losses).mean()
    z_loss = torch.stack(z_losses).mean()
    spec_loss = torch.stack(spec_losses).mean()
    mod_loss = torch.stack(mod_losses).mean()
    id_loss = torch.stack(id_losses).mean()

    return lb_loss, z_loss, spec_loss, mod_loss, id_loss