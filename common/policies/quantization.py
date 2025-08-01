import gc
from dataclasses import dataclass

import torch
import torch.nn as nn
import bitsandbytes as bnb
from bitsandbytes.nn import Linear4bit

from common.policies.pretrained import PreTrainedPolicy

@dataclass
class QuantizationConfig():
    quant_type: str = 'fp4'
    compute_dtype: torch.dtype = torch.bfloat16
    compress_statistics: bool = False
    quant_storage: torch.dtype = torch.uint8


def quantize_model(policy: PreTrainedPolicy, quant_cfg_obj: QuantizationConfig):
    def _convert_to_4bit(model):
        for name, module in model.named_children():
            if isinstance(module, nn.Linear):
                qlinear = bnb.nn.Linear4bit(
                    input_features=module.in_features,
                    output_features=module.out_features,
                    bias=module.bias is not None,
                    quant_type=quant_cfg_obj.quant_type,
                    compute_dtype=quant_cfg_obj.compute_dtype,
                    compress_statistics=quant_cfg_obj.compress_statistics,
                    quant_storage=quant_cfg_obj.quant_storage,
                )
                qlinear.load_state_dict(module.state_dict())
                qlinear = qlinear.to(policy.device)
                setattr(model, name, qlinear)
            else:
                _convert_to_4bit(module)  # 재귀적으로 내부 모듈도 탐색
        return model

    policy = _convert_to_4bit(policy)
    policy = policy.to(policy.device)

    gc.collect()
    if policy.device != torch.device('cpu'):
        torch.cuda.empty_cache()

    return policy