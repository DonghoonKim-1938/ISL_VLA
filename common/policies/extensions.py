from dataclasses import dataclass, field
from typing import Optional
from pathlib import Path

from common.policies.lora import LoraConfig
from common.policies.lora_moe import LoraMoEConfig
from common.policies.lora_msp import LoraMSPConfig


@dataclass
class ExtendedConfig:
    core: str = "vanilla"
    target_keywords: list[str] = field(default_factory=lambda: ["all-linear"])
    pretrained_expert: bool = False
    lora_cfg: Optional[LoraConfig] = None

    adapter_file_path: Optional[list[str | Path]] = None
    aux_loss_cfg: Optional[dict] = None

    expert_source: Optional[str] = "lora"

    def match_cfg(self):
        if self.core in ["vanilla"]:
            return None
        elif self.core in ["lora", "qlora"]:
            return LoraConfig()
        elif self.core in ["lora_moe", "qlora_moe"]:
            return LoraMoEConfig()
        elif self.core in ["lora_msp", "qlora_msp"]:
            return LoraMSPConfig()
        else:
            raise ValueError(f"Unknown core: {self.core}")

    @property
    def use_moe(self) -> bool:
        if self.core in ["vanilla", "lora", "qlora"]:
            return False
        elif self.core in ["lora_moe", "qlora_moe", 'lora_msp', 'qlora_msp']:
            return True
        else:
            raise ValueError(f"Unknown core: {self.core}")

    @property
    def use_adapters(self) -> bool:
        return self.core in ["lora", "qlora", "lora_moe", "qlora_moe", "lora_msp", "qlora_msp"]