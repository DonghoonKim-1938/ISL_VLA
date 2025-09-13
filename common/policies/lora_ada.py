from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Optional

import torch
import torch.nn as nn

from common.policies.lora import LoraConfig, LoraLinear

@dataclass
class LoraADAConfig(LoraConfig):
    layer_type: str = "lora_ada"
    merge_weights: bool = False

class LoraADALinear(LoraLinear):
    def __init__(self, base: nn.Linear, cfg: LoraADAConfig):
        super().__init__(base, cfg)
        if not isinstance(base, nn.Linear):
            raise TypeError("AdaLoraLinear expects an nn.Linear to wrap")

        self.cfg = cfg
        self._load_base(base, cfg.quantize)

        in_f, out_f = self.base.in_features, self.base.out_features
        if cfg.fan_in_fan_out:
            in_f, out_f = out_f, in_f

        self.A = nn.Parameter(torch.zeros(cfg.r, in_f, dtype=self.base.weight.dtype))
        self.E = nn.Parameter(torch.zeros(cfg.r, 1, dtype=self.base.weight.dtype))
        self.B = nn.Parameter(torch.zeros(out_f, cfg.r, dtype=self.base.weight.dtype))
        self.rank_num = nn.Parameter(torch.zeros(1, dtype=self.base.weight.dtype, requires_grad=False))

        self.rank_num.data.fill_(float(cfg.r))
        self.rank_num.requires_grad = False

        nn.init.kaiming_uniform_(self.A, a=math.sqrt(5))
        nn.init.kaiming_uniform_(self.B, a=math.sqrt(5))
        nn.init.zeros_(self.E)

    def extra_repr(self) -> str:
        return (
            f"in_features={self.base.in_features}, out_features={self.base.out_features}, "
            f"r={self.cfg.r}, alpha={self.cfg.alpha}"
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        base_out = self.base(x)
        x_dp = self.dropout(x)

        if self.cfg.r > 0:
            proj = self.A * self.E
            proj_r = torch.matmul(x_dp, proj.T)
            lora_out = torch.matmul(proj_r, self.B.T)

            scaled_lora_out = lora_out * self.cfg.scale
        else:
            scaled_lora_out = torch.zeros_like(base_out)

        return base_out + scaled_lora_out


class RankAllocator(object):
    """
    The RankAllocator for AdaLoRA Model that will be called every training step. 
    Paper: https://openreview.net/pdf?id=lq62uWRJjiY

    Args:
        model: the model that we apply AdaLoRA to.
        r (`int`): The initial rank for each incremental matrix.
        target_rank (`int`): The target average rank of incremental matrix.
        init_warmup (`int`): The steps of initial fine-tuning warmup.
        final_warmup (`int`): The step of final fine-tuning.
        mask_interval (`int`): The time internval between two budget allocations.
        beta1 (`float`): The hyperparameter of EMA for sensitivity smoothing.
        beta2 (`float`): The hyperparameter of EMA for undertainty quantification.
        total_step (`int`): The total training steps, correctly configured before training.
        target_total_rank (`Optinal[int]`): The speficified final total rank. 
        tb_writter (`SummaryWriter`): Tensorboard SummaryWriter. 
        tb_writter_loginterval (`int`): The logging interval of SummaryWriter. 
    """
    def __init__(
        self,
        model: nn.Module,
        r: int = 128,
        target_rank: int = 76,
        init_warmup: int = 1000,
        final_warmup: int = 20000,
        mask_interval: int = 10,
        beta1: float = 0.85,
        beta2: float = 0.85,
        total_step:Optional[int]=None,
        target_total_rank:Optional[int]=None,
        tb_writter=None,
        tb_writter_loginterval:int=500,
    ):
        self.ave_target_rank = target_rank
        self.target_rank = target_total_rank
        self.lora_init_rank = r
        self.initial_warmup = init_warmup
        self.final_warmup = final_warmup
        self.mask_interval = mask_interval
        self.beta1 = beta1
        self.beta2 = beta2
        self.total_step = total_step

        self.model = model
        self.ipt = {}
        self.exp_avg_ipt = {}
        self.exp_avg_unc = {}
        self.cat_ipt = {}
        self.rank_pattern = {}
        self.get_lora_param_name()

        self.tb_writter = tb_writter
        self.log_interval = tb_writter_loginterval

        assert (self.beta1<1 and self.beta1>0)
        assert (self.beta2<1 and self.beta2>0)

    def set_total_step(self, total_step:int):
        # Set total step number
        self.total_step = total_step
        assert self.total_step>self.initial_warmup+self.final_warmup

    def get_rank_pattern(self):
        # Return rank pattern
        return self.rank_pattern

    def get_lora_param_name(self):
        # Prepare the budget scheduler
        self.name_set = set()
        self.total_rank = 0
        self.shape_dict = {}
        for n,p in self.model.named_parameters():
            if "A" in n:
                name_mat = n.replace("A", "%s")
                self.name_set.add(name_mat)
                self.total_rank += p.size(0)
                self.shape_dict[n] = p.shape
            if "B" in n:
                self.shape_dict[n] = p.shape
        self.name_set = list(sorted(self.name_set))
        if self.target_rank is None:
            self.target_rank = self.ave_target_rank * len(self.name_set)

    def schedule_threshold(self, step:int):
        # Global budget schedule
        mask_ind = False
        target_rank = self.target_rank
        initial_warmup = self.initial_warmup
        final_warmup = self.final_warmup
        total_step = self.total_step
        self.global_step = step
        if step <= initial_warmup:
            # Initial warmup
            curr_rank = self.total_rank
            mask_ind = False
        elif step > total_step - final_warmup:
            # Final fine-tuning
            curr_rank = self.target_rank
            # Fix the rank pattern by
            # always masking the same unimportant singluar values
            mask_ind = True
        else:
            # Budget decreasing
            mul_coeff = 1-(step-initial_warmup)/(total_step-final_warmup-initial_warmup)
            curr_rank = target_rank + (self.total_rank-target_rank)*(mul_coeff**3)
            curr_rank = int(curr_rank)
            mask_ind = True if step % self.mask_interval == 0 else False
        return curr_rank, mask_ind

    def update_ipt(self, model):
        for n,p in model.named_parameters():
            if any(x in n for x in [".A", ".B", ".E"]):
                if n not in self.ipt:
                    self.ipt[n] = torch.zeros_like(p)
                    self.exp_avg_ipt[n] = torch.zeros_like(p)
                    self.exp_avg_unc[n] = torch.zeros_like(p)
                with torch.no_grad():
                    # Calculate sensitivity
                    if p is None or p.grad is None:
                        continue
                    self.ipt[n] = (p * p.grad).abs().detach()
                    # Update sensitivity
                    self.exp_avg_ipt[n] = self.beta1 * self.exp_avg_ipt[n] + \
                                        (1-self.beta1)*self.ipt[n]
                    # Update uncertainty
                    self.exp_avg_unc[n] = self.beta2 * self.exp_avg_unc[n] + \
                                        (1-self.beta2)*(self.ipt[n]-self.exp_avg_ipt[n]).abs()

    def calculate_score(self, n, p=None, metric="ipt"):
        if metric == "ipt":
            # Combine the senstivity and uncertainty
            ipt_score = self.exp_avg_ipt[n] * self.exp_avg_unc[n]
        elif metric == "mag":
            ipt_score = p.abs().detach().clone()
        else:
            raise ValueError("Unexcptected Metric: %s"%metric)
        return ipt_score

    def _combine_ipt(self, ipt_E, ipt_AB):
        ipt_AB = ipt_AB.sum(dim=1, keepdim=False)
        sum_ipt = ipt_E.view(-1) + ipt_AB.view(-1)
        return sum_ipt

    def mask_to_target_rank(self, model, curr_rank):
        is_dict = {}
        combine_dict = {}
        singular_dict = {}
        # Calculate the importance score for each sub matrix
        for n,p in model.named_parameters():
            if "A" in n:
                rdim, hdim_a = p.shape
                ipt_score = self.calculate_score(n, metric="ipt")
                comb_ipt = torch.mean(ipt_score, dim=1, keepdim=True)
                name_mat = n.replace("A", "%s")
                if name_mat not in combine_dict:
                    combine_dict[name_mat] = [comb_ipt]
                else:
                    combine_dict[name_mat].append(comb_ipt)
            if "B" in n:
                hdim_b, rdim = p.shape
                ipt_score = self.calculate_score(n, metric="ipt")
                comb_ipt = torch.mean(ipt_score, dim=0, keepdim=False).view(-1, 1)
                name_mat = n.replace("B", "%s")
                if name_mat not in combine_dict:
                    combine_dict[name_mat] = [comb_ipt]
                else:
                    combine_dict[name_mat].append(comb_ipt)
            if "E" in n:
                ipt_score = self.calculate_score(n, p=p, metric="ipt")
                name_mat = n.replace("E", "%s")
                singular_dict[name_mat] = ipt_score

        # Combine the importance scores
        all_is = []
        for name_mat in combine_dict:
            ipt_E = singular_dict[name_mat]
            ipt_AB = torch.cat(combine_dict[name_mat], dim=1)
            sum_ipt = self._combine_ipt(ipt_E, ipt_AB)
            name_E = name_mat%"E"
            is_dict[name_E] = sum_ipt.view(-1, 1)
            all_is.append(sum_ipt.view(-1))

        # Calculate the masking threshold
        mask_threshold = torch.kthvalue(torch.cat(all_is), (self.total_rank-curr_rank))[0].item()

        # Mask out unimportant singular values
        with torch.no_grad():
            curr_sum_rank = 0
            sum_param = 0
            for n,p in model.named_parameters():
                if "E" in n:
                    p.data.masked_fill_(is_dict[n]<=mask_threshold, 0.0)
                    ranknum = (is_dict[n]>mask_threshold).sum().item()

                    if self.tb_writter is not None and self.global_step%self.log_interval==0:
                        self.tb_writter.add_scalar("Ranknum/%s"%(n,), ranknum, self.global_step)
                        self.rank_pattern[n] = ranknum
                        curr_sum_rank += ranknum
                        sum_param += ranknum*self.shape_dict[n.replace("E", "A")][1]
                        sum_param += ranknum*self.shape_dict[n.replace("E", "B")][0]

            if self.tb_writter is not None and self.global_step%self.log_interval==0:
                self.tb_writter.add_scalar("Budget/total_rank", curr_sum_rank, self.global_step)
                self.tb_writter.add_scalar("Budget/mask_threshold", mask_threshold, self.global_step)
                self.tb_writter.add_scalar("Budget/sum_param", sum_param, self.global_step)

        return mask_threshold


    def update_and_mask(self, model, global_step):
        if global_step<self.total_step-self.final_warmup:
            # Update importance scores element-wise
            self.update_ipt(model)
            # do not update ipt during final fine-tuning
        # Budget schedule
        curr_rank, mask_ind = self.schedule_threshold(global_step)
        if mask_ind:
            # Mask to target budget
            mask_threshold = self.mask_to_target_rank(model, curr_rank)
        else:
            mask_threshold = None
        self._maybe_tb_writter_log(model)
        return curr_rank, mask_threshold

    def _maybe_tb_writter_log(self, model):
        if self.tb_writter is not None and self.global_step%self.log_interval==0:
            with torch.no_grad():
                regu_loss = []
                for n,p in model.named_parameters():
                    if "A" in n or "B" in n:
                        mat = p.data.detach().clone()
                        mat_cov = mat @ mat.T if "A" in n else mat.T @ mat
                        I = torch.eye(*mat_cov.size(), out=torch.empty_like(mat_cov))
                        I.requires_grad = False
                        orth_regu = torch.norm(mat_cov-I, p="fro")
                        regu_loss.append(orth_regu.item())
                        self.tb_writter.add_scalar(
                            "Orth_regu_loss/%s"%n, orth_regu.item(), self.global_step
                        )
                self.tb_writter.add_scalar(
                    "train/orth_regu_loss", sum(regu_loss)/len(regu_loss), self.global_step
                )