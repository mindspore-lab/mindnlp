# Copyright 2023-present the HuggingFace Inc. team.
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
# ============================================================================
# pylint: disable=invalid-name
# pylint: disable=arguments-differ
# pylint: disable=too-many-arguments
# pylint: disable=unused-argument
# pylint: disable=missing-function-docstring
# pylint: disable=too-many-instance-attributes
# pylint: disable=consider-using-dict-items
# pylint: disable=too-many-locals
# pylint: disable=simplifiable-if-expression
"Adalora Layer"
import warnings
from typing import Any, List, Optional

from mindspore import nn, ops, Parameter, Tensor, get_grad
from mindspore.common.initializer import initializer,  Normal


from mindnlp.peft.utils import transpose
from mindnlp._legacy.abc import  ParameterDict, CellDict
from mindnlp.transformers.ms_utils import Conv1D

from ..tuners_utils import check_adapters_to_merge, BaseTunerLayer




class AdaLoraLayer(BaseTunerLayer):
    "AdaLoraLayer class for AdaLoraModel."
    # List all names of layers that may contain adapter weights
    # Note: ranknum doesn't need to be included as it is not an nn.Cell
    adapter_layer_names = ("lora_A", "lora_B", "lora_E", "lora_embedding_A", "lora_embedding_B")
    # other_param_names is defined in LoraLayer

    def __init__(self, base_layer: nn.Cell) -> None:
        self.base_layer = base_layer
        self.r = {}
        self.lora_alpha = {}
        self.scaling = {}
        self.lora_dropout = CellDict()
        self.lora_E = ParameterDict({})
        self.lora_A = ParameterDict({})
        self.lora_B = ParameterDict({})
        self.ranknum = ParameterDict({})
        # For Embedding layer
        self.lora_embedding_A = CellDict()
        self.lora_embedding_B = CellDict()
        if isinstance(base_layer, nn.Dense):
            in_features, out_features = base_layer.in_channels, base_layer.out_channels
        elif isinstance(base_layer, nn.Conv2d):
            in_features, out_features = base_layer.in_channels, base_layer.out_channels
        elif isinstance(base_layer, nn.Embedding):
            in_features, out_features = base_layer.vocab_size, base_layer.embedding_size
        elif isinstance(base_layer, Conv1D):
            in_features, out_features = (
                base_layer.weight.ds_shape if hasattr(base_layer.weight, "ds_shape") else base_layer.weight.shape
            )
        else:
            raise ValueError(f"Unsupported layer type {type(base_layer)}")
        self.in_features = in_features
        self.out_features = out_features


    def update_layer(self, adapter_name, r, lora_alpha, lora_dropout, init_lora_weights):
        if r < 0:
            # note: r == 0 is allowed for AdaLora, see #1539
            raise ValueError(f"`r` should be a positive integer or 0, but the value passed is {r}")

        self.r[adapter_name] = r
        self.lora_alpha[adapter_name] = lora_alpha
        if lora_dropout > 0.0:
            lora_dropout_layer = nn.Dropout(p=lora_dropout)
        else:
            lora_dropout_layer = nn.Identity()

        self.lora_dropout[adapter_name] = lora_dropout_layer
        # Actual trainable parameters
        # Right singular vectors
        if r > 0:
            weight_A = ops.randn((r, self.in_features))
            weight_E = ops.randn((r, 1))
            weight_B = ops.randn((self.out_features, r))
        else:
            rank_idx = Tensor([False])
            weight_A = ops.randn((1, self.in_features))
            weight_E = ops.randn((1, 1))
            weight_B = ops.randn((self.out_features, 1))
            weight_A = weight_A[rank_idx, :]
            weight_E = weight_E[rank_idx, :]
            weight_B = weight_B[:, rank_idx]
        self.lora_A.update({adapter_name: Parameter(weight_A)})
        # Singular values
        self.lora_E.update({adapter_name: Parameter(weight_E)})
        # Left singular vectors
        self.lora_B.update({adapter_name: Parameter(weight_B)})
        # The current rank
        self.ranknum.update({adapter_name: Parameter(Tensor(float(r)), requires_grad=False)})
        self.scaling[adapter_name] = lora_alpha if lora_alpha > 0 else float(r)
        if init_lora_weights and r > 0:
            self.reset_lora_parameters(adapter_name)
        # if hasattr(self.get_base_layer(), "qweight"):
        #     # QuantLinear
        #     self.to(self.get_base_layer().qweight.device)
        # else:
        #     self.to(self.get_base_layer().weight.device)
        self.set_adapter(self.active_adapters)

    def reset_lora_parameters(self, adapter_name):
        if adapter_name in self.lora_A.keys():
            self.lora_E[adapter_name].set_data(initializer(
                Normal(sigma=0.02, mean=0.0),
                self.lora_E[adapter_name].shape,
                self.lora_E[adapter_name].dtype
            ))
            self.lora_A[adapter_name].set_data(initializer(
                Normal(sigma=0.02, mean=0.0),
                self.lora_A[adapter_name].shape,
                self.lora_A[adapter_name].dtype
            ))
            self.lora_B[adapter_name].set_data(initializer(
                Normal(sigma=0.02, mean=0.0),
                self.lora_B[adapter_name].shape,
                self.lora_B[adapter_name].dtype
            ))



class SVDLinear(nn.Cell, AdaLoraLayer):
    "SVD-based adaptation by a dense layer"
    # SVD-based adaptation by a dense layer
    def __init__(
        self,
        base_layer: nn.Cell,
        adapter_name: str,
        r: int = 0,
        lora_alpha: int = 1,
        lora_dropout: float = 0.0,
        fan_in_fan_out: bool = False,
        init_lora_weights: bool = True,
        **kwargs,
    ) -> None:
        super().__init__()
        AdaLoraLayer.__init__(self, base_layer)
        # Freezing the pre-trained weight matrix
        self.get_base_layer().requires_grad = False

        self.fan_in_fan_out = fan_in_fan_out
        self._active_adapter = adapter_name
        self.update_layer(adapter_name, r, lora_alpha, lora_dropout, init_lora_weights)

    def merge(self, safe_merge: bool = False, adapter_names: Optional[List[str]] = None) -> None:
        """
        Merge the active adapter weights into the base weights

        Args:
            safe_merge (`bool`, *optional*):
                If True, the merge operation will be performed in a copy of the original weights and check for NaNs
                before merging the weights. This is useful if you want to check if the merge operation will produce
                NaNs. Defaults to `False`.
            adapter_names (`List[str]`, *optional*):
                The list of adapter names that should be merged. If None, all active adapters will be merged. Defaults
                to `None`.
        """
        adapter_names = check_adapters_to_merge(self, adapter_names)
        if not adapter_names:
            # no adapter to merge
            return

        for active_adapter in adapter_names:
            base_layer = self.get_base_layer()
            if active_adapter in self.lora_A.keys():
                if safe_merge:
                    # Note that safe_merge will be slower than the normal merge
                    # because of the copy operation.
                    orig_weights = base_layer.weight.data.clone()
                    orig_weights += self.get_delta_weight(active_adapter)

                    if not ops.isfinite(orig_weights).all():
                        raise ValueError(
                            f"NaNs detected in the merged weights. The adapter {active_adapter} seems to be broken"
                        )

                    base_layer.weight.data = orig_weights
                else:
                    base_layer.weight.data += self.get_delta_weight(active_adapter)
                self.merged_adapters.append(active_adapter)

    def unmerge(self) -> None:
        """
        This method unmerges all merged adapter layers from the base weights.
        """
        if not self.merged:
            warnings.warn("Already unmerged. Nothing to do.")
            return
        while len(self.merged_adapters) > 0:
            active_adapter = self.merged_adapters.pop()
            if active_adapter in self.lora_A.keys():
                self.get_base_layer().weight.data -= self.get_delta_weight(active_adapter)

    def get_delta_weight(self, adapter) -> Tensor:
        return (
            transpose(self.lora_B[adapter] @ (self.lora_A[adapter] * self.lora_E[adapter]), self.fan_in_fan_out)
            * self.scaling[adapter]
            / (self.ranknum[adapter] + 1e-5)
        )

    def construct(self, x: Tensor, *args: Any, **kwargs: Any) -> Tensor:
        if self.disable_adapters:
            if self.merged:
                self.unmerge()
            result = self.base_layer(x, *args, **kwargs)
        elif self.merged:
            result = self.base_layer(x, *args, **kwargs)
        else:
            result = self.base_layer(x, *args, **kwargs)
            for active_adapter in self.active_adapters:
                if active_adapter not in self.lora_A.keys():
                    continue
                lora_A = self.lora_A[active_adapter]
                lora_B = self.lora_B[active_adapter]
                lora_E = self.lora_E[active_adapter]
                dropout = self.lora_dropout[active_adapter]
                scaling = self.scaling[active_adapter]
                ranknum = self.ranknum[active_adapter] + 1e-5

                x = x.to(lora_A.dtype)
                result += (dropout(x) @ (lora_A * lora_E).T @ lora_B.T) * scaling / ranknum

        return result

    def __repr__(self) -> str:
        rep = super().__repr__()
        return "adalora." + rep


class RankAllocator:
    """
    The RankAllocator for AdaLoraModel. Paper: https://openreview.net/pdf?id=lq62uWRJjiY

    Args:
        config ([`AdaLoraConfig`]): The configuration of the AdaLora model.
        model: the model that we apply AdaLoRA to.

    """

    def __init__(self, model, peft_config, adapter_name):
        self.peft_config = peft_config
        self.adapter_name = adapter_name
        self.beta1 = peft_config.beta1
        self.beta2 = peft_config.beta2
        assert self.beta1 > 0 and self.beta1 < 1
        assert self.beta2 > 0 and self.beta2 < 1

        self.reset_ipt()
        self._set_budget_scheduler(model)

    def set_total_step(self, total_step):
        self.peft_config.total_step = total_step

    def reset_ipt(self):
        self.ipt = {}
        self.exp_avg_ipt = {}
        self.exp_avg_unc = {}

    def _set_budget_scheduler(self, model):
        self.init_bgt = 0
        self.name_set = set()
        for n, p in model.parameters_and_names():
            if f"lora_A.{self.adapter_name}" in n:
                self.init_bgt += p.data.shape[0]
                self.name_set.add(n.replace("lora_A", "%s"))
        self.name_set = sorted(self.name_set)
        # The total final rank budget
        self.target_bgt = self.peft_config.target_r * len(self.name_set)

    def budget_schedule(self, step: int):
        tinit = self.peft_config.tinit
        tfinal = self.peft_config.tfinal
        total_step = self.peft_config.total_step
        # Initial warmup
        if step <= tinit:
            budget = self.init_bgt
            mask_ind = False
        # Final fine-tuning
        elif step > total_step - tfinal:
            budget = self.target_bgt
            mask_ind = True
        else:
            # Budget decreasing with a cubic scheduler
            mul_coeff = 1 - (step - tinit) / (total_step - tfinal - tinit)
            budget = int((self.init_bgt - self.target_bgt) * (mul_coeff**3) + self.target_bgt)
            mask_ind = True if step % self.peft_config.deltaT == 0 else False
        return budget, mask_ind

    def update_ipt(self, model,gradient):
        # Update the sensitivity and uncertainty for every weight
        for n, p in model.parameters_and_names():
            if "lora_" in n and self.adapter_name in n:
                if n not in self.ipt:
                    grad = get_grad(gradient, p)
                    self.ipt[n] = ops.zeros_like(p)
                    self.exp_avg_ipt[n] = ops.zeros_like(p)
                    self.exp_avg_unc[n] = ops.zeros_like(p)
                    self.ipt[n] = (p * grad).abs()
                    # Sensitivity smoothing
                    self.exp_avg_ipt[n] = self.beta1 * self.exp_avg_ipt[n] + (1 - self.beta1) * self.ipt[n]
                    # Uncertainty quantification
                    self.exp_avg_unc[n] = (
                        self.beta2 * self.exp_avg_unc[n] + (1 - self.beta2) * (self.ipt[n] - self.exp_avg_ipt[n]).abs()
                    )

    def _element_score(self, n):
        return self.exp_avg_ipt[n] * self.exp_avg_unc[n]

    def _combine_ipt(self, ipt_E, ipt_AB):
        ipt_AB = ipt_AB.sum(axis=1, keepdims=False)
        sum_ipt = ipt_E.view(-1) + ipt_AB.view(-1)
        return sum_ipt

    def mask_to_budget(self, model, budget):
        value_ipt = {}
        vector_ipt = {}
        triplet_ipt = {}
        # Get the importance score for A, E, B
        for n, p in model.parameters_and_names():
            if f"lora_A.{self.adapter_name}" in n:
                entry_ipt = self._element_score(n)
                comb_ipt = ops.mean(entry_ipt, axis=1, keep_dims=True)
                name_m = n.replace("lora_A", "%s")
                if name_m not in vector_ipt:
                    vector_ipt[name_m] = [comb_ipt]
                else:
                    vector_ipt[name_m].append(comb_ipt)
            if f"lora_B.{self.adapter_name}" in n:
                entry_ipt = self._element_score(n)
                comb_ipt = ops.mean(entry_ipt, axis=0, keep_dims=False).view(-1, 1)
                name_m = n.replace("lora_B", "%s")
                if name_m not in vector_ipt:
                    vector_ipt[name_m] = [comb_ipt]
                else:
                    vector_ipt[name_m].append(comb_ipt)
            if f"lora_E.{self.adapter_name}" in n:
                entry_ipt = self._element_score(n)
                name_m = n.replace("lora_E", "%s")
                value_ipt[name_m] = entry_ipt

        all_score = []
        # Calculate the score for each triplet
        for name_m in vector_ipt:
            ipt_E = value_ipt[name_m]
            ipt_AB = ops.cat(vector_ipt[name_m], axis=1)
            sum_ipt = self._combine_ipt(ipt_E, ipt_AB)
            name_E = name_m % "lora_E"
            triplet_ipt[name_E] = sum_ipt.view(-1, 1)
            all_score.append(sum_ipt.view(-1))

        # Get the threshold by ranking ipt
        mask_threshold = ops.topk(
            ops.cat(all_score),
            k=self.init_bgt - budget,
            largest=False
        )[0][self.init_bgt - budget-1].item()

        rank_pattern = {}
        # Mask the unimportant triplets
        for n, p in model.parameters_and_names():
            if f"lora_E.{self.adapter_name}" in n:
                p.masked_fill(triplet_ipt[n] <= mask_threshold, 0.0)
                rank_pattern[n] = (~(triplet_ipt[n] <= mask_threshold)).view(-1).asnumpy().tolist()
        return rank_pattern

    def update_and_allocate(self, model, global_step, gradient, force_mask=False):
        # # Update the importance score and allocate the budget
        if global_step < self.peft_config.total_step - self.peft_config.tfinal:
            self.update_ipt(model,gradient)
        budget, mask_ind = self.budget_schedule(global_step)
        # Allocate the budget according to importance scores
        if mask_ind or force_mask:
            rank_pattern = self.mask_to_budget(model, budget)
        else:
            rank_pattern = None
        return budget, rank_pattern

    def mask_using_rank_pattern(self, model, rank_pattern):
        # Mask the unimportant triplets
        is_adapter_name_truncated = False
        if self.adapter_name not in next(iter(rank_pattern.keys())):
            is_adapter_name_truncated = True

        for n, p in model.parameters_and_names():
            if f"lora_E.{self.adapter_name}" in n:
                key = n if not is_adapter_name_truncated else n.replace(f".{self.adapter_name}", "")
                mask = Tensor(rank_pattern[key]).unsqueeze(-1)
                p.masked_fill(~mask.bool(), 0.0)
