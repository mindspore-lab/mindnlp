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
# pylint: disable=arguments-differ
# pylint: disable=arguments-renamed
# pylint: disable=useless-parent-delegation
# pylint: disable=invalid-name
# pylint: disable=unused-variable
# pylint: disable=unused-argument
# pylint: disable=too-many-arguments
"Adalora Model"
import warnings
from mindspore import nn, ops, Tensor, Parameter
from mindnlp.transformers.ms_utils import Conv1D

from mindnlp.peft.tuners.lora import LoraConfig, LoraModel
from mindnlp.peft.utils import (
    TRANSFORMERS_MODELS_TO_ADALORA_TARGET_MODULES_MAPPING,
    _freeze_adapter,
    _get_submodules,
)

from ..tuners_utils import BaseTunerLayer
from .layer import AdaLoraLayer, RankAllocator, SVDLinear


class AdaLoraModel(LoraModel):
    """
    Creates AdaLoRA (Adaptive LoRA) model from a pretrained transformers model. Paper:
    https://openreview.net/forum?id=lq62uWRJjiY

    Args:
        model ([`mindspore.nn.Cell`]): The model to be adapted.
        config ([`AdaLoraConfig`]): The configuration of the AdaLora model.
        adapter_name (`str`): The name of the adapter, defaults to `"default"`.

    Returns:
        AdaLoraModel ([`mindspore.nn.Cell`]): The AdaLora model.

    Example::

        >>> from transformers import AutoModelForSeq2SeqLM, LoraConfig >>> from peft import AdaLoraModel, AdaLoraConfig
        >>> config = AdaLoraConfig(
                peft_type="ADALORA", task_type="SEQ_2_SEQ_LM", r=8, lora_alpha=32, target_modules=["q", "v"],
                lora_dropout=0.01,
            )
        >>> model = AutoModelForSeq2SeqLM.from_pretrained("t5-base") >>> model = AdaLoraModel(model, config, "default")

    > **Attributes**:  

    >   - **model** ([`transformers.PreTrainedModel`])— The model to be adapted. 

    >   - **peft_config** ([`AdaLoraConfig`]): The configuration of the AdaLora model. 
    """

    # Note: don't redefine prefix here, it should be inherited from LoraModel

    def __init__(self, model, config, adapter_name):
        super().__init__(model, config, adapter_name)

        traininable_mode_counter = 0
        for peft_config in self.peft_config.values():
            if not peft_config.inference_mode:
                traininable_mode_counter += 1

        if traininable_mode_counter > 1:
            raise ValueError(
                "AdaLoraModel supports only 1 trainable adapter. "
                "When using multiple adapters, set inference_mode to True for all adapters except the one you want to train."
            )

        if self.peft_config[adapter_name].inference_mode:
            _freeze_adapter(self.model, adapter_name)
        else:
            self.trainable_adapter_name = adapter_name
            self.rankallocator = RankAllocator(self.model, self.peft_config[adapter_name], self.trainable_adapter_name)

    def _check_new_adapter_config(self, config: LoraConfig) -> None:
        """
        A helper method to check the config when a new adapter is being added.

        Raise a ValueError if there is something wrong with the config or if it conflicts with existing adapters.

        """
        super()._check_new_adapter_config(config)

        traininable_mode_counter = 0
        for config_ in self.peft_config.values():
            if not config_.inference_mode:
                traininable_mode_counter += 1

        if traininable_mode_counter > 1:
            raise ValueError(
                f"{self.__class__.__name__} supports only 1 trainable adapter. "
                "When using multiple adapters, set inference_mode to True for all adapters except the one "
                "you want to train."
            )
    def _mark_only_adapters_as_trainable(self, model: nn.Cell) -> None:
        for n, p in model.parameters_and_names():
            if "lora_" not in n:
                p.requires_grad = False

        for active_adapter in self.active_adapters:
            bias = self.peft_config[active_adapter].bias
            if bias == "none":
                continue

            if bias == "all":
                for n, p in model.parameters_and_names():
                    if "bias" in n:
                        p.requires_grad = True
            elif bias == "lora_only":
                for m in model.cells():
                    if isinstance(m, AdaLoraLayer) and hasattr(m, "bias") and m.bias is not None:
                        m.bias.requires_grad = True
            else:
                raise NotImplementedError(f"Requested bias: {bias}, is not implemented.")
    def _create_and_replace(
        self,
        lora_config,
        adapter_name,
        target,
        target_name,
        parent,
        current_key,
        **optionnal_kwargs,
    ):
        kwargs = {
            "r": lora_config.init_r,
            "lora_alpha": lora_config.lora_alpha,
            "lora_dropout": lora_config.lora_dropout,
            "fan_in_fan_out": lora_config.fan_in_fan_out,
            "init_lora_weights": lora_config.init_lora_weights,
        }
        kwargs["loaded_in_8bit"] = optionnal_kwargs.pop("loaded_in_8bit", False)
        kwargs["loaded_in_4bit"] = optionnal_kwargs.pop("loaded_in_4bit", False)
        # if (kwargs["loaded_in_8bit"] or kwargs["loaded_in_4bit"]) and not is_bnb_available():
        #     raise ImportError(
        #         "To use AdaLora with 8-bit quantization, please install the `bitsandbytes` package. "
        #         "You can install it with `pip install bitsandbytes`."
        #     )
        # quantization_config = get_quantization_config(self.model, method="gptq")
        # if quantization_config is not None:
        #     kwargs["gptq_quantization_config"] = quantization_config

        # If it is not an AdaLoraLayer, create a new module, else update it with new adapters
        if not isinstance(target, AdaLoraLayer):
            new_module = self._create_new_module(lora_config, adapter_name, target, **kwargs)
            self._replace_module(parent, target_name, new_module, target)
        else:
            target.update_layer(
                adapter_name,
                lora_config.init_r,
                lora_config.lora_alpha,
                lora_config.lora_dropout,
                lora_config.init_lora_weights,
            )

    @staticmethod
    def _create_new_module(lora_config, adapter_name, target, **kwargs):
        # avoid eager bnb import
        # if is_bnb_available():
        #     import bitsandbytes as bnb

        #     from .bnb import SVDLinear8bitLt
        # if is_bnb_4bit_available():
        #     from .bnb import SVDLinear4bit

        # gptq_quantization_config = kwargs.get("gptq_quantization_config", None)
        # AutoGPTQQuantLinear = get_auto_gptq_quant_linear(gptq_quantization_config)

        # loaded_in_8bit = kwargs.pop("loaded_in_8bit", False)
        # loaded_in_4bit = kwargs.pop("loaded_in_4bit", False)

        if isinstance(target, BaseTunerLayer):
            target_base_layer = target.get_base_layer()
        else:
            target_base_layer = target

        # if loaded_in_8bit and isinstance(target_base_layer, bnb.nn.Linear8bitLt):
        #     kwargs.update(
        #         {
        #             "has_fp16_weights": target_base_layer.state.has_fp16_weights,
        #             "memory_efficient_backward": target_base_layer.state.memory_efficient_backward,
        #             "threshold": target_base_layer.state.threshold,
        #             "index": target_base_layer.index,
        #         }
        #     )
        #     new_module = SVDLinear8bitLt(target, adapter_name, **kwargs)
        # elif loaded_in_4bit and is_bnb_4bit_available() and isinstance(target_base_layer, bnb.nn.Linear4bit):
        #     fourbit_kwargs = kwargs.copy()
        #     fourbit_kwargs.update(
        #         {
        #             "compute_dtype": target_base_layer.compute_dtype,
        #             "compress_statistics": target_base_layer.weight.compress_statistics,
        #             "quant_type": target_base_layer.weight.quant_type,
        #         }
        #     )
        #     new_module = SVDLinear4bit(target, adapter_name, **fourbit_kwargs)
        # elif AutoGPTQQuantLinear is not None and isinstance(target, AutoGPTQQuantLinear):
        #     new_module = SVDQuantLinear(target, adapter_name, **kwargs)
        if isinstance(target_base_layer, nn.Dense):
            if kwargs["fan_in_fan_out"]:
                warnings.warn(
                    "fan_in_fan_out is set to True but the target module is `torch.nn.Linear`. "
                    "Setting fan_in_fan_out to False."
                )
                kwargs["fan_in_fan_out"] = lora_config.fan_in_fan_out = False
        elif isinstance(target_base_layer, Conv1D):
            if not kwargs["fan_in_fan_out"]:
                warnings.warn(
                    "fan_in_fan_out is set to False but the target module is `Conv1D`. "
                    "Setting fan_in_fan_out to True."
                )
                kwargs["fan_in_fan_out"] = lora_config.fan_in_fan_out = True
        else:
            raise ValueError(
                f"Target module {target} is not supported. "
                f"Currently, only `torch.nn.Linear` and `Conv1D` are supported."
            )
        new_module = SVDLinear(target, adapter_name, **kwargs)

        return new_module
    def _replace_module(self, parent, child_name, new_module, child):
        setattr(parent, child_name, new_module)

        # child layer wraps the original module, unpack it
        if hasattr(child, "base_layer"):
            child = child.base_layer

        # layers with base_layer don't need the weight to be copied, as they have a reference already
        if not hasattr(new_module, "base_layer"):
            new_module.weight = child.weight
            if hasattr(child, "bias"):
                new_module.bias = child.bias

        if getattr(child, "state", None) is not None:
            if hasattr(new_module, "base_layer"):
                new_module.base_layer.state = child.state
            else:
                new_module.state = child.state
    @staticmethod
    def _prepare_adapter_config(peft_config, model_config):
        if peft_config.target_modules is None:
            if model_config["model_type"] not in TRANSFORMERS_MODELS_TO_ADALORA_TARGET_MODULES_MAPPING:
                raise ValueError("Please specify `target_modules` in `peft_config`")
            peft_config.target_modules = TRANSFORMERS_MODELS_TO_ADALORA_TARGET_MODULES_MAPPING[
                model_config["model_type"]
            ]
        return peft_config

    def __getattr__(self, name: str):
        """Forward missing attributes to the wrapped module."""
        try:
            return super().__getattr__(name)  # defer to nn.Module's logic
        except AttributeError:
            return getattr(self.model, name)

    def construct(self, *args, **kwargs):
        """The construct method of the model"""
        outputs = self.model(*args, **kwargs)

        if (getattr(outputs, "loss", None) is not None) and isinstance(outputs.loss, Tensor):
            # Calculate the orthogonal regularization
            orth_reg_weight = self.peft_config[self.trainable_adapter_name].orth_reg_weight

            if orth_reg_weight <= 0:
                raise ValueError("orth_reg_weight should be greater than 0. ")

            regu_loss = 0
            num_param = 0
            for n, p in self.model.parameters_and_names():
                if ("lora_A" in n or "lora_B" in n) and self.trainable_adapter_name in n:
                    para_cov = p @ p.T if "lora_A" in n else p.T @ p
                    I = ops.eye(*para_cov.shape)  # noqa: E741
                    I = ops.stop_gradient(I)
                    num_param += 1
                    regu_loss += ops.norm(para_cov - I, ord="fro")
            if num_param > 0:
                regu_loss = regu_loss / num_param
            else:
                regu_loss = 0
            outputs.loss += orth_reg_weight * regu_loss
        return outputs

    def resize_modules_by_rank_pattern(self, rank_pattern, adapter_name):
        "resize the modules by rank pattern"
        lora_config = self.peft_config[adapter_name]
        for name, rank_idx in rank_pattern.items():
            if isinstance(rank_idx, list):
                rank = sum(rank_idx)
                rank_idx = Tensor(rank_idx).view(-1)
            elif isinstance(rank_idx, Tensor):
                rank_idx = rank_idx.view(-1)
                rank = rank_idx.sum().item()
            else:
                raise ValueError("Unexpected type of rank_idx")
            key = ".".join(name.split(".")[0:-2]) if adapter_name in name else ".".join(name.split(".")[0:-1])
            _, target, _ = _get_submodules(self.model, key)
            lora_E_weights = target.lora_E[adapter_name][rank_idx]
            lora_A_weights = target.lora_A[adapter_name][rank_idx]
            lora_B_weights = target.lora_B[adapter_name][:, rank_idx]
            ranknum = target.ranknum[adapter_name]
            target.update_layer(
                adapter_name,
                rank,
                lora_config.lora_alpha,
                lora_config.lora_dropout,
                lora_config.init_lora_weights,
            )
            if rank > 0:
                target.lora_E.update({adapter_name: Parameter(lora_E_weights)})
                target.lora_A.update({adapter_name: Parameter(lora_A_weights)})
                target.lora_B.update({adapter_name: Parameter(lora_B_weights)})
                # The scaling is exactly as the previous
                target.ranknum.update({adapter_name: Parameter(ranknum)})


    def resize_state_dict_by_rank_pattern(self, rank_pattern, state_dict, adapter_name):
        "resize the state_dict by rank pattern"
        for name, rank_idx in rank_pattern.items():
            rank = sum(rank_idx)
            prefix = ".".join(name.split(".")[0:-2]) if adapter_name in name else ".".join(name.split(".")[0:-1])
            for layer in ["lora_E", "lora_A", "lora_B"]:
                key = f"base_model.model.{prefix}.{layer}.{adapter_name}"
                if layer != "lora_B":
                    if rank != state_dict[key][2].reshape(state_dict[key][0]).shape[0]:
                        dims = []
                        data = state_dict[key][2].reshape(state_dict[key][0])
                        data = data[rank_idx]
                        state_dict[key][2] = data.reshape(-1)
                        for dim in data.shape:
                            dims.append(dim)
                        state_dict[key][0] = dims
                else:
                    if rank != state_dict[key][2].reshape(state_dict[key][0]).shape[1]:
                        dims = []
                        data = state_dict[key][2].reshape(state_dict[key][0])
                        data = data[:, rank_idx]
                        state_dict[key][2] = data.reshape(-1)
                        for dim in data.shape:
                            dims.append(dim)
                        state_dict[key][0] = dims
        return state_dict

    def update_and_allocate(self, global_step, gradient):
        """
        This method updates Adalora budget and mask.

        This should be called in every training step after `loss.backward()` and before `zero_grad()`.

        `tinit`, `tfinal` and `deltaT` are handled with in the method.

        Args:
            global_step (`int`): The current training step, it is used to calculate adalora budget.

        Example:

        ```python
        >>> loss = model(**input).loss
        >>> loss.backward()
        >>> optimizer.step()
        >>> model.base_model.update_and_allocate(i_step)
        >>> optimizer.zero_grad()
        ```
        """
        lora_config = self.peft_config[self.trainable_adapter_name]
        # Update the importance score and allocate the budget
        if global_step < lora_config.total_step - lora_config.tfinal:
            _, rank_pattern = self.rankallocator.update_and_allocate(self.model, global_step, gradient)
            if rank_pattern:
                lora_config.rank_pattern = rank_pattern
        # Finalize the budget allocation
        elif global_step == lora_config.total_step - lora_config.tfinal:
            _, rank_pattern = self.rankallocator.update_and_allocate(self.model, global_step, gradient,force_mask=True)
            # for some reason, this freezes the trainable parameters and nothing gets updates
            # self.resize_modules_by_rank_pattern(rank_pattern, self.trainable_adapter_name)
            lora_config.rank_pattern = rank_pattern
            self.rankallocator.reset_ipt()
        # Currently using inefficient way to mask the unimportant weights using the rank pattern
        #  due to problem mentioned above
        elif global_step > lora_config.total_step - lora_config.tfinal:
            self.rankallocator.mask_using_rank_pattern(self.model, lora_config.rank_pattern)
        # Pass the function and do forward propagation
