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
"Adalora Model"
import warnings
from mindspore import Tensor
from mindnlp.core.nn import Parameter

from mindnlp.core import nn, ops
from mindnlp.transformers.ms_utils import Conv1D
from mindnlp.peft.tuners.lora import LoraConfig, LoraModel
from mindnlp.peft.utils import (
    TRANSFORMERS_MODELS_TO_ADALORA_TARGET_MODULES_MAPPING,
    _freeze_adapter,
    _get_subcells,
)

from ..tuners_utils import BaseTunerLayer
from .layer import AdaLoraLayer, RankAllocator, SVDLinear


class AdaLoraModel(LoraModel):
    """
    Creates AdaLoRA (Adaptive LoRA) model from a pretrained transformers model. Paper:
    https://openreview.net/forum?id=lq62uWRJjiY

    Args:
        model ([`mindspore.nn.Module`]): The model to be adapted.
        config ([`AdaLoraConfig`]): The configuration of the AdaLora model.
        adapter_name (`str`): The name of the adapter, defaults to `"default"`.

    Returns:
        AdaLoraModel ([`mindspore.nn.Module`]): The AdaLora model.

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
        r"""
        Initializes an instance of the AdaLoraModel class.
        
        Args:
            self (AdaLoraModel): The current instance of the AdaLoraModel.
            model: The underlying model to be used.
            config: The configuration object for the AdaLoraModel.
            adapter_name: The name of the adapter to be used.
        
        Returns:
            None.
        
        Raises:
            ValueError: If more than one trainable adapter is specified.
            TypeError: If the adapter specified by 'adapter_name' is not in the configuration.
            AttributeError: If the specified adapter is in inference mode.
        """
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
    def _mark_only_adapters_as_trainable(self, model: nn.Module) -> None:
        """
        Marks only specific adapters in the model as trainable based on the specified bias configuration.
        
        Args:
            self: The instance of the AdaLoraModel class.
            model (nn.Module): The neural network model for which adapters should be marked as trainable.
        
        Returns:
            None. This method does not return any value.
        
        Raises:
            NotImplementedError: If the requested bias configuration is not implemented.
        """
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
        r"""
        This method '_create_and_replace' is defined within the 'AdaLoraModel' class and is responsible for creating and replacing a cell based on the provided parameters.
        
        Args:
            self (object): The instance of the 'AdaLoraModel' class.
            lora_config (object): An object containing LoRa configuration parameters.
            adapter_name (str): The name of the adapter.
            target (object): The target object on which the cell will be created and replaced.
            target_name (str): The name of the target.
            parent (object): The parent object where the cell will be replaced.
            current_key: Additional optional keyword arguments.
        
        Returns:
            None: This method does not return any value.
        
        Raises:
            TypeError: If the 'target' parameter is not an instance of the 'AdaLoraLayer' class.
            Exception: Any other unexpected exceptions may be raised during the execution of this method.
        """
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

        # If it is not an AdaLoraLayer, create a new cell, else update it with new adapters
        if not isinstance(target, AdaLoraLayer):
            new_cell = self._create_new_cell(lora_config, adapter_name, target, **kwargs)
            self._replace_cell(parent, target_name, new_cell, target)
        else:
            target.update_layer(
                adapter_name,
                lora_config.init_r,
                lora_config.lora_alpha,
                lora_config.lora_dropout,
                lora_config.init_lora_weights,
            )

    @staticmethod
    def _create_new_cell(lora_config, adapter_name, target, **kwargs):
        r"""
        This method creates a new cell for the AdaLoraModel.
        
        Args:
            lora_config (LoraConfig): The configuration for the LoRa model.
            adapter_name (str): The name of the adapter.
            target (Union[BaseTunerLayer, nn.Module]): The target layer for which the new cell is being created.
        
        Returns:
            None. This method returns None.
        
        Raises:
            - ValueError: If the target cell is not supported. Currently, only `torch.nn.Linear` and `Conv1D` are supported.
            - Warning: If the 'fan_in_fan_out' parameter needs to be adjusted based on the type of the target cell.
        """
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
        #     new_cell = SVDLinear8bitLt(target, adapter_name, **kwargs)
        # elif loaded_in_4bit and is_bnb_4bit_available() and isinstance(target_base_layer, bnb.nn.Linear4bit):
        #     fourbit_kwargs = kwargs.copy()
        #     fourbit_kwargs.update(
        #         {
        #             "compute_dtype": target_base_layer.compute_dtype,
        #             "compress_statistics": target_base_layer.weight.compress_statistics,
        #             "quant_type": target_base_layer.weight.quant_type,
        #         }
        #     )
        #     new_cell = SVDLinear4bit(target, adapter_name, **fourbit_kwargs)
        # elif AutoGPTQQuantLinear is not None and isinstance(target, AutoGPTQQuantLinear):
        #     new_cell = SVDQuantLinear(target, adapter_name, **kwargs)
        if isinstance(target_base_layer, nn.Linear):
            if kwargs["fan_in_fan_out"]:
                warnings.warn(
                    "fan_in_fan_out is set to True but the target cell is `torch.nn.Linear`. "
                    "Setting fan_in_fan_out to False."
                )
                kwargs["fan_in_fan_out"] = lora_config.fan_in_fan_out = False
        elif isinstance(target_base_layer, Conv1D):
            if not kwargs["fan_in_fan_out"]:
                warnings.warn(
                    "fan_in_fan_out is set to False but the target cell is `Conv1D`. "
                    "Setting fan_in_fan_out to True."
                )
                kwargs["fan_in_fan_out"] = lora_config.fan_in_fan_out = True
        else:
            raise ValueError(
                f"Target cell {target} is not supported. "
                f"Currently, only `torch.nn.Linear` and `Conv1D` are supported."
            )
        new_cell = SVDLinear(target, adapter_name, **kwargs)

        return new_cell
    def _replace_cell(self, parent, child_name, new_cell, child):
        r"""
        This method '_replace_cell' is defined within the 'AdaLoraModel' class.
        It replaces a cell within the model with a new cell, transferring relevant attributes from the original cell to the new cell.
        
        Args:
        - self (object): The instance of the AdaLoraModel class.
        - parent (object): The parent object where the cell is to be replaced.
        - child_name (str): The name of the child attribute within the parent object.
        - new_cell (object): The new cell object that will replace the original cell.
        - child (object): The original cell object that is being replaced.
        
        Returns:
        None. This method does not return any value.
        
        Raises:
        This method does not explicitly raise any exceptions. However, it may raise AttributeError if the attributes being accessed do not exist in the provided objects.
        """
        setattr(parent, child_name, new_cell)

        # child layer wraps the original cell, unpack it
        if hasattr(child, "base_layer"):
            child = child.base_layer

        # layers with base_layer don't need the weight to be copied, as they have a reference already
        if not hasattr(new_cell, "base_layer"):
            new_cell.weight = child.weight
            if hasattr(child, "bias"):
                new_cell.bias = child.bias

        if getattr(child, "state", None) is not None:
            if hasattr(new_cell, "base_layer"):
                new_cell.base_layer.state = child.state
            else:
                new_cell.state = child.state
    @staticmethod
    def _prepare_adapter_config(peft_config, model_config):
        r"""
        This method '_prepare_adapter_config' in the class 'AdaLoraModel' prepares the adapter configuration based on the provided 'peft_config' and 'model_config' parameters.
        
        Args:
        - peft_config (dict): A dictionary containing the configuration details for the adapter. It should include information about the target cells. If 'target_modules' is not specified, it is inferred based
on the 'model_type' from the 'model_config' parameter.
        - model_config (dict): A dictionary containing the configuration details specific to the model. It is used to determine the 'model_type' which is then used to infer the 'target_modules' if not explicitly
provided in 'peft_config'.
        
        Returns:
        None: This method does not return any value but updates the 'peft_config' parameter with the inferred or provided 'target_modules' based on the 'model_type'.
        
        Raises:
        - ValueError: Raised if 'target_modules' is not specified in 'peft_config' and the 'model_type' from 'model_config' does not have a corresponding mapping in
TRANSFORMERS_MODELS_TO_ADALORA_TARGET_MODULES_MAPPING.
        """
        if peft_config.target_modules is None:
            if model_config["model_type"] not in TRANSFORMERS_MODELS_TO_ADALORA_TARGET_MODULES_MAPPING:
                raise ValueError("Please specify `target_modules` in `peft_config`")
            peft_config.target_modules = TRANSFORMERS_MODELS_TO_ADALORA_TARGET_MODULES_MAPPING[
                model_config["model_type"]
            ]
        return peft_config

    def __getattr__(self, name: str):
        """Forward missing attributes to the wrapped cell."""
        try:
            return super().__getattr__(name)  # defer to nn.Module's logic
        except AttributeError:
            return getattr(self.model, name)

    def forward(self, *args, **kwargs):
        """The forward method of the model"""
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
                    regu_loss += ops.norm(para_cov - I, p="fro")
            if num_param > 0:
                regu_loss = regu_loss / num_param
            else:
                regu_loss = 0
            outputs.loss += orth_reg_weight * regu_loss
        return outputs

    def resize_cells_by_rank_pattern(self, rank_pattern, adapter_name):
        "resize the cells by rank pattern"
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
            _, target, _ = _get_subcells(self.model, key)
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
            # self.resize_cells_by_rank_pattern(rank_pattern, self.trainable_adapter_name)
            lora_config.rank_pattern = rank_pattern
            self.rankallocator.reset_ipt()
        # Currently using inefficient way to mask the unimportant weights using the rank pattern
        #  due to problem mentioned above
        elif global_step > lora_config.total_step - lora_config.tfinal:
            self.rankallocator.mask_using_rank_pattern(self.model, lora_config.rank_pattern)
        # Pass the function and do forward propagation
