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
# pylint: disable=line-too-long
# pylint: disable=unused-variable
# pylint: disable=unused-argument
# pylint: disable=too-many-arguments
"IA3 Model"
from __future__ import annotations

import re
import warnings
from dataclasses import asdict
from enum import Enum
from typing import Optional

from mindnlp.core import nn

from mindnlp.transformers.ms_utils import Conv1D
from mindnlp.peft.utils import (
    TRANSFORMERS_MODELS_TO_IA3_FEEDFORWARD_MODULES_MAPPING,
    TRANSFORMERS_MODELS_TO_IA3_TARGET_MODULES_MAPPING,
    ModulesToSaveWrapper,
    _get_subcells,
)
from ..tuners_utils import BaseTuner, BaseTunerLayer, check_target_cell_exists
from .layer import Conv2d, IA3Layer, Linear


class IA3Model(BaseTuner):
    """
    Creates a Infused Adapter by Inhibiting and Amplifying Inner Activations ((IA)^3) model from a pretrained
    transformers model. The method is described in detail in https://arxiv.org/abs/2205.05638

    Args:
        model ([`~transformers.PreTrainedModel`]): The model to be adapted.
        config ([`IA3Config`]): The configuration of the (IA)^3 model.
        adapter_name (`str`): The name of the adapter, defaults to `"default"`.

    Returns:
        IA3Model ([`mindspore.nn.Module`]): The IA3Lora model.

    Example:

        ```py
        >>> from transformers import AutoModelForSeq2SeqLM, ia3Config
        >>> from peft import IA3Model, IA3Config

        >>> config = IA3Config(
        ...     peft_type="IA3",
        ...     task_type="SEQ_2_SEQ_LM",
        ...     target_modules=["k", "v", "w0"],
        ...     feedforward_cells=["w0"],
        ... )

        >>> model = AutoModelForSeq2SeqLM.from_pretrained("t5-base")
        >>> ia3_model = IA3Model(config, model)
        ```
    > **Attributes**:  

    >   - **model** ([`transformers.PreTrainedModel`])— The model to be adapted. 

    >   - **peft_config** ([`IA3Config`]): The configuration of the (IA)^3  model. 
    """
    prefix: str = "ia3_"

    def __init__(self, model, config, adapter_name):
        r"""
        Initializes an instance of the IA3Model class.
        
        Args:
            self: The instance of the IA3Model class.
            model: The model object to be initialized.
            config: The configuration settings for the model.
            adapter_name: The name of the adapter.
        
        Returns:
            None. This method does not return any value.
        
        Raises:
            No specific exceptions are raised by this method.
        """
        super().__init__(model, config, adapter_name)

    @staticmethod
    def _create_new_cell(ia3_config, adapter_name, target, **kwargs):
        r"""
        Creates a new cell based on the provided parameters.
        
        Args:
            ia3_config (IA3Config): The configuration object for IA3Model.
            adapter_name (str): The name of the adapter.
            target (object): The target cell for which a new cell needs to be created.
        
        Returns:
            None
        
        Raises:
            ValueError: If the target cell is not supported. Only `torch.nn.Linear`, `torch.nn.Conv2d`, and `Conv1D` are supported.
            TypeError: If the target is not an instance of `BaseTunerLayer` or `nn.Conv2d`.
            TypeError: If the target base layer is not an instance of `nn.Linear` or `Conv1D`.
          
        Note:
            - The `loaded_in_8bit`, `loaded_in_4bit`, and `is_feedforward` parameters are optional and can be provided as keyword arguments.
            - The `fan_in_fan_out` parameter is expected to be present in the `kwargs` dictionary.
            - Depending on the type of `target` and `target_base_layer`, the appropriate cell (Conv2d or Linear) is created.
            - If `target` is an instance of `BaseTunerLayer`, `target_base_layer` is obtained using `get_base_layer()` method.
            - If `target` is `nn.Conv2d`, a new instance of `Conv2d` is created with the provided arguments.
            - If `target_base_layer` is `nn.Linear`, a new instance of `Linear` is created with the provided arguments.
            - If `target_base_layer` is `Conv1D`, a new instance of `Linear` is created with additional arguments indicating that the target is a Conv1D layer.
            - The created cell is returned.
        
        """
        # avoid eager bnb import
        # if is_bnb_available():
        #     import bitsandbytes as bnb

        #     from .bnb import Linear8bitLt

        # if is_bnb_4bit_available():
        #     from .bnb import Linear4bit

        loaded_in_8bit = kwargs.pop("loaded_in_8bit", False)
        loaded_in_4bit = kwargs.pop("loaded_in_4bit", False)
        is_feedforward = kwargs.pop("is_feedforward", False)

        if isinstance(target, BaseTunerLayer):
            target_base_layer = target.get_base_layer()
        else:
            target_base_layer = target

        # if loaded_in_8bit and isinstance(target_base_layer, bnb.nn.Linear8bitLt):
        #     eightbit_kwargs = kwargs.copy()
        #     eightbit_kwargs.update(
        #         {
        #             "has_fp16_weights": target_base_layer.state.has_fp16_weights,
        #             "memory_efficient_backward": target_base_layer.state.memory_efficient_backward,
        #             "threshold": target_base_layer.state.threshold,
        #             "index": target_base_layer.index,
        #         }
        #     )
        #     new_cell = Linear8bitLt(target, adapter_name, is_feedforward=is_feedforward, **eightbit_kwargs)
        # elif loaded_in_4bit and isinstance(target_base_layer, bnb.nn.Linear4bit):
        #     fourbit_kwargs = kwargs.copy()
        #     fourbit_kwargs.update(
        #         {
        #             "compute_dtype": target_base_layer.compute_dtype,
        #             "compress_statistics": target_base_layer.weight.compress_statistics,
        #             "quant_type": target_base_layer.weight.quant_type,
        #         }
        #     )
        #     new_cell = Linear4bit(target, adapter_name, is_feedforward=is_feedforward, **fourbit_kwargs)
        if isinstance(target, nn.Conv2d):
            new_cell = Conv2d(target, adapter_name, is_feedforward=is_feedforward, **kwargs)
        elif isinstance(target_base_layer, nn.Linear):
            if kwargs["fan_in_fan_out"]:
                warnings.warn(
                    "fan_in_fan_out is set to True but the target cell is `torch.nn.Linear`. "
                    "Setting fan_in_fan_out to False."
                )
                kwargs["fan_in_fan_out"] = ia3_config.fan_in_fan_out = False
            new_cell = Linear(target, adapter_name, is_feedforward=is_feedforward, **kwargs)
        elif isinstance(target_base_layer, Conv1D):
            if not kwargs["fan_in_fan_out"]:
                warnings.warn(
                    "fan_in_fan_out is set to False but the target cell is `Conv1D`. "
                    "Setting fan_in_fan_out to True."
                )
                kwargs["fan_in_fan_out"] = ia3_config.fan_in_fan_out = True
            new_cell = Linear(
                target, adapter_name, is_feedforward=is_feedforward, is_target_conv_1d_layer=True, **kwargs
            )
        else:
            raise ValueError(
                f"Target cell {target} is not supported. "
                f"Currently, only `torch.nn.Linear`, `torch.nn.Conv2d`, and `Conv1D` are supported."
            )
        return new_cell

    @staticmethod
    def _check_target_cell_exists(ia3_config, key):
        r"""
        Checks if the target cell exists in the IA3 configuration.
        
        Args:
            ia3_config (dict): The IA3 configuration dictionary.
                This dictionary contains the configuration information for the IA3Model.
                The target cell is checked against this configuration.
            key (str): The target cell key to be checked.
                This key represents the target cell to be verified against the IA3 configuration.
        
        Returns:
            None: This method does not return any value.
        
        Raises:
            None: This method does not raise any exceptions.
        """
        return check_target_cell_exists(ia3_config, key)

    def _mark_only_adapters_as_trainable(self, model: nn.Module) -> None:
        r"""
        Marks only the adapters in the given model as trainable.
        
        Args:
            self (IA3Model): The instance of the IA3Model class.
            model (nn.Module): The model for which the adapters need to be marked as trainable.
        
        Returns:
            None. This method does not return any value.
        
        Raises:
            None.
        """
        for name, param in model.parameters_and_names():
            if self.prefix not in name:
                param.requires_grad = False

    def _create_and_replace(
        self,
        ia3_config,
        adapter_name,
        target,
        target_name,
        parent,
        **optionnal_kwargs,
    ):
        r"""
        Creates a new cell and replaces the target cell with it.
        
        Args:
            self (IA3Model): The current instance of the IA3Model class.
            ia3_config: The configuration settings for the IA3 model.
            adapter_name: The name of the adapter.
            target: The target cell to be replaced.
            target_name: The name of the target cell.
            parent: The parent cell of the target cell.
        
        Returns:
            None
        
        Raises:
            None
        """
        def _create_and_replace(self, ia3_config, adapter_name, target, target_name, parent):
            """
            Creates a new cell and replaces the target cell with it.
        
            Args:
                self (IA3Model): The current instance of the IA3Model class.
                ia3_config: The configuration settings for the IA3 model.
                adapter_name: The name of the adapter.
                target: The target cell to be replaced.
                target_name: The name of the target cell.
                parent: The parent cell of the target cell.
        
            Returns:
                None
        
            Raises:
                None
            """
            current_key = optionnal_kwargs.pop('current_key')
            is_feedforward = self._check_target_cell_feedforward(ia3_config, current_key)
            kwargs = {'fan_in_fan_out': ia3_config.fan_in_fan_out, 'init_ia3_weights': ia3_config.init_ia3_weights, 'is_feedforward': is_feedforward}
            kwargs['loaded_in_8bit'] = optionnal_kwargs.pop('loaded_in_8bit', False)
            kwargs['loaded_in_4bit'] = optionnal_kwargs.pop('loaded_in_4bit', False)
            if isinstance(target, IA3Layer):
                target.update_layer(adapter_name, ia3_config.init_ia3_weights)
            else:
                new_cell = self._create_new_cell(ia3_config, adapter_name, target, **kwargs)
                if adapter_name not in self.active_adapters:
                    new_cell.requires_grad = False
                self._replace_cell(parent, target_name, new_cell, target)
        # check if target cell is in feedforward_cells
        current_key = optionnal_kwargs.pop("current_key")
        is_feedforward = self._check_target_cell_feedforward(ia3_config, current_key)

        kwargs = {
            "fan_in_fan_out": ia3_config.fan_in_fan_out,
            "init_ia3_weights": ia3_config.init_ia3_weights,
            "is_feedforward": is_feedforward,
        }
        kwargs["loaded_in_8bit"] = optionnal_kwargs.pop("loaded_in_8bit", False)
        kwargs["loaded_in_4bit"] = optionnal_kwargs.pop("loaded_in_4bit", False)
        if isinstance(target, IA3Layer):
            target.update_layer(
                adapter_name,
                ia3_config.init_ia3_weights,
            )
        else:
            new_cell = self._create_new_cell(ia3_config, adapter_name, target, **kwargs)
            if adapter_name not in self.active_adapters:
                # adding an additional adapter: it is not automatically trainable
                new_cell.requires_grad = False
            self._replace_cell(parent, target_name, new_cell, target)

    @staticmethod
    def _check_target_cell_feedforward(ia3_config, key) -> bool:
        """
        A helper private method that checks if the target cell `key` matches with a feedforward cell specified in
        `ia3_config`
        """
        if isinstance(ia3_config.feedforward_cells, str):
            is_feedforward = bool(re.fullmatch(ia3_config.feedforward_cells, key))
        else:
            is_feedforward = any(key.endswith(target_key) for target_key in ia3_config.feedforward_cells)
        return is_feedforward

    def _replace_cell(self, parent, child_name, new_cell, child):
        r"""
        Replaces a specified child object in the parent object with a new object.
        
        Args:
            self (IA3Model): The instance of the IA3Model class.
            parent: The parent object where the child object is located.
            child_name: The name of the child object to be replaced.
            new_cell: The new object that will replace the child object.
            child: The child object to be replaced.
        
        Returns:
            None. This method does not return any value.
        
        Raises:
            None.
        
        Note:
            This method replaces the child object in the parent object with the new object. If the child object has a 'base_layer'
            attribute, the method updates the child object to be the 'base_layer'. If the new object does not have a 'base_layer'
            attribute, the method copies the weight and bias attributes from the child object to the new object. If the child object
            has a 'state' attribute, the method updates the 'state' attribute of the new object to match the child object's 'state'.
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

    def __getattr__(self, name: str):
        """Forward missing attributes to the wrapped cell."""
        try:
            return super().__getattr__(name)  # defer to nn.Module's logic
        except AttributeError:
            return getattr(self.model, name)

    def get_peft_config_as_dict(self, inference: bool = False):
        """Get the configuration of the (IA)^3 model as a dictionary."""
        config_dict = {}
        for key, value in self.peft_config.items():
            config = {k: v.value if isinstance(v, Enum) else v for k, v in asdict(value).items()}
            if inference:
                config["inference_mode"] = True
            config_dict[key] = config
        return config

    def _set_adapter_layers(self, enabled=True):
        r"""
        Method to set the adapter layers in the IA3Model.
        
        Args:
            self (IA3Model): The instance of the IA3Model class.
            enabled (bool, optional): A flag indicating whether to enable the adapter layers. Default is True.
        
        Returns:
            None. This method does not return any value.
        
        Raises:
            - TypeError: If the 'enabled' parameter is not a boolean.
            - AttributeError: If the 'IA3Model' instance does not have a 'cells' method.
            - ValueError: If the 'IA3Model' instance's cells include a cell that is not an IA3Layer or a ModulesToSaveWrapper.
        """
        for cell in self.model.cells():
            if isinstance(cell, (IA3Layer, ModulesToSaveWrapper)):
                cell.enable_adapters(enabled)

    def enable_adapter_layers(self) -> None:
        """Enable all adapters.

        Call this if you have previously disabled all adapters and want to re-enable them.
        """
        self._set_adapter_layers(enabled=True)

    def disable_adapter_layers(self) -> None:
        """Disable all adapters.

        When disabling all adapters, the model output corresponds to the output of the base model.
        """
        self._set_adapter_layers(enabled=False)

    def set_adapter(self, adapter_name: str | list[str]) -> None:
        """Set the active adapter(s).

        Additionally, this function will set the specified adapters to trainable (i.e., requires_grad=True). If this is
        not desired, use the following code.

        ```py
        >>> for name, param in model_peft.named_parameters():
        ...     if ...:  # some check on name (ex. if 'lora' in name)
        ...         param.requires_grad = False
        ```

        Args:
            adapter_name (`str` or `list[str]`): Name of the adapter(s) to be activated.
        """
        for cell in self.model.cells():
            if isinstance(cell, IA3Layer):
                if cell.merged:
                    warnings.warn("Adapter cannot be set when the model is merged. Unmerging the model first.")
                    cell.unmerge()
                cell.set_adapter(adapter_name)
        self.active_adapter = adapter_name

    def _prepare_adapter_config(self, peft_config, model_config):
        r"""
        Prepare the adapter configuration for the IA3Model.
        
        Args:
            self (IA3Model): The instance of the IA3Model class.
            peft_config (object): The configuration object for the adapter.
            model_config (dict): The configuration dictionary for the model.
        
        Returns:
            None: This method does not return any value.
        
        Raises:
            ValueError: If `peft_config.target_modules` is None and `model_config['model_type']` is not found in TRANSFORMERS_MODELS_TO_IA3_TARGET_MODULES_MAPPING.
            ValueError: If `peft_config.feedforward_cells` is None and `model_config['model_type']` is not found in TRANSFORMERS_MODELS_TO_IA3_FEEDFORWARD_MODULES_MAPPING.
        """
        if peft_config.target_modules is None:
            if model_config["model_type"] not in TRANSFORMERS_MODELS_TO_IA3_TARGET_MODULES_MAPPING:
                raise ValueError("Please specify `target_modules` in `peft_config`")
            peft_config.target_modules = TRANSFORMERS_MODELS_TO_IA3_TARGET_MODULES_MAPPING[model_config["model_type"]]
        if peft_config.feedforward_cells is None:
            if model_config["model_type"] not in TRANSFORMERS_MODELS_TO_IA3_FEEDFORWARD_MODULES_MAPPING:
                raise ValueError("Please specify `feedforward_cells` in `peft_config`")
            peft_config.feedforward_cells = TRANSFORMERS_MODELS_TO_IA3_FEEDFORWARD_MODULES_MAPPING[
                model_config["model_type"]
            ]
        return peft_config

    def _unload_and_optionally_merge(
        self, merge: bool = True, safe_merge: bool = False, adapter_names: Optional[list[str]] = None
    ):
        r"""
        This method merges the (IA)^3 layers into the base model. This is needed if someone wants to use the base model
        as a standalone model.

        Args:
            safe_merge (`bool`, `optional`, defaults to `False`):
                If True, the merge operation will be performed in a copy of the original weights and check for NaNs
                before merging the weights. This is useful if you want to check if the merge operation will produce
                NaNs. Defaults to `False`.
            adapter_names (`List[str]`, *optional*):
                The list of adapter names that should be merged. If None, all active adapters will be merged. Defaults
                to `None`.
        """
        if getattr(self.model, "is_loaded_in_8bit", False):
            raise ValueError("Cannot merge ia3 layers when the model is loaded in 8-bit mode")

        if getattr(self.model, "is_loaded_in_4bit", False):
            raise ValueError("Cannot merge ia3 layers when the model is loaded in 4-bit mode")

        self._unloading_checks(adapter_names)
        key_list = [key for key, _ in self.model.name_cells() if self.prefix not in key]
        for key in key_list:
            try:
                parent, target, target_name = _get_subcells(self.model, key)
            except AttributeError:
                continue

            if hasattr(target, "base_layer"):
                if merge:
                    target.merge(safe_merge=safe_merge, adapter_names=adapter_names)
                self._replace_cell(parent, target_name, target.get_base_layer(), target)
            elif isinstance(target, ModulesToSaveWrapper):
                # save any additional trainable cells part of `modules_to_save`
                new_cell = target.modules_to_save[target.active_adapter]
                if hasattr(new_cell, "base_layer"):
                    # check if the cell is itself a tuner layer
                    if merge:
                        new_cell.merge(safe_merge=safe_merge, adapter_names=adapter_names)
                    new_cell = new_cell.get_base_layer()
                setattr(parent, target_name, new_cell)

        return self.model

    def merge_and_unload(self, safe_merge: bool = False, adapter_names: Optional[list[str]] = None) -> nn.Module:
        r"""
        This method merges the IA³ layers into the base model. This is needed if someone wants to use the base model as
        a standalone model.

        Args:
            safe_merge (`bool`):
                whether to activate the safe merging check to check if there is any potential Nan in the adapter
                weights
            adapter_names (`List[str]`, *optional*):
                The list of adapter names that should be merged. If None, all active adapters will be merged. Defaults
                to `None`.

        Example:

        ```py
        >>> from transformers import AutoModelForCausalLM
        >>> from peft import PeftModel

        >>> base_model = AutoModelForCausalLM.from_pretrained("tiiuae/falcon-40b")
        >>> peft_model_id = "smangrul/falcon-40B-int4-peft-lora-sfttrainer-sample"
        >>> model = PeftModel.from_pretrained(base_model, peft_model_id)
        >>> merged_model = model.merge_and_unload()
        ```
        """
        return self._unload_and_optionally_merge(safe_merge=safe_merge, adapter_names=adapter_names)

    def unload(self) -> nn.Module:
        """
        Gets back the base model by removing all the IA³ cells without merging. This gives back the original base
        model.
        """
        return self._unload_and_optionally_merge(merge=False)

    def delete_adapter(self, adapter_name: str) -> None:
        """
        Deletes an existing adapter.

        Args:
            adapter_name (str): Name of the adapter to be deleted.
        """
        if adapter_name not in self.peft_config:
            raise ValueError(f"Adapter {adapter_name} does not exist")
        del self.peft_config[adapter_name]

        key_list = [key for key, _ in self.model.name_cells() if self.prefix not in key]
        new_adapter = None
        for key in key_list:
            _, target, _ = _get_subcells(self.model, key)
            if isinstance(target, IA3Layer):
                target.delete_adapter(adapter_name)
                if new_adapter is None:
                    new_adapter = target.active_adapters[:]

        self.active_adapter = new_adapter or []
