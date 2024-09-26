# Copyright 2023 Huawei Technologies Co., Ltd
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ============================================================================
"""Lokr."""
import re
from typing import Optional, Union, Dict, Type, List
from itertools import chain
from tqdm import tqdm

from mindnlp.core import nn

from ...utils import (
    ModulesToSaveWrapper,
    _get_subcells,
)

from ..tuners_utils import (
    BaseTuner,
    BaseTunerLayer,
    check_target_cell_exists,
)
from .layer import Conv2d, Dense, LoKrLayer
from .config import LoKrConfig


class LoKrModel(BaseTuner):
    """
    Creates Low-Rank Kronecker Product model from a pretrained model. The original method is partially described in
    https://arxiv.org/abs/2108.06098 and in https://arxiv.org/abs/2309.14859 Current implementation heavily borrows
    from
    https://github.com/KohakuBlueleaf/LyCORIS/blob/eb460098187f752a5d66406d3affade6f0a07ece/lycoris/cells/lokr.py

    Args:
        model (`mindspore.nn.Module`): The model to which the adapter tuner layers will be attached.
        peft_config ([`LoKrConfig`]): The configuration of the LoKr model.
        adapter_name (`str`): The name of the adapter, defaults to `"default"`.

    Returns:
        LoKrModel ([`mindspore.nn.Module`]): The LoKr model.

    Example:
        ```py
        >>> from diffusers import StableDiffusionPipeline
        >>> from peft import LoKrModel, LoKrConfig

        >>> config_te = LoKrConfig(
        ...     r=8,
        ...     lora_alpha=32,
        ...     target_modules=["k_proj", "q_proj", "v_proj", "out_proj", "fc1", "fc2"],
        ...     rank_dropout=0.0,
        ...     cell_dropout=0.0,
        ...     init_weights=True,
        ... )
        >>> config_unet = LoKrConfig(
        ...     r=8,
        ...     lora_alpha=32,
        ...     target_modules=[
        ...         "proj_in",
        ...         "proj_out",
        ...         "to_k",
        ...         "to_q",
        ...         "to_v",
        ...         "to_out.0",
        ...         "ff.net.0.proj",
        ...         "ff.net.2",
        ...     ],
        ...     rank_dropout=0.0,
        ...     cell_dropout=0.0,
        ...     init_weights=True,
        ...     use_effective_conv2d=True,
        ... )

        >>> model = StableDiffusionPipeline.from_pretrained("runwayml/stable-diffusion-v1-5")
        >>> model.text_encoder = LoKrModel(model.text_encoder, config_te, "default")
        >>> model.unet = LoKrModel(model.unet, config_unet, "default")
        ```

    > **Attributes**:  

    >   - **model** ([`~nn.Module`])— The model to be adapted. 

    >   - **peft_config** ([`LoKrConfig`]): The configuration of the LoKr  model. 

    """
    prefix: str = "lokr_"
    layers_mapping: Dict[Type[nn.Module], Type[LoKrLayer]] = {
        nn.Conv2d: Conv2d,
        nn.Linear: Dense,
    }

    def _create_and_replace(
        self,
        config: LoKrConfig,
        adapter_name: str,
        target: Union[LoKrLayer, nn.Module],
        target_name: str,
        parent: nn.Module,
        current_key: str,
        loaded_in_8bit: Optional[bool] = False,
        loaded_in_4bit: Optional[bool] = False,
    ) -> None:
        """
        A private method to create and replace the target cell with the adapter cell.
        """
        # Regexp matching - Find key which matches current target_name in patterns provided
        pattern_keys = list(
            chain(config.rank_pattern.keys(), config.alpha_pattern.keys())
        )
        target_name_key = next(
            filter(lambda key: re.match(rf"(.*\.)?{key}$", current_key), pattern_keys),
            target_name,
        )

        kwargs = config.to_dict()
        kwargs["r"] = config.rank_pattern.get(target_name_key, config.r)
        kwargs["alpha"] = config.alpha_pattern.get(target_name_key, config.lora_alpha)

        if isinstance(target, LoKrLayer):
            target.update_layer(adapter_name, **kwargs)
        else:
            new_cell = self._create_new_cell(config, adapter_name, target, **kwargs)
            self._replace_cell(parent, target_name, new_cell, target)

    @classmethod
    def _create_new_cell(
        cls, config: LoKrConfig, adapter_name: str, target: nn.Module, **kwargs
    ) -> LoKrLayer:
        r"""
        This method creates a new LoKrLayer instance based on the provided parameters.
        
        Args:
            cls (class): The class reference. It is used to access the class-level layers_mapping attribute.
            config (LoKrConfig): The configuration object used for creating the new cell.
            adapter_name (str): The name of the adapter to be associated with the new cell.
            target (nn.Module): The target cell for which the new cell is being created.
        
        Returns:
            LoKrLayer: Returns a new instance of LoKrLayer representing the created cell.
        
        Raises:
            ValueError: If the target cell type is not supported, an exception is raised, indicating the unsupported cell type. 
                This occurs when the target cell type does not match any of the supported cell types in the layers_mapping attribute.
        """
        # Find corresponding subtype of provided target cell
        new_cell_cls = None
        for subtype, target_cls in cls.layers_mapping.items():
            if (
                hasattr(target, "base_layer")
                and isinstance(target.get_base_layer(), subtype)
                and isinstance(target, BaseTunerLayer)
            ):
                # nested tuner layers are allowed
                new_cell_cls = target_cls
                break
            elif isinstance(target, subtype):
                new_cell_cls = target_cls
                break

        # We didn't find corresponding type, so adapter for this layer is not supported
        if new_cell_cls is None:
            supported_cells = ", ".join(
                layer.__name__ for layer in cls.layers_mapping.keys()
            )
            raise ValueError(
                f"Target cell of type {type(target)} not supported, "
                f"currently only adapters for {supported_cells} are supported"
            )

        if isinstance(target, BaseTunerLayer):
            target_base_layer = target.get_base_layer()
        else:
            target_base_layer = target

        if isinstance(target_base_layer, nn.Module):
            new_cell = new_cell_cls(target, adapter_name=adapter_name, **kwargs)
        elif isinstance(target_base_layer, nn.Module):
            new_cell = new_cell_cls(target, adapter_name=adapter_name, **kwargs)
        else:
            supported_cells = ", ".join(
                layer.__name__ for layer in cls.layers_mapping.keys()
            )
            raise ValueError(
                f"Target cell of type {type(target)} not supported, "
                f"currently only adapters for {supported_cells} are supported"
            )

        return new_cell

    def __getattr__(self, name: str):
        """Forward missing attributes to the wrapped cell."""
        try:
            return super().__getattr__(name)  # defer to nn.Module's logic
        except AttributeError:
            return getattr(self.model, name)

    def _replace_cell(self, parent, child_name, new_cell, child):
        r"""
        Replaces a cell in the LoKrModel with a new cell.
        
        Args:
            self (LoKrModel): The instance of the LoKrModel class.
            parent: The parent object containing the cell to be replaced.
            child_name: The name of the child object to be replaced.
            new_cell: The new cell object to be assigned.
            child: The child object to be replaced.
        
        Returns:
            None. This method does not return any value.
        
        Raises:
            None.
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

    def _mark_only_adapters_as_trainable(self, model: nn.Module) -> None:
        r"""
        The _mark_only_adapters_as_trainable method in the LoKrModel class marks only the adapters in the provided model as trainable, by setting the requires_grad attribute to False for parameters not
containing the specified prefix.
        
        Args:
            self (LoKrModel): The instance of the LoKrModel class.
            model (nn.Module): The model for which the adapters are to be marked as trainable.
        
        Returns:
            None: This method does not return any value.
        
        Raises:
            None
        """
        for n, p in model.parameters_and_names():
            if self.prefix not in n:
                p.requires_grad = False

    def _set_adapter_layers(self, enabled=True):
        r"""
        Sets the adapter layers in the LoKrModel by enabling or disabling them.
        
        Args:
            self (LoKrModel): The instance of the LoKrModel class.
            enabled (bool, optional): Indicates whether to enable or disable the adapter layers. Defaults to True.
        
        Returns:
            None. This method does not return any value.
        
        Raises:
            None.
        """
        for cell in self.model.cells():
            if isinstance(cell, (BaseTunerLayer, ModulesToSaveWrapper)):
                cell.enable_adapters(enabled)

    def _unload_and_optionally_merge(
        self,
        merge: bool = True,
        progressbar: bool = False,
        safe_merge: bool = False,
        adapter_names: Optional[List[str]] = None,
    ):
        """
        Method to unload and optionally merge the model.
        
        Args:
            self (LoKrModel): The current instance of the LoKrModel class.
            merge (bool): A flag indicating whether to merge the model. Defaults to True.
            progressbar (bool): A flag indicating whether to display a progress bar. Defaults to False.
            safe_merge (bool): A flag indicating whether to perform a safe merge. Defaults to False.
            adapter_names (Optional[List[str]]): A list of adapter names. Defaults to None.
        
        Returns:
            None: This method does not return any value.
        
        Raises:
            ValueError: If the model is gptq quantized and merge is True, it raises a ValueError with the message 
            "Cannot merge LOHA layers when the model is gptq quantized".
            AttributeError: If an attribute error occurs during the method execution.
        """
        if merge:
            if getattr(self.model, "quantization_method", None) == "gptq":
                raise ValueError(
                    "Cannot merge LOHA layers when the model is gptq quantized"
                )

        self._unloading_checks(adapter_names)
        key_list = [
            key for key, _ in self.model.named_cells() if self.prefix not in key
        ]
        desc = "Unloading " + ("and merging " if merge else "") + "model"
        for key in tqdm(key_list, disable=not progressbar, desc=desc):
            try:
                parent, target, target_name = _get_subcells(self.model, key)
            except AttributeError:
                continue

            if hasattr(target, "base_layer"):
                if merge:
                    target.merge(safe_merge=safe_merge, adapter_names=adapter_names)
                self._replace_cell(
                    parent, target_name, target.get_base_layer(), target
                )
            elif isinstance(target, ModulesToSaveWrapper):
                # save any additional trainable cells part of `modules_to_save`
                new_cell = target.modules_to_save[target.active_adapter]
                if hasattr(new_cell, "base_layer"):
                    # check if the cell is itself a tuner layer
                    if merge:
                        new_cell.merge(
                            safe_merge=safe_merge, adapter_names=adapter_names
                        )
                    new_cell = new_cell.get_base_layer()
                setattr(parent, target_name, new_cell)

        return self.model

    def _unloading_checks(self, adapter_names: Optional[List[str]]):
        r"""
        Perform unloading checks for the LoKrModel class.
        
        This method checks if multiple adapters with `modules_to_save` specified can be unloaded.
        If any of the specified adapters have cells to save, unloading multiple adapters is not allowed.
        
        Args:
            self (LoKrModel): An instance of the LoKrModel class.
            adapter_names (Optional[List[str]]): A list of adapter names to consider for unloading. If not provided, all active adapters will be considered.
        
        Returns:
            None. This method does not return any value.
        
        Raises:
            ValueError: If multiple adapters with `modules_to_save` specified are attempted to be unloaded.
        
        """
        adapters_to_consider = adapter_names or self.active_adapters
        is_modules_to_save_available = any(
            self.peft_config[adapter].modules_to_save
            for adapter in adapters_to_consider
        )
        if is_modules_to_save_available and len(adapters_to_consider) > 1:
            raise ValueError(
                "Cannot unload multiple adapters that specify `modules_to_save`."
            )

    @staticmethod
    def _prepare_adapter_config(peft_config, model_config):
        r"""
        Prepare adapter configuration based on PEFT and model configurations.
        
        Args:
            peft_config (object): The configuration object for PEFT.
                It should contain information about the target cells.
                Required parameter. Must not be None.
            model_config (object): The configuration object for the model.
        
        Returns:
            None. This method does not return any value.
        
        Raises:
            ValueError: If `target_modules` is not specified in `peft_config`.
        """
        if peft_config.target_modules is None:
            raise ValueError("Please specify `target_modules` in `peft_config`")
        return peft_config

    @staticmethod
    def _check_target_cell_exists(LoKR_config, key):
        r"""
        Checks if a target cell exists in the LoKR configuration.
        
        Args:
            LoKR_config (dict): The LoKR configuration dictionary containing information about the target cells.
            key (str): The key corresponding to the target cell to be checked.
        
        Returns:
            None. Returns None if the target cell exists in the LoKR configuration; otherwise, raises an exception.
        
        Raises:
            This method does not raise any exceptions explicitly. However, if the target cell does not exist in the LoKR configuration, further handling may be required based on the context in which this
method is used.
        """
        return check_target_cell_exists(LoKR_config, key)
