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
"""
BaseTuner class and BaseTunerLayer class.
"""
from __future__ import annotations
import re
import logging
import warnings
import copy
from typing import Any, Optional, Union
from abc import ABC
from contextlib import contextmanager
from mindspore import Tensor
from mindnlp.core import nn

from ..config import PeftConfig
from ..utils import _get_subcells

logger = logging.getLogger(__name__)


@contextmanager
def onload_layer(layer):
    r"""
    A utility for modifying a cell containing one or more tuners and a base layer, any of which are offloaded to the
    CPU or disk. Moves a cell's sub-cells to the execution device before some action is performed, after that the
    base layer state dictionary is re-assigned (if that layer was offloaded to the disk) and finally the parameters are
    offloaded.

    If the cell has no offloaded sub-cells, this function does nothing.

    Args:
        layer ('mindspore.nn.Module'):
            layer with tuners to be merged
    """
    offloaded_cells = []
    for name, cell in layer.cells_and_names():
        if name in ["", "base_layer"]:
            continue
        # if hasattr(cell, "_hf_hook") and isinstance(cell._hf_hook, AlignDevicesHook) and cell._hf_hook.offload:
        #     cell._hf_hook.pre_forward(cell)
        #     offloaded_cells.append(cell)

    # base_layer_offload = False
    # if hasattr(layer, "base_layer") and (
    #     hasattr(layer.base_layer, "_hf_hook")
    #     and isinstance(layer.base_layer._hf_hook, AlignDevicesHook)
    #     and layer.base_layer._hf_hook.offload
    # ):
    #     # check if the base layer is disk-offloaded (must contain a 'dataset' and an offload index)
    #     if torch.device("meta") in layer.base_layer._hf_hook.original_devices.values() and hasattr(
    #         layer.base_layer._hf_hook.weights_map, "dataset"
    #     ):
    #         # find the disk-offload index (maps cells to safetensors) from the `dataset` (OffloadedWeightsLoader object)
    #         index = layer.base_layer._hf_hook.weights_map.dataset.index
    #         cell_name = list(dict(layer.base_layer._hf_hook.weights_map.dataset).keys())[0]  # any cell will do
    #         file_name = index[cell_name]["safetensors_file"]
    #         base_name_arr = []
    #         # get effective dir name
    #         for i in os.path.split(file_name):
    #             if "--" in i:
    #                 base_name_arr.append(i)
    #                 break
    #             base_name_arr.append(i)
    #         base_name = os.path.join(*base_name_arr)
    #         safetensors_filename = base_name + "-merged"
    #     layer.base_layer._hf_hook.pre_forward(layer.base_layer)
    #     base_layer_offload = True

    # yield

    # for cell in offloaded_cells:
    #     cell._hf_hook.post_forward(cell, torch.tensor([]))

    # if base_layer_offload:
    #     # re-make weights map (must be on cpu to send params to the disk via memmap if disk offload)
    #     layer.base_layer._hf_hook.weights_map = {
    #         name: param.to("cpu") for name, param in named_cell_tensors(layer.base_layer)
    #     }
    #     # offload weights map to disk if original device is the disk
    #     if torch.device("meta") in layer.base_layer._hf_hook.original_devices.values() and hasattr(
    #         layer.base_layer._hf_hook.weights_map, "dataset"
    #     ):
    #         # rewrite directory with merged weights
    #         offload_state_dict(safetensors_filename, layer.base_layer._hf_hook.weights_map)
    #     layer.base_layer._hf_hook.post_forward(layer.base_layer, torch.tensor([]))


class BaseTuner(nn.Module):
    r"""
    A base tuner model that provides the common methods and attributes for all tuners that are injectable into a
    torch.nn.Module

    For adding a new Tuner class, one needs to overwrite the following methods:

    - **_prepare_adapter_config**:
        A private method to eventually prepare the adapter config, for example in case the field `target_modules` is
        missing.
    - **_check_new_adapter_config**:
        A helper private method to check if the passed cell's key name matches any of the target cells in the
        adatper_config.
    - **_create_and_replace**:
        A private method to create and replace the target cell with the adapter cell.
    - **_check_target_cell_exists**:
        A private helper method to check if the passed cell's key name matches any of the target cells in the
        adatper_config.

    The easiest is to check what is done in the `peft.tuners.lora.LoraModel` class.

    Attributes:
        model (`mindspore.nn.Module`):
            The model to which the adapter tuner layers will be attached.
        forward (`Callable`):
            The forward method of the model.
        peft_config (`Union[`PeftConfig`, dict[str, PeftConfig]]`):
            The adapter configuration object, it should be a dictionary of `str` to `PeftConfig` objects. One can also
            pass a PeftConfig object and a new adapter will be created with the default name `adapter` or create a new
            dictionary with a key `adapter_name` and a value of that peft config.
        config (`dict[str, Any]`):
            The model configuration object, it should be a dictionary of `str` to `Any` objects.
    """
    def __init__(self, model, peft_config: Union[PeftConfig, dict[str, PeftConfig]], adapter_name: str) -> None:
        r"""
        __init__
        
        Initializes an instance of the BaseTuner class.
        
        Args:
        - self: The instance of the class.
        - model: The model to be tuned.
        - peft_config: A Union of PeftConfig or a dictionary of adapter names to PeftConfig objects. It specifies the configuration for the adapter.
        - adapter_name: A string representing the name of the adapter.
        
        Returns:
        None. The method initializes the instance of the BaseTuner class.
        
        Raises:
        - AttributeError: If the 'peft_config' attribute is already found in the model, indicating the presence of multiple adapters in the model.
        - TypeError: If the 'peft_config' parameter is not of type PeftConfig or dictionary of adapter names to PeftConfig objects.
        """
        super().__init__()
        # self.peft_config = config
        # self.add_adapter(adapter_name, self.peft_config[adapter_name])

        self.model = model

        # For advanced developers, if you want to attach multiple adapters to your
        # model, just add a `peft_config` dict attribute to your model.
        if not hasattr(self, "peft_config"):
            self.peft_config = {adapter_name: peft_config} if isinstance(peft_config, PeftConfig) else peft_config
        else:
            logger.info(
                "Already found a `peft_config` attribute in the model. This will lead to having multiple adapters"
                " in the model. Make sure to know what you are doing!"
            )
            if isinstance(peft_config, PeftConfig):
                self.peft_config[adapter_name] = peft_config
            else:
                # user is adding a dict of PeftConfigs
                self.peft_config.update(peft_config)

        # transformers models have a .config attribute, whose presence is assumed later on
        # if not hasattr(self, "config"):
        #     self.config = {"model_type": "custom"}

        self.active_adapter: str | list[str] = adapter_name
        self.inject_adapter(self.model, adapter_name)

        # Copy the peft_config in the injected model.
        self.model.peft_config = self.peft_config

    @property
    def active_adapters(self) -> list[str]:
        r"""
        Method to retrieve the active adapters.
        
        Args:
            self: BaseTuner object. The instance of the BaseTuner class.
            
        Returns:
            list[str]: A list of active adapters. If the active_adapter attribute is a string, it is returned as a single-element list. 
            Otherwise, the active_adapter attribute itself is returned.
            
        Raises:
            None
        """
        if isinstance(self.active_adapter, str):
            return [self.active_adapter]
        # is already a list of str
        return self.active_adapter

    def forward(self, *args: Any, **kwargs: Any):
        r"""
        This method forwards an instance of the BaseTuner class.
        
        Args:
            self: The instance of the BaseTuner class.
        
        Returns:
            None. This method does not return any value.
        
        Raises:
            None. This method does not raise any exceptions.
        """
        return self.model.forward(*args, **kwargs)

    def _prepare_adapter_config(self, peft_config: PeftConfig, model_config: dict) -> PeftConfig:
        r"""
        A private method to eventually prepare the adapter config. For transformers based models, if
        `peft_config.target_modules` is None, we can automatically infer the target cells from the
        `TRANSFORMERS_MODELS_TO_XXX_TARGET_MODULES_MAPPING`. This method can be further refactored in the future to
        automatically infer it for all tuner models.

        Check out `peft.tuner.lora.LoraModel._prepare_adapter_config` for an example.

        Args:
            peft_config (`str`):
                The adapter config.
            model_config (`str`):
                The transformers model config, that config should contain the `model_type` key.
        """
        return None

    @staticmethod
    def _check_target_cell_exists(peft_config: PeftConfig, key: str) -> bool:
        r"""
        A helper private method to check if the passed cell's key name matches any of the target cells in the
        `peft_config.target_modules` list. If it does, return `True`, else return `False`.

        Args:
            peft_config (`PeftConfig`):
                The adapter config.
            key (`str`):
                The cell's key name.
        """
    def _create_and_replace(
        self,
        peft_config: PeftConfig,
        adapter_name: str,
        target: nn.Module,
        target_name: str,
        parent: nn.Module,
        current_key: str,
    ) -> None:
        r"""
        Inplace replacement of the target cell with the adapter layer. This method needs to be overriden by all the
        tuner classes.

        Check `peft.tuners.lora.LoraModel._create_and_replace` for an example.

        Args:
            peft_config (`PeftConfig`):
                The adapter config.
            adapter_name (`str`):
                The adapter name.
            target (`nn.Module`):
                The target cell.
            target_name (`str`):
                The target cell's name.
            parent (`nn.Module`):
                The parent cell.
            **optionnal_kwargs (`dict`):
                The optional keyword arguments to pass to deal with particular cases (e.g. 8bit, 4bit quantization)
        """
    def _mark_only_adapters_as_trainable(self, model):
        r"""
        A helper method to mark only the adapter layers as trainable (i.e. cell.requires_grad = False) This needs to
        be overriden for all tuner classes to match the correct key names.

        Check `peft.tuners.lora.LoraModel._mark_only_adapters_as_trainable` for an example.
        """
    def _check_new_adapter_config(self, config: PeftConfig) -> None:
        """
        A helper method to check the config when a new adapter is being added.

        Raise a ValueError if there is something wrong with the config or if it conflicts with existing adapters.

        """
    # def add_adapter(self, adapter_name, config=None):
    #     """add adapter"""
    #     if config is not None:
    #         model_config = self.model.config.to_dict() if hasattr(self.model.config, "to_dict") else self.model.config
    #         config = self._prepare_lora_config(config, model_config)
    #         self.peft_config[adapter_name] = config

    #     self._find_and_replace(adapter_name)

    #     if len(self.peft_config) > 1 and self.peft_config[adapter_name].bias != "none":
    #         raise ValueError(
    #             "LoraModel supports only 1 adapter with bias. When using multiple adapters, set bias to 'none' for all adapters."
    #         )
    #     # only lora trainable
    #     self._mark_only_adapters_as_trainable()
    #     if self.peft_config[adapter_name].inference_mode:
    #         # freeze adapter
    #         _freeze_adapter(self.model, adapter_name)
    def inject_adapter(self, model: nn.Module, adapter_name: str):
        r"""
        Creates adapter layers and replaces the target cells with the adapter layers. This method is called under the
        hood by `peft.mapping.get_peft_model` if a non-prompt tuning adapter class is passed, e.g. LoRA.

        The corresponding PEFT config is directly retrieved from the `peft_config` attribute of the BaseTuner class.
        Rename add_adapter -> inject_adapter.

        Args:
            model (`nn.Module`):
                The model to be tuned.
            adapter_name (`str`):
                The adapter name.
        """
        peft_config = self.peft_config[adapter_name]
        # Note: If possible, all checks should be performed *at the start of this method*.
        # This way, we can raise early if something goes wrong, without leaving the model
        # in a bad (half-initialized) state.
        self._check_new_adapter_config(peft_config)

        is_target_modules_in_base_model = False
        key_list = [key for key, _ in model.cells_and_names()]  # named_cells

        model_config = getattr(model, "config", {"model_type": "custom"})
        if hasattr(model_config, "to_dict"):
            model_config = model_config.to_dict()

        peft_config = self._prepare_adapter_config(peft_config, model_config) # pylint: disable=assignment-from-none
        for key in key_list:
            if not self._check_target_cell_exists(peft_config, key):
                continue

            is_target_modules_in_base_model = True
            parent, target, target_name = _get_subcells(model, key)

            optionnal_kwargs = {
                "loaded_in_8bit": getattr(model, "is_loaded_in_8bit", False),  
                "loaded_in_4bit": getattr(model, "is_loaded_in_4bit", False),
                "current_key": key,
            }
            # **finally create or replace target cell.**
            self._create_and_replace(peft_config, adapter_name, target, target_name, parent, current_key=key)

        if not is_target_modules_in_base_model:
            raise ValueError(
                f"Target cells {peft_config.target_modules} not found in the base model. "
                f"Please check the target cells and try again."
            )

        self._mark_only_adapters_as_trainable(model)

        if self.peft_config[adapter_name].inference_mode:
            for name, param in self.model.parameters_and_names():
                if adapter_name in name:
                    param.requires_grad = False

    def merge_adapter(self):
        """
        This method merges the LoRa layers into the base model.
        """
        for cell in self.model.cells():
            if isinstance(cell, BaseTunerLayer):
                cell.merge()

    def unmerge_adapter(self):
        """
        This method unmerges the LoRa layers from the base model.
        """
        for cell in self.model.cells():
            if isinstance(cell, BaseTunerLayer):
                cell.unmerge()


class BaseTunerLayer(ABC):
    r"""
    A tuner layer mixin that provides the common methods and attributes for all tuners.

    Args:
        is_pluggable (`bool`, *optional*):
            Whether the adapter layer can be plugged to any pytorch cell
        active_adapters (Union[List[`str`], `str`], *optional*):
            The name of the active adapter.
    """
    # All names of layers that may contain adapter (trainable) weights
    adapter_layer_names: tuple[str] = ()
    # All names of other parameters that may contain adapter-related parameters
    other_param_names: tuple[str] = ()

    # indicates whether all adapters should be disabled
    _disable_adapters: bool = False

    # the currently active adapter(s)
    _active_adapter: str | list[str] = "default"

    # List all merged adapters
    merged_adapters: list[str] = []

    def get_base_layer(self) -> nn.Module:
        """
        (Recursively) get the base_layer.

        This is necessary for the case that the tuner layer wraps another tuner layer.

        """
        base_layer = self
        while hasattr(base_layer, "base_layer"):
            base_layer = base_layer.base_layer
        return base_layer

    @property
    def weight(self) -> Tensor:
        r"""
        Returns the weight of the base layer.
        
        Args:
            self: The instance of the BaseTunerLayer class.
        
        Returns:
            A Tensor object representing the weight of the base layer.
        
        Raises:
            None.
        """
        # This is required for some transformers code, e.g. for T5, weight is accessed as:
        #     self.wo.weight
        # where "wo" is the adapter layer.
        # https://github.com/huggingface/transformers/blob/78f6ed6c70b29c1560780e3869a7ad4c6b3d2710/src/transformers
        # /models/t5/modeling_t5.py#L292
        base_layer = self.get_base_layer()
        weight = base_layer.weight
        return weight

    @property
    def bias(self) -> Tensor:
        r"""
        This method retrieves the bias tensor from the base layer.
        
        Args:
            self: An instance of the BaseTunerLayer class.
        
        Returns:
            Tensor: The bias tensor obtained from the base layer.
        
        Raises:
            This method does not raise any exceptions.
        """
        base_layer = self.get_base_layer()
        return base_layer.bias

    def merge(self, safe_merge: bool = False, adapter_names: Optional[list[str]] = None) -> None:
        r"""
        Merge the current layer with other layers.
        
        Args:
            self (BaseTunerLayer): The instance of the BaseTunerLayer class.
            safe_merge (bool): A flag indicating whether to perform a safe merge. Defaults to False.
            adapter_names (Optional[list[str]]): A list of adapter names. Defaults to None.
        
        Returns:
            None. This method does not return any value.
        
        Raises:
            NotImplementedError: If the method is called without being implemented.
        """
        raise NotImplementedError

    def unmerge(self) -> None:
        r"""
        unmerge(self)
            This method unmerges the current instance of BaseTunerLayer.
        
        Args:
            self: BaseTunerLayer - The instance of BaseTunerLayer to be unmerged.
        
        Returns:
            None: This method does not return any value.
        
        Raises:
            NotImplementedError: If the method is called, a NotImplementedError is raised as this method is not implemented.
        """
        raise NotImplementedError

    @property
    def merged(self) -> bool:
        r"""
        Returns whether the current instance of the BaseTunerLayer class has merged adapters.
        
        Args:
            self (BaseTunerLayer): The current instance of the BaseTunerLayer class.
        
        Returns:
            bool: A boolean value indicating whether the current instance has merged adapters. 
            Returns True if there are merged adapters, and False otherwise.
        
        Raises:
            None.
        
        """
        return bool(self.merged_adapters)

    @property
    def disable_adapters(self) -> bool:
        r"""
        Disables the adapters in the BaseTunerLayer.
        
        Args:
            self: An instance of the BaseTunerLayer class.
        
        Returns:
            bool: Returns a boolean value indicating whether the adapters were successfully disabled.
        
        Raises:
            None.
        
        This method disables the adapters in the BaseTunerLayer. Adapters are components that allow communication between different systems or modules. By disabling the adapters, the BaseTunerLayer restricts
any further communication or interaction with external systems.
        
        Note:
            The disable_adapters method does not remove or delete the adapters from the BaseTunerLayer instance. It only disables their functionality temporarily. To enable the adapters again, use the
enable_adapters method.
        
        Example:
            >>> tuner_layer = BaseTunerLayer()
            >>> tuner_layer.disable_adapters()
            True
        """
        # use a property to ensure that disable_adapters is not set directly, instead use the enable_adapters method
        return self._disable_adapters

    @property
    def active_adapter(self) -> str | list[str]:
        r"""Return the active adapter.
        
        This method is a property of the BaseTunerLayer class. It returns the active adapter, which can be either a string or a list of strings.
        
        Args:
            self: An instance of the BaseTunerLayer class.
        
        Returns:
            str | list[str]: The active adapter. If there is only one active adapter, it is returned as a string. If there are multiple active adapters, they are returned as a list of strings.
        
        Raises:
            None.
        
        """
        # use a property to ensure that active_adapter is not set directly, instead use the set_adapter method
        return self._active_adapter

    @property
    def active_adapters(self):
        r"""
        Returns a list of active adapters.
        
        Args:
            self (BaseTunerLayer): The instance of the BaseTunerLayer class.
        
        Returns:
            list: A list of active adapters. If the active_adapter attribute is a string, it will be returned as a single-item list. Otherwise, the active_adapter attribute will be returned as is.
        
        Raises:
            None.
        """
        if isinstance(self.active_adapter, str):
            return [self.active_adapter]
        # is already a list of str
        return self.active_adapter

    def enable_adapters(self, enabled: bool) -> None:
        """Toggle the enabling and disabling of adapters

        Takes care of setting the requires_grad flag for the adapter weights.

        Args:
            enabled (bool): True to enable adapters, False to disable adapters
        """
        if enabled:
            self.set_adapter(self.active_adapters)
            self._disable_adapters = False
        else:
            # disable grads on all adapter layers
            for layer_name in self.adapter_layer_names:
                layer = getattr(self, layer_name)
                layer.set_grad(requires_grad=False)
            self._disable_adapters = True

    def set_adapter(self, adapter_names: str | list[str]) -> None:
        """Set the active adapter(s).

        Additionally, this function will set the specified adapters to trainable (i.e., requires_grad=True). If this is
        not desired, use the following code.

        ```py
        >>> for name, param in model_peft.named_parameters():
        ...     if ...:  # some check on name (ex. if 'lora' in name)
        ...         param.requires_grad = False
        ```

        Args:
            adapter_name (`str` or `List[str]`): Name of the adapter(s) to be activated.
        """
        if isinstance(adapter_names, str):
            adapter_names = [adapter_names]

        # Deactivate grads on the inactive adapter and activate grads on the active adapter
        for layer_name in self.adapter_layer_names:
            cell_dict = getattr(self, layer_name)
            for key, layer in cell_dict.items():
                if key in adapter_names:
                    # Note: It is possible that not a single layer is called with requires_grad_(True) here. This may
                    # happen if a completely different adapter layer is being activated.
                    layer.requires_grad = True
                else:
                    layer.requires_grad = False

        self._active_adapter = adapter_names

    def _all_available_adapter_names(self) -> list[str]:
        """Return a sorted list of all available adapter names"""
        adapter_names = set()
        for name in self.adapter_layer_names + self.other_param_names:
            # we check each possible attribute and if it's a dict or ModuleDict, we assume that the keys are the adapter
            # names
            attr = getattr(self, name)
            if hasattr(attr, "keys"):
                adapter_names.update(attr.keys())
        return sorted(adapter_names)

    def delete_adapter(self, adapter_name: str) -> None:
        """
        Delete an adapter from the layer

        This should be called on all adapter layers, or else we will get an inconsistent state.

        This method will also set a new active adapter if the deleted adapter was an active adapter. It is important
        that the new adapter is chosen in a deterministic way, so that the same adapter is chosen on all layers.

        Args:
            adapter_name (`str`): The name of the adapter to delete

        """
        for attr in self.adapter_layer_names + self.other_param_names:
            if adapter_name in getattr(self, attr):
                del getattr(self, attr)[adapter_name]

        if adapter_name in self.active_adapters:
            # choose a new active adapter
            active_adapters = self.active_adapters[:]
            active_adapters.remove(adapter_name)
            if active_adapters:
                self.set_adapter(active_adapters)
            else:
                # no active adapters left, set a new default adapter
                # here we get the list of all adapters existing adapter names and choose the first one
                remaining_adapters = self._all_available_adapter_names()
                if not remaining_adapters:
                    self.set_adapter([])
                else:
                    new_active_adapter = remaining_adapters[0]
                    warnings.warn(
                        f"Adapter {adapter_name} was active which is now deleted. Setting active adapter to "
                        f"{new_active_adapter}."
                    )
                    self.set_adapter(remaining_adapters[0])

def check_adapters_to_merge(cell: BaseTunerLayer, adapter_names: Optional[list[str]] = None) -> list[str]:
    """
    Helper function to check which adapters should be merged.

    Only return those adapters that are not already merged. Give a warning if some or all of the adapters are already
    merged.

    """
    if adapter_names is None:
        adapter_names = cell.active_adapters

    if cell.merged:
        merged_adapters = set(cell.merged_adapters)
        adapter_names = [name for name in adapter_names if name not in merged_adapters]

        if adapter_names:
            warnings.warn(
                f"Already following adapters were merged {','.join(cell.merged_adapters)}. "
                f"You are now additionally merging {','.join(adapter_names)}."
            )
        else:
            warnings.warn("All adapters are already merged, nothing to do.")

    return adapter_names

def check_target_cell_exists(config, key: str) -> bool | re.Match[str] | None:
    """A helper method to check if the passed cell's key name matches any of the target cells in the adapter_config.

    Args:
        config (`LoraConfig` | `LycorisConfig`): A config to match target cells from
        key (`str`): A key to search any matches in config

    Returns:
        `bool` | `re.Match[str]` | `None`: True of match object if key matches any target cells from config, False or
        None if no match found
    """
    if isinstance(config.target_modules, str):
        target_cell_found = re.fullmatch(config.target_modules, key)
    elif key in config.target_modules:
        # this cell is specified directly in target_modules
        target_cell_found = True
    else:
        target_cell_found = any(key.endswith(f".{target_key}") for target_key in config.target_modules)

        layer_indexes = getattr(config, "layers_to_transform", None)
        layers_pattern = getattr(config, "layers_pattern", None)

        is_using_layer_indexes = layer_indexes is not None and (
            len(layer_indexes) != 0 if isinstance(layer_indexes, list) else True
        )
        if is_using_layer_indexes and target_cell_found:
            layer_index = None
            if layers_pattern is None or len(layers_pattern) == 0:
                layer_index = re.match(r".*\.[^.]*\.(\d+)\.", key)
            else:
                layers_pattern = [layers_pattern] if isinstance(layers_pattern, str) else layers_pattern
                for pattern in layers_pattern:
                    layer_index = re.match(rf".*\.{pattern}\.(\d+)\.", key)
                    if layer_index is not None:
                        break

            if layer_index is None:
                target_cell_found = False
            else:
                layer_index = int(layer_index.group(1))
                if isinstance(layer_indexes, int):
                    target_cell_found = layer_index == layer_indexes
                else:
                    target_cell_found = layer_index in layer_indexes

    return target_cell_found


def clone_cell(cell: nn.Module, share_weights=False):
    """Clone a cell in a pytorch model.

    Clones a cell of a model, optionally sharing all the parameters between the original and the clone. Simplifies
    reusing a cell when manipulating the architecture of a model.
    """
    clone = copy.deepcopy(cell)

    def _share_weights(src: nn.Module, dst: nn.Module):
        for name, param in src.parameters_and_names(expand=False):
            setattr(dst, name, param)

    if share_weights:
        for name, subcell in cell.parameters_and_names():
            _share_weights(subcell, clone.get_subcell(name))

    return clone

def replicate_layers(model: nn.Module, layer_map: list[tuple[int, int]]):
    """Replicate layers in a transfomer model with weight sharing.

    This function looks for a cell list attribute at model[(.model)*].layers and replicates the layers in the cell
    list according to the layer map. For example the map `[[0, 4], [2, 5]]` will take the set of layers `[0, 1, 2, 3,
    4]` and replace them with a cell list containing `[0, 1, 2, 3, 2, 3, 4]`.
    """
    while hasattr(model, "model"):
        model = model.model
    # Some variants of the bert model nest the main model under the bert attribute.
    if hasattr(model, "bert"):
        model = model.bert

    model_type = None
    layers: nn.ModuleList = None
    if hasattr(model, "layers"):
        model_type = "llama"
        layers = model.layers
    elif hasattr(model, "encoder") and hasattr(model.encoder, "layer"):
        model_type = "bert"
        layers = model.encoder.layer
    elif hasattr(model, "h"):
        model_type = "falcon"
        layers = model.h
    if not model_type or not isinstance(layers, nn.ModuleList):
        raise ValueError(
            "Could not locate the layers attribute in the model. "
            "Expected Llama, Bert or Falcon compatible architectures."
        )

    new_layers = []
    for start, end in layer_map:
        for i in range(start, end):
            current_idx = len(new_layers)
            new_layers.append(clone_cell(layers[i], share_weights=True))
            # This is a hack needed to work around the layer_idx introduced in HF transformers.
            for subcell in new_layers[-1].cells():
                if hasattr(subcell, "layer_idx"):
                    subcell.layer_idx = current_idx
    layers = nn.ModuleList(new_layers)
    if model_type == "llama":
        model.layers = layers
    elif model_type == "bert":
        model.encoder.layer = layers
    elif model_type == "falcon":
        model.h = layers
    else:
        raise ValueError("Unexpected model type, need to handle post-processing of layers.")
    if hasattr(model.config, "num_hidden_layers"):  # Common to Llama, Bert, Falcon.
        model.config.num_hidden_layers = len(new_layers)
