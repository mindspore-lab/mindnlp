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
"""lora model"""
from __future__ import annotations

import math
import operator
import re
import warnings
from contextlib import contextmanager
from dataclasses import asdict, replace
from enum import Enum
from functools import partial, reduce
from itertools import chain
from typing import Optional
try:
    from typing import Literal
except:
    from typing_extensions import Literal

from tqdm import tqdm
import mindspore

from mindnlp.core import nn, ops
from ..tuners_utils import (
    BaseTuner,
    BaseTunerLayer,
    check_target_cell_exists,
    # onload_layer,
    replicate_layers,
)
from ...utils import (
    TRANSFORMERS_MODELS_TO_LORA_TARGET_MODULES_MAPPING,
    ModulesToSaveWrapper,
    _freeze_adapter,
    _get_subcells,
)
from ...utils.merge_utils import dare_linear, dare_ties, magnitude_prune, task_arithmetic, ties

from .config import LoraConfig
from .layer import Conv2d, LoraLayer, dispatch_default


def _adapter_names_pre_forward_hook(target, args, kwargs, adapter_names):
    """
    Args:
        target (object): The target object to which the hook is applied.
        args (tuple): The positional arguments passed to the function.
        kwargs (dict): The keyword arguments passed to the function.
        adapter_names (list): The list of adapter names.
    
    Returns:
        None: This function does not return any value.
    
    Raises:
        None
    """
    # pre-forward hook to inject the adapter_names argument when using mixed adapter batches inference
    kwargs["adapter_names"] = adapter_names
    return args, kwargs


class LoraModel(BaseTuner):
    """
    Creates Low Rank Adapter (LoRA) model from a pretrained transformers model.

    The method is described in detail in https://arxiv.org/abs/2106.09685.

    Args:
        model ([`nn.Module`]): The model to be adapted.
        config ([`LoraConfig`]): The configuration of the Lora model.
        adapter_name (`str`): The name of the adapter, defaults to `"default"`.

    Returns:
        LoraModel ([`mindspore.nn.Module`]): The Lora model.

    Example:

        ```py
        >>> from transformers import AutoModelForSeq2SeqLM
        >>> from peft import LoraModel, LoraConfig

        >>> config = LoraConfig(
        ...     task_type="SEQ_2_SEQ_LM",
        ...     r=8,
        ...     lora_alpha=32,
        ...     target_cells=["q", "v"],
        ...     lora_dropout=0.01,
        ... )

        >>> model = AutoModelForSeq2SeqLM.from_pretrained("t5-base")
        >>> lora_model = LoraModel(model, config, "default")
        ```

        ```py
        >>> import torch
        >>> import transformers
        >>> from peft import LoraConfig, PeftModel, get_peft_model, prepare_model_for_kbit_training

        >>> rank = ...
        >>> target_cells = ["q_proj", "k_proj", "v_proj", "out_proj", "fc_in", "fc_out", "wte"]
        >>> config = LoraConfig(
        ...     r=4, lora_alpha=16, target_cells=target_cells, lora_dropout=0.1, bias="none", task_type="CAUSAL_LM"
        ... )
        >>> quantization_config = transformers.BitsAndBytesConfig(load_in_8bit=True)

        >>> tokenizer = transformers.AutoTokenizer.from_pretrained(
        ...     "kakaobrain/kogpt",
        ...     revision="KoGPT6B-ryan1.5b-float16",  # or float32 version: revision=KoGPT6B-ryan1.5b
        ...     bos_token="[BOS]",
        ...     eos_token="[EOS]",
        ...     unk_token="[UNK]",
        ...     pad_token="[PAD]",
        ...     mask_token="[MASK]",
        ... )
        >>> model = transformers.GPTJForCausalLM.from_pretrained(
        ...     "kakaobrain/kogpt",
        ...     revision="KoGPT6B-ryan1.5b-float16",  # or float32 version: revision=KoGPT6B-ryan1.5b
        ...     pad_token_id=tokenizer.eos_token_id,
        ...     use_cache=False,
        ...     device_map={"": rank},
        ...     ms_dtype=torch.float16,
        ...     quantization_config=quantization_config,
        ... )
        >>> model = prepare_model_for_kbit_training(model)
        >>> lora_model = get_peft_model(model, config)
        ```

    > **Attributes**:  

    >   - **model** ([`transformers.PreTrainedModel`])— The model to be adapted. 

    >   - **peft_config** ([`LoraConfig`]): The configuration of the Lora model.
    """
    prefix: str = "lora_"

    def _check_new_adapter_config(self, config: LoraConfig) -> None:
        """
        A helper method to check the config when a new adapter is being added.

        Raise a ValueError if there is something wrong with the config or if it conflicts with existing adapters.

        """
        # TODO: there should be a check if any of the existing adapters actually has bias != "none", or else the check
        # does not fully correspond to the error message.
        if (len(self.peft_config) > 1) and (config.bias != "none"):
            raise ValueError(
                f"{self.__class__.__name__} supports only 1 adapter with bias. When using multiple adapters, "
                "set bias to 'none' for all adapters."
            )

    @staticmethod
    def _check_target_cell_exists(lora_config, key):
        r"""
        Checks if the target cell exists in the LoRa configuration.
        
        Args:
            lora_config (dict): A dictionary containing the LoRa configuration.
                This dictionary should have the following structure:
                {
                    "target_cells": {
                        "cell1": {
                            ...
                        },
                        "cell2": {
                            ...
                        },
                        ...
                    },
                    ...
                }
                The 'target_cells' key should contain the target cell information.
            key (str): The key to identify the target cell.
                The key should be a string that matches the key used in the 'target_cells' dictionary.
        
        Returns:
            None: This method does not return any value.
        
        Raises:
            None: This method does not raise any exceptions.
        """
        return check_target_cell_exists(lora_config, key)

    def _prepare_model(self, peft_config: LoraConfig, model: nn.Module):
        r"""
        A private method to modify the model structure before adapter is applied.

        Args:
            peft_config (`PeftConfig`):
                The prepared adapter config.
            model (`nn.Module`):
                The model that is going to be adapted.
        """
        if peft_config.layer_replication:
            replicate_layers(model, peft_config.layer_replication)

    def _create_and_replace(
        self,
        lora_config,
        adapter_name,
        target,
        target_name,
        parent,
        current_key,
    ):
        r"""
        Creates a new cell and replaces an existing cell in the LoraModel.
        
        Args:
            self (LoraModel): The instance of the LoraModel class.
            lora_config (LoraConfig): The LoraConfig object containing Lora configuration parameters.
            adapter_name (str): The name of the adapter.
            target (LoraLayer): The target LoraLayer or AdaLoraLayer object to update or replace.
            target_name (str): The name of the target layer.
            parent (nn.Module): The parent module to which the target layer belongs.
            current_key: The current key used for matching patterns.
        
        Returns:
            None. The method modifies the LoraModel by creating and replacing cells.
        
        Raises:
            ValueError: If the current_key is None.
        
        Note:
            This method dynamically determines the appropriate rank (r) and alpha (lora_alpha) values
            based on the current_key and the pattern keys defined in the lora_config. It then creates
            a new cell with the specified lora configuration parameters and replaces the existing
            cell with the new cell in the LoraModel.
        
            If the target is an instance of LoraLayer (but not AdaLoraLayer), the method updates
            the layer with the specified adapter_name, rank (r), lora_alpha, lora_dropout,
            init_lora_weights, use_rslora, and use_dora parameters.
        
            If the target is not an instance of LoraLayer, the method creates a new cell using the
            _create_new_cell method with the specified lora configuration parameters. If the adapter_name
            is not in the active_adapters list, the requires_grad attribute of the new cell is set to False.
        
            The method then replaces the existing cell in the parent module with the new cell using
            the _replace_cell method.
        """
        if current_key is None:
            raise ValueError("Current Key shouldn't be `None`")

        # Regexp matching - Find key which matches current target_name in patterns provided
        pattern_keys = list(chain(lora_config.rank_pattern.keys(), lora_config.alpha_pattern.keys()))
        target_name_key = next(filter(lambda key: re.match(rf".*\.{key}$", current_key), pattern_keys), current_key)
        r = lora_config.rank_pattern.get(target_name_key, lora_config.r)
        alpha = lora_config.alpha_pattern.get(target_name_key, lora_config.lora_alpha)

        kwargs = {
            "r": r,
            "lora_alpha": alpha,
            "lora_dropout": lora_config.lora_dropout,
            "fan_in_fan_out": lora_config.fan_in_fan_out,
            "init_lora_weights": lora_config.init_lora_weights,
            "use_rslora": lora_config.use_rslora,
            "use_dora": lora_config.use_dora,
            "loaded_in_8bit": getattr(self.model, "is_loaded_in_8bit", False),
            "loaded_in_4bit": getattr(self.model, "is_loaded_in_4bit", False),
        }

        # quant_methods = ["gptq", "aqlm", "awq"]
        # for quant_method in quant_methods:
        #     quantization_config = get_quantization_config(self.model, method=quant_method)
        #     if quantization_config is not None:
        #         kwargs[f"{quant_method}_quantization_config"] = quantization_config

        # note: AdaLoraLayer is a subclass of LoraLayer, we need to exclude it
        from ..adalora import AdaLoraLayer

        if isinstance(target, LoraLayer) and not isinstance(target, AdaLoraLayer):
            target.update_layer(
                adapter_name,
                r,
                lora_alpha=alpha,
                lora_dropout=lora_config.lora_dropout,
                init_lora_weights=lora_config.init_lora_weights,
                use_rslora=lora_config.use_rslora,
                use_dora=lora_config.use_dora,
            )
        else:
            new_cell = self._create_new_cell(lora_config, adapter_name, target, **kwargs)
            if adapter_name not in self.active_adapters:
                # adding an additional adapter: it is not automatically trainable
                new_cell.requires_grad = False
            self._replace_cell(parent, target_name, new_cell, target)

    def _replace_cell(self, parent, child_name, new_cell, child):
        r"""
        This method replaces a cell within the LoraModel by updating the specified child of the parent with a new cell.
        
        Args:
            self (object): The instance of the LoraModel class.
            parent (object): The parent object where the cell replacement will occur.
            child_name (str): The name of the child attribute within the parent object.
            new_cell (object): The new cell object that will replace the existing child within the parent.
            child (object): The existing child object that will be replaced by the new_cell.
        
        Returns:
            None. This method does not return any value.
        
        Raises:
            No specific exceptions are raised within this method.
        """
        setattr(parent, child_name, new_cell)
        # It's not necessary to set requires_grad here, as that is handled by
        # _mark_only_adapters_as_trainable

        # child layer wraps the original cell, unpack it
        if hasattr(child, "base_layer"):
            child = child.base_layer

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
        Marks only specific adapters in the model as trainable based on the specified bias configuration.
        
        Args:
            self (LoraModel): The instance of the LoraModel class.
            model (nn.Module): The neural network model on which to apply the trainable markings.
        
        Returns:
            None. This method does not return any value.
        
        Raises:
            NotImplementedError: If the requested bias configuration is not implemented.
        """
        for n, p in model.parameters_and_names():
            if self.prefix not in n:
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
                    if isinstance(m, LoraLayer) and hasattr(m, "bias") and m.bias is not None:
                        m.bias.requires_grad = True
            else:
                raise NotImplementedError(f"Requested bias: {bias}, is not implemented.")

    @staticmethod
    def _create_new_cell(lora_config, adapter_name, target, **kwargs):
        r"""
        Method to create a new cell based on the provided parameters.
        
        Args:
            lora_config (dict): The configuration parameters for the Lora model.
            adapter_name (str): The name of the adapter to be used.
            target (torch.nn.Module): The target cell for which a new cell needs to be created.
        
        Returns:
            None. Returns the newly created cell based on the specified target.
        
        Raises:
            ValueError: If the target cell is not supported. Currently supported cells include `torch.nn.Linear`, `torch.nn.Embedding`, `torch.nn.Conv2d`, and `transformers.pytorch_utils.Conv1D`.
        """
        # Collect dispatcher functions to decide what backend to use for the replaced LoRA layer. The order matters,
        # because the first match is always used. Therefore, the default layers should be checked last.
        dispatchers = [dispatch_default]

        new_cell = None
        for dispatcher in dispatchers:
            new_cell = dispatcher(target, adapter_name, lora_config=lora_config, **kwargs)
            if new_cell is not None:  # first match wins
                break

        if new_cell is None:
            # no cell could be matched
            raise ValueError(
                f"Target cell {target} is not supported. Currently, only the following cells are supported: "
                "`torch.nn.Linear`, `torch.nn.Embedding`, `torch.nn.Conv2d`, `transformers.pytorch_utils.Conv1D`."
            )

        return new_cell

    def __getattr__(self, name: str):
        """Forward missing attributes to the wrapped cell."""
        try:
            return super().__getattr__(name)  # defer to nn.Module's logic
        except AttributeError:
            return getattr(self.model, name)

    def get_peft_config_as_dict(self, inference: bool = False):
        r"""
        Returns a dictionary representation of the PEFT config.
        
        Args:
            self: An instance of the LoraModel class.
            inference (bool): A flag indicating whether the method is called for inference. Default is False.
        
        Returns:
            dict: A dictionary containing the PEFT config. The keys represent the configuration options, and the values
                  represent their corresponding values. If 'inference' is True, the dictionary will also include the
                  'inference_mode' key set to True.
        
        Raises:
            None.
        
        Note:
            - The method uses the 'peft_config' attribute of the LoraModel instance to create the dictionary.
            - If a value in the 'peft_config' attribute is an instance of Enum, its value will be extracted using the
              'value' attribute.
            - The 'config_dict' dictionary will only contain one key-value pair. If the 'inference' flag is True, the
              'config_dict' will be updated to include the 'inference_mode' key.
        
        Example usage:
            model = LoraModel()
            config = model.get_peft_config_as_dict(inference=True)
            print(config)  # {'inference_mode': True}
        
            config = model.get_peft_config_as_dict()
            print(config)  # {}
        
        """
        config_dict = {}
        for key, value in self.peft_config.items():
            config = {k: v.value if isinstance(v, Enum) else v for k, v in asdict(value).items()}
            if inference:
                config["inference_mode"] = True
        config_dict[key] = config # pylint: disable=undefined-loop-variable
        return config

    def _set_adapter_layers(self, enabled: bool = True) -> None:
        r"""
        Sets the adapter layers for the LoraModel.
        
        Args:
            self (LoraModel): The instance of the LoraModel class.
            enabled (bool, optional): A flag to enable or disable the adapter layers. Defaults to True.
        
        Returns:
            None. This method does not return any value.
        
        Raises:
            None.
        """
        for cell in self.model.cells():
            if isinstance(cell, (BaseTunerLayer, ModulesToSaveWrapper)):
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
        for active_adapter in self.active_adapters:
            val = self.peft_config[active_adapter].bias
            if val != "none":
                msg = (
                    f"Careful, disabling adapter layers with bias configured to be '{val}' does not produce the same "
                    "output as the the base model would without adaption."
                )
                warnings.warn(msg)
        self._set_adapter_layers(enabled=False)

    def set_adapter(self, adapter_name: str | list[str]) -> None:
        """Set the active adapter(s).

        Additionally, this function will set the specified adapters to trainable (i.e., requires_grad=True). If this is
        not desired, use the following code.

        ```py
        >>> for name, param in model_peft.parameters_and_names():
        ...     if ...:  # some check on name (ex. if 'lora' in name)
        ...         param.requires_grad = False
        ```

        Args:
            adapter_name (`str` or `list[str]`): Name of the adapter(s) to be activated.
        """
        for cell in self.model.cells():
            if isinstance(cell, LoraLayer):
                if cell.merged:
                    warnings.warn("Adapter cannot be set when the model is merged. Unmerging the model first.")
                    cell.unmerge()
                cell.set_adapter(adapter_name)
        self.active_adapter = adapter_name

    @contextmanager
    def _enable_peft_forward_hooks(self, *args, **kwargs):
        r"""
        Enable PEFT forward hooks for the LoraModel class.
        
        Args:
            self (LoraModel): The instance of the LoraModel class.
            
        Returns:
            None. This method is intended to be used as a context manager and does not explicitly return a value.
        
        Raises:
            ValueError: If the 'adapter_names' parameter is provided while the model is in training mode.
        """
        # If adapter_names is passed as an argument, we inject it into the forward arguments.
        adapter_names = kwargs.pop("adapter_names", None)
        if adapter_names is None:
            # nothing to do
            yield
            return

        if self.training:
            raise ValueError("Cannot pass `adapter_names` when the model is in training mode.")

        hook_handles = []
        for cell in self.cells():
            if isinstance(cell, LoraLayer):
                pre_forward = partial(_adapter_names_pre_forward_hook, adapter_names=adapter_names)
                handle = cell.register_forward_pre_hook(pre_forward, with_kwargs=True)
                hook_handles.append(handle)

        yield

        for handle in hook_handles:
            handle.remove()

    def _check_merge_allowed(self):
        """Verify that the configuration supports merging.

        Currently gptq quantization and replicated layers do not support merging.
        """
        if getattr(self.model, "quantization_method", None) == "gptq":
            raise ValueError("Cannot merge LORA layers when the model is gptq quantized")
        if self.peft_config.get("layer_replication"):
            raise ValueError("Cannot merge LORA layers when base model layers are replicated")

    @staticmethod
    def _prepare_adapter_config(peft_config, model_config):
        r"""
        Prepare the adapter configuration for a LoraModel.
        
        This method takes two parameters, peft_config and model_config, and returns None.
        
        Args:
            peft_config (PeftConfig): The configuration for the adapter.
                - target_cells (set): The target cells for the adapter. If not specified, it will be determined based on the model type.
            model_config (dict): The configuration for the model.
                - model_type (str): The type of the model.
        
        Returns:
            None. The method does not return any value.
        
        Raises:
            ValueError: If the target_cells is not specified in peft_config and the model_type is not found in the TRANSFORMERS_MODELS_TO_LORA_TARGET_MODULES_MAPPING.
        
        """
        if peft_config.target_cells is None:
            if model_config["model_type"] not in TRANSFORMERS_MODELS_TO_LORA_TARGET_MODULES_MAPPING:
                raise ValueError("Please specify `target_cells` in `peft_config`")
            peft_config.target_cells = set(
                TRANSFORMERS_MODELS_TO_LORA_TARGET_MODULES_MAPPING[model_config["model_type"]]
            )
        return peft_config

    def _unload_and_optionally_merge(
        self,
        merge=True,
        progressbar: bool = False,
        safe_merge: bool = False,
        adapter_names: Optional[list[str]] = None,
    ):
        r"""
        Method to unload and optionally merge a LoraModel.
        
        Args:
        - self: The instance of the LoraModel class.
        - merge (bool): Flag indicating whether to perform a merge operation.
        - progressbar (bool): Flag indicating whether to display a progress bar during unloading.
        - safe_merge (bool): Flag indicating whether to perform a safe merge operation.
        - adapter_names (Optional[list[str]]): List of names of adapters to consider during unloading.
        
        Returns:
        None. The method modifies the model in place.
        
        Raises:
        - AttributeError: If an attribute error occurs during the unloading process.
        """
        if merge:
            self._check_merge_allowed()

        key_list = [key for key, _ in self.model.cells_and_names() if self.prefix not in key]
        desc = "Unloading " + ("and merging " if merge else "") + "model"
        for key in tqdm(key_list, disable=not progressbar, desc=desc):
            try:
                parent, target, target_name = _get_subcells(self.model, key)
            except AttributeError:
                continue
            # with onload_layer(target):
            #     if hasattr(target, "base_layer"):
            #         if merge:
            #             target.merge(safe_merge=safe_merge, adapter_names=adapter_names)
            #         self._replace_cell(parent, target_name, target.get_base_layer(), target)
            #     elif isinstance(target, ModulesToSaveWrapper):
            #         # save any additional trainable cells part of `cells_to_save`
            #         new_cell = target.cells_to_save[target.active_adapter]
            #         if hasattr(new_cell, "base_layer"):
            #             # check if the cell is itself a tuner layer
            #             if merge:
            #                 new_cell.merge(safe_merge=safe_merge, adapter_names=adapter_names)
            #             new_cell = new_cell.get_base_layer()
            #         setattr(parent, target_name, new_cell)

        return self.model

    def _check_add_weighted_adapter(
        self, adapters: list[str], combination_type: str, svd_rank: int | None
    ) -> tuple[str, int, str]:
        """
        Helper function to check if the arguments to add_weighted_adapter are valid and compatible with the underlying
        model.
        """
        for adapter in adapters:
            if adapter not in list(self.peft_config.keys()):
                raise ValueError(f"Adapter {adapter} does not exist")

        # If more than one of the adapters targets the same cell with cells_to_save, raise an error, as these
        # cells cannot be merged. First, find the ModulesToSaveWrapper instances in the model, then check if they
        # have cells for the adapters to be merged.
        cells_to_save_wrappers = [cell for cell in self.cells() if isinstance(cell, ModulesToSaveWrapper)]
        problematic_wrappers = [
            wrapper
            for wrapper in cells_to_save_wrappers
            if sum(adapter in wrapper.cells_to_save for adapter in adapters) > 1
        ]
        if problematic_wrappers:
            raise ValueError(
                "Cannot add weighted adapters if they target the same cell with cells_to_save, but found "
                f"{len(problematic_wrappers)} such instance(s)."
            )

        # if there is only one adapter, we can only use linear merging
        combination_type = "linear" if len(adapters) == 1 else combination_type

        adapters_ranks = [self.peft_config[adapter].r for adapter in adapters]
        if combination_type in ("linear", "ties", "dare_ties", "dare_linear", "magnitude_prune"):
            # all adapters ranks should be same, new rank is just this value
            if len(set(adapters_ranks)) != 1:
                raise ValueError(
                    "All adapters must have the same r value when using combination_type linear, ties, dare_ties or "
                    "dare_linear."
                )
            new_rank = adapters_ranks[0]
        elif combination_type == "cat":
            # adapters ranks may be different, new rank is sum of all ranks
            # be careful, because output adapter rank may be really big if mixing a lot of adapters
            new_rank = sum(adapters_ranks)
        elif combination_type.endswith("svd"):
            # new rank is the max of all ranks of the adapters if not provided
            new_rank = svd_rank or max(adapters_ranks)
        else:
            raise ValueError(f"Invalid combination_type: {combination_type}")

        target_cell_types = [type(self.peft_config[adapter].target_cells) for adapter in adapters]
        if not target_cell_types:
            raise ValueError(f"Found no adapter matching the names in {adapters}")
        if len(set(target_cell_types)) > 1:
            raise ValueError(
                "all adapter configs should follow the same target cells type. "
                "Combining adapters with `target_cells` type being a mix of list/set and string is not supported."
            )

        if target_cell_types[0] == str:
            new_target_cells = "|".join(f"({self.peft_config[adapter].target_cells})" for adapter in adapters)
        elif target_cell_types[0] == set:
            new_target_cells = reduce(
                operator.or_, (self.peft_config[adapter].target_cells for adapter in adapters)
            )
        else:
            raise TypeError(f"Invalid type {target_cell_types[0]} found in target_cells")

        return combination_type, new_rank, new_target_cells

    def add_weighted_adapter(
        self,
        adapters: list[str],
        weights: list[float],
        adapter_name: str,
        combination_type: str = "svd",
        svd_rank: int | None = None,
        svd_clamp: int | None = None,
        svd_full_matrices: bool = True,
        density: float | None = None,
        majority_sign_method: Literal["total", "frequency"] = "total",
    ) -> None:
        """
        This method adds a new adapter by merging the given adapters with the given weights.

        When using the `cat` combination_type you should be aware that rank of the resulting adapter will be equal to
        the sum of all adapters ranks. So it's possible that the mixed adapter may become too big and result in OOM
        errors.

        Args:
            adapters (`list`):
                List of adapter names to be merged.
            weights (`list`):
                List of weights for each adapter.
            adapter_name (`str`):
                Name of the new adapter.
            combination_type (`str`):
                The merging type can be one of [`svd`, `linear`, `cat`, `ties`, `ties_svd`, `dare_ties`, `dare_linear`,
                `dare_ties_svd`, `dare_linear_svd`, `magnitude_prune`, `magnitude_prune_svd`]. When using the `cat`
                combination_type, the rank of the resulting adapter is equal to the sum of all adapters ranks (the
                mixed adapter may be too big and result in OOM errors).
            svd_rank (`int`, *optional*):
                Rank of output adapter for svd. If None provided, will use max rank of merging adapters.
            svd_clamp (`float`, *optional*):
                A quantile threshold for clamping SVD decomposition output. If None is provided, do not perform
                clamping. Defaults to None.
            svd_full_matrices (`bool`, *optional*):
                Controls whether to compute the full or reduced SVD, and consequently, the shape of the returned
                tensors U and Vh. Defaults to True.
            density (`float`, *optional*):
                Value between 0 and 1. 0 means all values are pruned and 1 means no values are pruned. Should be used
                with [`ties`, `ties_svd`, `dare_ties`, `dare_linear`, `dare_ties_svd`, `dare_linear_svd`,
                `magnintude_prune`, `magnitude_prune_svd`]
            majority_sign_method (`str`):
                The method, should be one of ["total", "frequency"], to use to get the magnitude of the sign values.
                Should be used with [`ties`, `ties_svd`, `dare_ties`, `dare_ties_svd`]
        """
        if adapter_name in list(self.peft_config.keys()):
            return
        for adapter in adapters:
            if adapter not in list(self.peft_config.keys()):
                raise ValueError(f"Adapter {adapter} does not exist")

        combination_type, new_rank, new_target_cells = self._check_add_weighted_adapter(
            adapters=adapters,
            combination_type=combination_type,
            svd_rank=svd_rank,
        )

        self.peft_config[adapter_name] = replace(
            self.peft_config[adapters[0]],
            r=new_rank,
            lora_alpha=new_rank,
            target_cells=new_target_cells,
        )
        self.inject_adapter(self.model, adapter_name)

        # Do we really need that?
        _freeze_adapter(self.model, adapter_name)

        key_list = [key for key, _ in self.model.cells_and_names() if self.prefix not in key]
        for key in key_list:
            _, target, _ = _get_subcells(self.model, key)
            if isinstance(target, LoraLayer):
                if adapter_name in target.lora_A:
                    target_lora_A = target.lora_A[adapter_name].weight
                    target_lora_B = target.lora_B[adapter_name].weight
                elif adapter_name in target.lora_embedding_A:
                    target_lora_A = target.lora_embedding_A[adapter_name]
                    target_lora_B = target.lora_embedding_B[adapter_name]
                else:
                    continue

                target_lora_A.data = target_lora_A.data * 0.0
                target_lora_B.data = target_lora_B.data * 0.0
                if combination_type == "cat":
                    loras_A, loras_B = [], []
                    for adapter, weight in zip(adapters, weights):
                        if adapter in target.lora_A:
                            current_adapter_lora_A = target.lora_A[adapter].weight
                            current_adapter_lora_B = target.lora_B[adapter].weight
                        elif adapter in target.lora_embedding_A:
                            current_adapter_lora_A = target.lora_embedding_A[adapter]
                            current_adapter_lora_B = target.lora_embedding_B[adapter]
                        else:
                            continue
                        loras_A.append(current_adapter_lora_A.data * weight * target.scaling[adapter])
                        loras_B.append(current_adapter_lora_B.data)

                    if len(loras_A) == 0:
                        raise ValueError("No matching LoRAs found. Please raise an issue on GitHub.")
                    loras_A = ops.cat(loras_A, dim=0)
                    loras_B = ops.cat(loras_B, dim=1)
                    target_lora_A.data[: loras_A.shape[0], :] = loras_A
                    target_lora_B.data[:, : loras_B.shape[1]] = loras_B
                elif combination_type in [
                    "svd",
                    "ties_svd",
                    "dare_linear_svd",
                    "dare_ties_svd",
                    "magnitude_prune_svd",
                ]:
                    target_lora_A.data, target_lora_B.data = self._svd_generalized_task_arithmetic_weighted_adapter(
                        combination_type,
                        adapters,
                        weights,
                        new_rank,
                        target,
                        target_lora_A,
                        target_lora_B,
                        density,
                        majority_sign_method,
                        svd_clamp,
                        full_matrices=svd_full_matrices,
                    )
                elif combination_type in ["linear", "ties", "dare_linear", "dare_ties", "magnitude_prune"]:
                    target_lora_A.data, target_lora_B.data = self._generalized_task_arithmetic_weighted_adapter(
                        combination_type, adapters, weights, target, density, majority_sign_method
                    )

    def _svd_generalized_task_arithmetic_weighted_adapter(
        self,
        combination_type,
        adapters,
        weights,
        new_rank,
        target,
        target_lora_A,
        target_lora_B,
        density,
        majority_sign_method,
        clamp=None,
        full_matrices=True,
    ):
        r"""Perform a Singular Value Decomposition (SVD) with various combination types on the given parameters.
        
        Args:
            self (LoraModel): The instance of the LoraModel class.
            combination_type (str): The type of combination to perform. Valid options are:
                - 'svd': Standard SVD combination.
                - 'ties_svd': Combination with ties.
                - 'dare_linear_svd': Combination with DARE (Density-Aware Ranking Evaluation) using linear interpolation.
                - 'dare_ties_svd': Combination with DARE (Density-Aware Ranking Evaluation) using ties.
                - 'magnitude_prune_svd': Combination with magnitude pruning.
            adapters (list): A list of adapters to consider for the combination.
            weights (list): A list of weights corresponding to the adapters.
            new_rank (int): The desired new rank after the combination.
            target: The target object.
            target_lora_A: The target LoRA A object.
            target_lora_B: The target LoRA B object.
            density (float): The density parameter used in combination types 'ties_svd', 'dare_linear_svd', 'dare_ties_svd', and 'magnitude_prune_svd'.
            majority_sign_method (str): The majority sign method used in combination types 'ties_svd' and 'dare_ties_svd'. Valid options are:
                - 'positive': Majority sign is positive.
                - 'negative': Majority sign is negative.
                - 'absolute': Majority sign is absolute.
            clamp (float, optional): The clamping value. Defaults to None.
            full_matrices (bool, optional): Whether to compute full matrices in the SVD computation. Defaults to True.
        
        Returns:
            None
        
        Raises:
            ValueError: If no matching LoRAs are found.
            ValueError: If an invalid value is passed to the combination_type parameter.
        
        """
        valid_adapters = []
        valid_weights = []
        is_embedding = any(adapter in target.lora_embedding_A for adapter in adapters)
        for adapter, weight in zip(adapters, weights):
            if adapter in target.lora_A or adapter in target.lora_embedding_A:
                valid_adapters.append(adapter)
                valid_weights.append(weight * target.scaling[adapter])

        # if no valid adapter, nothing to do
        if len(valid_adapters) == 0:
            raise ValueError("No matching LoRAs found. Please raise an issue on Github.")
        delta_weight = [target.get_delta_weight(adapter) for adapter in valid_adapters]
        valid_weights = mindspore.tensor(valid_weights)
        if combination_type == "svd":
            delta_weight = task_arithmetic(delta_weight, valid_weights)
        elif combination_type == "ties_svd":
            delta_weight = ties(delta_weight, valid_weights, density, majority_sign_method)
        elif combination_type == "dare_linear_svd":
            delta_weight = dare_linear(delta_weight, valid_weights, density)
        elif combination_type == "dare_ties_svd":
            delta_weight = dare_ties(delta_weight, valid_weights, density, majority_sign_method)
        elif combination_type == "magnitude_prune_svd":
            delta_weight = magnitude_prune(delta_weight, valid_weights, density)
        else:
            raise ValueError(f"Invalid value passed to combination type: {combination_type}")

        conv2d = isinstance(target, Conv2d)
        if conv2d:
            conv2d_1x1 = target.weight.shape[2:4] == (1, 1)
            if not conv2d_1x1:
                delta_weight = delta_weight.flatten(start_dim=1)
            else:
                delta_weight = delta_weight.squeeze()
        if (hasattr(target, "fan_in_fan_out") and target.fan_in_fan_out) or is_embedding:
            delta_weight = delta_weight.T

        # based on https://github.com/kohya-ss/sd-scripts/blob/main/networks/svd_merge_lora.py#L114-L131
        U, S, Vh = ops.svd(delta_weight, full_matrices=full_matrices)
        U = U[:, :new_rank]
        S = S[:new_rank]
        U = U @ ops.diag(S)
        Vh = Vh[:new_rank, :]
        if clamp is not None:
            dist = ops.cat([U.flatten(), Vh.flatten()])
            hi_val = ops.quantile(dist, clamp)
            low_val = -hi_val
            U = U.clamp(low_val, hi_val)
            Vh = Vh.clamp(low_val, hi_val)
        if conv2d:
            U = U.reshape(target_lora_B.data.shape)
            Vh = Vh.reshape(target_lora_A.data.shape)
        return Vh, U

    def _generalized_task_arithmetic_weighted_adapter(
        self,
        combination_type,
        adapters,
        weights,
        target,
        density,
        majority_sign_method,
    ):
        r"""
        Generalized Task Arithmetic Weighted Adapter.
        
        This method performs a weighted combination of task arithmetic operations on the given adapters and their corresponding weights.
        The combination type determines the specific arithmetic operation to be applied.
        
        Args:
            self (LoraModel): The instance of the LoraModel class.
            combination_type (str): The type of combination to be performed. Valid values are:
                - 'linear': Perform a linear combination of the task tensors.
                - 'ties': Perform a combination of task tensors with tie handling.
                - 'dare_linear': Perform a linear combination of task tensors with density-aware regularization.
                - 'dare_ties': Perform a combination of task tensors with tie handling and density-aware regularization.
                - 'magnitude_prune': Perform a combination of task tensors with magnitude pruning.
            adapters (list): A list of adapter names.
            weights (list): A list of weights corresponding to the adapters.
            target (Target): The target object containing the lora_A, lora_B, lora_embedding_A, and lora_embedding_B attributes.
            density (float): The density parameter for density-aware regularization.
            majority_sign_method (str): The method to determine the sign of the majority in tie handling. Valid values are:
                - 'positive': The majority is considered positive.
                - 'negative': The majority is considered negative.
        
        Returns:
            list: A list containing the combined task tensors for lora_A and lora_B.
        
        Raises:
            ValueError: If the combination_type parameter is not one of the valid combination types.
        """
        # account weights for LoRA A and B layers.
        valid_weights = []
        lora_A_deltas = []
        lora_B_deltas = []
        for adapter, weight in zip(adapters, weights):
            if adapter in target.lora_A:
                current_adapter_lora_A = target.lora_A[adapter].weight
                current_adapter_lora_B = target.lora_B[adapter].weight
            elif adapter in target.lora_embedding_A:
                current_adapter_lora_A = target.lora_embedding_A[adapter]
                current_adapter_lora_B = target.lora_embedding_B[adapter]
            else:
                continue
            valid_weights.append(math.sqrt(weight * target.scaling[adapter]))
            lora_A_deltas.append(current_adapter_lora_A.data)
            lora_B_deltas.append(current_adapter_lora_B.data)
        valid_weights = mindspore.tensor(valid_weights)
        lora_deltas = [lora_A_deltas, lora_B_deltas]
        dtype = lora_A_deltas[0].dtype
        for i, task_tensors in enumerate(lora_deltas):
            if combination_type == "linear":
                lora_deltas[i] = task_arithmetic(task_tensors, valid_weights)
            elif combination_type == "ties":
                lora_deltas[i] = ties(task_tensors, valid_weights, density, majority_sign_method)
            elif combination_type == "dare_linear":
                lora_deltas[i] = dare_linear(task_tensors, valid_weights, density)
            elif combination_type == "dare_ties":
                lora_deltas[i] = dare_ties(task_tensors, valid_weights, density, majority_sign_method)
            elif combination_type == "magnitude_prune":
                lora_deltas[i] = magnitude_prune(task_tensors, valid_weights, density)
            else:
                raise ValueError("Invalid combination type")
        lora_deltas = [delta.to(dtype) for delta in lora_deltas]
        return lora_deltas

    def delete_adapter(self, adapter_name: str) -> None:
        """
        Deletes an existing adapter.

        Args:
            adapter_name (str): Name of the adapter to be deleted.
        """
        if adapter_name not in list(self.peft_config.keys()):
            raise ValueError(f"Adapter {adapter_name} does not exist")
        del self.peft_config[adapter_name]

        key_list = [key for key, _ in self.model.cells_and_names() if self.prefix not in key]
        new_adapter = None
        for key in key_list:
            _, target, _ = _get_subcells(self.model, key)
            if isinstance(target, LoraLayer):
                target.delete_adapter(adapter_name)
                if new_adapter is None:
                    new_adapter = target.active_adapters[:]

        self.active_adapter = new_adapter or []

    def merge_and_unload(
        self, progressbar: bool = False, safe_merge: bool = False, adapter_names: Optional[list[str]] = None
    ) -> nn.Module:
        r"""
        This method merges the LoRa layers into the base model. This is needed if someone wants to use the base model
        as a standalone model.

        Args:
            progressbar (`bool`):
                whether to show a progressbar indicating the unload and merge process
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
        return self._unload_and_optionally_merge(
            progressbar=progressbar, safe_merge=safe_merge, adapter_names=adapter_names
        )

    def unload(self) -> nn.Module:
        """
        Gets back the base model by removing all the lora cells without merging. This gives back the original base
        model.
        """
        return self._unload_and_optionally_merge(merge=False)
