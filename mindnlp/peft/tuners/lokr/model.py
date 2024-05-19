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

from mindspore import nn

from ...utils import (
    ModulesToSaveWrapper,
    _get_submodules,
)

from ..tuners_utils import (
    BaseTuner,
    BaseTunerLayer,
    check_target_module_exists,
)
from .layer import Conv2d, Dense, LoKrLayer
from .config import LoKrConfig


class LoKrModel(BaseTuner):
    """
    Creates Low-Rank Kronecker Product model from a pretrained model. The original method is partially described in
    https://arxiv.org/abs/2108.06098 and in https://arxiv.org/abs/2309.14859 Current implementation heavily borrows
    from
    https://github.com/KohakuBlueleaf/LyCORIS/blob/eb460098187f752a5d66406d3affade6f0a07ece/lycoris/modules/lokr.py

    Args:
        model (`mindspore.nn.Cell`): The model to which the adapter tuner layers will be attached.
        peft_config ([`LoKrConfig`]): The configuration of the LoKr model.
        adapter_name (`str`): The name of the adapter, defaults to `"default"`.

    Returns:
        LoKrModel ([`mindspore.nn.Cell`]): The LoKr model.

    Example:
        ```py
        >>> from diffusers import StableDiffusionPipeline
        >>> from peft import LoKrModel, LoKrConfig

        >>> config_te = LoKrConfig(
        ...     r=8,
        ...     lora_alpha=32,
        ...     target_modules=["k_proj", "q_proj", "v_proj", "out_proj", "fc1", "fc2"],
        ...     rank_dropout=0.0,
        ...     module_dropout=0.0,
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
        ...     module_dropout=0.0,
        ...     init_weights=True,
        ...     use_effective_conv2d=True,
        ... )

        >>> model = StableDiffusionPipeline.from_pretrained("runwayml/stable-diffusion-v1-5")
        >>> model.text_encoder = LoKrModel(model.text_encoder, config_te, "default")
        >>> model.unet = LoKrModel(model.unet, config_unet, "default")
        ```

    > **Attributes**:  

    >   - **model** ([`~nn.Cell`])— The model to be adapted. 

    >   - **peft_config** ([`LoKrConfig`]): The configuration of the LoKr  model. 

    """

    prefix: str = "lokr_"
    layers_mapping: Dict[Type[nn.Cell], Type[LoKrLayer]] = {
        nn.Conv2d: Conv2d,
        nn.Dense: Dense,
    }

    def _create_and_replace(
        self,
        config: LoKrConfig,
        adapter_name: str,
        target: Union[LoKrLayer, nn.Cell],
        target_name: str,
        parent: nn.Cell,
        current_key: str,
        loaded_in_8bit: Optional[bool] = False,
        loaded_in_4bit: Optional[bool] = False,
    ) -> None:
        """
        A private method to create and replace the target module with the adapter module.
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
            new_module = self._create_new_module(config, adapter_name, target, **kwargs)
            self._replace_module(parent, target_name, new_module, target)

    @classmethod
    def _create_new_module(
        cls, config: LoKrConfig, adapter_name: str, target: nn.Cell, **kwargs
    ) -> LoKrLayer:
        # Find corresponding subtype of provided target module
        new_module_cls = None
        for subtype, target_cls in cls.layers_mapping.items():
            if (
                hasattr(target, "base_layer")
                and isinstance(target.get_base_layer(), subtype)
                and isinstance(target, BaseTunerLayer)
            ):
                # nested tuner layers are allowed
                new_module_cls = target_cls
                break
            elif isinstance(target, subtype):
                new_module_cls = target_cls
                break

        # We didn't find corresponding type, so adapter for this layer is not supported
        if new_module_cls is None:
            supported_modules = ", ".join(
                layer.__name__ for layer in cls.layers_mapping.keys()
            )
            raise ValueError(
                f"Target module of type {type(target)} not supported, "
                f"currently only adapters for {supported_modules} are supported"
            )

        if isinstance(target, BaseTunerLayer):
            target_base_layer = target.get_base_layer()
        else:
            target_base_layer = target

        if isinstance(target_base_layer, nn.Cell):
            new_module = new_module_cls(target, adapter_name=adapter_name, **kwargs)
        elif isinstance(target_base_layer, nn.Cell):
            new_module = new_module_cls(target, adapter_name=adapter_name, **kwargs)
        else:
            supported_modules = ", ".join(
                layer.__name__ for layer in cls.layers_mapping.keys()
            )
            raise ValueError(
                f"Target module of type {type(target)} not supported, "
                f"currently only adapters for {supported_modules} are supported"
            )

        return new_module

    def __getattr__(self, name: str):
        """Forward missing attributes to the wrapped module."""
        try:
            return super().__getattr__(name)  # defer to nn.Module's logic
        except AttributeError:
            return getattr(self.model, name)

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

    def _mark_only_adapters_as_trainable(self, model: nn.Cell) -> None:
        for n, p in model.parameters_and_names():
            if self.prefix not in n:
                p.requires_grad = False

    def _set_adapter_layers(self, enabled=True):
        for module in self.model.modules():
            if isinstance(module, (BaseTunerLayer, ModulesToSaveWrapper)):
                module.enable_adapters(enabled)

    def _unload_and_optionally_merge(
        self,
        merge: bool = True,
        progressbar: bool = False,
        safe_merge: bool = False,
        adapter_names: Optional[List[str]] = None,
    ):
        if merge:
            if getattr(self.model, "quantization_method", None) == "gptq":
                raise ValueError(
                    "Cannot merge LOHA layers when the model is gptq quantized"
                )

        self._unloading_checks(adapter_names)
        key_list = [
            key for key, _ in self.model.named_modules() if self.prefix not in key
        ]
        desc = "Unloading " + ("and merging " if merge else "") + "model"
        for key in tqdm(key_list, disable=not progressbar, desc=desc):
            try:
                parent, target, target_name = _get_submodules(self.model, key)
            except AttributeError:
                continue

            if hasattr(target, "base_layer"):
                if merge:
                    target.merge(safe_merge=safe_merge, adapter_names=adapter_names)
                self._replace_module(
                    parent, target_name, target.get_base_layer(), target
                )
            elif isinstance(target, ModulesToSaveWrapper):
                # save any additional trainable modules part of `modules_to_save`
                new_module = target.modules_to_save[target.active_adapter]
                if hasattr(new_module, "base_layer"):
                    # check if the module is itself a tuner layer
                    if merge:
                        new_module.merge(
                            safe_merge=safe_merge, adapter_names=adapter_names
                        )
                    new_module = new_module.get_base_layer()
                setattr(parent, target_name, new_module)

        return self.model

    def _unloading_checks(self, adapter_names: Optional[List[str]]):
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
        if peft_config.target_modules is None:
            raise ValueError("Please specify `target_modules` in `peft_config`")
        return peft_config

    @staticmethod
    def _check_target_module_exists(LoKR_config, key):
        return check_target_module_exists(LoKR_config, key)
