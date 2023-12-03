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
# pylint: disable=C0103
# pylint: disable=R1705
# pylint: disable=R1702
# pylint: disable=W0631
# pylint: disable=W0613
# pylint: disable=E1111
# pylint: disable=W1401
# pylint: disable=W0237
"""Lora."""
import math
import re
import warnings
from dataclasses import asdict, dataclass, field
from enum import Enum
from typing import List, Optional, Union

import mindspore
from mindspore import nn, ops
from mindspore.common.initializer import initializer, HeUniform, Zero, Normal

# import mindnlp._legacy.functional as F
from mindnlp.transformers.ms_utils import Conv1D
from mindnlp.abc import CellDict

from ..config import PeftConfig
# from ..import_utils import is_bnb_4bit_available, is_bnb_available
from ..utils import (
    # CLAMP_QUANTILE,
    COMMON_LAYERS_PATTERN,
    TRANSFORMERS_MODELS_TO_LORA_TARGET_MODULES_MAPPING,
    ModulesToSaveWrapper,
    PeftType,
    # _freeze_adapter,
    _get_submodules,
    transpose,
)

from .tuners_utils import BaseTuner, BaseTunerLayer

# if is_bnb_available():
#     import bitsandbytes as bnb


@dataclass
class LoraConfig(PeftConfig):
    """
    This is the configuration class to store the configuration of a [`LoraModel`].

    Args:
        r (`int`): Lora attention dimension.
        target_modules (`Union[List[str],str]`): The names of the modules to apply Lora to.
        lora_alpha (`float`): The alpha parameter for Lora scaling.
        lora_dropout (`float`): The dropout probability for Lora layers.
        fan_in_fan_out (`bool`): Set this to True if the layer to replace stores weight like (fan_in, fan_out).
        For example, gpt-2 uses `Conv1D` which stores weights like (fan_in, fan_out) and hence this should be set to `True`.:
        bias (`str`): Bias type for Lora. Can be 'none', 'all' or 'lora_only'
        modules_to_save (`List[str]`):List of modules apart from LoRA layers to be set as trainable
            and saved in the final checkpoint.
        layers_to_transform (`Union[List[int],int]`):
            The layer indexes to transform, if this argument is specified, it will apply the LoRA transformations on
            the layer indexes that are specified in this list. If a single integer is passed, it will apply the LoRA
            transformations on the layer at this index.
        layers_pattern (`str`):
            The layer pattern name, used only if `layers_to_transform` is different from `None` and if the layer
            pattern is not in the common layers pattern.
    """

    r: int = field(default=8, metadata={"help": "Lora attention dimension"})
    target_modules: Optional[Union[List[str], str]] = field(
        default=None,
        metadata={
            "help": "List of module names or regex expression of the module names to replace with Lora."
            "For example, ['q', 'v'] or '.*decoder.*(SelfAttention|EncDecAttention).*(q|v)$' "
        },
    )
    lora_alpha: int = field(default=8, metadata={"help": "Lora alpha"})
    lora_dropout: float = field(default=0.0, metadata={"help": "Lora dropout"})
    fan_in_fan_out: bool = field(
        default=False,
        metadata={"help": "Set this to True if the layer to replace stores weight like (fan_in, fan_out)"},
    )
    bias: str = field(default="none", metadata={"help": "Bias type for Lora. Can be 'none', 'all' or 'lora_only'"})
    modules_to_save: Optional[List[str]] = field(
        default=None,
        metadata={
            "help": "List of modules apart from LoRA layers to be set as trainable and saved in the final checkpoint. "
            "For example, in Sequence Classification or Token Classification tasks, "
            "the final layer `classifier/score` are randomly initialized and as such need to be trainable and saved."
        },
    )
    init_lora_weights: bool = field(
        default=True,
        metadata={"help": "Whether to initialize the weights of the Lora layers."},
    )
    layers_to_transform: Optional[Union[List, int]] = field(
        default=None,
        metadata={
            "help": "The layer indexes to transform, is this argument is specified, \
                PEFT will transform only the layers indexes that are specified inside this list. \
                If a single integer is passed, PEFT will transform only the layer at this index."
        },
    )
    layers_pattern: Optional[str] = field(
        default=None,
        metadata={
            "help": "The layer pattern name, used only if `layers_to_transform` is different to None and \
                  if the layer pattern is not in the common layers pattern."
        },
    )

    def __post_init__(self):
        self.peft_type = PeftType.LORA

    @property
    def is_prompt_learning(self):
        r"""
        Utility method to check if the configuration is for prompt learning.
        """
        return False

class LoraModel(BaseTuner):
    """
    Creates Low Rank Adapter (Lora) model from a pretrained transformers model.

    Args:
        model ([`~mindnlp.PreTrainedModel`]): The model to be adapted.
        config ([`LoraConfig`]): The configuration of the Lora model.

    Returns:
        `mindspore.nn.Cell`: The Lora model.

    Example:

        ```py
        >>> from mindnlp.transformers import GPTForSequenceClassification
        >>> from mindnlp.modules import LoraModel, LoraConfig

        >>> config = LoraConfig(
        ...     peft_type="LORA",
        ...     task_type="SEQ_2_SEQ_LM",
        ...     r=8,
        ...     lora_alpha=32,
        ...     target_modules=["q", "v"],
        ...     lora_dropout=0.01,
        ... )

        >>> model = AutoModelForSeq2SeqLM.from_pretrained("t5-base")
        >>> lora_model = LoraModel(config, model)
        ```
    """

    def __init__(self, model: nn.Cell, config, adapter_name):
        # call BaseTuner.__init__
        # setup config and inject lora adapter
        super().__init__(model, config, adapter_name)

    @staticmethod
    def _prepare_adapter_config(peft_config, model_config):
        if peft_config.target_modules is None:
            # If target_modules is not specified, use the default target_modules for the model type
            if model_config["model_type"] not in TRANSFORMERS_MODELS_TO_LORA_TARGET_MODULES_MAPPING:
                raise ValueError("Please specify `target_modules` in `peft_config`")
            peft_config.target_modules = TRANSFORMERS_MODELS_TO_LORA_TARGET_MODULES_MAPPING[model_config["model_type"]]
        return peft_config
    def _check_new_adapter_config(self, config: LoraConfig):
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
    def _check_target_module_exists(lora_config, key):
        if isinstance(lora_config.target_modules, str):
            target_module_found = re.fullmatch(lora_config.target_modules, key)
        else:
            target_module_found = any(
                re.match(f".*\.{target_key}$", key) for target_key in lora_config.target_modules
            ) or any(target_key == key for target_key in lora_config.target_modules)
            is_using_layer_indexes = getattr(lora_config, "layers_to_transform", None) is not None
            layer_indexing_pattern = getattr(lora_config, "layers_pattern", None)

            if is_using_layer_indexes and target_module_found:
                layers_pattern = COMMON_LAYERS_PATTERN if layer_indexing_pattern is None else layer_indexing_pattern
                layers_pattern = [layers_pattern] if isinstance(layers_pattern, str) else layers_pattern

                for pattern in layers_pattern:
                    layer_index = re.match(f".*.{pattern}\.(\d+)\.*", key)
                    if layer_index is not None:  # pylint: disable=R1723
                        layer_index = int(layer_index.group(1))
                        if isinstance(lora_config.layers_to_transform, int):
                            target_module_found = layer_index == lora_config.layers_to_transform
                        else:
                            target_module_found = layer_index in lora_config.layers_to_transform

                        break
                    else:
                        target_module_found = False
        return target_module_found

    def _create_and_replace(
        self,
        lora_config,
        adapter_name,
        target,
        target_name,
        parent,
        **optionnal_kwargs,
    ):
        bias = hasattr(target, "bias") and target.bias is not None
        kwargs = {
            "r": lora_config.r,
            "lora_alpha": lora_config.lora_alpha,
            "lora_dropout": lora_config.lora_dropout,
            "fan_in_fan_out": lora_config.fan_in_fan_out,
            "init_lora_weights": lora_config.init_lora_weights,
        }

        kwargs["loaded_in_8bit"] = optionnal_kwargs.pop("loaded_in_8bit", False)
        kwargs["loaded_in_4bit"] = optionnal_kwargs.pop("loaded_in_4bit", False)
        kwargs["bias"] = bias

        # TODO: better deal with that
        # if isinstance(target, LoraLayer) and isinstance(target, torch.nn.Conv2d):
        #     target.update_layer_conv2d(
        #         adapter_name,
        #         lora_config.r,
        #         lora_config.lora_alpha,
        #         lora_config.lora_dropout,
        #         lora_config.init_lora_weights,
        #     )
        if isinstance(target, LoraLayer) and isinstance(target, mindspore.nn.Embedding):
            target.update_layer_embedding(
                adapter_name,
                lora_config.r,
                lora_config.lora_alpha,
                lora_config.lora_dropout,
                lora_config.init_lora_weights,
            )

        elif isinstance(target, LoraLayer):
            target.update_layer(
                adapter_name,
                lora_config.r,
                lora_config.lora_alpha,
                lora_config.lora_dropout,
                lora_config.init_lora_weights,
            )
        else:
            new_module = self._create_new_module(lora_config, adapter_name, target, **kwargs)
            self._replace_module(parent, target_name, new_module, target)

    @staticmethod
    def _replace_module(parent, child_name, new_module, child):
        setattr(parent, child_name, new_module)
        if isinstance(parent, nn.SequentialCell):
            parent.cell_list = list(parent._cells.values())

        new_module.weight = child.weight
        if hasattr(child, "bias"):
            if child.bias is not None:
                new_module.bias = child.bias

        if getattr(child, "state", None) is not None:
            new_module.state = child.state
            # TODO: .to(device) not support in mindspore
            # new_module.to(child.weight.device)   # error

        # TODO: dispatch to correct device
        # for name, module in new_module.parameters_and_names():
        #     if "lora_" in name:
        #         module.to(child.weight.device)   # error
        #     if "ranknum" in name:
        #         module.to(child.weight.device)   # error


    def __getattr__(self, name: str):
        """Forward missing attributes to the wrapped module."""
        try:
            return super().__getattr__(name)  # defer to nn.Module's logic
        except AttributeError:
            return getattr(self.model, name)

    def get_peft_config_as_dict(self, inference: bool = False):
        """get peft config as dict"""
        config_dict = {}
        for key, value in self.peft_config.items():
            config = {k: v.value if isinstance(v, Enum) else v for k, v in asdict(value).items()}
            if inference:
                config["inference_mode"] = True
        config_dict[key] = config
        return config

    def _set_adapter_layers(self, enabled=True):
        for module in self.model.cells():
            module.disable_adapters = not isinstance(module, LoraLayer)

    def enable_adapter_layers(self):
        """enable_adapter_layers"""
        self._set_adapter_layers(enabled=True)

    def disable_adapter_layers(self):
        """disable_adapter_layers"""
        self._set_adapter_layers(enabled=False)

    def set_adapter(self, adapter_name):
        """set_adapter"""
        for module in self.model.modules():
            if isinstance(module, LoraLayer):
                if module.merged:
                    warnings.warn("Adapter cannot be set when the model is merged. Unmerging the model first.")
                    module.unmerge()
                module.active_adapter = adapter_name

    def merge_adapter(self):
        """merge_adapter"""
        for module in self.model.modules():
            if isinstance(module, LoraLayer):
                module.merge()

    def unmerge_adapter(self):
        """unmerge_adapter"""
        for module in self.model.modules():
            if isinstance(module, LoraLayer):
                module.unmerge()

    @staticmethod
    def _prepare_lora_config(peft_config, model_config):
        if peft_config.target_modules is None:
            if model_config["model_type"] not in TRANSFORMERS_MODELS_TO_LORA_TARGET_MODULES_MAPPING:
                raise ValueError("Please specify `target_modules` in `peft_config`")
            peft_config.target_modules = TRANSFORMERS_MODELS_TO_LORA_TARGET_MODULES_MAPPING[model_config["model_type"]]

        if peft_config.inference_mode:
            peft_config.merge_weights = True

        return peft_config

    def merge_and_unload(self):
        r"""
        This method merges the LoRa layers into the base model. This is needed if someone wants to use the base model
        as a standalone model.
        """
        if getattr(self.config, "model_type", None) == "gpt2":
            raise ValueError("GPT2 models are not supported for merging LORA layers")

        if getattr(self.model, "is_loaded_in_8bit", False):
            raise ValueError("Cannot merge LORA layers when the model is loaded in 8-bit mode")

        key_list = [key for key, _ in self.model.named_modules() if "lora" not in key]
        for key in key_list:
            try:
                parent, target, target_name = _get_submodules(self.model, key)
            except AttributeError:
                continue
            if isinstance(target, LoraLayer):
                if isinstance(target, nn.Embedding):
                    new_module = nn.Embedding(target.in_channels, target.out_channels)
                elif isinstance(target, nn.Conv2d):
                    new_module = nn.Conv2d(
                        target.in_channels,
                        target.out_channels,
                        kernel_size=target.kernel_size,
                        stride=target.stride,
                        padding=target.padding,
                        dilation=target.dilation,
                    )
                elif isinstance(target, nn.Dense):
                    bias = target.bias is not None
                    new_module = nn.Dense(target.in_channels, target.out_channels, has_bias=bias)
                else:
                    raise ValueError(f"Not support {type(target)}.")
                target.merge()
                self._replace_module(parent, target_name, new_module, target)

            # save any additional trainable modules part of `modules_to_save`
            if isinstance(target, ModulesToSaveWrapper):
                setattr(parent, target_name, target.modules_to_save[target.active_adapter])

        return self.model

    # def add_weighted_adapter(self, adapters, weights, adapter_name):
    #     """add_weighted_adapter"""
    #     if len({self.peft_config[adapter].r for adapter in adapters}) != 1:
    #         raise ValueError("All adapters must have the same r value")
    #     self.peft_config[adapter_name] = replace(
    #         self.peft_config[adapters[0]], lora_alpha=self.peft_config[adapters[0]].r
    #     )
    #     self._find_and_replace(adapter_name)
    #     mark_only_lora_as_trainable(self.model, self.peft_config[adapter_name].bias)
    #     _freeze_adapter(self.model, adapter_name)
    #     key_list = [key for key, _ in self.model.named_modules() if "lora" not in key]
    #     for key in key_list:
    #         _, target, _ = _get_submodules(self.model, key)
    #         if isinstance(target, LoraLayer):
    #             if adapter_name in target.lora_A:
    #                 target.lora_A[adapter_name].weight.data = target.lora_A[adapter_name].weight.data * 0.0
    #                 target.lora_B[adapter_name].weight.data = target.lora_B[adapter_name].weight.data * 0.0
    #                 for adapter, weight in zip(adapters, weights):
    #                     if adapter not in target.lora_A:
    #                         continue
    #                     target.lora_A[adapter_name].weight.data += (
    #                         target.lora_A[adapter].weight.data * weight * target.scaling[adapter]
    #                     )
    #                     target.lora_B[adapter_name].weight.data += target.lora_B[adapter].weight.data * weight

    #             elif adapter_name in target.lora_embedding_A:
    #                 target.lora_embedding_A[adapter_name].data = target.lora_embedding_A[adapter_name].data * 0.0
    #                 target.lora_embedding_B[adapter_name].data = target.lora_embedding_B[adapter_name].data * 0.0
    #                 for adapter, weight in zip(adapters, weights):
    #                     if adapter not in target.lora_embedding_A:
    #                         continue
    #                     target.lora_embedding_A[adapter_name].data += (
    #                         target.lora_embedding_A[adapter].data * weight * target.scaling[adapter]
    #                     )
    #                     target.lora_embedding_B[adapter_name].data += target.lora_embedding_B[adapter].data * weight

    def _get_active_adapter(self) -> str:
        active_adapter = None
        for _, module in self.model.cells_and_names():
            if isinstance(module, LoraLayer):
                active_adapter = module.active_adapter

        if active_adapter is None:
            raise ValueError(
                "Something went wrong, no active adapter could be found, please report the issue on GitHub"
            )
        return active_adapter

    def _mark_only_adapters_as_trainable(self) -> None:
        """mark_only_lora_as_trainable"""
        # get bias
        active_adapter = self._get_active_adapter()
        bias = self.peft_config[active_adapter].bias

        for n, p in self.model.parameters_and_names():  # named_parameters() -> parameters_and_names()
            if "lora_" not in n:
                p.requires_grad = False
                # print(n, p, "requires_grad = False")
        if bias == "none":
            return
        elif bias == "all":
            for n, p in self.model.parameters_and_names():
                if "bias" in n:
                    p.requires_grad = True
        elif bias == "lora_only":
            for m in self.model.cells():  # .cells() for modules()
                if isinstance(m, LoraLayer) and hasattr(m, "bias") and m.bias is not None:
                    m.bias.requires_grad = True
        else:
            raise NotImplementedError


    @staticmethod
    def _create_new_module(
            lora_config: PeftConfig,
            adapter_name: str,
            target: mindspore.nn.Cell,
            **kwargs
        ):
        """"""
        # TODO: support loaded_in_8bit & loaded_in_4bit later, just pop now.
        # pylint: disable=W0612
        loaded_in_8bit = kwargs.pop("loaded_in_8bit", False)
        loaded_in_4bit = kwargs.pop("loaded_in_4bit", False)
        bias = kwargs.pop("bias", False)

        # if loaded_in_8bit and isinstance(target, bnb.nn.Linear8bitLt):
        #     eightbit_kwargs = kwargs.copy()
        #     eightbit_kwargs.update(
        #         {
        #             "has_fp16_weights": target.state.has_fp16_weights,
        #             "memory_efficient_backward": target.state.memory_efficient_backward,
        #             "threshold": target.state.threshold,
        #             "index": target.index,
        #         }
        #     )
        #     new_module = Linear8bitLt(
        #         adapter_name, target.in_channels, target.out_channels, bias=bias, **eightbit_kwargs
        #     )
        # elif loaded_in_4bit and is_bnb_4bit_available() and isinstance(target, bnb.nn.Linear4bit):
        #     fourbit_kwargs = kwargs.copy()
        #     fourbit_kwargs.update(
        #         {
        #             "compute_dtype": target.compute_dtype,
        #             "compress_statistics": target.weight.compress_statistics,
        #             "quant_type": target.weight.quant_type,
        #         }
        #     )
        #     new_module = Linear4bit(adapter_name, target.in_channels, target.out_channels, bias=bias, **fourbit_kwargs)
        if isinstance(target, nn.Embedding):
            embedding_kwargs = kwargs.copy()
            embedding_kwargs.pop("fan_in_fan_out", None)
            in_features, out_features = target.vocab_size, target.embedding_size  # target.num_embeddings, target.embedding_dim
            new_module = Embedding(adapter_name, in_features, out_features, **embedding_kwargs)
        # elif isinstance(target, torch.nn.Conv2d):
        #     out_channels, in_channels = target.weight.size()[:2]
        #     kernel_size = target.weight.size()[2:]
        #     stride = target.stride``
        #     padding = target.padding
        #     new_module = Conv2d(adapter_name, in_channels, out_channels, kernel_size, stride, padding, **kwargs)
        else:
            if isinstance(target, nn.Dense): # Linear
                # get
                in_features, out_features = target.in_channels, target.out_channels
                if kwargs["fan_in_fan_out"]:
                    warnings.warn(
                        "fan_in_fan_out is set to True but the target module is `torch.nn.Linear`. "
                        "Setting fan_in_fan_out to False."
                    )
                    kwargs["fan_in_fan_out"] = lora_config.fan_in_fan_out = False
            elif isinstance(target, Conv1D):
                in_features, out_features = (
                    target.weight.ds_shape if hasattr(target.weight, "ds_shape") else target.weight.shape
                )
                kwargs["is_target_conv_1d_layer"] = True
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
            new_module = Linear(adapter_name, in_features, out_features, has_bias=bias, **kwargs)

        return new_module


class LoraLayer(BaseTunerLayer):
    """Lora Layer"""
    # TODO add CellDict Support
    def __init__(self, in_features: int, out_features: int, **kwargs):
        self.r = {}
        self.lora_alpha = {}
        self.scaling = {}
        # TODO: there is no nn.CellDict() in mindspore
        self.lora_dropout = CellDict()
        self.lora_A = CellDict()
        self.lora_B = CellDict()
        # For Embedding layer
        self.lora_embedding_A = CellDict()
        self.lora_embedding_B = CellDict()

        # Mark the weight as unmerged
        self.merged = False
        self.disable_adapters = False
        self.in_features = in_features
        self.out_features = out_features
        self.kwargs = kwargs

    def update_layer(self, adapter_name, r, lora_alpha, lora_dropout, init_lora_weights):
        """
        update lora layer.
        """
        self.r[adapter_name] = r
        self.lora_alpha[adapter_name] = lora_alpha
        if lora_dropout > 0.0:
            lora_dropout_layer = nn.Dropout(p=lora_dropout)
        else:
            lora_dropout_layer = nn.Identity()

        # self.lora_dropout.append({adapter_name: lora_dropout_layer})
        self.lora_dropout.update(CellDict({adapter_name: lora_dropout_layer}))
        # Actual trainable parameters
        if r > 0:
            self.lora_A.update({adapter_name: nn.Dense(self.in_features, r, has_bias=False)})
            self.lora_B.update({adapter_name: nn.Dense(r, self.out_features, has_bias=False)})
            # self.lora_A.append(nn.Dense(self.in_features, r, has_bias=False))
            # self.lora_B.append(nn.Dense(r, self.out_features, has_bias=False))
            self.scaling[adapter_name] = lora_alpha / r
        if init_lora_weights:
            self.reset_lora_parameters(adapter_name)
        # TODO: to device
        # self.to(self.weight.device)

    # TODO: add conv2d
    # def update_layer_conv2d(self, adapter_name, r, lora_alpha, lora_dropout, init_lora_weights):
    #     self.r[adapter_name] = r
    #     self.lora_alpha[adapter_name] = lora_alpha
    #     if lora_dropout > 0.0:
    #         lora_dropout_layer = nn.Dropout(p=lora_dropout)
    #     else:
    #         lora_dropout_layer = nn.Identity()

    #     self.lora_dropout.update(CellList({adapter_name: lora_dropout_layer}))
    #     # Actual trainable parameters
    #     if r > 0:
    #         kernel_size = self.kwargs["kernel_size"]
    #         stride = self.kwargs["stride"]
    #         padding = self.kwargs["padding"]
    #         self.lora_A.update(
    #             CellList({adapter_name: nn.Conv2d(self.in_features, r, kernel_size, stride, padding, bias=False)})
    #         )
    #         self.lora_B.update(
    #             CellList({adapter_name: nn.Conv2d(r, self.out_features, (1, 1), (1, 1), bias=False)})
    #         )
    #         self.scaling[adapter_name] = lora_alpha / r
    #     if init_lora_weights:
    #         self.reset_lora_parameters(adapter_name)
    #     self.to(self.weight.device)

    def update_layer_embedding(self, adapter_name, r, lora_alpha, lora_dropout, init_lora_weights):
        """
        update layer embedding.
        """
        self.r[adapter_name] = r
        self.lora_alpha[adapter_name] = lora_alpha
        if lora_dropout > 0.0:
            lora_dropout_layer = nn.Dropout(p=lora_dropout)
        else:
            lora_dropout_layer = nn.Identity()

        self.lora_dropout.update({adapter_name: lora_dropout_layer})
        # Actual trainable parameters
        if r > 0:
            weight_A = mindspore.ops.randn((r, self.in_features))  # dtype=self.weight.dtype, device=self.weight.device
            weight_B = mindspore.ops.randn((self.out_features, r))  # dtype=self.weight.dtype, device=self.weight.device
            self.lora_embedding_A.update({adapter_name: mindspore.Parameter(weight_A)})
            self.lora_embedding_B.update({adapter_name: mindspore.Parameter(weight_B)})
            self.scaling[adapter_name] = lora_alpha / r
        if init_lora_weights:
            self.reset_lora_parameters(adapter_name)
        # self.to(self.weight.device)

    def reset_lora_parameters(self, adapter_name):
        """
        reset lora parameters.
        """
        # pylint: disable=C0201
        if adapter_name in self.lora_A.keys():
            self.lora_A[adapter_name].weight.set_data(initializer(
                HeUniform(negative_slope=math.sqrt(5)),
                self.lora_A[adapter_name].weight.shape,
                self.lora_A[adapter_name].weight.dtype
            ))
            self.lora_B[adapter_name].weight.set_data(initializer(
                Zero(),
                self.lora_B[adapter_name].weight.shape,
                self.lora_B[adapter_name].weight.dtype
            ))

        if adapter_name in self.lora_embedding_A.keys():
            # initialize a the same way as the default for nn.linear and b to zero
            self.lora_embedding_A[adapter_name].weight.set_data(initializer(
                Zero(),
                self.lora_embedding_A[adapter_name].weight.shape,
                self.lora_embedding_A[adapter_name].weight.dtype
            ))
            self.lora_embedding_B[adapter_name].weight.set_data(initializer(
                Normal(),
                self.lora_embedding_B[adapter_name].weight.shape,
                self.lora_embedding_B[adapter_name].weight.dtype
            ))
        # TODO embedding not ok
        # if adapter_name in self.lora_embedding_A.keys():
        #     # initialize a the same way as the default for nn.Dense and b to zero
        #     Zero()(self.lora_embedding_A[adapter_name])
        #     Normal(mean=0, sigma=1.)(self.lora_embedding_B[adapter_name])


class Linear(nn.Dense, LoraLayer):
    """Lora implemented in a dense layer"""
    def __init__(
        self,
        adapter_name: str,
        in_features: int,
        out_features: int,
        r: int = 0,
        lora_alpha: int = 1,
        lora_dropout: float = 0.0,
        fan_in_fan_out: bool = False,  # Set this to True if the layer to replace stores weight like (fan_in, fan_out)
        is_target_conv_1d_layer: bool = False,
        **kwargs,
    ):
        init_lora_weights = kwargs.pop("init_lora_weights", True)

        nn.Dense.__init__(self, in_features, out_features, **kwargs)
        LoraLayer.__init__(self, in_features=in_features, out_features=out_features)
        # Freezing the pre-trained weight matrix
        self.weight.requires_grad = False

        self.fan_in_fan_out = fan_in_fan_out
        if fan_in_fan_out:
            self.weight.data = self.weight.data.T

        # nn.Linear.reset_parameters(self)
        self.update_layer(adapter_name, r, lora_alpha, lora_dropout, init_lora_weights)  # call # LoraLayer.update_layer
        self.active_adapter = adapter_name
        self.is_target_conv_1d_layer = is_target_conv_1d_layer

    def merge(self):
        """merge"""
        if self.active_adapter not in self.lora_A.keys():
            return
        if self.merged:
            warnings.warn("Already merged. Nothing to do.")
            return
        if self.r[self.active_adapter] > 0:
            self.weight.data += self.get_delta_weight(self.active_adapter)
            self.merged = True

    def unmerge(self):
        """unmerge"""
        if self.active_adapter not in self.lora_A.keys():
            return
        if not self.merged:
            warnings.warn("Already unmerged. Nothing to do.")
            return
        if self.r[self.active_adapter] > 0:
            self.weight.data -= self.get_delta_weight(self.active_adapter)
            self.merged = False

    def get_delta_weight(self, adapter):
        """
        get delta weight. Add or Sub to origin.
        """
        return (
            transpose(
                self.lora_B[adapter].weight @ self.lora_A[adapter].weight,
                self.fan_in_fan_out,
            )
            * self.scaling[adapter]
        )

    def extend_repr(self):
        s = f'input_channels={self.in_channels}, output_channels={self.out_channels}'
        if self.has_bias:
            s += f', has_bias={self.has_bias}'
        if self.activation_flag:
            s += f', activation={self.activation}'
        s += f', requires_grad={self.weight.requires_grad}'
        return s

    def _linear(self, x: mindspore.Tensor) -> mindspore.Tensor:
        return ops.dense(x, transpose(self.weight, self.fan_in_fan_out), bias=self.bias)

    def construct(self, x: mindspore.Tensor):
        if self.active_adapter not in self.lora_A.keys():
            return self._linear(x)

        previous_dtype = x.dtype

        if self.disable_adapters:
            if (self.r[self.active_adapter] > 0) and self.merged:
                self.unmerge()
            result = self._linear(x)
        elif (self.r[self.active_adapter] == 0) or self.merged:
            result = self._linear(x)
        else:
            lora_A = self.lora_A[self.active_adapter]
            lora_B = self.lora_B[self.active_adapter]
            dropout = self.lora_dropout[self.active_adapter]
            scaling = self.scaling[self.active_adapter]

            result = self._linear(x)
            x = x.to(lora_A.weight.dtype)
            result += lora_B(lora_A(dropout(x))) * scaling

        result = result.to(previous_dtype)
        return result


class Embedding(nn.Embedding, LoraLayer):
    """LoRA implemented in a Embedding layer"""
    def __init__(
        self,
        adapter_name: str,
        num_embeddings: int,
        embedding_dim: int,
        r: int = 0,
        lora_alpha: int = 1,
        lora_dropout: float = 0.0,
        **kwargs,
    ):
        init_lora_weights = kwargs.pop("init_lora_weights", True)

        nn.Embedding.__init__(self, num_embeddings, embedding_dim, **kwargs)
        LoraLayer.__init__(self, in_features=num_embeddings, out_features=embedding_dim)

        self.weight.requires_grad = False

        # TODO: check nesissary
        # check the api of mindspore.nn.Embedding initialization
        # nn.Embedding.reset_parameters(self)
        self.update_layer_embedding(adapter_name, r, lora_alpha, lora_dropout, init_lora_weights)
        self.active_adapter = adapter_name

    def unmerge(self, mode: bool = True):
        """unmerge"""
        if not self.merged:
            warnings.warn("Already unmerged. Nothing to do.")
            return
        if self.r[self.active_adapter] > 0:
            self.weight.data -= (
                transpose(
                    self.lora_embedding_B[self.active_adapter] @ self.lora_embedding_A[self.active_adapter], True
                )
                * self.scaling[self.active_adapter]
            )
            self.merged = False

    def merge(self):
        """merge"""
        if self.merged:
            warnings.warn("Already merged. Nothing to do.")
            return
        if self.r[self.active_adapter] > 0:
            self.weight.data += (
                transpose(
                    self.lora_embedding_B[self.active_adapter] @ self.lora_embedding_A[self.active_adapter], True
                )
                * self.scaling[self.active_adapter]
            )
            self.merged = True

    def construct(self, ids: mindspore.Tensor):
        if self.disable_adapters:
            if self.r[self.active.adapter] > 0 and self.merged:
                self.weight.data -= (
                    transpose(
                        self.lora_embedding_B[self.active_adapter].weight
                        @ self.lora_embedding_A[self.active_adapter].weight,
                        True,
                    )
                    * self.scaling[self.active_adapter]
                )
                self.merged = False
            return nn.Embedding.construct(self, ids)

        elif self.r[self.active_adapter] > 0 and not self.merged:
            result = nn.Embedding.construct(self, ids)
            if self.r[self.active_adapter] > 0:
                after_A = ops.gather(
                    self.lora_embedding_A[self.active_adapter].T,
                    ids,
                    0
                )
                result += (after_A @ self.lora_embedding_B[self.active_adapter].T) * self.scaling[self.active_adapter]
            return result
        else:
            return nn.Embedding.construct(self, ids)
