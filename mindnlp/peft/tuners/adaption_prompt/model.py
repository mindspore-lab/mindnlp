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
"""model for adaption prompt tuners."""
from typing import Dict, List

from mindnlp.core import nn

from mindnlp.peft.utils import _freeze_adapter, _get_subcells

from .config import AdaptionPromptConfig, prepare_config
from .layer import AdaptedAttention
from .utils import is_adaption_prompt_trainable


class AdaptionPromptModel(nn.Module):
    """
    Implements adaption prompts as described in https://arxiv.org/pdf/2303.16199.pdf.

    The top L attention cells are replaced with AdaptedAttention cells that wrap the original ones, but insert
    trainable prompts with gates (for zero init).

    Notes on the multi-adapter pattern:
    - We store the states of different adapters by keeping a dictionary of AdaptedAttention cells indexed by adapter
      name.
    - Every time we switch adapters, we remove the cells of the currently active adapter from the model, store them
      in the dictionary, and replace them with the cells of the new adapter.
    - To avoid duplicated and potentially inconsistent state, the currently active adapter is always removed from the
      dictionary.
    - Disabling the adapter would also result in the cells being removed from the model.
    """
    def __init__(self, model, configs: Dict, adapter_name: str):
        r"""
        Initializes an instance of the AdaptionPromptModel class.
        
        Args:
            self: The current instance of the class.
            model: The underlying model to be used for adaption prompts. Expected to be an object of a specific model class.
            configs: A dictionary containing configuration details for the adaption prompt model.
                - Type: Dict
                - Purpose: Specifies various configurations required for the adaption prompt model.
                - Restrictions: None
            adapter_name: The name of the adapter to be added.
                - Type: str
                - Purpose: Identifies the adapter which needs to be added to the adaption prompt model.
                - Restrictions: None
        
        Returns:
            None
        
        Raises:
            None
        """
        super(AdaptionPromptModel, self).__init__()
        self.model = model
        self.peft_config = {}
        self._parents = {}
        self._cached_adapters = {}
        self._active_adapter = None
        self._enabled = True
        self.forward = self.model.forward
        self.add_adapter(adapter_name, configs[adapter_name])
        self._mark_only_adaption_prompts_as_trainable(self.model)

    def add_adapter(self, adapter_name: str, config: AdaptionPromptConfig) -> None:
        """Add an adapter with the given name and config."""
        config = prepare_config(config, self.model)
        if adapter_name in self.peft_config:
            raise ValueError(f"Adapter named '{adapter_name}' already exists.")

        parents = []
        # 获取模型的所有子模块及其名称
        for name, subcell in self.model.name_cells().items():
            if name.endswith(config.target_modules):
                # 对每个符合条件的子模块调用 _get_subcells 函数
                parent, target, target_name = _get_subcells(self.model, name)
                if target == subcell:
                    parents.append(parent)

        if len(parents) < config.adapter_layers:
            raise ValueError("Config specifies more adapter layers than available in the model.")

        parents = parents[-config.adapter_layers:]
        self._parents[adapter_name] = parents

        if self._active_adapter and self._enabled:
            self._remove_adapted_attentions(self._active_adapter)
        self._active_adapter = adapter_name
        self.peft_config[adapter_name] = config
        self._create_adapted_attentions(config, parents)
        if not self._enabled:
            self._remove_adapted_attentions(adapter_name)

        if config.inference_mode:
            _freeze_adapter(self.model, adapter_name)

    def set_adapter(self, adapter_name: str) -> None:
        """Set the model to use the adapter with the given name."""
        if self._active_adapter == adapter_name:
            return
        if adapter_name not in self.peft_config:
            raise ValueError(f"Adapter with name '{adapter_name}' does not exist.")

        if self._enabled:
            self._remove_adapted_attentions(self._active_adapter)
            self._set_adapted_attentions(adapter_name)

        self._active_adapter = adapter_name

    def enable_adapter_layers(self):
        """Enable adapter layers by swapping in cached AdaptedAttention cells."""
        self._enabled = True
        self._set_adapted_attentions(self._active_adapter)

    def disable_adapter_layers(self):
        """Disable adapter layers by swapping out AdaptedAttention cells."""
        self._enabled = False
        self._remove_adapted_attentions(self._active_adapter)

    def _create_adapted_attentions(self, config: AdaptionPromptConfig, parents: List[nn.Module]) -> None:
        """Wrap LlamaAttention cells with newly created AdaptedAttention cells."""
        for par in parents:
            attn = AdaptedAttention(
                model_type=self.model.config.model_type,
                adapter_len=config.adapter_len,
                model=getattr(par, config.target_modules),
            )
            setattr(par, config.target_modules, attn)

    def _set_adapted_attentions(self, adapter_name: str) -> None:
        """Replace LlamaAttention cells with cached AdaptedAttention cells."""
        cached = self._cached_adapters[adapter_name]
        del self._cached_adapters[adapter_name]
        config = self.peft_config[adapter_name]
        for i, par in enumerate(self._parents[adapter_name]):
            setattr(par, config.target_modules, cached[i])

    def _remove_adapted_attentions(self, adapter_name: str) -> None:
        """Remove AdaptedAttention cells from the model and store them in the cache."""
        config = self.peft_config[adapter_name]
        adapted_attentions = []
        for par in self._parents[adapter_name]:
            attn = getattr(par, config.target_modules)
            adapted_attentions.append(attn)
            setattr(par, config.target_modules, attn.model)
        self._cached_adapters[adapter_name] = adapted_attentions

    def _mark_only_adaption_prompts_as_trainable(self, model: nn.Module) -> None:
        r"""Marks only adaption prompts as trainable in the given model.
        
        Args:
            self (AdaptionPromptModel): The instance of AdaptionPromptModel class.
            model (nn.Module): The model for which adaption prompts need to be marked as trainable.
        
        Returns:
            None. This method does not return any value.
        
        Raises:
            None.
        """
        for param in model.trainable_params():
            if not is_adaption_prompt_trainable(param.name):
                param.requires_grad = False
    def __getattr__(self, name: str):
        """Forward missing attributes to the wrapped cell."""
        try:
            return super().__getattr__(name)  # defer to nn.Module's logic
        except AttributeError:
            # This is necessary as e.g. causal models have various methods that we
            # don't want to re-implement here.
            return getattr(self.model, name)
