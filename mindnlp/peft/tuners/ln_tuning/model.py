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
"""ln_tuning model"""
from __future__ import annotations

import warnings
from typing import Optional

from mindspore import nn
from mindspore.nn.cell import Cell
from tqdm import tqdm

from mindnlp.peft.config import PeftConfig
from mindnlp.peft.tuners.tuners_utils import (
    BaseTuner,
    _get_subcells,
    check_target_cell_exists,
)
from mindnlp.peft.utils import (
    TRANSFORMERS_MODELS_TO_LNTUNING_TARGET_MODULES_MAPPING,
    ModulesToSaveWrapper,
)

from .layer import LNTuningLayer


class LNTuningModel(BaseTuner):
    """
    Creates LayerNorm tuning from a pretrained transformer model.

    The method is described in detail in https://arxiv.org/abs/2312.11420.

    Args:
        model ([`mindspore.nn.Module`]): The model to be adapted.
        config ([`LNTuningConfig`]): The configuration of the Lora model.
        adapter_name (`str`): The name of the adapter, defaults to `"default"`.

    Returns:
        'mindspore.nn.Module': The adapted model with LayerNorm tuned on.

    Example:

        ```py
        >>> from mindnlp.transformers import AutoModelForCausalLM
        >>> from mindnlp.peft import get_peft_model, TaskType, LNTuningConfig

        >>> peft_config = LNTuningConfig(
        ...     task_type=TaskType.CAUSAL_LM,
        ... )

        >>> model = AutoModelForCausalLM.from_pretrained("meta-llama/Llama-2-7b-hf")
        >>> model = get_peft_model(model, peft_config)
        >>> model.print_trainable_parameters()
        ```

    **Attributes**:
        - **model** ([`~transformers.PreTrainedModel`]) -- The model to be adapted.
        - **peft_config** ([`LNTuningConfig`]): The configuration of the Lora model.
    """

    prefix: str = "ln_tuning_"

    # def __init__(self, model, config, adapter_name) -> None:
    #     # self.adapter_name = adapter_name
    #     super().__init__(model, config, adapter_name)

    def __getattr__(self, name: str):
        """Forward missing attributes to the wrapped module."""
        try:
            return super().__getattr__(name)  # defer to nn.Module's logic
        except AttributeError:
            return getattr(self.model, name)

    # TODO: here need to handle the modules_to_save rather than the target_modules
    @staticmethod
    def _prepare_adapter_config(
        peft_config: PeftConfig, model_config: dict
    ) -> PeftConfig:
        if peft_config.target_modules is None:
            if (
                model_config["model_type"]
                not in TRANSFORMERS_MODELS_TO_LNTUNING_TARGET_MODULES_MAPPING
            ):
                raise ValueError("Please specify `target_modules` in `peft_config`")
            peft_config.target_modules = set(
                TRANSFORMERS_MODELS_TO_LNTUNING_TARGET_MODULES_MAPPING[
                    model_config["model_type"]
                ]
            )
        return peft_config

    def _create_and_replace(
        self,
        peft_config: PeftConfig,
        adapter_name: str,
        target: Cell,
        target_name: str,
        parent: Cell,
        current_key: str,
    ) -> None:
        # replace the original cell with a same new cell
        new_cell = self._create_new_cell(peft_config, target, adapter_name)
        if adapter_name != self.active_adapter:
            new_cell.requires_grad = False
        self._replace_module(parent, target_name, new_cell, target)

    def _create_new_cell(
        self,
        peft_config: PeftConfig,
        target: Cell,
        adapter_name: str,
    ) -> Cell:
        if not isinstance(target, LNTuningLayer):
            new_cell = LNTuningLayer(target, adapter_name)
        else:
            new_cell = target
            new_cell.update_layer(target.base_layer, adapter_name)
        return new_cell

    def _replace_module(
        self, parent: Cell, child_name: str, new_cell: Cell, child: Cell
    ) -> None:
        setattr(parent, child_name, new_cell)

        if hasattr(child, "base_layer"):
            child = child.base_layer

        if getattr(child, "state", None) is not None:
            if hasattr(new_cell, "base_layer"):
                new_cell.base_layer.state = child.state
            else:
                new_cell.state = child.state

    def _mark_only_adapters_as_trainable(self, model: Cell):
        for n, p in model.parameters_and_names():
            if self.prefix not in n:
                p.requires_grad = False
            else:
                p.requires_grad = True

    def _check_target_cell_exists(self, peft_config: PeftConfig, key: str) -> bool:
        return check_target_cell_exists(peft_config, key)

    def _set_adapter_layers(self, enabled: bool) -> None:
        for cell in self.model.cells():
            if isinstance(cell, (LNTuningLayer, ModulesToSaveWrapper)):
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

    def set_adapter(self, adapter_name: str) -> None:
        for cell in self.model.cells():
            if isinstance(cell, LNTuningLayer):
                if cell.merged:
                    warnings.warn(
                        "Adapter cannot be set when the model is merged. Unmerging the model first."
                    )
                    cell.unmerge()
                cell.set_adapter(adapter_name)
        self.active_adapter = adapter_name

    def _unload_and_optionally_merge(
        self,
        merge=True,
        progressbar: bool = False,
        safe_merge: bool = False,
        adapter_names: Optional[list[str]] = None,
    ):
        self._unloading_checks(adapter_names)
        key_list = [
            key for key, _ in self.model.named_cells() if self.prefix not in key
        ]
        desc = "Unloading adapters " + ("and merging " if merge else "") + "model"

        for key in tqdm(key_list, disable=not progressbar, desc=desc):
            try:
                parent, target, target_name = _get_subcells(self.model, key)
            except AttributeError:
                continue

            if hasattr(target, "base_layer"):
                if merge:
                    target.merge(adapter_names)
                self._replace_module(
                    parent, target_name, target.get_base_layer(), target
                )

        return self.model

    def unload(self):
        return self._unload_and_optionally_merge(merge=False)

    def merge_and_unload(
        self,
        progressbar: bool = False,
        safe_merge: bool = False,
        adapter_names: Optional[list[str]] = None,
    ) -> nn.Module:
        return self._unload_and_optionally_merge(merge=True)
