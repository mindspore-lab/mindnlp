# Copyright 2023 Huawei Technologies Co., Ltd
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
"""poly model"""


from contextlib import contextmanager
from dataclasses import asdict
from enum import Enum
from typing import Any

from mindnlp.core import nn

from mindnlp.peft.tuners.tuners_utils import (
    BaseTuner,
    BaseTunerLayer,
    check_target_cell_exists,
)
from mindnlp.peft.utils import (
    TRANSFORMERS_MODELS_TO_LORA_TARGET_MODULES_MAPPING,
    ModulesToSaveWrapper,
)

from .config import PolyConfig
from .layer import Dense, PolyLayer


class PolyModel(BaseTuner):
    prefix: str = "poly_"

    # def __init__(self, model, config, adapter_name) -> None:
    #     super().__init__(model, config, adapter_name)

    @staticmethod
    def _check_target_cell_exists(poly_config, key):
        return check_target_cell_exists(poly_config, key)

    def _create_and_replace(
        self,
        poly_config: PolyConfig,
        adapter_name: str,
        target: nn.Module,
        target_name: str,
        parent: nn.Module,
        **optional_kwargs: Any,
    ):
        if isinstance(target, PolyLayer):
            target.update_layer(adapter_name, poly_config)
        else:
            new_cell = self._create_new_cell(
                poly_config,
                adapter_name,
                target,
            )
            if adapter_name not in self.active_adapters:
                # adding an additional adapter: it is not automatically trainable
                new_cell.requires_grad = False
            self._replace_cell(parent, target_name, new_cell, target)

    def _replace_cell(self, parent, child_name, new_cell, child):
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
            # new_cell.to(child.weight.device)

        # dispatch to correct device
        # for name, cell in new_cell.parameters_and_names():
        #     if (self.prefix in name) or ("ranknum" in name):
        #         weight = child.qweight if hasattr(child, "qweight") else child.weight
        #         cell.to(weight.device)

    def _mark_only_adapters_as_trainable(self, model: nn.Module) -> None:
        for name, cell in model.parameters_and_names():
            if self.prefix not in name:
                cell.requires_grad = False

    @staticmethod
    def _create_new_cell(poly_config, adapter_name, target, **kwargs):
        if isinstance(target, BaseTunerLayer):
            target_base_layer = target.get_base_layer()
        else:
            target_base_layer = target

        if isinstance(target_base_layer, nn.Linear):
            return Dense(target, adapter_name, poly_config, **kwargs)
        else:
            raise ValueError(
                f"Target cell {target} is not supported. Currently, only the following cells are supported: "
                "`nn.Linear`."
            )

    def __getattr__(self, name: str):
        """Construct missing attributes to the wrapped cell."""
        try:
            return super().__getattr__(name)  # defer to nn.Module's logic
        except AttributeError:
            return getattr(self.model, name)

    def get_peft_config_as_dict(self, inference: bool = False):
        for _, value in self.peft_config.items():
            config = {
                k: v.value if isinstance(v, Enum) else v
                for k, v in asdict(value).items()
            }
            if inference:
                config["inference_mode"] = True
        return config

    def _set_adapter_layers(self, enabled=True):
        for cell in self.model.cells():
            if isinstance(cell, (PolyLayer, ModulesToSaveWrapper)):
                cell.enable_adapters(enabled)

    def enable_adapter_layers(self):
        self._set_adapter_layers(enabled=True)

    def disable_adapter_layers(self):
        self._set_adapter_layers(enabled=False)

    def set_adapter(self, adapter_name):
        for cell in self.model.cells():
            if isinstance(cell, PolyLayer):
                cell.set_adapter(adapter_name)

    def _prepare_adapter_config(self, peft_config, model_config):
        if peft_config.target_modules is None:
            if (
                model_config["model_type"]
                not in TRANSFORMERS_MODELS_TO_LORA_TARGET_MODULES_MAPPING
            ):
                raise ValueError("Please specify `target_modules` in `peft_config`")
            peft_config.target_modules = set(
                TRANSFORMERS_MODELS_TO_LORA_TARGET_MODULES_MAPPING[
                    model_config["model_type"]
                ]
            )
        return peft_config

    def _register_pre_hooks(self, task_ids):
        """Helper method to register pre hooks."""
        if task_ids is None:
            return []

        def pre_hook(_, inputs):
            args, kwargs = inputs
            kwargs["task_ids"] = task_ids
            return args, kwargs

        handles = []
        for cell in self.model.cells():
            if isinstance(cell, Dense):
                handle = cell.register_forward_pre_hook(pre_hook)
                handles.append(handle)

        return handles

    @contextmanager
    def _manage_pre_hooks(self, task_ids):
        """Context manager to handle the lifecycle of pre hooks."""
        handles = self._register_pre_hooks(task_ids)
        try:
            yield
        finally:
            for handle in handles:
                handle.remove()

    def forward(self, *args, task_ids=None, **kwargs):
        with self._manage_pre_hooks(task_ids):
            return self.model(*args, **kwargs)

    def generate(self, *args, task_ids=None, **kwargs):
        with self._manage_pre_hooks(task_ids):
            return self.model.generate(*args, **kwargs)
