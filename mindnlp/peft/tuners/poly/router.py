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
"""poly router"""
from abc import ABC, abstractmethod

import mindspore

from mindnlp.core import nn, ops
from mindnlp.core.nn import Parameter
from mindnlp.core.distributions.relaxed_bernoulli import RelaxedBernoulli
from .config import PolyConfig

EPS = 1e-12


def get_router(poly_config: PolyConfig) -> nn.Module:
    if poly_config.poly_type == "poly":
        return PolyRouter(poly_config)
    else:
        raise ValueError(
            f"Unsupported poly_type: {poly_config.poly_type}. "
            "Currently, only the following types are supported: "
            "`poly`."
        )


class Router(nn.Module, ABC):
    @abstractmethod
    def reset(self): ...

    @abstractmethod
    def forward(self, task_ids: mindspore.Tensor, input_ids: mindspore.Tensor): ...


class PolyRouter(Router):
    # It's a simplified implementation of
    # https://github.com/microsoft/mttl/blob/ce4ca51dbca73be656feb9b3e5233633e3c5dec7/mttl/models/poly.py#L138
    def __init__(self, poly_config: PolyConfig):
        super().__init__()

        self.poly_type = poly_config.poly_type
        self.n_tasks = poly_config.n_tasks
        self.n_skills = poly_config.n_skills
        self.n_splits = poly_config.n_splits

        self.cell_logits = Parameter(
            ops.zeros(self.n_tasks, self.n_splits * self.n_skills)
        )

    def reset(self):
        nn.init.uniform_(self.module_logits, -1e-3, 1e-3)

    def forward(self, task_ids: mindspore.Tensor, input_ids: mindspore.Tensor):
        if task_ids is None:
            raise ValueError("task_ids should not be None.")
        if task_ids.max().item() >= self.n_tasks:
            raise ValueError(
                f"Only {self.n_tasks} tasks available. Found task id = {task_ids.max().item()}"
            )

        # move task id to input's device
        # task_ids = task_ids.to(self.cell_logits.device)

        cell_logits = self.cell_logits[task_ids]
        cell_logits = cell_logits.view(-1, self.n_splits, self.n_skills)

        if self.training:
            cell_logits = RelaxedBernoulli(
                temperature=1.0, logits=cell_logits
            ).rsample()
        else:
            cell_logits = ops.sigmoid(cell_logits)

        cell_weights = cell_logits / (cell_logits.sum(dim=-1, keepdim=True) + EPS)

        return cell_weights
