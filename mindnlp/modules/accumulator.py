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
# pylint: disable=C0412
"""gradient accumulator"""

import mindspore
from mindspore import ops
from mindspore import Tensor, Parameter
from mindnlp.utils import less_min_pynative_first

if less_min_pynative_first:
    from mindspore import ms_class as jit_class
else:
    from mindspore import jit_class


@jit_class
class Accumulator():
    """Gradient Accumulator."""
    def __init__(self, optimizer, accumulate_step, clip_norm=1.0):
        self.optimizer = optimizer
        self.clip_norm = clip_norm
        self.inner_grads = optimizer.parameters.clone(prefix="accumulate_", init='zeros')
        self.zeros = optimizer.parameters.clone(prefix="zeros_", init='zeros')
        self.counter = Parameter(Tensor(1, mindspore.int32), 'counter_')
        assert accumulate_step > 0
        self.accumulate_step = accumulate_step
        self.map = ops.HyperMap()

    def __call__(self, grads):
        # 将单步获得的梯度累加至Accumulator的inner_grads
        self.map(ops.partial(ops.assign_add), self.inner_grads, grads)
        if self.counter % self.accumulate_step == 0:
            # 如果达到累积步数，进行参数优化更新
            self.optimizer(self.inner_grads)
            # 完成参数优化更新后，清零inner_grads
            self.map(ops.partial(ops.assign), self.inner_grads, self.zeros)
        # 计算步数加一
        ops.assign_add(self.counter, Tensor(1, mindspore.int32))

        return True

__all__ = ['Accumulator']
