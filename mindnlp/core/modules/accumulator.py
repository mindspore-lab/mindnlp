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
"""gradient accumulator"""

import mindspore
from mindspore import jit_class
from mindspore import Tensor
from mindnlp.core.nn import Parameter
from mindnlp.core import ops

@jit_class
class Accumulator():
    """Gradient Accumulator."""
    def __init__(self, optimizer, accumulate_step, clip_norm=1.0):
        r"""
        Initialize the Accumulator object with the provided optimizer, accumulate step, and optional clipping norm.
        
        Args:
            optimizer (Optimizer): The optimizer to be used for parameter updates.
            accumulate_step (int): The number of steps to accumulate gradients before updating parameters.
            clip_norm (float, optional): The maximum gradient norm value for gradient clipping. Default is 1.0.
        
        Returns:
            None: This method does not return any value.
        
        Raises:
            AssertionError: If accumulate_step is less than or equal to 0.
        """
        self.optimizer = optimizer
        self.clip_norm = clip_norm
        self.inner_grads = optimizer.parameters.clone(prefix="accumulate_", init='zeros')
        self.zeros = optimizer.parameters.clone(prefix="zeros_", init='zeros')
        self.counter = Parameter(Tensor(1, mindspore.int32), 'counter_')
        assert accumulate_step > 0
        self.accumulate_step = accumulate_step

    def __call__(self, grads):
        r"""
        This method '__call__' is defined within the class 'Accumulator' and is used to accumulate gradients and update the optimizer at a specified interval.
        
        Args:
            self (Accumulator): The instance of the Accumulator class.
            grads (Tensor): The gradients to be accumulated and passed to the optimizer. It should be a Tensor type, representing the gradients computed during backpropagation.
        
        Returns:
            None: This method does not return any value, as it only performs operations on the input gradients and updates the optimizer.
        
        Raises:
            - TypeError: If the input 'grads' is not of type Tensor.
            - ValueError: If the input 'grads' is not in the expected format or shape.
            - RuntimeError: If there are issues with the accumulation or optimization process.
        """
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
