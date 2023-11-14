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
# pylint: disable=protected-access
# pylint: disable=invalid-name
"""
Auto mixed precision api.
"""

import mindspore
from mindspore import nn, ops
from mindspore import Tensor, Parameter, context, jit_class
from mindspore.ops import constexpr
import mindspore.common.dtype as mstype
from mindnlp.modules import StaticGRU, StaticLSTM
from mindnlp._legacy.nn import Matmul
from mindnlp.utils import less_min_pynative_first

# For AMP white list
AMP_WHITE_LIST = (
    nn.Dense,
    nn.Conv1d,
    nn.Conv2d,
    nn.Conv3d,
    nn.Conv1dTranspose,
    nn.Conv2dTranspose,
    nn.Conv3dTranspose,
    nn.GRUCell,
    nn.LSTMCell,
    nn.RNNCell,
    nn.LSTM,
    nn.RNN,
    nn.GRU,
    nn.PReLU,
    StaticGRU,
    StaticLSTM,
    Matmul
)

AMP_BLACK_LIST = (
    nn.BatchNorm1d,
    nn.BatchNorm2d,
    nn.LayerNorm
)

class _OutputTo32(nn.Cell):
    "Wrap cell for amp. Cast network output back to float32"

    def __init__(self, op):
        super().__init__(auto_prefix=False)
        self._op = op

    def construct(self, *x):
        return ops.cast(self._op(*x), mstype.float32)

class _OutputTo16(nn.Cell):
    "Wrap cell for amp. Cast network output back to float32"

    def __init__(self, op):
        super().__init__(auto_prefix=False)
        self._op = op

    def construct(self, *x):
        return ops.cast(self._op(*x), mstype.float16)

def auto_mixed_precision(network, amp_level='O1'):
    """auto mixed precision cast."""
    if amp_level == 'O0':
        pass
    elif amp_level == 'O1':
        auto_white_list(network)
    elif amp_level == 'O2':
        auto_black_list(network)
    elif amp_level == 'O3':
        network.to_float(mstype.float16)
    else:
        raise ValueError(f"the amp_level '{amp_level}' is not supported.")
    return network

def auto_white_list(network, white_list=None):
    """auto cast based on white list"""
    if white_list is None:
        white_list = AMP_WHITE_LIST
    cells = network.name_cells()
    change = False
    for name in cells:
        subcell = cells[name]
        if subcell == network:
            continue
        if isinstance(subcell, white_list):
            network._cells[name] = _OutputTo32(subcell.to_float(mstype.float16))
            change = True
        else:
            auto_white_list(subcell, white_list)

    if isinstance(network, nn.SequentialCell) and change:
        network.cell_list = list(network.cells())

def auto_black_list(network, black_list=None):
    """auto cast based on black list"""
    if black_list is None:
        black_list = AMP_BLACK_LIST
    network.to_float(mstype.float16)
    cells = network.name_cells()
    change = False
    for name in cells:
        subcell = cells[name]
        if subcell == network:
            continue
        if isinstance(subcell, black_list):
            network._cells[name] = _OutputTo16(subcell.to_float(mstype.float32))
            change = True
        else:
            auto_black_list(subcell, black_list)

    if isinstance(network, nn.SequentialCell) and change:
        network.cell_list = list(network.cells())


_hypermap = ops.HyperMap()
_partial = ops.Partial()

@constexpr
def _ascend_target():
    return context.get_context("device_target") == "Ascend"


@constexpr
def _gpu_target():
    return context.get_context("device_target") == "GPU"

def _grad_unscale(scale, grad):
    """grad unscale."""
    return grad * ops.Reciprocal()(scale).astype(grad.dtype)

def _grad_scale(scale, grad):
    """grad scale."""
    return grad * scale.astype(grad.dtype)

def _is_finite(inputs):
    """whether input tensor is finite."""
    if _gpu_target():
        return ops.FloatStatus()(inputs)[0] == 0
    status = ops.isfinite(inputs)
    return status.all()

def init_status():
    r"""
    Returns a Tensor indicating initialized status for overflow detection.
    """
    if _ascend_target() and less_min_pynative_first:
        status = ops.NPUAllocFloatStatus()()
        clear_status = ops.NPUClearFloatStatus()(status)
        status = ops.depend(status, clear_status)
    else:
        status = Tensor([0, 0, 0, 0, 0, 0, 0, 0], mstype.float32)

    return status

def all_finite(inputs, status=None):
    """whether all inputs tensor are finite."""
    if _ascend_target():
        if not less_min_pynative_first:
            return mindspore.amp.all_finite(inputs)
        if status is None:
            raise ValueError("The status must be initialized on Ascend, but get 'None'.")
        status = ops.depend(status, inputs)
        get_status = ops.NPUGetFloatStatus()(status)
        status = ops.depend(status, get_status)
        status_finite = status.sum() == 0
        _ = ops.NPUClearFloatStatus()(status)
        return status_finite
    outputs = _hypermap(_partial(_is_finite), inputs)
    return ops.stack(outputs).all()

@jit_class
class LossScaler():
    """
    Basic LossScaler.
    """
    def __init__(self, scale_value):
        super().__init__()
        self.scale_value = Parameter(Tensor(scale_value, dtype=mstype.float32), name="scale_value")
        self.counter = Parameter(Tensor(0, dtype=mstype.int32), name="counter")

    def scale(self, inputs):
        """scale inputs tensor."""
        raise NotImplementedError

    def unscale(self, inputs):
        """unscale inputs tensor."""
        raise NotImplementedError

    def adjust(self, grads_finite):
        """adjust scale value."""
        raise NotImplementedError

class NoLossScaler(LossScaler):
    """
    No LossScaler
    """
    def __init__(self):
        super().__init__(1)

    def scale(self, inputs):
        return inputs

    def unscale(self, inputs):
        return inputs

    def adjust(self, grads_finite):
        return

class StaticLossScaler(LossScaler):
    """
    Static LossScaler.
    """
    def scale(self, inputs):
        return _hypermap(_partial(_grad_scale, self.scale_value), inputs)

    def unscale(self, inputs):
        return _hypermap(_partial(_grad_unscale, self.scale_value), inputs)

    def adjust(self, grads_finite):
        return

class DynamicLossScaler(LossScaler):
    """
    Dynamic LossScaler
    """
    def __init__(self, scale_value, scale_factor, scale_window):
        super().__init__(scale_value)
        self.scale_factor = scale_factor
        self.scale_window = scale_window

    def scale(self, inputs):
        return _hypermap(_partial(_grad_scale, self.scale_value), inputs)

    def unscale(self, inputs):
        return _hypermap(_partial(_grad_unscale, self.scale_value), inputs)

    def adjust(self, grads_finite):
        one = ops.ones((), self.scale_value.dtype)
        scale_mul_factor = self.scale_value * self.scale_factor
        scale_value = ops.select(
            grads_finite,
            # When grads are finite increase loss scale periodically.
            ops.select(
                self.counter == (self.scale_window - 1),
                ops.select(_is_finite(scale_mul_factor),
                           scale_mul_factor,
                           self.scale_value),
                self.scale_value),
            # If grads are non finite reduce loss scale.
            ops.maximum(one, self.scale_value / self.scale_factor))
        ops.assign(self.scale_value, scale_value)

        counter = ((self.counter + 1) % self.scale_window) * grads_finite
        ops.assign(self.counter, counter)
        return True
