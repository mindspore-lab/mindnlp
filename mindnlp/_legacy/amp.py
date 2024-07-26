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
"""
Auto mixed precision api.
"""

import mindspore
from mindnlp.core import nn, ops
from mindspore import Tensor, Parameter, context, jit_class
from mindspore.ops import constexpr
import mindspore.common.dtype as mstype
from mindnlp.modules import StaticGRU, StaticLSTM
from mindnlp._legacy.nn import Matmul
from mindnlp.utils import less_min_pynative_first

# For AMP white list
AMP_WHITE_LIST = (
    nn.Linear,
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

class _OutputTo32(nn.Module):
    "Wrap cell for amp. Cast network output back to float32"
    def __init__(self, op):
        r"""
        Initializes an instance of the '_OutputTo32' class.
        
        Args:
            self: The instance of the class.
            op: An object representing the operation to be performed.
        
        Returns:
            None.
        
        Raises:
            None.
        """
        super().__init__(auto_prefix=False)
        self._op = op

    def forward(self, *x):
        r"""
        Constructs and returns a float32 tensor.
        
        Args:
            *x (tuple): The input tensor(s) to be cast to float32. Variable number of tensor inputs can be passed as arguments.
        
        Returns:
            None. The method returns the forwarded float32 tensor.
        
        Raises:
            - TypeError: If the input tensor(s) cannot be cast to float32.
            - ValueError: If the input tensor(s) are of inappropriate shape or type for casting to float32.
        """
        return ops.cast(self._op(*x), mstype.float32)

class _OutputTo16(nn.Module):
    "Wrap cell for amp. Cast network output back to float32"
    def __init__(self, op):
        r"""
        Initialize an instance of the _OutputTo16 class.
        
        Args:
            self (object): The instance of the class.
            op (any): The operation to be assigned to the instance.
            
        Returns:
            None. This method initializes the _OutputTo16 instance with the provided operation.
            
        Raises:
            No specific exceptions are raised by this method.
        """
        super().__init__(auto_prefix=False)
        self._op = op

    def forward(self, *x):
        r"""
        This method forwards a new instance of _OutputTo16 class.
        
        Args:
            *x: Variable length argument list. The input values to be cast to float16.
                 The type of each input value should be compatible with the cast operation.
        
        Returns:
            None: This method does not return any value.
        
        Raises:
            - TypeError: If the input values are not compatible with the cast operation.
        """
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
    r"""
    Checks if the current device target is set to 'Ascend'.
    
    Returns:
        None
    
    Raises:
        None
    """
    return context.get_context("device_target") == "Ascend"


@constexpr
def _gpu_target():
    r"""
    This function checks whether the device target context is set to 'GPU'.
    
    Returns:
        None: This function does not return any value.
    
    """
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
        r"""
        __init__
        
        Initializes a new instance of the LossScaler class.
        
        Args:
            self: The instance of the LossScaler class.
            scale_value (float): The value used to scale the loss. Must be a floating point number.
        
        Returns:
            None. This method does not return any value.
        
        Raises:
            TypeError: If the scale_value is not a floating point number.
            ValueError: If the scale_value is invalid or out of range.
        """
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
        r"""
        Initializes the NoLossScaler class instance.
        
        Args:
            self: The instance of the NoLossScaler class.
        
        Returns:
            None. This method does not return any value.
        
        Raises:
            This method does not explicitly raise any exceptions.
        """
        super().__init__(1)

    def scale(self, inputs):
        r"""
        Method: scale
        
        Description:
        This method scales the input data.
        
        Args:
        - self (object): The instance of the NoLossScaler class.
        - inputs (object): The input data to be scaled.
        
        Returns:
        - None: This method does not return any value.
        
        Raises:
        - None: This method does not raise any exceptions.
        """
        return inputs

    def unscale(self, inputs):
        """
        Unscales the given inputs.
        
        Args:
            self (NoLossScaler): An instance of the NoLossScaler class.
                - This parameter represents the current instance of the NoLossScaler class.
                - It is used to access the attributes and methods of the class.
        
            inputs: The inputs to be unscaled.
                - This parameter represents the inputs that need to be unscaled.
                - It can be of any type.
        
        Returns:
            None. This method modifies the inputs in place.
                - This method does not return any value.
                - It modifies the 'inputs' parameter directly.
        
        Raises:
            N/A
                - This method does not raise any exceptions.
        """
        def unscale(self, inputs):
            """
            Unscales the given inputs.
        
            Args:
                self (NoLossScaler): An instance of the NoLossScaler class.
                inputs: The inputs to be unscaled.
        
            Returns:
                None. This method modifies the inputs in place.
        
            Raises:
                N/A
            """
        return inputs

    def adjust(self, grads_finite):
        r"""
        Adjusts the value of the finite gradients.
        
        Args:
            self (NoLossScaler): An instance of the NoLossScaler class.
            grads_finite (float): The finite gradients to be adjusted.
        
        Returns:
            None: This method does not return any value.
        
        Raises:
            None: This method does not raise any exceptions.
        """
        return

class StaticLossScaler(LossScaler):
    """
    Static LossScaler.
    """
    def scale(self, inputs):
        r"""
        Method to scale the inputs using a static loss scaler.
        
        Args:
            self (StaticLossScaler): Instance of the StaticLossScaler class.
                The StaticLossScaler object that contains the scale value used for scaling.
            inputs (list): List of input values to be scaled.
                The inputs to be scaled using the specified scale value.
        
        Returns:
            None
            This method does not return any value. It scales the input values in place.
        
        Raises:
            None
            This method does not raise any exceptions.
        """
        return _hypermap(_partial(_grad_scale, self.scale_value), inputs)

    def unscale(self, inputs):
        r"""
        Unscale the inputs using the specified scale value.
        
        Args:
            self (StaticLossScaler): An instance of the StaticLossScaler class.
            inputs: The inputs to be unscaled.
        
        Returns:
            None: This method does not return any value.
        
        Raises:
            None: This method does not raise any exceptions.
        """
        return _hypermap(_partial(_grad_unscale, self.scale_value), inputs)

    def adjust(self, grads_finite):
        r"""
        Method 'adjust' in the class 'StaticLossScaler'.
        
        Args:
            self (object): The instance of the StaticLossScaler class.
            grads_finite (list): A list of gradients for adjustment.
        
        Returns:
            None. This method does not return any value.
        
        Raises:
            No specific exceptions are raised by this method.
        """
        return

class DynamicLossScaler(LossScaler):
    """
    Dynamic LossScaler
    """
    def __init__(self, scale_value, scale_factor, scale_window):
        r"""
        Initializes an instance of the DynamicLossScaler class.
        
        Args:
            self (DynamicLossScaler): The current instance of the DynamicLossScaler class.
            scale_value (float): The initial scale value for the loss scaling.
            scale_factor (float): The factor by which the scale value is multiplied after each successful iteration.
            scale_window (int): The number of iterations for which the scale value remains unchanged before it is updated.
        
        Returns:
            None. This method does not return any value.
        
        Raises:
            None. This method does not raise any exceptions.
        """
        super().__init__(scale_value)
        self.scale_factor = scale_factor
        self.scale_window = scale_window

    def scale(self, inputs):
        """
        This method scales the inputs using a dynamic loss scaler value.
        
        Args:
            self (DynamicLossScaler): An instance of the DynamicLossScaler class.
                This parameter represents the current instance of the DynamicLossScaler class.
            inputs (list): A list of input values to be scaled.
                The inputs should be in a format compatible with the scaling operation.
        
        Returns:
            None. This method does not return any value explicitly but modifies the inputs in-place.
        
        Raises:
            - TypeError: If the inputs are not provided in a list format.
            - ValueError: If there is an issue with the scaling process.
        """
        return _hypermap(_partial(_grad_scale, self.scale_value), inputs)

    def unscale(self, inputs):
        r"""
        Unscale the input values using the specified scaling factor.
        
        Args:
            self (DynamicLossScaler): An instance of the DynamicLossScaler class.
                This parameter is automatically passed when calling the method.
            inputs: The input values to be unscaled. It can be a single value or an iterable.
                The values should be of a numeric type and within the range supported by the scaling factor.
        
        Returns:
            None. This method does not return anything.
        
        Raises:
            None. This method does not raise any exceptions.
        """
        return _hypermap(_partial(_grad_unscale, self.scale_value), inputs)

    def adjust(self, grads_finite):
        """
        Adjusts the dynamic loss scaling value based on the gradients and internal state.
        
        Args:
            self (DynamicLossScaler): The instance of the DynamicLossScaler class.
            grads_finite (bool): A boolean indicating whether the gradients are finite.
        
        Returns:
            None: This method does not return any value.
        
        Raises:
            None: This method does not raise any exceptions.
        """
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
