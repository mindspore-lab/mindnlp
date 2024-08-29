# Copyright 2022 Huawei Technologies Co., Ltd
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
"""MindNLP Activations"""
import math
from collections import OrderedDict
from mindspore import Tensor
from mindnlp.core import nn, ops


class QuickGELUActivation(nn.Module):
    """
    Applies GELU approximation that is fast but somewhat inaccurate. See: https://github.com/hendrycks/GELUs
    """
    def forward(self, input: Tensor) -> Tensor:
        r"""
        forwards the QuickGELU activation function.
        
        Args:
            self (QuickGELUActivation): The instance of the QuickGELUActivation class.
            input (Tensor): The input tensor to apply the QuickGELU activation to.
        
        Returns:
            Tensor: The tensor resulting from applying the QuickGELU activation to the input tensor.
        
        Raises:
            None
        """
        return input * ops.sigmoid(1.702 * input)


class ClippedGELUActivation(nn.Module):
    """
    Clip the range of possible GeLU outputs between [min, max]. This is especially useful for quantization purpose, as
    it allows mapping negatives values in the GeLU spectrum. For more information on this trick, please refer to
    https://arxiv.org/abs/2004.09602.

    Gaussian Error Linear Unit. Original Implementation of the gelu activation function in Google Bert repo when
    initially created.

    For information: OpenAI GPT's gelu is slightly different (and gives slightly different results): 0.5 * x * (1 +
    torch.tanh(math.sqrt(2 / math.pi) * (x + 0.044715 * torch.pow(x, 3)))). See https://arxiv.org/abs/1606.08415
    """
    def __init__(self, min: float, max: float):
        r"""
        Initializes an instance of the ClippedGELUActivation class.
        
        Args:
            self: The instance of the ClippedGELUActivation class.
            min (float): The minimum value for clipping.
                The value of 'min' should be less than 'max'.
            max (float): The maximum value for clipping.
                The value of 'max' should be greater than 'min'.
        
        Returns:
            None.
        
        Raises:
            ValueError: If 'min' is greater than 'max', a ValueError is raised with a detailed error message.
        """
        if min > max:
            raise ValueError(f"min should be < max (got min: {min}, max: {max})")

        super().__init__()
        self.min = min
        self.max = max

    def forward(self, x: Tensor) -> Tensor:
        r"""
        forwards a ClippedGELUActivation function with input clipping.
        
        Args:
            self: ClippedGELUActivation
                The instance of the ClippedGELUActivation class.
        
            x: Tensor
                The input tensor to the activation function.
        
        Returns:
            Tensor:
                The tensor resulting from applying the ClippedGELUActivation function to the input tensor,
                with values clipped to the range [min, max].
        
        Raises:
            None
        """
        return ops.clip(gelu(x), self.min, self.max)


class AccurateGELUActivation(nn.Module):
    """
    Applies GELU approximation that is faster than default and more accurate than QuickGELU. See:
    https://github.com/hendrycks/GELUs

    Implemented along with MEGA (Moving Average Equipped Gated Attention)
    """
    def __init__(self):
        r"""
        Initializes an instance of the AccurateGELUActivation class.
        
        Args:
            self: The instance of the class itself.
        
        Returns:
            None.
        
        Raises:
            None.
        """
        super().__init__()
        self.precomputed_constant = math.sqrt(2 / math.pi)

    def forward(self, input: Tensor) -> Tensor:
        r"""
        This method 'forward' is responsible for applying the Accurate Gaussian Error Linear Unit (GELU)
        activation function to the input tensor.
        
        Args:
            self (AccurateGELUActivation): The instance of the AccurateGELUActivation class.
            input (Tensor):
                The input tensor on which the GELU activation function will be applied.
                It represents the input values to be transformed. It should be a tensor of numerical values.
        
        Returns:
            Tensor:
                A tensor of the same shape as the input tensor, containing the output values after applying the
                Accurate GELU activation function.
                The transformed tensor represents the non-linearity applied to the input tensor.
        
        Raises:
            TypeError: If the input tensor is not of type Tensor.
            ValueError: If the dimensions of the input tensor are not compatible with the operations within the method.
            RuntimeError: If there is an issue during the computation of the GELU activation function.
        """
        return 0.5 * input * (1 + ops.tanh(self.precomputed_constant * (input + 0.044715 * ops.pow(input, 3))))


class MishActivation(nn.Module):
    """
    See Mish: A Self-Regularized Non-Monotonic Activation Function (Misra., https://arxiv.org/abs/1908.08681). Also
    visit the official repository for the paper: https://github.com/digantamisra98/Mish
    """
    def forward(self, input: Tensor) -> Tensor:
        r"""
        forwards a Mish activation function on the input tensor.
        
        Args:
            self (MishActivation): An instance of the MishActivation class.
            input (Tensor): The input tensor to apply the activation function on.
        
        Returns:
            Tensor: The tensor with the Mish activation function applied.
        
        Raises:
            None.
        
        The Mish activation function is defined as the element-wise product of the input tensor and
        the hyperbolic tangent of the softplus function applied to the input tensor.
        This activation function introduces a non-linearity that helps in capturing more complex patterns in the data.
        
        Note:
            - The input tensor should have a shape that is compatible with the activation function.
        """
        return input * ops.tanh(ops.softplus(input))


class LinearActivation(nn.Module):
    """
    Applies the linear activation function, i.e. forwarding input directly to output.
    """
    def forward(self, input: Tensor) -> Tensor:
        r"""
        forward method in the LinearActivation class.
        
        Args:
            self (object): The instance of the LinearActivation class.
            input (Tensor): The input tensor to be processed.
        
        Returns:
            Tensor: The processed tensor as per the implementation.
        
        Raises:
            None.
        """
        return input


class LaplaceActivation(nn.Module):
    """
    Applies elementwise activation based on Laplace function, introduced in MEGA as an attention activation. See
    https://arxiv.org/abs/2209.10655

    Inspired by squared relu, but with bounded range and gradient for better stability
    """
    def forward(self, input, mu=0.707107, sigma=0.282095):
        r"""
        This method 'forward' in the class 'LaplaceActivation' performs a Laplace activation function transformation
        on the input data.
        
        Args:
            self (object): The instance of the class.
            input (tensor): The input data to be transformed using the Laplace activation function.
            mu (float, optional): The mean value used for normalization. Default is 0.707107.
            sigma (float, optional): The standard deviation value used for normalization. Default is 0.282095.
        
        Returns:
            None.
        
        Raises:
            ValueError: If the input data is not a valid tensor.
            TypeError: If the input data or the normalization parameters are of incorrect types.
            ZeroDivisionError: If sigma is set to zero, resulting in division by zero.
        """
        input = (input - mu).div(sigma * math.sqrt(2.0))
        return 0.5 * (1.0 + ops.erf(input))


class ReLUSquaredActivation(nn.Module):
    """
    Applies the relu^2 activation introduced in https://arxiv.org/abs/2109.08668v2
    """
    def forward(self, input):
        r"""
        forwards the ReLU squared activation of the input.
        
        Args:
            self (object): Instance of the ReLUSquaredActivation class.
            input (numeric): The input value to be processed by the activation function.
        
        Returns:
            None: This method returns None as it updates the internal state of the object.
        
        Raises:
            None.
        """
        # relu_applied = ops.relu(input)
        relu_applied = nn.functional.relu(input)
        squared = ops.square(relu_applied)
        return squared


class ClassInstantier(OrderedDict):
    r"""
    Class Instantier
    """
    def __getitem__(self, key):
        r"""
        Retrieve an item from the ClassInstantier object using the specified key.
        
        Args:
            self (ClassInstantier): The ClassInstantier object itself.
            key: The key used to retrieve the item from the object.
        
        Returns:
            None.
        
        Raises:
            None.
        """
        content = super().__getitem__(key)
        cls, kwargs = content if isinstance(content, tuple) else (content, {})
        return cls(**kwargs)


ACT2CLS = {
    """
    Excitation equation matrix
    """
    'relu': nn.ReLU,
    'gelu': nn.GELU,
    'gelu_new': (nn.GELU, {'approximate': 'tanh'}),
    'gelu_approximate': nn.GELU,
    'gelu_pytorch_tanh': nn.GELU,
    "swish": nn.SiLU,
    "gelu_10": nn.GELU,
    "gelu_fast": (nn.GELU, {'approximate': 'tanh'}),
    "gelu_python": nn.GELU,
    "linear": nn.ReLU,
    "mish": nn.Mish,
    "quick_gelu": QuickGELUActivation,
    "relu": nn.ReLU,
    "relu2": ReLUSquaredActivation,
    "relu6": nn.ReLU6,
    "sigmoid": nn.Sigmoid,
    "silu": nn.SiLU,
    "tanh": nn.Tanh,
}
ACT2FN = ClassInstantier(ACT2CLS)


def get_activation(activation_string):
    """
    Obtained parameters required for outputting self. activation in the SequenceSummary class
    :param activation_string:
    :return:
    """
    if activation_string in ACT2FN:
        return ACT2FN[activation_string]
    raise KeyError(f"function {activation_string} not found in ACT2FN mapping {list(ACT2FN.keys())}")

gelu_python = get_activation("gelu_python")
gelu_new = get_activation("gelu_new")
gelu = get_activation("gelu")
gelu_fast = get_activation("gelu_fast")
quick_gelu = get_activation("quick_gelu")
silu = get_activation("silu")
mish = get_activation("mish")
linear_act = get_activation("linear")
