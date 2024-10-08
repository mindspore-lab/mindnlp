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
"""loha layer"""
import math
from typing import Any, Set, Tuple

import mindspore
from mindnlp.core import nn, ops
from mindnlp.core.nn import ParameterDict, Parameter

from mindnlp.peft.tuners.lycoris_utils import LycorisLayer


class LoHaLayer(nn.Module, LycorisLayer):

    r"""
    The LoHaLayer class represents a layer that applies Local Harmonic Adaptation (LoHa) to a base layer. LoHaLayer inherits from nn.Module and LycorisLayer. It provides methods to create, reset, and update
adapter parameters, as well as to calculate delta weights and apply the adaptation to input data. 
    
    Attributes:
        base_layer (nn.Module): The base layer for which LoHa adaptation is applied.
    
    Methods:
        - create_adapter_parameters(adapter_name, r, shape): Creates adapter parameters for the specified adapter name, rank, and shape.
        - reset_adapter_parameters(adapter_name): Resets adapter parameters for the specified adapter name with initialized weights.
        - reset_adapter_parameters_random(adapter_name): Resets adapter parameters for the specified adapter name with random weights.
        - update_layer(adapter_name, r, alpha, rank_dropout, module_dropout, init_weights, use_effective_conv2d, **kwargs): Updates the layer with a new adapter using the specified parameters.
        - get_delta_weight(adapter_name): Retrieves the delta weight for the specified adapter name.
        - forward(x, *args, **kwargs): Constructs the layer by applying the base layer and active adapters to the input data.
    
    The class also provides internal functions for managing adapter parameters, updating the layer with new adapters, and applying the adaptation to the input data. 
    
    Note: Detailed parameter descriptions for each method are available in the method signatures in the source code.
    """
    # All names of layers that may contain adapter weights
    adapter_layer_names = (
        "hada_w1_a",
        "hada_w1_b",
        "hada_w2_a",
        "hada_w2_b",
        "hada_t1",
        "hada_t2",
    )
    # other_param_names is defined on parent class

    def __init__(self, base_layer: nn.Module):
        r"""
        Initializes the LoHaLayer class.
        
        Args:
            self: The instance of the class.
            base_layer (nn.Module): The base layer to be initialized with. It should be an instance of nn.Module.
        
        Returns:
            None. This method does not return any value.
        
        Raises:
            N/A
        """
        super().__init__()
        LycorisLayer.__init__(self, base_layer)

        # LoHa info
        self.hada_w1_a = ParameterDict({})
        self.hada_w1_b = ParameterDict({})
        self.hada_w2_a = ParameterDict({})
        self.hada_w2_b = ParameterDict({})
        self.hada_t1 = ParameterDict({})
        self.hada_t2 = ParameterDict({})

    @property
    def _available_adapters(self) -> Set[str]:
        """
        Method to retrieve the set of available adapters in the LoHaLayer class.
        
        Args:
            self: Instance of the LoHaLayer class.
        
        Returns:
            Returns a set of strings representing the available adapters in the LoHaLayer instance.
        
        Raises:
            No specific exceptions are raised by this method.
        """
        return {
            *self.hada_w1_a,
            *self.hada_w1_b,
            *self.hada_w2_a,
            *self.hada_w2_b,
            *self.hada_t1,
            *self.hada_t2,
        }

    def create_adapter_parameters(
        self, adapter_name: str, r: int, shape: Tuple[int, ...]
    ):
        r"""
        This method creates adapter parameters for the LoHaLayer class.
        
        Args:
            self: An instance of the LoHaLayer class.
            adapter_name (str): The name of the adapter.
            r (int): The value of 'r' parameter.
            shape (Tuple[int, ...]): The shape of the parameters.
        
        Returns:
            None: This method does not return any value.
        
        Raises:
            None: This method does not raise any exceptions.
        '''
        
        The method's code is: 
        def create_adapter_parameters(self, adapter_name: str, r: int, shape: Tuple[int, ...]):
            if len(shape) == 4:
                self.hada_t1[adapter_name] = Parameter(ops.zeros(r, r, shape[2], shape[3]))
                self.hada_w1_a[adapter_name] = Parameter(ops.zeros(r, shape[0]))
                self.hada_w1_b[adapter_name] = Parameter(ops.zeros(r, shape[1]))
                self.hada_t2[adapter_name] = Parameter(ops.zeros(r, r, shape[2], shape[3]))
                self.hada_w2_a[adapter_name] = Parameter(ops.zeros(r, shape[0]))
                self.hada_w2_b[adapter_name] = Parameter(ops.zeros(r, shape[1]))
            else:
                self.hada_w1_a[adapter_name] = Parameter(ops.zeros(shape[0], r))
                self.hada_w1_b[adapter_name] = Parameter(ops.zeros(r, shape[1]))
                self.hada_w2_a[adapter_name] = Parameter(ops.zeros(shape[0], r))
                self.hada_w2_b[adapter_name] = Parameter(ops.zeros(r, shape[1]))
        """
        # https://github.com/KohakuBlueleaf/LyCORIS/blob/eb460098187f752a5d66406d3affade6f0a07ece/lycoris/modules/loha.py#L130C9-L143C75
        if len(shape) == 4:
            self.hada_t1[adapter_name] = Parameter(
                ops.zeros(r, r, shape[2], shape[3])
            )
            self.hada_w1_a[adapter_name] = Parameter(
                ops.zeros(r, shape[0])
            )  # out_dim, 1-mode
            self.hada_w1_b[adapter_name] = Parameter(
                ops.zeros(r, shape[1])
            )  # in_dim , 2-mode

            self.hada_t2[adapter_name] = Parameter(
                ops.zeros(r, r, shape[2], shape[3])
            )
            self.hada_w2_a[adapter_name] = Parameter(
                ops.zeros(r, shape[0])
            )  # out_dim, 1-mode

            self.hada_w2_b[adapter_name] = Parameter(
                ops.zeros(r, shape[1])
            )  # in_dim , 2-mode

        else:
            self.hada_w1_a[adapter_name] = Parameter(ops.zeros(shape[0], r))
            self.hada_w1_b[adapter_name] = Parameter(ops.zeros(r, shape[1]))

            self.hada_w2_a[adapter_name] = Parameter(ops.zeros(shape[0], r))
            self.hada_w2_b[adapter_name] = Parameter(ops.zeros(r, shape[1]))

    def reset_adapter_parameters(self, adapter_name: str):
        # Original implementation performs initialization with normal distribution
        # https://github.com/KohakuBlueleaf/LyCORIS/blob/3549fdef8f564761d68b695a08ef88b1122fdedc/lycoris/modules/loha.py#L158

        # FedPara paper proposes to perform He initialization, let's stick with it
        # It is enough to initialize only single matrix with zeros to make adapter do nothing after initialization
        if adapter_name in self.hada_w1_a.keys():
            nn.init.kaiming_uniform_(self.hada_w1_a[adapter_name], a=math.sqrt(5))
            nn.init.kaiming_uniform_(self.hada_w1_b[adapter_name], a=math.sqrt(5))
            nn.init.kaiming_uniform_(self.hada_w2_a[adapter_name], a=math.sqrt(5))
            nn.init.zeros_(self.hada_w2_b[adapter_name])
        if adapter_name in self.hada_t1.keys():
            nn.init.kaiming_uniform_(self.hada_t1[adapter_name], a=math.sqrt(5))
            nn.init.kaiming_uniform_(self.hada_t2[adapter_name], a=math.sqrt(5))

    def reset_adapter_parameters_random(self, adapter_name: str):
        # Original implementation performs initialization with normal distribution
        # https://github.com/KohakuBlueleaf/LyCORIS/blob/3549fdef8f564761d68b695a08ef88b1122fdedc/lycoris/modules/loha.py#L158

        # FedPara paper proposes to perform He initialization, let's stick with it
        # It is enough to initialize only single matrix with zeros to make adapter do nothing after initialization
        if adapter_name in self.hada_w1_a.keys():
            nn.init.kaiming_uniform_(self.hada_w1_a[adapter_name], a=math.sqrt(5))
            nn.init.kaiming_uniform_(self.hada_w1_b[adapter_name], a=math.sqrt(5))
            nn.init.kaiming_uniform_(self.hada_w2_a[adapter_name], a=math.sqrt(5))
            nn.init.kaiming_uniform_(self.hada_w2_b[adapter_name], a=math.sqrt(5))
        if adapter_name in self.hada_t1.keys():
            nn.init.kaiming_uniform_(self.hada_t1[adapter_name], a=math.sqrt(5))
            nn.init.kaiming_uniform_(self.hada_t2[adapter_name], a=math.sqrt(5))

    def update_layer(
        self,
        adapter_name: str,
        r: int,
        alpha: float,
        rank_dropout: float,
        module_dropout: float,
        init_weights: bool,
        use_effective_conv2d: bool = False,
        **kwargs,
    ) -> None:
        """Internal function to create loha adapter

        Args:
            adapter_name (`str`): Name for the adapter to add.
            r (`int`): Rank for the added adapter.
            alpha (`float`): Alpha for the added adapter.
            rank_dropout (`float`): The dropout probability for rank dimension during training.
            module_dropout (`float`): The dropout probability for disabling adapter during training.
            init_weights (`bool`): Whether to initialize weights.
            use_effective_conv2d (`bool`, *optional*, defaults to `False`):
                Use parameter effective decomposition for Conv2d with ksize > 1.
        """
        if r <= 0:
            raise ValueError(
                f"`r` should be a positive integer value but the value passed is {r}"
            )

        self.r[adapter_name] = r
        self.alpha[adapter_name] = alpha
        self.scaling[adapter_name] = alpha / r
        self.rank_dropout[adapter_name] = rank_dropout
        self.module_dropout[adapter_name] = module_dropout

        # Determine shape of LoHa weights
        base_layer = self.get_base_layer()
        if isinstance(base_layer, nn.Linear):
            shape = tuple(base_layer.weight.shape)
        elif isinstance(base_layer, nn.Conv2d):
            use_effective_conv2d = use_effective_conv2d and base_layer.kernel_size != (
                1,
                1,
            )
            if use_effective_conv2d:
                shape = (
                    base_layer.out_channels,
                    base_layer.in_channels,
                    *base_layer.kernel_size,
                )
            else:
                shape = (
                    base_layer.out_channels,
                    base_layer.in_channels
                    * base_layer.kernel_size[0]
                    * base_layer.kernel_size[1],
                )
        else:
            raise TypeError(
                f"LoHa is not implemented for base layers of type {type(base_layer).__name__}"
            )

        # Create weights with provided shape
        self.create_adapter_parameters(adapter_name, r, shape)

        # Initialize weights
        if init_weights:
            self.reset_adapter_parameters(adapter_name)
        else:
            self.reset_adapter_parameters_random(adapter_name)
        # TODO
        self.set_adapter(self.active_adapters)

    def get_delta_weight(self, adapter_name: str) -> mindspore.Tensor:
        r"""
        This method calculates the delta weight for a given adapter.
        
        Args:
            self: The LoHaLayer object.
            adapter_name (str): The name of the adapter for which to calculate the delta weight.
        
        Returns:
            mindspore.Tensor: The delta weight tensor calculated for the specified adapter.
        
        Raises:
            None.
        
        """
        # https://github.com/KohakuBlueleaf/LyCORIS/blob/eb460098187f752a5d66406d3affade6f0a07ece/lycoris/modules/loha.py#L178
        if adapter_name in self.hada_t1.keys():
            weight = make_weight_cp(
                self.hada_t1[adapter_name],
                self.hada_w1_a[adapter_name],
                self.hada_w1_b[adapter_name],
                self.hada_t2[adapter_name],
                self.hada_w2_a[adapter_name],
                self.hada_w2_b[adapter_name],
                scale=mindspore.tensor(self.scaling[adapter_name]),
            )
        else:
            weight = make_weight(
                self.hada_w1_a[adapter_name],
                self.hada_w1_b[adapter_name],
                self.hada_w2_a[adapter_name],
                self.hada_w2_b[adapter_name],
                scale=mindspore.tensor(self.scaling[adapter_name]),
            )

        base_layer = self.get_base_layer()
        weight = weight.reshape(base_layer.weight.shape)

        # Perform rank dropout during training - drop rows of addition weights
        rank_dropout = self.rank_dropout[adapter_name]
        if self.training and rank_dropout:
            drop = (ops.rand(weight.size(0)) > rank_dropout).to(weight.dtype)
            drop = drop.view(-1, *[1] * len(weight.shape[1:]))
            # TODO: Investigate if there should be a scaler like in normal dropout during training
            # Original implementation doesn't have it
            # https://github.com/KohakuBlueleaf/LyCORIS/blob/eb460098187f752a5d66406d3affade6f0a07ece/lycoris/modules/loha.py#L193
            drop /= drop.mean()
            weight *= drop
        return weight

    def forward(self, x: mindspore.Tensor, *args, **kwargs) -> mindspore.Tensor:
        r"""
        This method forwards the output tensor by applying various operations based on the input tensor and layer configurations.
        
        Args:
            self: An instance of the LoHaLayer class.
            x (mindspore.Tensor): The input tensor on which the operations will be applied.
        
        Returns:
            mindspore.Tensor: The output tensor after applying the specified operations.
        
        Raises:
            None
        """
        previous_dtype = x.dtype

        if self.disable_adapters:
            if self.merged:
                self.unmerge()
            result = self.base_layer(x, *args, **kwargs)
        elif self.merged:
            result = self.base_layer(x, *args, **kwargs)
        else:
            result = self.base_layer(x, *args, **kwargs)

            # Execute all the adapters
            for active_adapter in self.active_adapters:
                if active_adapter not in self._available_adapters:
                    continue

                module_dropout = self.module_dropout[active_adapter]

                # Modify current execution weights
                if (not self.training) or (
                    self.training and ops.rand(1) > module_dropout
                ):
                    result = result + self._get_delta_activations(
                        active_adapter, x, *args, **kwargs
                    )

        result = result.to(previous_dtype)
        return result


class Linear(LoHaLayer):
    """LoHa implemented in Linear layer"""
    def __init__(
        self,
        base_layer: nn.Module,
        adapter_name: str = "default",
        r: int = 0,
        alpha: float = 0.0,
        rank_dropout: float = 0.0,
        module_dropout: float = 0.0,
        init_weights: bool = True,
        **kwargs,
    ):
        r"""
        __init__
        
        Initializes the Linear class.
        
        Args:
            self: The instance of the class itself.
            base_layer (nn.Module): The base layer for the linear adapter.
            adapter_name (str, optional): The name of the adapter. Defaults to 'default'.
            r (int, optional): The value for r. Defaults to 0.
            alpha (float, optional): The value for alpha. Defaults to 0.0.
            rank_dropout (float, optional): The value for rank dropout. Defaults to 0.0.
            module_dropout (float, optional): The value for module dropout. Defaults to 0.0.
            init_weights (bool, optional): If True, initializes the weights. Defaults to True.
            **kwargs: Additional keyword arguments.
        
        Returns:
            None: This method does not return any value.
        
        Raises:
            None
        """
        super().__init__(base_layer)

        # Create adapter and set it active
        self._active_adapter = adapter_name
        self.update_layer(
            adapter_name, r, alpha, rank_dropout, module_dropout, init_weights, **kwargs
        )

    def _get_delta_activations(
        self, adapter_name: str, input: mindspore.Tensor, *args: Any, **kwargs: Any
    ) -> mindspore.Tensor:
        r"""
        Get the delta activations for the specified adapter and input.
        
        Args:
            self (Linear): The Linear instance.
            adapter_name (str): The name of the adapter.
            input (mindspore.Tensor): The input tensor.
        
        Returns:
            mindspore.Tensor: The delta activations tensor.
        
        Raises:
            ValueError: If the adapter name is not valid.
            TypeError: If the input tensor is not of type mindspore.Tensor.
        """
        delta_weight = self.get_delta_weight(adapter_name)
        # don't add bias here, because the bias is already included in the output of the base_layer
        return ops.dense(input, delta_weight)

    def __repr__(self) -> str:
        r"""
        Docstring for __repr__ method in the class Linear.
        
        Args:
            self: Linear object. Represents the instance of the Linear class.
            
        Returns:
            str: A string representation of the Linear object with the prefix 'loha.' added to the default representation obtained using super().
        
        Raises:
            No specific exceptions are raised within this method.
        """
        rep = super().__repr__()
        return "loha." + rep


class Conv2d(LoHaLayer):
    """LoHa implemented in Conv2d layer"""
    def __init__(
        self,
        base_layer: nn.Module,
        adapter_name: str = "default",
        r: int = 0,
        alpha: float = 0.0,
        rank_dropout: float = 0.0,
        module_dropout: float = 0.0,
        use_effective_conv2d: bool = False,
        init_weights: bool = True,
        **kwargs,
    ):
        r"""
        Initializes an instance of the Conv2d class.
        
        Args:
            self: The instance of the class.
            base_layer (nn.Module): The base layer used for convolutional operations.
            adapter_name (str, optional): The name of the adapter. Defaults to 'default'.
            r (int, optional): The value of r. Defaults to 0.
            alpha (float, optional): The value of alpha. Defaults to 0.0.
            rank_dropout (float, optional): The value of rank dropout. Defaults to 0.0.
            module_dropout (float, optional): The value of module dropout. Defaults to 0.0.
            use_effective_conv2d (bool, optional): Boolean flag indicating whether to use effective conv2d. Defaults to False.
            init_weights (bool, optional): Boolean flag indicating whether to initialize the weights. Defaults to True.
            **kwargs: Additional keyword arguments.
        
        Returns:
            None: This method does not return any value.
        
        Raises:
            None: This method does not raise any exceptions.
        """
        super().__init__(base_layer)

        # Create adapter and set it active
        self._active_adapter = adapter_name
        self.update_layer(
            adapter_name,
            r,
            alpha,
            rank_dropout,
            module_dropout,
            init_weights,
            use_effective_conv2d,
            **kwargs,
        )

    def _get_delta_activations(
        self, adapter_name: str, input: mindspore.Tensor, *args: Any, **kwargs: Any
    ) -> mindspore.Tensor:
        r"""
        This method calculates the delta activations for the given input tensor using the specified adapter_name.
        
        Args:
            self (Conv2d): The instance of the Conv2d class.
            adapter_name (str): The name of the adapter used to obtain the delta weight.
            input (mindspore.Tensor): The input tensor for which the delta activations need to be calculated.
        
        Returns:
            mindspore.Tensor: A tensor containing the delta activations calculated based on the input and delta weight.
        
        Raises:
            - ValueError: If the adapter_name is invalid or not found.
            - RuntimeError: If there is an issue in obtaining the delta weight or base layer.
            - TypeError: If the input tensor is not of type mindspore.Tensor.
        """
        delta_weight = self.get_delta_weight(adapter_name)
        # don't add bias here, because the bias is already included in the output of the base_layer
        base_layer = self.get_base_layer()
        return ops.conv2d(
            input,
            delta_weight,
            stride=base_layer.stride,
            padding=base_layer.padding,
            dilation=base_layer.dilation,
            groups=base_layer.groups,
        )

    def __repr__(self) -> str:
        r"""
        This method returns a string representation of the object.
        
        Args:
            self: The instance of the Conv2d class.
        
        Returns:
            str: A string representation of the object with the prefix 'loha.'.
        
        Raises:
            No specific exceptions are raised by this method.
        """
        rep = super().__repr__()
        return "loha." + rep


# Below code is a direct copy from https://github.com/KohakuBlueleaf/LyCORIS/blob/eb460098187f752a5d66406d3affade6f0a07ece/lycoris/modules/loha.py#L9


class HadaWeight(nn.Module):

    r"""
    The HadaWeight class represents a module that calculates the Hadamard product of two sets of weights in a neural network. 
    
    This class inherits from nn.Module and provides methods for forwarding the Hadamard product of two weight matrices, backpropagating through the operation, and computing the gradients with respect to the
input weights. 
    
    The forward method computes the Hadamard product of two sets of weights scaled by a specified factor. 
    
    The bprop method calculates the gradients of the input weights with respect to the output gradients, considering the Hadamard product operation and the scaling factor. 
    
    Usage:
        hw = HadaWeight()
        diff_weight = hw.forward(w1a, w1b, w2a, w2b, scale)
        grad_w1a, grad_w1b, grad_w2a, grad_w2b, scale = hw.bprop(w1a, w1b, w2a, w2b, scale, out, dout)
    """
    def forward(self, w1a, w1b, w2a, w2b, scale=mindspore.tensor(1)):
        r"""
        Constructs the Hadamard weight for the given inputs.
        
        Args:
            self (HadaWeight): An instance of the HadaWeight class.
            w1a (tensor): The first weight matrix (w1a) of shape (m, n).
            w1b (tensor): The second weight matrix (w1b) of shape (n, p).
            w2a (tensor): The third weight matrix (w2a) of shape (p, q).
            w2b (tensor): The fourth weight matrix (w2b) of shape (q, r).
            scale (tensor, optional): The scale factor to be applied to the result. Defaults to mindspore.tensor(1).
        
        Returns:
            tensor: The resulting Hadamard weight matrix of shape (m, r).
        
        Raises:
            TypeError: If any of the input parameters (w1a, w1b, w2a, w2b, scale) are not of type 'tensor'.
            ValueError: If the shapes of the weight matrices (w1a, w1b, w2a, w2b) are not compatible.
        """
        diff_weight = ((w1a @ w1b) * (w2a @ w2b)) * scale
        return diff_weight

    def bprop(self, w1a, w1b, w2a, w2b, scale, out, dout):
        r"""
        This method, bprop, is a part of the HadaWeight class and is used for backpropagation calculations. It takes in the following parameters:
        
        Args:
        - self: Represents the instance of the HadaWeight class.
        - w1a: A numpy array representing weights for the first layer's input units.
        - w1b: A numpy array representing weights for the first layer's output units.
        - w2a: A numpy array representing weights for the second layer's input units.
        - w2b: A numpy array representing weights for the second layer's output units.
        - scale: A scalar value used for scaling the gradient.
        - out: A numpy array representing the output of the forward pass.
        - dout: A numpy array representing the gradient of the output layer.
        
        Returns:
        - grad_w1a: A numpy array representing the gradient of the weights w1a.
        - grad_w1b: A numpy array representing the gradient of the weights w1b.
        - grad_w2a: A numpy array representing the gradient of the weights w2a.
        - grad_w2b: A numpy array representing the gradient of the weights w2b.
        - scale: A scalar value representing the updated scale.
        
        Raises:
        - None
        
        """
        dout = dout * scale
        temp = dout * (w2a @ w2b)
        grad_w1a = temp @ w1b.T
        grad_w1b = w1a.T @ temp

        temp = dout * (w1a @ w1b)
        grad_w2a = temp @ w2b.T
        grad_w2b = w2a.T @ temp

        return grad_w1a, grad_w1b, grad_w2a, grad_w2b, scale


class HadaWeightCP(nn.Module):

    r"""
    The HadaWeightCP class represents a cell for performing HadaWeightCP (Hadamard product with weight and channel permutation) operations. This class inherits from nn.Module and provides methods for
forwarding the HadaWeightCP operation and its backward propagation.
    
    The forward method takes input tensors t1, w1a, w1b, t2, w2a, w2b, and optional scale, and returns the result of the HadaWeightCP operation. The HadaWeightCP operation involves performing einsum
operations on the input tensors and scaling the result by the provided scale.
    
    The bprop method takes input tensors t1, w1a, w1b, t2, w2a, w2b, scale, out, and dout, and computes the gradients with respect to the input tensors and weight matrices. The method involves performing
einsum operations and computing gradients for w1a, w1b, t1, w2a, w2b, and t2.
    
    This class is designed to be used as a building block for neural network models that involve HadaWeightCP operations and provides an efficient and optimized implementation for such operations.
    """
    def forward(self, t1, w1a, w1b, t2, w2a, w2b, scale=mindspore.tensor(1)):
        r"""
        Constructs a weighted tensor product using the HadaWeightCP method.
        
        Args:
            self: An instance of the HadaWeightCP class.
            t1: A tensor of shape (i, j, k, l), representing the first input tensor.
            w1a: A tensor of shape (j, r), representing the first weight tensor (a).
            w1b: A tensor of shape (i, p), representing the first weight tensor (b).
            t2: A tensor of shape (i, j, k, l), representing the second input tensor.
            w2a: A tensor of shape (j, r), representing the second weight tensor (a).
            w2b: A tensor of shape (i, p), representing the second weight tensor (b).
            scale: A tensor of shape (), representing the scaling factor applied to the product (default: 1).
        
        Returns:
            None. This method does not return any value.
        
        Raises:
            None.
        """
        rebuild1 = ops.einsum("i j k l, j r, i p -> p r k l", t1, w1b, w1a)
        rebuild2 = ops.einsum("i j k l, j r, i p -> p r k l", t2, w2b, w2a)
        return rebuild1 * rebuild2 * scale

    def bprop(self, t1, w1a, w1b, t2, w2a, w2b, scale, out, dout):
        r"""
        This method calculates the backward propagation for the HadaWeightCP class.
        
        Args:
            self (HadaWeightCP): An instance of the HadaWeightCP class.
            t1 (numpy.ndarray): Input tensor 1.
            w1a (numpy.ndarray): Weight tensor 1a.
            w1b (numpy.ndarray): Weight tensor 1b.
            t2 (numpy.ndarray): Input tensor 2.
            w2a (numpy.ndarray): Weight tensor 2a.
            w2b (numpy.ndarray): Weight tensor 2b.
            scale (float): Scaling factor.
            out (numpy.ndarray): Output tensor.
            dout (numpy.ndarray): Gradient of the output tensor.
        
        Returns:
            Tuple[numpy.ndarray, numpy.ndarray, numpy.ndarray, numpy.ndarray, numpy.ndarray, numpy.ndarray, None]:
            - grad_t1 (numpy.ndarray): Gradient for input tensor 1.
            - grad_w1a (numpy.ndarray): Gradient for weight tensor 1a.
            - grad_w1b (numpy.ndarray): Gradient for weight tensor 1b.
            - grad_t2 (numpy.ndarray): Gradient for input tensor 2.
            - grad_w2a (numpy.ndarray): Gradient for weight tensor 2a.
            - grad_w2b (numpy.ndarray): Gradient for weight tensor 2b.
            - None: This method does not return any value.
        
        Raises:
            None: This method does not raise any exceptions.
        """
        dout = dout * scale

        temp = ops.einsum("i j k l, j r -> i r k l", t2, w2b)
        rebuild = ops.einsum("i j k l, i r -> r j k l", temp, None)

        grad_w = rebuild * dout

        grad_w1a = ops.einsum("r j k l, i j k l -> r i", temp, grad_w)
        grad_temp = ops.einsum("i j k l, i r -> r j k l", grad_w, w1a.T)

        grad_w1b = ops.einsum("i r k l, i j k l -> r j", t1, grad_temp)
        grad_t1 = ops.einsum("i j k l, j r -> i r k l", grad_temp, w1b.T)

        temp = ops.einsum("i j k l, j r -> i r k l", t1, w1b)
        rebuild = ops.einsum("i j k l, i r -> r j k l", temp, w1a)

        grad_w = rebuild * dout

        grad_w2a = ops.einsum("r j k l, i j k l -> r i", temp, grad_w)
        grad_temp = ops.einsum("i j k l, i r -> r j k l", grad_w, w2a.T)

        grad_w2b = ops.einsum("i r k l, i j k l -> r j", t2, grad_temp)
        grad_t2 = ops.einsum("i j k l, j r -> i r k l", grad_temp, w2b.T)
        return (grad_t1, grad_w1a, grad_w1b, grad_t2, grad_w2a, grad_w2b, None)


def make_weight(w1a, w1b, w2a, w2b, scale):
    """
    Args:
        w1a (float): The weight value for the first item in the first set.
        w1b (float): The weight value for the second item in the first set.
        w2a (float): The weight value for the first item in the second set.
        w2b (float): The weight value for the second item in the second set.
        scale (float): The scale factor for the weights.
    
    Returns:
        None: This function does not return any value.
    
    Raises:
        None: This function does not explicitly raise any exceptions.
    """
    hadaweight = HadaWeight()
    return hadaweight(w1a, w1b, w2a, w2b, scale)


def make_weight_cp(t1, w1a, w1b, t2, w2a, w2b, scale):
    r"""
    This function takes in seven parameters: t1, w1a, w1b, t2, w2a, w2b, and scale.
    
    Args:
        t1 (type): The first parameter representing some value.
        w1a (type): The second parameter representing a weight value.
        w1b (type): The third parameter representing another weight value.
        t2 (type): The fourth parameter representing a different value.
        w2a (type): The fifth parameter representing a weight value.
        w2b (type): The sixth parameter representing another weight value.
        scale (type): The seventh parameter representing a scaling factor.
    
    Returns:
        None: This function does not return any value.
    
    Raises:
        None: This function does not raise any exceptions.
    """
    hadaweightcp = HadaWeightCP()
    return hadaweightcp(t1, w1a, w1b, t2, w2a, w2b, scale)
