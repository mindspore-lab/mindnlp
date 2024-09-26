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
# ============================================================================
# pylint: disable=missing-function-docstring
# pylint: disable=missing-class-docstring
# pylint: disable=line-too-long
# pylint: disable=arguments-differ
# pylint: disable=unused-argument
# pylint: disable=too-many-arguments
"IA3 Layer"
import warnings
from typing import Any, List, Optional

from mindspore import Tensor
from mindspore.common.initializer import initializer,  Constant

from mindnlp.core import nn, ops
from mindnlp.core.nn import Parameter
from mindnlp.transformers.ms_utils import Conv1D
from mindnlp.peft.utils import transpose
from mindnlp.core.nn import  ParameterDict

from ..tuners_utils import BaseTunerLayer, check_adapters_to_merge


class IA3Layer(BaseTunerLayer):

    r"""
    The `IA3Layer` class represents a layer used in the IA3 (Incremental Adapters for Adapting Adapters) framework. This class inherits from the `BaseTunerLayer` class.
    
    Attributes:
        base_layer (nn.Module): The base layer of the IA3Layer.
        ia3_l (ParameterDict): A dictionary containing the IA3 layer parameters.
        _disable_adapters (bool): A flag indicating whether adapters are disabled.
        merged_adapters (List): A list of merged adapters.
        is_feedforward (bool): A flag indicating whether the IA3Layer is a feedforward layer.
        in_features (int): The number of input features for the IA3Layer.
        out_features (int): The number of output features for the IA3Layer.
    
    Methods:
        __init__(self, base_layer: nn.Module, is_feedforward: bool, **kwargs) -> None:
            Initializes an instance of the IA3Layer class.
    
        update_layer(self, adapter_name, init_ia3_weights):
            Updates the IA3Layer with the specified adapter name and initializes IA3 weights.
    
        reset_ia3_parameters(self, adapter_name):
            Resets the IA3Layer parameters for the specified adapter name.
    
    """
    # All names of layers that may contain adapter weights
    adapter_layer_names = ("ia3_l",)

    def __init__(self, base_layer: nn.Module, is_feedforward: bool, **kwargs) -> None:
        r"""
        Initialize the IA3Layer class.
        
        Args:
            self: The instance of the IA3Layer class.
            base_layer (nn.Module): The base layer used in the IA3Layer.
                This parameter specifies the base layer (e.g., nn.Linear, nn.Conv2d, nn.Embedding, Conv1D) to be used in the IA3Layer.
            is_feedforward (bool): A boolean flag indicating whether the IA3Layer is feedforward or not.
                Set to True if the IA3Layer is feedforward, False otherwise.
        
        Returns:
            None: This method does not return any value.
        
        Raises:
            ValueError: If the provided base_layer is not supported or of an unsupported type.
        """
        self.base_layer = base_layer
        self.ia3_l = ParameterDict({})
        # Mark the weight as unmerged
        self._disable_adapters = False
        self.merged_adapters = []
        self.is_feedforward = is_feedforward
        base_layer = self.get_base_layer()
        if isinstance(base_layer, nn.Linear):
            in_features, out_features = base_layer.in_features, base_layer.out_features
        elif isinstance(base_layer, nn.Conv2d):
            in_features, out_features = base_layer.in_channels, base_layer.out_channels
        elif isinstance(base_layer, nn.Embedding):
            in_features, out_features = base_layer.vocab_size, base_layer.embedding_size
        elif isinstance(base_layer, Conv1D):
            in_features, out_features = (
                base_layer.weight.ds_shape if hasattr(base_layer.weight, "ds_shape") else base_layer.weight.shape
            )
        else:
            raise ValueError(f"Unsupported layer type {type(base_layer)}")
        self.in_features = in_features
        self.out_features = out_features

    def update_layer(self, adapter_name, init_ia3_weights):
        r"""
        Updates the IA3 layer with the given adapter name and initializes its weights if specified.
        
        Args:
            self (IA3Layer): The IA3Layer instance.
            adapter_name (str): The name of the adapter to update.
            init_ia3_weights (bool): Flag indicating whether to initialize the IA3 weights.
        
        Returns:
            None
        
        Raises:
            None
        """
        # This code works for linear layers, override for other layer types
        # Actual trainable parameters
        if self.is_feedforward:
            weight = ops.randn((1, self.in_features))
        else:
            weight = ops.randn((self.out_features, 1))
        self.ia3_l[adapter_name] = Parameter(weight)
        if init_ia3_weights:
            self.reset_ia3_parameters(adapter_name)
        self.set_adapter(self.active_adapters)

    def reset_ia3_parameters(self, adapter_name):
        r"""
        Resets the IA3 parameters for a given adapter in the IA3Layer.
        
        Args:
            self: The instance of the IA3Layer class.
            adapter_name (str): The name of the adapter whose parameters need to be reset.
        
        Returns:
            None. This method does not return any value.
        
        Raises:
            None.
        
        This method resets the IA3 parameters for the specified adapter by setting its data to a constant value of 1.0 using the initializer function. The adapter_name parameter is used to identify the adapter
in the ia3_l dictionary. If the adapter_name is not found in the dictionary, no action is taken.
        """
        if adapter_name in self.ia3_l.keys():
            # initialize learned vector with torch.ones
            self.ia3_l[adapter_name].assign_value(initializer(
                Constant(1.0),
                self.ia3_l[adapter_name].shape,
                self.ia3_l[adapter_name].dtype
            ))


class Linear(nn.Module, IA3Layer):

    r"""
    The `Linear` class represents a linear layer that inherits from `nn.Module` and `IA3Layer`.
    
    Summary:
        This class implements a linear layer that can merge and unmerge adapter weights into the base weights.
    
    Attributes:
        - `base_layer`: An instance of `nn.Module` representing the base layer.
        - `adapter_name`: A string specifying the active adapter name.
        - `fan_in_fan_out`: A boolean indicating whether to transpose the adapter weights.
        - `is_feedforward`: A boolean indicating whether the layer is feedforward.
        - `is_target_conv_1d_layer`: A boolean indicating whether the layer is a target convolutional 1D layer.
        - `init_ia3_weights`: A boolean indicating whether to initialize IA3 weights.
        - `merged_adapters`: A list of merged adapter names.
    
    Methods:
        - `__init__(self, base_layer: nn.Module, adapter_name: str, fan_in_fan_out: bool = False, is_feedforward: bool = False, is_target_conv_1d_layer: bool = False, init_ia3_weights: bool = True, **kwargs) ->
None`:
            Initializes a `Linear` instance with the given parameters.
    
        - `merge(self, safe_merge: bool = False, adapter_names: Optional[List[str]] = None) -> None`:
            Merges the active adapter weights into the base weights.
    
        - `unmerge(self) -> None`:
            Unmerges all merged adapter layers from the base weights.
    
        - `forward(self, x: Tensor, *args, **kwargs) -> Tensor`:
            Constructs the linear layer with the given input tensor.
    
    Note:
        - The `merge` method merges the active adapter weights into the base weights, allowing for adaptation.
        - The `unmerge` method unmerges all merged adapter layers from the base weights.
        - The `forward` method forwards the linear layer, taking into account adapter weights if applicable.
    
    """
    # (IA)^3 implemented in a dense layer
    def __init__(
        self,
        base_layer: nn.Module,
        adapter_name: str,
        fan_in_fan_out: bool = False,  # Set this to True if the layer to replace stores weight like (fan_in, fan_out)
        is_feedforward: bool = False,  # Set to True if the layer is treated as a feedforward layer
        is_target_conv_1d_layer: bool = False,  # whether target cell is a conv1d layer. useful while unloading later
        init_ia3_weights: bool = True,  # whether to initialize IA3 weights
        **kwargs,
    ) -> None:
        r"""
        Initializes a Linear object.
        
        Args:
            self: The instance of the Linear class.
            base_layer (nn.Module): The base layer to be used for the Linear layer.
            adapter_name (str): The name of the adapter.
            fan_in_fan_out (bool): A flag indicating whether to use fan-in/fan-out weights.
            is_feedforward (bool): A flag indicating whether the layer is feedforward.
            is_target_conv_1d_layer (bool): A flag indicating whether the layer is a 1D convolutional layer.
            init_ia3_weights (bool): A flag indicating whether to initialize IA3 weights.
            **kwargs: Additional keyword arguments.
        
        Returns:
            None. This method does not return any value.
        
        Raises:
            None.
        """
        super().__init__()
        IA3Layer.__init__(self, base_layer, is_feedforward=is_feedforward)
        self.fan_in_fan_out = fan_in_fan_out
        self.is_target_conv_1d_layer = is_target_conv_1d_layer
        self._active_adapter = adapter_name
        self.update_layer(adapter_name, init_ia3_weights)

    def merge(self, safe_merge: bool = False, adapter_names: Optional[List[str]] = None) -> None:
        """
        Merge the active adapter weights into the base weights

        Args:
            safe_merge (`bool`, *optional*):
                If True, the merge operation will be performed in a copy of the original weights and check for NaNs
                before merging the weights. This is useful if you want to check if the merge operation will produce
                NaNs. Defaults to `False`.
            adapter_names (`List[str]`, *optional*):
                The list of adapter names that should be merged. If None, all active adapters will be merged. Defaults
                to `None`.
        """
        adapter_names = check_adapters_to_merge(self, adapter_names)
        if not adapter_names:
            # no adapter to merge
            return

        for active_adapter in adapter_names:
            if active_adapter in self.ia3_l.keys():
                base_layer = self.get_base_layer()
                ia3_l = transpose(self.ia3_l[active_adapter].data, self.fan_in_fan_out)
                if safe_merge:
                    orig_weights = base_layer.weight.data
                    orig_weights = ops.mul(orig_weights, ia3_l)

                    if not ops.isfinite(orig_weights).all():
                        raise ValueError(
                            f"NaNs detected in the merged weights. The adapter {active_adapter} seems to be broken"
                        )
                    base_layer.weight.data = orig_weights
                else:
                    base_layer.weight.data = ops.mul(base_layer.weight.data, ia3_l)

                if not self.is_feedforward and (base_layer.bias is not None):
                    scaling = self.ia3_l[active_adapter].reshape(base_layer.bias.shape)
                    base_layer.bias.data = ops.mul(base_layer.bias.data, scaling.data)

                self.merged_adapters.append(active_adapter)

    def unmerge(self) -> None:
        """
        This method unmerges all merged adapter layers from the base weights.
        """
        if not self.merged:
            warnings.warn("Already unmerged. Nothing to do.")
            return

        warnings.warn("Unmerge result can be inaccurate for (IA)^3.")
        while len(self.merged_adapters) > 0:
            active_adapter = self.merged_adapters.pop()
            if active_adapter in self.ia3_l.keys():
                base_layer = self.get_base_layer()
                # Add tolerace to avoid division by zero
                ia3_l = transpose(self.ia3_l[active_adapter].data, self.fan_in_fan_out) + 1e-8
                base_layer.weight.data = ops.div(base_layer.weight.data, ia3_l)

                if not self.is_feedforward and (base_layer.bias is not None):
                    scaling = self.ia3_l[active_adapter].reshape(base_layer.bias.shape)
                    base_layer.bias.data = ops.div(base_layer.bias.data, scaling.data + 1e-8)

    def forward(self, x: Tensor, *args: Any, **kwargs: Any) -> Tensor:
        r"""
        This method forwards a tensor using the input tensor 'x' and additional arguments and keyword arguments. It adapts the input tensor based on the configuration of the Linear class, including the use
of adapters and merging layers.
        
        Args:
            x (Tensor): The input tensor to be processed. It should be of the type Tensor.
            *args: Additional positional arguments that can be passed to the method.
            **kwargs: Additional keyword arguments that can be passed to the method.
        
        Returns:
            Tensor: The forwarded tensor based on the input 'x' and the configuration of the Linear class.
        
        Raises:
            None: This method does not explicitly raise any exceptions.
        """
        dtype = previous_dtype = x.dtype
        if self.disable_adapters:
            if self.merged:
                self.unmerge()
            result = self.base_layer(x, *args, **kwargs)
        elif self.merged:
            result = self.base_layer(x, *args, **kwargs)
        else:
            ia3_scaling = 1
            for active_adapter in self.active_adapters:
                if active_adapter not in self.ia3_l.keys():
                    continue
                dtype = self.ia3_l[active_adapter].dtype
                ia3_scaling *= self.ia3_l[active_adapter].flatten()

            if self.is_feedforward:
                x = x.to(dtype)
                interm = (x * ia3_scaling).to(self.get_base_layer().weight.dtype)
                result = self.base_layer(interm, *args, **kwargs)
            else:
                result = self.base_layer(x, *args, **kwargs)
                result = result.to(dtype) * ia3_scaling

        result = result.to(previous_dtype)
        return result


class Conv2d(nn.Module, IA3Layer):

    r"""
    The Conv2d class represents a convolutional neural network layer with adaptive scaling capabilities for adapter layers. 
    This class inherits from nn.Module and IA3Layer, allowing for flexible integration with existing neural network architectures. 
    The class provides methods for updating, merging, and unmerging adapter layers, as well as forwarding the final output based on the input tensor.
    
    Methods:
    - __init__: Initialize the Conv2d layer with specified parameters and adapter settings.
    - update_layer: Update the adapter layer with new weights based on the provided adapter name.
    - merge: Merge active adapter weights into the base weights with optional safe merge checks.
    - unmerge: Unmerge all previously merged adapter layers from the base weights.
    - forward: Construct the output tensor based on the input tensor, considering adapter scaling and merging configurations.
    
    Note: The Conv2d class is designed to enhance neural network models with adaptive scaling functionality for improved performance and flexibility.
    """
    def __init__(
        self,
        base_layer: nn.Module,
        adapter_name: str,
        fan_in_fan_out: bool = False,  # Set this to True if the layer to replace stores weight like (fan_in, fan_out)
        is_feedforward: bool = False,  # Set to True if the layer is treated as a feedforward layer
        init_ia3_weights: bool = True,
        **kwargs,
    ) -> None:
        r"""
        Initializes a new instance of the Conv2d class.
        
        Args:
            self (Conv2d): The current instance of the Conv2d class.
            base_layer (nn.Module): The base layer for the Conv2d operation.
            adapter_name (str): The name of the adapter.
            fan_in_fan_out (bool, optional): Flag indicating whether to use fan-in/fan-out initialization. Defaults to False.
            is_feedforward (bool, optional): Flag indicating whether the Conv2d operation is feedforward. Defaults to False.
            init_ia3_weights (bool, optional): Flag indicating whether to initialize IA3 weights. Defaults to True.
            **kwargs: Additional keyword arguments.
        
        Returns:
            None
        
        Raises:
            None
        """
        super().__init__()
        IA3Layer.__init__(self, base_layer, is_feedforward=is_feedforward)
        self.fan_in_fan_out = fan_in_fan_out
        self._active_adapter = adapter_name

        self.update_layer(adapter_name, init_ia3_weights)

    def update_layer(self, adapter_name, init_ia3_weights):
        r"""
        Updates the layer of the Conv2d class with the specified adapter name and initialization of IA3 weights.
        
        Args:
            self (Conv2d): The instance of the Conv2d class.
            adapter_name (str): The name of the adapter to be updated.
            init_ia3_weights (bool): Indicates whether to initialize IA3 weights or not.
        
        Returns:
            None
        
        Raises:
            None
        """
        # Actual trainable parameters
        if self.is_feedforward:
            weight = ops.randn((1, self.in_features, 1, 1))
        else:
            weight = ops.randn((1, self.out_features, 1, 1))
        self.ia3_l[adapter_name] = Parameter(weight)
        if init_ia3_weights:
            self.reset_ia3_parameters(adapter_name)
        self.set_adapter(self.active_adapters)

    def merge(self, safe_merge: bool = False, adapter_names: Optional[List[str]] = None) -> None:
        """
        Merge the active adapter weights into the base weights

        Args:
            safe_merge (`bool`, *optional*):
                If True, the merge operation will be performed in a copy of the original weights and check for NaNs
                before merging the weights. This is useful if you want to check if the merge operation will produce
                NaNs. Defaults to `False`.
            adapter_names (`List[str]`, *optional*):
                The list of adapter names that should be merged. If None, all active adapters will be merged. Defaults
                to `None`.
        """
        adapter_names = check_adapters_to_merge(self, adapter_names)
        if not adapter_names:
            # no adapter to merge
            return

        for active_adapter in adapter_names:
            if active_adapter in self.ia3_l.keys():
                base_layer = self.get_base_layer()
                ia3_scaling = self.ia3_l[active_adapter].data
                if not self.is_feedforward:
                    ia3_scaling = ia3_scaling.permute(1, 0, 2, 3)

                if safe_merge:
                    output_weight = ops.mul(base_layer.weight.data, ia3_scaling).clone()

                    if not ops.isfinite(output_weight).all():
                        raise ValueError(
                            f"NaNs detected in the merged weights. The adapter {active_adapter} seems to be broken"
                        )

                    base_layer.weight.data = output_weight
                else:
                    base_layer.weight.data = ops.mul(base_layer.weight.data, ia3_scaling)

                if not self.is_feedforward and (base_layer.bias is not None):
                    scaling = self.ia3_l[active_adapter].reshape(base_layer.bias.shape)
                    base_layer.bias.data = ops.mul(base_layer.bias.data, scaling.data)

                self.merged_adapters.append(active_adapter)

    def unmerge(self) -> None:
        """
        This method unmerges all merged adapter layers from the base weights.
        """
        if not self.merged:
            warnings.warn("Already unmerged. Nothing to do.")
            return

        warnings.warn("Unmerge result can be inaccurate for (IA)^3.")
        while len(self.merged_adapters) > 0:
            active_adapter = self.merged_adapters.pop()
            if active_adapter in self.ia3_l.keys():
                base_layer = self.get_base_layer()
                # divide by (IA)^3 vector. Add tolerace to avoid division by zero
                ia3_scaling = self.ia3_l[active_adapter].data
                if not self.is_feedforward:
                    ia3_scaling = ia3_scaling.permute(1, 0, 2, 3)
                base_layer.weight.data = ops.div(base_layer.weight.data, ia3_scaling + 1e-8)

                if not self.is_feedforward and (base_layer.bias is not None):
                    scaling = self.ia3_l[active_adapter].reshape(base_layer.bias.shape)
                    base_layer.bias.data = ops.mul(base_layer.bias.data, scaling.data)

    def forward(self, x: Tensor, *args: Any, **kwargs: Any) -> Tensor:
        r"""
        Construct method for the Conv2d class.
        
        Args:
            self: The instance of the Conv2d class.
            x (Tensor): The input tensor representing the input data. It is the primary input to the forward method.
        
        Returns:
            Tensor: The output tensor after processing the input data through the forward method. The type and shape of the tensor is dependent on the operation performed within the method.
        
        Raises:
            N/A
        """
        dtype = previous_dtype = x.dtype

        if self.disable_adapters:
            if self.merged:
                self.unmerge()
            result = self.base_layer(x, *args, **kwargs)
        elif self.merged:
            result = self.base_layer(x, *args, **kwargs)
        else:
            ia3_scaling = 1
            for active_adapter in self.active_adapters:
                if active_adapter not in self.ia3_l.keys():
                    continue
                dtype = self.ia3_l[active_adapter].dtype
                ia3_scaling *= self.ia3_l[active_adapter]

            if self.is_feedforward:
                x = x.to(dtype)
                interm = (x * ia3_scaling).to(self.get_base_layer().weight.dtype)
                result = self.base_layer(interm, *args, **kwargs)
            else:
                result = self.base_layer(x, *args, **kwargs)
                result = result.to(dtype) * ia3_scaling

        result = result.to(previous_dtype)
        return result
