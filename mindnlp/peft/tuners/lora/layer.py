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
"""lora layer"""
from __future__ import annotations

import math
import warnings
from typing import Any, Optional, Union

import mindspore
from mindspore import Parameter
from mindnlp.core import nn, ops
from mindnlp.core.nn import ParameterDict, functional as F
from ....transformers.ms_utils import Conv1D
from ..tuners_utils import BaseTunerLayer, check_adapters_to_merge
from ...utils.other import transpose

from .config import LoraConfig


class LoraLayer(BaseTunerLayer):

    r"""
    The `LoraLayer` class represents a layer that implements LOcal Response Adjustment (LORA) for neural network models. It inherits from the `BaseTunerLayer` class and provides methods for updating and
scaling the layer's parameters, as well as performing mixed batch forward operations.
    
    Attributes:
        base_layer (nn.Module): The base layer used for computation.
        r (dict): Dictionary of adapter names and associated integer values representing the r parameter in LORA.
        lora_alpha (dict): Dictionary of adapter names and associated float values representing the alpha parameter in LORA.
        scaling (dict): Dictionary of adapter names and associated float values representing the scaling factor in LORA.
        lora_dropout (nn.ModuleDict): Dictionary of adapter names and associated dropout layers used in LORA.
        lora_A (nn.ModuleDict): Dictionary of adapter names and associated nn.Linear layers used in LORA for input transformation.
        lora_B (nn.ModuleDict): Dictionary of adapter names and associated nn.Linear layers used in LORA for output transformation.
        lora_embedding_A (ParameterDict): Dictionary of adapter names and associated parameter dictionaries used in LORA for input embedding.
        lora_embedding_B (ParameterDict): Dictionary of adapter names and associated parameter dictionaries used in LORA for output embedding.
        _disable_adapters (bool): Boolean flag indicating whether adapters are disabled.
        merged_adapters (list): List of merged adapters.
        use_dora (dict): Dictionary of adapter names and associated boolean values indicating whether DoRA (Distributed Orthogonal Random Access) is enabled.
        lora_magnitude_vector (Optional[ParameterDict]): Optional parameter dictionary for storing the magnitude vector in LORA.
        _caches (dict): Dictionary for caching intermediate values during computation.
        kwargs (dict): Additional keyword arguments.
    
    Methods:
        update_layer(adapter_name, r, lora_alpha, lora_dropout, init_lora_weights, use_rslora, use_dora=False): Updates the LORA layer with the specified adapter parameters.
        reset_lora_parameters(adapter_name, init_lora_weights): Resets the LORA layer parameters based on the specified initialization method.
        _get_weight_norm(weight, lora_weight, scaling): Computes the normalized weight using LORA parameters.
        _cache_store(key, value): Stores a value in the cache.
        _cache_pop(key): Retrieves and removes a value from the cache.
        set_scale(adapter, scale): Sets the scaling factor for a specific adapter.
        scale_layer(scale): Scales the layer by the specified factor.
        unscale_layer(scale=None): Unscales the layer by the specified factor or to its original scaling.
        _check_forward_args(x, *args, **kwargs): Checks the compatibility of arguments with the model's configuration and state.
        _mixed_batch_forward(x, *args, adapter_names, **kwargs): Performs a mixed batch forward operation considering the specified adapter names.
    
    Raises:
        ValueError: If unsupported layer types or incorrect adapter configurations are encountered.
    """
    # All names of layers that may contain (trainable) adapter weights
    adapter_layer_names = ("lora_A", "lora_B", "lora_embedding_A", "lora_embedding_B")
    # All names of other parameters that may contain adapter-related parameters
    other_param_names = ("r", "lora_alpha", "scaling", "lora_dropout")

    def __init__(self, base_layer: nn.Module, **kwargs) -> None:
        r"""
        __init__
        
        This method initializes the LoraLayer class.
        
        Args:
            self: LoraLayer object
                The instance of the LoraLayer class.
            base_layer: nn.Module
                The base layer to be used for the LoraLayer. It can be an instance of nn.Linear, nn.Conv2d, nn.Embedding, Conv1D, or other supported layer types.
        
        Returns:
            None
            This method does not return any value.
        
        Raises:
            ValueError
                If the base_layer type is not supported or recognized.
        """
        self.base_layer = base_layer
        self.r = {}
        self.lora_alpha = {}
        self.scaling = {}
        self.lora_dropout = nn.ModuleDict({})
        self.lora_A = nn.ModuleDict({})
        self.lora_B = nn.ModuleDict({})
        # For Embedding layer
        self.lora_embedding_A = ParameterDict({})
        self.lora_embedding_B = ParameterDict({})
        # Mark the weight as unmerged
        self._disable_adapters = False
        self.merged_adapters = []
        self.use_dora: dict[str, bool] = {}
        self.lora_magnitude_vector: Optional[ParameterDict] = None  # for DoRA
        self._caches: dict[str, Any] = {}
        self.kwargs = kwargs

        base_layer = self.get_base_layer()
        if isinstance(base_layer, nn.Linear):
            in_features, out_features = base_layer.in_features, base_layer.out_features
        elif isinstance(base_layer, nn.Conv2d):
            in_features, out_features = base_layer.in_channels, base_layer.out_channels
        elif isinstance(base_layer, nn.Embedding):
            in_features, out_features = base_layer.num_embeddings, base_layer.embedding_dim
        elif isinstance(base_layer, Conv1D):
            in_features, out_features = (
                base_layer.weight.ds_shape if hasattr(base_layer.weight, "ds_shape") else base_layer.weight.shape
            )
        elif hasattr(base_layer, "infeatures") and hasattr(base_layer, "outfeatures"):
            # QuantLinear
            in_features, out_features = base_layer.infeatures, base_layer.outfeatures
        elif hasattr(base_layer, "input_size") and hasattr(base_layer, "output_size"):
            # Megatron ColumnParallelLinear,RowParallelLinear
            in_features, out_features = base_layer.input_size, base_layer.output_size
        elif hasattr(base_layer, "codebooks") and base_layer.__class__.__name__ == "QuantizedLinear":
            # AQLM QuantLinear
            in_features, out_features = base_layer.in_features, base_layer.out_features
        elif hasattr(base_layer, "w_bit") and base_layer.__class__.__name__ == "WQLinear_GEMM":
            # Awq layers
            in_features, out_features = base_layer.in_features, base_layer.out_features
        elif base_layer.__class__.__name__ == "EetqLinear":
            # Eetq layers
            in_features, out_features = base_layer.in_features, base_layer.out_features
        elif hasattr(base_layer, "W_q") and base_layer.__class__.__name__ == "HQQLinear":
            # HQQ layers
            in_features, out_features = base_layer.in_features, base_layer.out_features
        else:
            raise ValueError(f"Unsupported layer type {type(base_layer)}")

        self.in_features = in_features
        self.out_features = out_features

    def update_layer(
        self, adapter_name, r, lora_alpha, lora_dropout, init_lora_weights, use_rslora, use_dora: bool = False
    ):
        r"""
        Update the layer configuration for the specified adapter in the LoraLayer class.
        
        Args:
            self (LoraLayer): The LoraLayer instance.
            adapter_name (str): The name of the adapter to be updated.
            r (int): The number of units in the layer. Should be a positive integer.
            lora_alpha (float): The alpha value for Lora scaling.
            lora_dropout (float): The dropout rate for the Lora layer. Should be in the range [0.0, 1.0].
            init_lora_weights (str or bool): The method for initializing Lora weights. Can be 'loftq' or a boolean value.
            use_rslora (bool): Flag indicating whether to use RS-Lora scaling.
            use_dora (bool, optional): Flag indicating whether to use Dora. Defaults to False.
        
        Returns:
            None. The method updates the internal state of the LoraLayer instance.
        
        Raises:
            ValueError: If the value of 'r' is not a positive integer.
        """
        # This code works for linear layers, override for other layer types
        if r <= 0:
            raise ValueError(f"`r` should be a positive integer value but the value passed is {r}")

        self.r[adapter_name] = r
        self.lora_alpha[adapter_name] = lora_alpha
        if lora_dropout > 0.0:
            lora_dropout_layer = nn.Dropout(p=lora_dropout)
        else:
            lora_dropout_layer = nn.Identity()

        self.lora_dropout.update(nn.ModuleDict({adapter_name: lora_dropout_layer}))
        # Actual trainable parameters
        self.lora_A[adapter_name] = nn.Linear(self.in_features, r, bias=False)
        self.lora_B[adapter_name] = nn.Linear(r, self.out_features, bias=False)
        if use_rslora:
            self.scaling[adapter_name] = lora_alpha / math.sqrt(r)
        else:
            self.scaling[adapter_name] = lora_alpha / r

        if init_lora_weights == "loftq":
            self.loftq_init(adapter_name)
        elif init_lora_weights:
            self.reset_lora_parameters(adapter_name, init_lora_weights)

        # check weight and qweight (for GPTQ)
        for weight_name in ("weight", "qweight"):
            weight = getattr(self.get_base_layer(), weight_name, None)
            if weight is not None:
                # the layer is already completely initialized, this is an update
                if ops.is_floating_point(weight) or ops.is_complex(weight):
                    for param in self.parameters():
                        param.set_data(param.astype(weight.dtype))
                break

        if use_dora:
            self.dora_init(adapter_name)
            self.use_dora[adapter_name] = True
        else:
            self.use_dora[adapter_name] = False

        self.set_adapter(self.active_adapters)

    def reset_lora_parameters(self, adapter_name, init_lora_weights):
        r"""
        Reset the LoRa parameters for a given adapter.
        
        Args:
            self (object): The instance of the LoraLayer class.
            adapter_name (str): The name of the LoRa adapter for which parameters need to be reset.
            init_lora_weights (bool/str): Specifies the type of initialization for LoRa weights.
                If False, no initialization is performed.
                If True, HeUniform initialization with sqrt(5) is applied.
                If 'gaussian', Normal initialization with a scale of 1 divided by r[adapter_name] is used.
        
        Returns:
            None. This method does not return any value.
        
        Raises:
            ValueError: If the init_lora_weights parameter is not recognized or has an unsupported value.
        """
        if init_lora_weights is False:
            return

        if adapter_name in self.lora_A.keys():
            if init_lora_weights is True:
                # initialize A the same way as the default for nn.Linear and B to zero
                # https://github.com/microsoft/LoRA/blob/a0a92e0f26c067cf94747bdbf1ce73793fa44d19/loralib/layers.py#L124
                nn.init.kaiming_uniform_(self.lora_A[adapter_name].weight, a=math.sqrt(5))
            elif init_lora_weights.lower() == "gaussian":
                nn.init.normal_(self.lora_A[adapter_name].weight, std=1 / self.r[adapter_name])
            else:
                raise ValueError(f"Unknown initialization {init_lora_weights}")
            nn.init.zeros_(self.lora_B[adapter_name].weight)
        if adapter_name in self.lora_embedding_A.keys():
            # initialize a the same way as the default for nn.Linear and b to zero
            nn.init.zeros_(self.lora_embedding_A[adapter_name])
            nn.init.normal_(self.lora_embedding_B[adapter_name])

    def _get_weight_norm(self, weight, lora_weight, scaling) -> mindspore.Tensor:
        r"""
        This method calculates the normalized weight for the LoraLayer.
        
        Args:
            self (LoraLayer): The instance of the LoraLayer class.
            weight (mindspore.Tensor): The weight tensor to be normalized.
            lora_weight (mindspore.Tensor): The Lora weight tensor to be added to the weight.
            scaling (float): The scaling factor to be applied to the lora_weight before adding to the weight.
        
        Returns:
            mindspore.Tensor: The normalized weight tensor after applying the LoraLayer normalization process.
        
        Raises:
            ValueError: If the weight or lora_weight tensors are invalid or incompatible for normalization.
            TypeError: If the input types are not as expected.
        """
        # calculate L2 norm of weight matrix, column-wise
        weight = transpose(weight, self.fan_in_fan_out)
        weight = weight + scaling * lora_weight
        weight_norm = F.normalize(weight, dim=1).to(weight.dtype)
        return weight_norm

    def _cache_store(self, key: str, value: Any) -> None:
        r"""
        Method _cache_store in the LoraLayer class.
        
        This method stores a key-value pair in the cache.
        
        Args:
            self (LoraLayer): The instance of the LoraLayer class.
            key (str): The key for the cache entry.
            value (Any): The value to be stored in the cache.
        
        Returns:
            None. This method does not return any value.
        
        Raises:
            No specific exceptions are raised by this method.
        """
        self._caches[key] = value

    def _cache_pop(self, key: str) -> Any:
        r"""
        Method _cache_pop in class LoraLayer.
        
        This method is responsible for popping the value associated with the specified key from the cache.
        
        Args:
            self (LoraLayer): The instance of the LoraLayer class.
            key (str): The key for which the associated value needs to be popped from the cache.
        
        Returns:
            Any: The value associated with the specified key in the cache.
        
        Raises:
            KeyError: If the specified key is not present in the cache.
            Exception: Any other unexpected exceptions during the operation.
        """
        value = self._caches.pop(key)
        return value

    def set_scale(self, adapter, scale):
        r"""
        This method sets the scale for a specific adapter in the LoraLayer class.
        
        Args:
            self (object): The instance of the LoraLayer class.
            adapter (str): The identifier of the adapter for which the scale is to be set.
            scale (float): The scale value to be set for the specified adapter. It is a floating point number.
        
        Returns:
            None: This method does not return any value.
        
        Raises:
            - KeyError: If the specified adapter is not found in the 'scaling' attribute of the LoraLayer instance.
            - ZeroDivisionError: If the scale calculation involves division by zero, such as when the 'r' attribute for the specified adapter is zero.
        """
        if adapter not in self.scaling:
            # Ignore the case where the adapter is not in the layer
            return
        self.scaling[adapter] = scale * self.lora_alpha[adapter] / self.r[adapter]

    def scale_layer(self, scale: float) -> None:
        r"""
        Scale the layer by a specified factor.
        
        Args:
            self (LoraLayer): The instance of the LoraLayer class.
            scale (float): The scaling factor to be applied to the layer. Must be a float value.
            
        Returns:
            None. This method does not return any value.
        
        Raises:
            - TypeError: If the scale parameter is not a float.
        """
        if scale == 1:
            return

        for active_adapter in self.active_adapters:
            if active_adapter not in self.lora_A.keys():
                continue

            self.scaling[active_adapter] *= scale

    def unscale_layer(self, scale=None) -> None:
        r"""
        This method unscales a layer by either calculating a new scaling factor or dividing the current scaling factor by a specified scale value.
        
        Args:
            self (LoraLayer): The instance of the LoraLayer class.
            scale (float, optional): The value by which to divide the current scaling factor. If set to None, a new scaling factor is calculated based on the existing values. Default is None.
        
        Returns:
            None: This method does not return any value.
        
        Raises:
            - KeyError: If the active_adapter is not found in the keys of the lora_A dictionary.
            - ZeroDivisionError: If the scale parameter is 0 and the current scaling factor needs to be divided by it.
        """
        for active_adapter in self.active_adapters:
            if active_adapter not in self.lora_A.keys():
                continue

            if scale is None:
                self.scaling[active_adapter] = self.lora_alpha[active_adapter] / self.r[active_adapter]
            else:
                self.scaling[active_adapter] /= scale

    def _check_forward_args(self, x, *args, **kwargs):
        """Check if the arguments are compatible with the configs and state of the model"""
        adapter_names = kwargs.get("adapter_names", None)
        if adapter_names is None:
            return

        if len(x) != len(adapter_names):
            msg = (
                "Length of `adapter_names` should be the same as the number of inputs, but got "
                f"{len(adapter_names)} and {len(x)} respectively."
            )
            raise ValueError(msg)

        if self.merged:
            # It is unclear what would be the right thing to do if users pass adapter_names and there are merged
            # adapters. Therefore, it is better to raise an error in this case.
            msg = "Cannot pass `adapter_names` when there are merged adapters, please call `unmerge_adapter` first."
            raise ValueError(msg)

        unique_adapters = set(self.active_adapters)
        for adapter_name in unique_adapters:
            if self.use_dora.get(adapter_name, False):
                msg = "Cannot pass `adapter_names` when DoRA is enabled."
                raise ValueError(msg)

    def _mixed_batch_forward(
        self, x: mindspore.Tensor, *args: Any, adapter_names: list[str], **kwargs: Any
    ) -> mindspore.Tensor:
        r""" 
        This method '_mixed_batch_forward' is defined in the class 'LoraLayer' and is responsible for performing mixed batch forward propagation.
        
        Args:
            self (LoraLayer): The instance of the LoraLayer class.
            x (mindspore.Tensor): The input tensor for the forward propagation.
        
        Returns:
            mindspore.Tensor: The output tensor after the forward propagation.
        
        Raises:
            - KeyError: If the specified active_adapter is not found in the self.lora_A keys.
            - TypeError: If the input parameters are not of the expected types.
            - IndexError: If there is an index error while accessing the sub_batch_indices_list.
        
        """
        # This is a special method that handles the case when users pass the argument `adapter_names`. This is an
        # extra argument that allows mixing different adapters in the same batch at inference time.
        result = self.base_layer(x, *args, **kwargs)
        torch_result_dtype = result.dtype

        unique_adapters = set(adapter_names)
        sub_batch_indices_list = []
        for adapter in unique_adapters:
            sub_batch_indices_list.append([index for index, item in enumerate(adapter_names) if item == adapter])

        for i, active_adapter in enumerate(unique_adapters):
            if active_adapter == "__base__":
                continue
            if active_adapter not in self.lora_A.keys():
                continue

            lora_A = self.lora_A[active_adapter]
            lora_B = self.lora_B[active_adapter]
            dropout = self.lora_dropout[active_adapter]
            scaling = self.scaling[active_adapter]

            # getting the sub-batch, passing it to LoRA layers and updating the corresponding indices of the linear
            # layer output
            sub_batch = x[sub_batch_indices_list[i]].to(lora_A.weight.dtype)
            lora_output = lora_B(lora_A(dropout(sub_batch))) * scaling
            result[sub_batch_indices_list[i]] += lora_output.to(torch_result_dtype)

        return result


# Below code is based on https://github.com/microsoft/LoRA/blob/main/loralib/layers.py
# and modified to work with PyTorch FSDP


#  ------------------------------------------------------------------------------------------
#  Copyright (c) Microsoft Corporation. All rights reserved.
#  Licensed under the MIT License (MIT). See LICENSE in the repo root for license information.
#  ------------------------------------------------------------------------------------------


class Linear(nn.Module, LoraLayer):

    r"""
    The Linear class represents a customizable linear layer with support for LoRA (Learned Optimizer Rate Annealing) adapters. This class inherits from the nn.Module and LoraLayer classes. 
    
    The class includes methods for initializing the layer, merging and unmerging adapter weights, computing delta weights for adapters, forwarding the layer's forward pass, and generating a string
representation of the class.
    
    The __init__ method initializes the Linear layer with specified parameters and configures the LoRA adapters. The merge method combines the active adapter weights into the base weights, with an option to
perform a safe merge operation. The unmerge method reverses the merge operation by unmerging all merged adapter layers from the base weights. The get_delta_weight method computes the delta weight for a given
adapter. The forward method applies the forwarded linear layer to input data, with support for adapter-specific adjustments. The __repr__ method returns a string representation of the Linear class prefixed
with 'lora.'.
    """
    # Lora implemented in a dense layer
    def __init__(
        self,
        base_layer,
        adapter_name: str,
        r: int = 0,
        lora_alpha: int = 1,
        lora_dropout: float = 0.0,
        fan_in_fan_out: bool = False,  # Set this to True if the layer to replace stores weight like (fan_in, fan_out)
        is_target_conv_1d_layer: bool = False,
        init_lora_weights: Union[bool, str] = True,
        use_rslora: bool = False,
        use_dora: bool = False,
        **kwargs,
    ) -> None:
        r"""
        Initializes a Linear object.
        
        Args:
            self: The instance of the Linear class.
            base_layer: The base layer to be used for the Linear object.
            adapter_name (str): The name of the adapter.
            r (int): The value of r.
            lora_alpha (int): The alpha value for lora.
            lora_dropout (float): The dropout value for lora.
            fan_in_fan_out (bool): Flag indicating if fan in fan out is enabled.
            is_target_conv_1d_layer (bool): Flag indicating if the layer is the target conv 1D layer.
            init_lora_weights (Union[bool, str]): Flag or string indicating if lora weights should be initialized.
            use_rslora (bool): Flag indicating if RSLora should be used.
            use_dora (bool): Flag indicating if Dora should be used.
            **kwargs: Additional keyword arguments.
        
        Returns:
            None. This method does not return any value.
        
        Raises:
            None.
        """
        super().__init__()
        LoraLayer.__init__(self, base_layer, **kwargs)
        self.fan_in_fan_out = fan_in_fan_out

        self._active_adapter = adapter_name
        self.update_layer(
            adapter_name,
            r,
            lora_alpha=lora_alpha,
            lora_dropout=lora_dropout,
            init_lora_weights=init_lora_weights,
            use_rslora=use_rslora,
            use_dora=use_dora,
        )
        self.is_target_conv_1d_layer = is_target_conv_1d_layer

    def merge(self, safe_merge: bool = False, adapter_names: Optional[list[str]] = None) -> None:
        """
        Merge the active adapter weights into the base weights

        Args:
            safe_merge (`bool`, *optional*):
                If True, the merge operation will be performed in a copy of the original weights and check for NaNs
                before merging the weights. This is useful if you want to check if the merge operation will produce
                NaNs. Defaults to `False`.
            adapter_names (`list[str]`, *optional*):
                The list of adapter names that should be merged. If None, all active adapters will be merged. Defaults
                to `None`.
        """
        adapter_names = check_adapters_to_merge(self, adapter_names)
        if not adapter_names:
            # no adapter to merge
            return

        for active_adapter in adapter_names:
            if active_adapter in self.lora_A.keys():
                base_layer = self.get_base_layer()
                if safe_merge:
                    # Note that safe_merge will be slower than the normal merge
                    # because of the copy operation.
                    orig_weights = base_layer.weight.data.clone()
                    delta_weight = self.get_delta_weight(active_adapter)
                    if not self.use_dora[active_adapter]:
                        orig_weights = orig_weights + delta_weight
                    else:
                        # handle dora
                        # since delta_weight already includes scaling, set it to 1 here
                        weight_norm = self._get_weight_norm(
                            orig_weights, transpose(delta_weight, self.fan_in_fan_out), scaling=1
                        )
                        # We need to cache weight_norm because it has to be based on the original weights. We
                        # cannot calculate it on the fly based on the merged weights when unmerging because its a
                        # different value
                        self._cache_store(f"{active_adapter}-weight_norm", weight_norm)
                        dora_factor = self.lora_magnitude_vector[active_adapter] / weight_norm
                        dora_factor = transpose(dora_factor.view(-1, 1), self.fan_in_fan_out)
                        orig_weights = dora_factor * (orig_weights + delta_weight)

                    if not ops.isfinite(orig_weights).all():
                        raise ValueError(
                            f"NaNs detected in the merged weights. The adapter {active_adapter} seems to be broken"
                        )

                    base_layer.weight.data = orig_weights
                else:
                    delta_weight = self.get_delta_weight(active_adapter)
                    if not self.use_dora[active_adapter]:
                        base_layer.weight.data = base_layer.weight.data + delta_weight
                    else:
                        # handle dora
                        # since delta_weight already includes scaling, set it to 1 here
                        weight_norm = self._get_weight_norm(
                            base_layer.weight, transpose(delta_weight, self.fan_in_fan_out), scaling=1
                        )
                        # We need to cache weight_norm because it has to be based on the original weights. We
                        # cannot calculate it on the fly based on the merged weights when unmerging because its a
                        # different value
                        self._cache_store(f"{active_adapter}-weight_norm", weight_norm)
                        dora_factor = self.lora_magnitude_vector[active_adapter] / weight_norm
                        dora_factor = transpose(dora_factor.view(-1, 1), self.fan_in_fan_out)
                        new_weight = dora_factor * (base_layer.weight.data + delta_weight)
                        base_layer.weight.data = new_weight

                self.merged_adapters.append(active_adapter)

    def unmerge(self) -> None:
        """
        This method unmerges all merged adapter layers from the base weights.
        """
        if not self.merged:
            warnings.warn("Already unmerged. Nothing to do.")
            return
        while len(self.merged_adapters) > 0:
            active_adapter = self.merged_adapters.pop()
            if active_adapter in self.lora_A.keys():
                weight = self.get_base_layer().weight
                delta_weight = self.get_delta_weight(active_adapter)
                if not self.use_dora[active_adapter]:
                    weight.data -= delta_weight
                else:
                    weight_norm = self._cache_pop(f"{active_adapter}-weight_norm")
                    dora_factor = self.lora_magnitude_vector[active_adapter] / weight_norm
                    weight_orig = weight.data / dora_factor.view(-1, 1) - delta_weight
                    weight.data = weight_orig

    def get_delta_weight(self, adapter) -> mindspore.Tensor:
        """
        Compute the delta weight for the given adapter.

        Args:
            adapter (str):
                The name of the adapter for which the delta weight should be computed.
        """
        dtype = self.lora_B[adapter].weight.dtype

        weight_A = self.lora_A[adapter].weight
        weight_B = self.lora_B[adapter].weight

        output_tensor = transpose(weight_B @ weight_A, self.fan_in_fan_out) * self.scaling[adapter]

        return output_tensor

    def forward(self, x: mindspore.Tensor, *args: Any, **kwargs: Any) -> mindspore.Tensor:
        r"""
        Constructs the forward pass of the Linear class.
        
        Args:
            self (Linear): The instance of the Linear class.
            x (mindspore.Tensor): The input tensor to be processed by the forward pass.
        
        Returns:
            mindspore.Tensor: The output tensor resulting from the forward pass.
        
        Raises:
            None.
        """
        self._check_forward_args(x, *args, **kwargs)
        adapter_names = kwargs.pop("adapter_names", None)

        if self.disable_adapters:
            if self.merged:
                self.unmerge()
            result = self.base_layer(x, *args, **kwargs)
        elif adapter_names is not None:
            result = self._mixed_batch_forward(x, *args, adapter_names=adapter_names, **kwargs)
        elif self.merged:
            result = self.base_layer(x, *args, **kwargs)
        else:
            result = self.base_layer(x, *args, **kwargs)
            torch_result_dtype = result.dtype
            for active_adapter in self.active_adapters:
                if active_adapter not in self.lora_A.keys():
                    continue
                lora_A = self.lora_A[active_adapter]
                lora_B = self.lora_B[active_adapter]
                dropout = self.lora_dropout[active_adapter]
                scaling = self.scaling[active_adapter]
                x = x.to(lora_A.weight.dtype)

                if not self.use_dora[active_adapter]:
                    result = result + lora_B(lora_A(dropout(x))) * scaling
                else:
                    x = dropout(x)
                    result = result + self._apply_dora(x, lora_A, lora_B, scaling, active_adapter)

            result = result.to(torch_result_dtype)

        return result

    def __repr__(self) -> str:
        r"""
        This method returns a string representation of the Linear class instance.
        
        Args:
            self (Linear): The instance of the Linear class for which the string representation is being generated.
        
        Returns:
            str: A string representation of the Linear class instance prefixed with 'lora.'.
        
        Raises:
            No specific exceptions are raised by this method.
        """
        rep = super().__repr__()
        return "lora." + rep


class Embedding(nn.Module, LoraLayer):

    r"""
    The 'Embedding' class represents a customizable adapter layer that can be integrated into neural network architectures. It inherits functionalities from the nn.Module and LoraLayer classes, providing a
flexible mechanism for adapting neural network behavior.
    
    The class includes methods for initializing the adapter layer, updating its parameters, merging adapter weights into base weights, unmerging adapter layers, computing delta weights, and performing mixed
batch forward passes. It also allows for embedding computations and the forwardion of the adapted network output.
    
    The 'Embedding' class is designed to enhance neural network performance by introducing adapter layers that can adapt to specific tasks or data characteristics, offering a versatile approach to model
adaptation and specialization.
    """
    # LoRA implemented in a Embedding layer
    def __init__(
        self,
        base_layer: nn.Module,
        adapter_name: str,
        r: int = 0,
        lora_alpha: int = 1,
        lora_dropout: float = 0.0,
        init_lora_weights: Union[bool, str] = True,
        use_rslora: bool = False,
        use_dora: bool = False,
        **kwargs,
    ) -> None:
        r"""
        Initializes an instance of the Embedding class.
        
        Args:
        - self: The instance of the class.
        - base_layer (nn.Module): The base layer to be used for initialization.
        - adapter_name (str): The name of the adapter.
        - r (int): The value of r.
        - lora_alpha (int): The value of lora alpha.
        - lora_dropout (float): The dropout rate for LORA.
        - init_lora_weights (Union[bool, str]): Flag to initialize LORA weights.
        - use_rslora (bool): Flag to indicate if RSLORA should be used.
        - use_dora (bool): Flag to indicate if DORA should be used.
        
        Returns:
        - None: This method does not return any value.
        
        Raises:
        - ValueError: If use_dora is set to True, as the class does not support DoRA yet. It advises to set use_dora to False.
        """
        super().__init__()
        LoraLayer.__init__(self, base_layer)

        if use_dora:
            raise ValueError(f"{self.__class__.__name__} does not support DoRA yet, please set it to False")

        self._active_adapter = adapter_name
        self.update_layer(
            adapter_name,
            r,
            lora_alpha=lora_alpha,
            lora_dropout=lora_dropout,
            init_lora_weights=init_lora_weights,
            use_rslora=use_rslora,
            use_dora=use_dora,
        )

    def update_layer(self, adapter_name, r, lora_alpha, lora_dropout, init_lora_weights, use_rslora, use_dora):
        """
        Updates the layer with the specified parameters for the given adapter.
        
        Args:
            self (Embedding): The instance of the Embedding class.
            adapter_name (str): The name of the adapter to update.
            r (int): The positive integer value representing the dimensionality of the adapter.
            lora_alpha (float): The alpha value for LoRA scaling.
            lora_dropout (float): The dropout probability for the LoRA layer. Should be in the range (0.0, 1.0).
            init_lora_weights (str or bool): The method for initializing LoRA weights. If 'loftq', initialize using loftq method. If True, reset using the provided method.
            use_rslora (bool): True to use RSLoRA scaling, False to use regular LoRA scaling.
            use_dora (bool): The flag to indicate whether DORA (Dynamic Operation Routing for Adapters) is used.
        
        Returns:
            None. The method updates the layer in place.
        
        Raises:
            ValueError: If the value of `r` is not a positive integer.
        """
        if r <= 0:
            raise ValueError(f"`r` should be a positive integer value but the value passed is {r}")

        self.r[adapter_name] = r
        self.lora_alpha[adapter_name] = lora_alpha
        if lora_dropout > 0.0:
            lora_dropout_layer = nn.Dropout(p=lora_dropout)
        else:
            lora_dropout_layer = nn.Identity()

        self.lora_dropout[adapter_name] = lora_dropout_layer
        # Actual trainable parameters
        weight_A = ops.randn((r, self.in_features))
        weight_B = ops.randn((self.out_features, r))
        self.lora_embedding_A[adapter_name] = Parameter(weight_A)
        self.lora_embedding_B[adapter_name] = Parameter(weight_B)
        if use_rslora:
            self.scaling[adapter_name] = lora_alpha / math.sqrt(r)
        else:
            self.scaling[adapter_name] = lora_alpha / r

        if init_lora_weights == "loftq":
            self.loftq_init(adapter_name)
        elif init_lora_weights:
            self.reset_lora_parameters(adapter_name, init_lora_weights)

        base_layer = self.get_base_layer()
        weight = getattr(base_layer, "weight", None)
        if weight is not None:
            # the layer is already completely initialized, this is an update
            self.to(dtype=weight.dtype)

        self.set_adapter(self.active_adapters)

    def merge(self, safe_merge: bool = False, adapter_names: Optional[list[str]] = None) -> None:
        """
        Merge the active adapter weights into the base weights

        Args:
            safe_merge (`bool`, *optional*):
                If True, the merge operation will be performed in a copy of the original weights and check for NaNs
                before merging the weights. This is useful if you want to check if the merge operation will produce
                NaNs. Defaults to `False`.
            adapter_names (`list[str]`, *optional*):
                The list of adapter names that should be merged. If None, all active adapters will be merged. Defaults
                to `None`.
        """
        adapter_names = check_adapters_to_merge(self, adapter_names)
        if not adapter_names:
            # no adapter to merge
            return

        for active_adapter in adapter_names:
            if active_adapter in self.lora_embedding_A.keys():
                base_layer = self.get_base_layer()
                if safe_merge:
                    # Note that safe_merge will be slower than the normal merge
                    # because of the copy operation.
                    orig_weights = base_layer.weight.data.clone()
                    orig_weights = orig_weights + self.get_delta_weight(active_adapter)

                    if not ops.isfinite(orig_weights).all():
                        raise ValueError(
                            f"NaNs detected in the merged weights. The adapter {active_adapter} seems to be broken"
                        )

                    base_layer.weight.data = orig_weights
                else:
                    base_layer.weight.data = base_layer.weight.data + self.get_delta_weight(active_adapter)
                self.merged_adapters.append(active_adapter)

    def unmerge(self) -> None:
        """
        This method unmerges all merged adapter layers from the base weights.
        """
        if not self.merged:
            warnings.warn("Already unmerged. Nothing to do.")
            return
        while len(self.merged_adapters) > 0:
            active_adapter = self.merged_adapters.pop()
            if active_adapter in self.lora_embedding_A.keys():
                self.get_base_layer().weight.data -= self.get_delta_weight(active_adapter)

    def get_delta_weight(self, adapter) -> mindspore.Tensor:
        """
        Compute the delta weight for the given adapter.

        Args:
            adapter (str):
                The name of the adapter for which the delta weight should be computed.
        """
        weight_A = self.lora_embedding_A[adapter]
        weight_B = self.lora_embedding_B[adapter]

        output_tensor = transpose(weight_B @ weight_A, True) * self.scaling[adapter]

        return output_tensor

    def _mixed_batch_forward(
        self, x: mindspore.Tensor, *args: Any, adapter_names: list[str], **kwargs: Any
    ) -> mindspore.Tensor:
        r"""
        This method '_mixed_batch_forward' is defined in the class 'Embedding' and is used to perform a mixed batch forward operation.
        
        Args:
            self: The instance of the 'Embedding' class.
            x (mindspore.Tensor): The input tensor on which the mixed batch forward operation is performed.
        
            *args: Variable length argument list.
            
            adapter_names (list[str]): A list of adapter names which are used to identify unique adapters.
        
            **kwargs: Variable keyword argument list.
        
        Returns:
            mindspore.Tensor: Returns the result of the mixed batch forward operation as a tensor of type 'mindspore.Tensor'.
        
        Raises:
            None
        """
        # This is a special method that handles the case when users pass the argument `adapter_names`. This is an
        # extra argument that allows mixing different adapters in the same batch at inference time.
        result = self.base_layer(x, *args, **kwargs)

        unique_adapters = set(adapter_names)
        sub_batch_indices_list = []
        for adapter in unique_adapters:
            sub_batch_indices_list.append([index for index, item in enumerate(adapter_names) if item == adapter])

        for i, active_adapter in enumerate(unique_adapters):
            if active_adapter == "__base__":
                continue
            if active_adapter not in self.lora_embedding_A.keys():
                continue

            embedding_A = self.lora_embedding_A[active_adapter].T
            embedding_B = self.lora_embedding_B[active_adapter].T
            scaling = self.scaling[active_adapter]

            # getting the sub-batch, passing it to LoRA layers and updating the corresponding indices of the linear
            # layer output
            sub_batch = x[sub_batch_indices_list[i]]
            after_A = self._embed(sub_batch, embedding_A)
            result[sub_batch_indices_list[i]] += (after_A @ embedding_B) * scaling

        return result

    def _embed(self, input: mindspore.Tensor, weight: mindspore.Tensor) -> mindspore.Tensor:
        r"""
        Method _embed in the class Embedding.
        
        This method is responsible for performing embedding using the input and weight tensors.
        
        Args:
            self (Embedding): The instance of the Embedding class.
            input (mindspore.Tensor): The input tensor containing the indices for embedding lookup.
            weight (mindspore.Tensor): The weight tensor containing the embedding vectors.
        
        Returns:
            mindspore.Tensor: A tensor resulting from the embedding lookup operation.
        
        Raises:
            None
        """
        base_layer = self.get_base_layer()
        return F.embedding(
            input,
            weight,
            padding_idx=base_layer.padding_idx,
            max_norm=base_layer.max_norm,
            norm_type=base_layer.norm_type,
            scale_grad_by_freq=base_layer.scale_grad_by_freq,
            # sparse=base_layer.sparse,
        )

    def forward(self, x: mindspore.Tensor, *args: Any, **kwargs: Any) -> mindspore.Tensor:
        r"""
        Constructs the embedding layer.
        
        Args:
            self (Embedding): The instance of the Embedding class.
            x (mindspore.Tensor): The input tensor to be embedded.
        
        Returns:
            mindspore.Tensor: The embedded tensor.
        
        Raises:
            TypeError: If the input arguments are not of the correct type.
            ValueError: If any of the input arguments are invalid or out of range.
            RuntimeError: If an error occurs while embedding the tensor.
        """
        # TODO: no dtype conversion here, unlike in Linear, is that correct?
        self._check_forward_args(x, *args, **kwargs)
        adapter_names = kwargs.pop("adapter_names", None)

        if self.disable_adapters:
            if self.merged:
                self.unmerge()
            result = self.base_layer(x, *args, **kwargs)
        elif adapter_names is not None:
            result = self._mixed_batch_forward(x, *args, adapter_names=adapter_names, **kwargs)
        elif self.merged:
            result = self.base_layer(x, *args, **kwargs)
        else:
            result = self.base_layer(x, *args, **kwargs)
            torch_result_dtype = result.dtype
            for active_adapter in self.active_adapters:
                if active_adapter not in self.lora_embedding_A:
                    continue
                embedding_A = self.lora_embedding_A[active_adapter].T
                embedding_B = self.lora_embedding_B[active_adapter].T
                scaling = self.scaling[active_adapter]
                after_A = self._embed(x, embedding_A)
                result = result + (after_A @ embedding_B) * scaling
            result = result.to(torch_result_dtype)

        return result

    def __repr__(self) -> str:
        r"""
        This method '__repr__' in the class 'Embedding' generates a string representation of the object.
        
        Args:
            self: An instance of the Embedding class.
                Purpose: Represents the current instance of the Embedding class.
                Restrictions: None.
        
        Returns:
            str: A string representation of the object.
                Purpose: Provides a textual representation of the object for debugging and logging purposes.
        
        Raises:
            None.
        """
        rep = super().__repr__()
        return "lora." + rep


class Conv2d(nn.Module, LoraLayer):

    r"""
    Represents a custom Conv2d class that incorporates LoRA (Locally Recurrent Adaptive) functionality for adaptive learning in neural networks. This class inherits from the nn.Module and LoraLayer classes.
    
    Attributes:
        - base_layer (nn.Module): The base layer for the Conv2d operation.
        - adapter_name (str): The name of the adapter associated with the Conv2d operation.
        - r (int): The parameter 'r' representing the number of features in the Conv2d operation.
        - lora_alpha (int): The alpha value used in LoRA operations.
        - lora_dropout (float): The dropout rate for LoRA operations.
        - init_lora_weights (Union[bool, str]): Indicates whether to initialize LoRA weights or use a specific initialization method.
        - use_rslora (bool): Flag indicating whether to use RSLora (Root-Sparse LoRA) functionality.
        - use_dora (bool): Flag indicating whether to use DoRA (Densely Recurrent Adaptive) functionality.
    
    Methods:
        - __init__: Initializes the Conv2d class with specified parameters and initializes LoRA operations.
        - update_layer: Updates the specified adapter with the provided parameters for LoRA operations.
        - merge: Merges the active adapter weights into the base weights, optionally performing a safe merge operation.
        - unmerge: Unmerges all previously merged adapter layers from the base weights.
        - get_delta_weight: Computes the delta weight for a given adapter based on LoRA weights.
        - _get_weight_norm: Computes the norm of the weights based on scaling factors.
        - _apply_dora: Calculates the output with DoRA applied for LoRA operations.
        - forward: Constructs the Conv2d operation, incorporating LoRA functionality based on active adapters.
        - __repr__: Returns a string representation of the Conv2d class prefixed with 'lora.'.
    
    Note: The Conv2d class extends the functionality of the underlying nn.Module and LoraLayer classes by incorporating adaptive learning mechanisms using LoRA operations.
    """
    # Lora implemented in a conv2d layer
    def __init__(
        self,
        base_layer: nn.Module,
        adapter_name: str,
        r: int = 0,
        lora_alpha: int = 1,
        lora_dropout: float = 0.0,
        init_lora_weights: Union[bool, str] = True,
        use_rslora: bool = False,
        use_dora: bool = False,
        **kwargs,
    ) -> None:
        r"""
        Initializes an instance of the Conv2d class.
        
        Args:
            self: The instance of the Conv2d class.
            base_layer (nn.Module): The base layer to be adapted.
            adapter_name (str): The name of the adapter.
            r (int, optional): The value of r. Defaults to 0.
            lora_alpha (int, optional): The value of lora_alpha. Defaults to 1.
            lora_dropout (float, optional): The value of lora_dropout. Defaults to 0.0.
            init_lora_weights (Union[bool, str], optional): The value to initialize Lora weights. Defaults to True.
            use_rslora (bool, optional): Flag to indicate whether to use RSLora. Defaults to False.
            use_dora (bool, optional): Flag to indicate whether to use Dora. Defaults to False.
            **kwargs: Additional keyword arguments.
        
        Returns:
            None
        
        Raises:
            None
        """
        super().__init__()
        LoraLayer.__init__(self, base_layer)

        self._active_adapter = adapter_name
        self.update_layer(
            adapter_name,
            r,
            lora_alpha=lora_alpha,
            lora_dropout=lora_dropout,
            init_lora_weights=init_lora_weights,
            use_rslora=use_rslora,
            use_dora=use_dora,
        )

    def update_layer(self, adapter_name, r, lora_alpha, lora_dropout, init_lora_weights, use_rslora, use_dora):
        r"""
        Update the layer for the Conv2d class with the provided parameters.
        
        Args:
        - self: The instance of the Conv2d class.
        - adapter_name (str): The name of the adapter.
        - r (int): The positive integer value representing the number of features for the adapter.
        - lora_alpha (float): The alpha value for the LORA mechanism.
        - lora_dropout (float): The dropout probability for the LORA mechanism. Should be in the range (0.0, 1.0].
        - init_lora_weights (str or bool): The method to initialize LORA weights. Can be 'loftq' or a boolean value.
        - use_rslora (bool): Flag indicating whether to use RS-LORA scaling.
        - use_dora (bool): Flag indicating whether to use DORA for the adapter.
        
        Returns:
        None. This method updates the Conv2d layer with the specified parameters.
        
        Raises:
        - ValueError: If the value of `r` is less than or equal to 0.
        """
        if r <= 0:
            raise ValueError(f"`r` should be a positive integer value but the value passed is {r}")

        self.r[adapter_name] = r
        self.lora_alpha[adapter_name] = lora_alpha
        if lora_dropout > 0.0:
            lora_dropout_layer = nn.Dropout(p=lora_dropout)
        else:
            lora_dropout_layer = nn.Identity()

        self.lora_dropout[adapter_name] = lora_dropout_layer
        # Actual trainable parameters
        base_layer = self.get_base_layer()
        kernel_size = base_layer.kernel_size
        stride = base_layer.stride
        padding = base_layer.padding
        self.lora_A[adapter_name] = nn.Conv2d(self.in_features, r, kernel_size, stride, padding, bias=False)
        self.lora_B[adapter_name] = nn.Conv2d(r, self.out_features, (1, 1), (1, 1), bias=False)
        if use_rslora:
            self.scaling[adapter_name] = lora_alpha / math.sqrt(r)
        else:
            self.scaling[adapter_name] = lora_alpha / r

        if init_lora_weights == "loftq":
            self.loftq_init(adapter_name)
        elif init_lora_weights:
            self.reset_lora_parameters(adapter_name, init_lora_weights)

        weight = getattr(base_layer, "weight", None)
        if weight is not None:
            # the layer is already completely initialized, this is an update
            self.to(dtype=weight.dtype)

        if use_dora:
            self.dora_init(adapter_name)
            self.use_dora[adapter_name] = True
        else:
            self.use_dora[adapter_name] = False

        self.set_adapter(self.active_adapters)

    def merge(self, safe_merge: bool = False, adapter_names: Optional[list[str]] = None) -> None:
        """
        Merge the active adapter weights inside the base weights

        Args:
            safe_merge (`bool`, *optional*):
                If True, the merge operation will be performed in a copy of the original weights and check for NaNs
                before merging the weights. This is useful if you want to check if the merge operation will produce
                NaNs. Defaults to `False`.
            adapter_names (`list[str]`, *optional*):
                The list of adapter names that should be merged. If None, all active adapters will be merged. Defaults
                to `None`.
        """
        adapter_names = check_adapters_to_merge(self, adapter_names)
        if not adapter_names:
            # no adapter to merge
            return

        for active_adapter in adapter_names:
            if active_adapter in self.lora_A.keys():
                base_layer = self.get_base_layer()
                if safe_merge:
                    # Note that safe_merge will be slower than the normal merge
                    # because of the copy operation.
                    orig_weights = base_layer.weight.data.clone()
                    delta_weight = self.get_delta_weight(active_adapter)

                    if not self.use_dora[active_adapter]:
                        orig_weights = orig_weights + delta_weight
                    else:
                        # handle dora
                        # since delta_weight already includes scaling, set it to 1 here
                        weight_norm = self._get_weight_norm(orig_weights, delta_weight, scaling=1)
                        # We need to cache weight_norm because it has to be based on the original weights. We
                        # cannot calculate it on the fly based on the merged weights when unmerging because its a
                        # different value
                        self._cache_store(f"{active_adapter}-weight_norm", weight_norm)
                        dora_factor = self.lora_magnitude_vector[active_adapter] / weight_norm
                        orig_weights = dora_factor.view(-1, 1, 1, 1) * (orig_weights + delta_weight)

                    if not ops.isfinite(orig_weights).all():
                        raise ValueError(
                            f"NaNs detected in the merged weights. The adapter {active_adapter} seems to be broken"
                        )
                    base_layer.weight.data = orig_weights
                else:
                    delta_weight = self.get_delta_weight(active_adapter)
                    if not self.use_dora[active_adapter]:
                        base_layer.weight.data = base_layer.weight.data + delta_weight
                    else:
                        # handle dora
                        # since delta_weight already includes scaling, set it to 1 here
                        weight_norm = self._get_weight_norm(base_layer.weight, delta_weight, scaling=1)
                        # We need to cache weight_norm because it has to be based on the original weights. We
                        # cannot calculate it on the fly based on the merged weights when unmerging because its a
                        # different value
                        self._cache_store(f"{active_adapter}-weight_norm", weight_norm)
                        dora_factor = self.lora_magnitude_vector[active_adapter] / weight_norm
                        new_weight = dora_factor.view(-1, 1, 1, 1) * (base_layer.weight.data + delta_weight)
                        base_layer.weight.data = new_weight

                self.merged_adapters.append(active_adapter)

    def unmerge(self) -> None:
        """
        This method unmerges all merged adapter layers from the base weights.
        """
        if not self.merged:
            warnings.warn("Already unmerged. Nothing to do.")
            return
        while len(self.merged_adapters) > 0:
            active_adapter = self.merged_adapters.pop()
            if active_adapter in self.lora_A.keys():
                weight = self.get_base_layer().weight
                delta_weight = self.get_delta_weight(active_adapter)
                if not self.use_dora[active_adapter]:
                    weight.data -= delta_weight
                else:
                    weight_norm = self._cache_pop(f"{active_adapter}-weight_norm")
                    dora_factor = self.lora_magnitude_vector[active_adapter] / weight_norm
                    weight_orig = weight.data / dora_factor.view(-1, 1, 1, 1) - delta_weight
                    weight.data = weight_orig

    def get_delta_weight(self, adapter) -> mindspore.Tensor:
        """
        Compute the delta weight for the given adapter.

        Args:
            adapter (str):
                The name of the adapter for which the delta weight should be computed.
        """
        dtype = self.lora_A[adapter].weight.dtype

        # In case users wants to merge the adapter weights that are in
        # float16 while being on CPU, we need to cast the weights to float32, perform the merge and then cast back to
        # float16 because the `@` and matmul operation in general is not supported in torch + cpu + fp16.
        weight_A = self.lora_A[adapter].weight
        weight_B = self.lora_B[adapter].weight

        # https://github.com/bmaltais/kohya_ss/blob/feb6728762a8f463d15ba936d189d4c3abfaa1ab/networks/lora.py#L117
        if self.get_base_layer().weight.shape[2:4] == (1, 1):
            # conv2d 1x1
            output_tensor = (weight_B.squeeze(3).squeeze(2) @ weight_A.squeeze(3).squeeze(2)).unsqueeze(2).unsqueeze(
                3
            ) * self.scaling[adapter]
        else:
            # conv2d 3x3
            output_tensor = (
                ops.conv2d(
                    weight_A.permute(1, 0, 2, 3),
                    weight_B,
                ).permute(1, 0, 2, 3)
                * self.scaling[adapter]
            )

        return output_tensor

    def _get_weight_norm(self, weight, lora_weight, scaling) -> mindspore.Tensor:
        r"""
        Calculates and returns the normalized weight tensor for the Conv2d layer.
        
        Args:
            self (Conv2d): The instance of the Conv2d class.
            weight (mindspore.Tensor): The weight tensor of the Conv2d layer.
            lora_weight (mindspore.Tensor): The additional weight tensor for LORA (Low-Rank Approximation).
            scaling (float): The scaling factor to adjust the impact of lora_weight.
        
        Returns:
            mindspore.Tensor: The normalized weight tensor after applying L2 normalization.
        
        Raises:
            None.
        
        This method takes the weight tensor of the Conv2d layer, the additional lora_weight tensor, and a scaling factor as input. It calculates the normalized weight tensor by adding the scaled lora_weight
tensor to the weight tensor. Then, it applies L2 normalization to the resulting tensor along dimensions (1, 2, 3) and returns the normalized weight tensor. The purpose of this method is to compute the weight
normalization required for the Conv2d layer's computations.
        """
        # calculate L2 norm of weight matrix, channel-wise
        weight = weight + scaling * lora_weight
        # the following is needed to have compatibility with the 4D weight tensors of Conv2D
        weight_norm = weight.norm(p=2, dim=(1, 2, 3), keepdim=True).swapaxes(1, 0)
        return weight_norm

    def _apply_dora(self, x, lora_A, lora_B, scaling, active_adapter):
        """
        For DoRA, calculate the extra output from LoRA with DoRA applied. This should be added on top of the base layer
        output.
        """
        base_layer = self.get_base_layer()
        weight = base_layer.weight
        lora_weight = ops.mm(lora_B.weight.flatten(start_dim=1), lora_A.weight.flatten(start_dim=1))
        lora_weight = lora_weight.reshape(weight.shape)
        magnitude = self.lora_magnitude_vector[active_adapter]
        weight_norm = self._get_weight_norm(weight, lora_weight, scaling)
        # see section 4.3 of DoRA (https://arxiv.org/abs/2402.09353)
        # "[...] we suggest treating ||V +∆V ||_c in
        # Eq. (5) as a constant, thereby detaching it from the gradient
        # graph. This means that while ||V + ∆V ||_c dynamically
        # reflects the updates of ∆V , it won’t receive any gradient
        # during backpropagation"
        mag_norm_scale = magnitude / weight_norm
        result_dora = (mag_norm_scale - 1) * (
            ops.conv2d(
                x,
                weight,
                bias=None,
                stride=base_layer.stride,
                padding=base_layer.padding,
                dilation=base_layer.dilation,
                groups=base_layer.groups,
            )
        ) + mag_norm_scale * lora_B(lora_A(x)) * scaling

        return result_dora

    def forward(self, x: mindspore.Tensor, *args, **kwargs) -> mindspore.Tensor:
        r"""
        Constructs a forward pass of the Conv2d layer.
        
        Args:
            self (Conv2d): An instance of the Conv2d class.
            x (mindspore.Tensor): The input tensor to the Conv2d layer.
                It should have a shape of (batch_size, channels, height, width).
        
        Returns:
            mindspore.Tensor: The output tensor after passing through the Conv2d layer.
                It has the same shape as the input tensor.
        
        Raises:
            ValueError: If the input tensor is not provided.
            TypeError: If the input tensor is not of type mindspore.Tensor.
        """
        self._check_forward_args(x, *args, **kwargs)
        adapter_names = kwargs.pop("adapter_names", None)

        if self.disable_adapters:
            if self.merged:
                self.unmerge()
            result = self.base_layer(x, *args, **kwargs)
        elif adapter_names is not None:
            result = self._mixed_batch_forward(x, *args, adapter_names=adapter_names, **kwargs)
        elif self.merged:
            result = self.base_layer(x, *args, **kwargs)
        else:
            result = self.base_layer(x, *args, **kwargs)
            torch_result_dtype = result.dtype

            for active_adapter in self.active_adapters:
                if active_adapter not in self.lora_A.keys():
                    continue
                lora_A = self.lora_A[active_adapter]
                lora_B = self.lora_B[active_adapter]
                dropout = self.lora_dropout[active_adapter]
                scaling = self.scaling[active_adapter]
                x = x.to(lora_A.weight.dtype)

                if not self.use_dora[active_adapter]:
                    result = result + lora_B(lora_A(dropout(x))) * scaling
                else:
                    x = dropout(x)
                    result = result + self._apply_dora(x, lora_A, lora_B, scaling, active_adapter)

            result = result.to(torch_result_dtype)
        return result

    def __repr__(self) -> str:
        r"""
        Method '__repr__' in the class 'Conv2d'.
        
        Args:
            self: Conv2d - The instance of the Conv2d class.
                Represents the current object instance.
        
        Returns:
            str - A string representation of the object.
            Returns a string prefixed with 'lora.', which is a concatenation of the superclass's string representation.
        
        Raises:
            No specific exceptions are raised by this method.
        """
        rep = super().__repr__()
        return "lora." + rep


def dispatch_default(
    target: nn.Module,
    adapter_name: str,
    lora_config: LoraConfig,
    **kwargs,
) -> Optional[nn.Module]:
    r"""
    Dispatches the default adapter for different types of neural network layers.
    
    Args: 
        target (nn.Module): The target neural network layer for which the adapter is being dispatched.
        adapter_name (str): The name of the adapter being used.
        lora_config (LoraConfig): Configuration object containing LoftQ configuration settings.
        
    Returns: 
        Optional[nn.Module]: The new cell representing the adapted version of the target neural network layer, or None if no adapter is dispatched.
        
    Raises: 
        - KeyError: If required keys are not found in the input kwargs.
        - TypeError: If the input target is not a valid neural network cell.
        - Warning: If conflicting settings are detected for fan_in_fan_out parameter.
    """
    new_cell = None

    if isinstance(target, BaseTunerLayer):
        target_base_layer = target.get_base_layer()
    else:
        target_base_layer = target

    if isinstance(target_base_layer, nn.Embedding):
        embedding_kwargs = kwargs.copy()
        embedding_kwargs.pop("fan_in_fan_out", None)
        embedding_kwargs.update(lora_config.loftq_config)
        new_cell = Embedding(target, adapter_name, **embedding_kwargs)
    elif isinstance(target_base_layer, nn.Conv2d):
        kwargs.update(lora_config.loftq_config)
        new_cell = Conv2d(target, adapter_name, **kwargs)
    elif isinstance(target_base_layer, nn.Linear):
        if kwargs["fan_in_fan_out"]:
            warnings.warn(
                "fan_in_fan_out is set to True but the target cell is `torch.nn.Linear`. "
                "Setting fan_in_fan_out to False."
            )
            kwargs["fan_in_fan_out"] = lora_config.fan_in_fan_out = False
        kwargs.update(lora_config.loftq_config)
        new_cell = Linear(target, adapter_name, **kwargs)
    elif isinstance(target_base_layer, Conv1D):
        if not kwargs["fan_in_fan_out"]:
            warnings.warn(
                "fan_in_fan_out is set to False but the target cell is `Conv1D`. Setting fan_in_fan_out to True."
            )
            kwargs["fan_in_fan_out"] = lora_config.fan_in_fan_out = True
        kwargs.update(lora_config.loftq_config)
        new_cell = Linear(target, adapter_name, is_target_conv_1d_layer=True, **kwargs)

    return new_cell
