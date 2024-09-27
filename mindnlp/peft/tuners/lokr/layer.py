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
"""Lokr."""
import math
import warnings
from typing import List, Optional, Union, Any, Set, Tuple
from abc import abstractmethod

import mindspore as ms
from mindspore.common.initializer import initializer, HeUniform, Zero

from mindnlp.core import nn, ops
from mindnlp.core.nn import ParameterDict

from ..tuners_utils import (
    BaseTunerLayer,
    check_adapters_to_merge,
)


class LoKrLayer(nn.Module, BaseTunerLayer):

    r"""
    LoKrLayer is a custom PyTorch class representing a layer that implements the Locally Kroneckerized Neural Network adaptation technique. This technique allows for adaptive modifications to be made on top of
the base layer's output. The class provides methods for creating, updating, merging, unmerging, and managing adaptive layers within the network.
    
    Attributes:
        - lokr_w1: Dictionary of parameters for the first adaptive layer.
        - lokr_w1_a: Dictionary of parameters for the first adaptive layer (alternative).
        - lokr_w1_b: Dictionary of parameters for the first adaptive layer (alternative).
        - lokr_w2: Dictionary of parameters for the second adaptive layer.
        - lokr_w2_a: Dictionary of parameters for the second adaptive layer (alternative).
        - lokr_w2_b: Dictionary of parameters for the second adaptive layer (alternative).
        - lokr_t2: Dictionary of parameters for the second adaptive layer (tensor version).
        - base_layer: The base layer on which the adaptive modifications are applied.
        - r: Dictionary storing the rank values for each adapter.
        - alpha: Dictionary storing alpha values for each adapter.
        - scaling: Dictionary storing scaling values for each adapter.
        - rank_dropout: Dictionary storing rank dropout probabilities for each adapter.
        - cell_dropout: Dictionary storing cell dropout probabilities for each adapter.
        - _disable_adapters: Boolean flag indicating whether adapters are disabled.
        - merged_adapters: List of names of merged adapters.
    
    Methods:
        - _get_delta_activations: Abstract method to retrieve activations added on top of the base layer output.
        - _available_adapters: Property returning a set of available adapter names.
        - active_adapter: Property returning the name of the active adapter.
        - disable_adapters: Property returning a boolean flag indicating whether adapters are disabled.
        - merged: Property returning a boolean value indicating if any adapters are merged.
        - active_adapters: Property returning a list of active adapter names.
        - get_base_layer: Method to recursively retrieve the base layer.
        - create_adapter_parameters: Method to create adapter parameters based on input configurations.
        - reset_adapter_parameters: Method to reset adapter parameters to initial values.
        - reset_adapter_parameters_random: Method to reset adapter parameters to random initial values.
        - update_layer: Method to update the layer with a new adapter based on specified parameters.
        - set_adapter: Method to set the active adapter(s) and mark them as trainable.
        - merge: Method to merge active adapter weights into the base weights.
        - unmerge: Method to unmerge all merged adapter layers from the base weights.
        - get_delta_weight: Method to calculate the delta weight for a specific adapter.
        - forward: Method to forward the output of the layer with adaptive modifications applied.
    
    Note:
        This class is intended for advanced neural network adaptation techniques and should be used in conjunction with PyTorch's nn.Module functionalities.
    """
    other_param_names = ("r", "alpha", "scaling", "rank_dropout", "cell_dropout")
    # All names of layers that may contain adapter weights
    adapter_layer_names = (
        "lokr_w1",
        "lokr_w1_a",
        "lokr_w1_b",
        "lokr_w2",
        "lokr_w2_a",
        "lokr_w2_b",
        "lokr_t2",
    )
    r"""
    A tuner layer mixin that provides the common methods and attributes for all tuners.

    Args:
        is_pluggable (`bool`, *optional*):
            Whether the adapter layer can be plugged to any pytorch cell
        active_adapters (Union[List[`str`], `str`], *optional*):
            The name of the active adapter.
    """

    # indicates whether all adapters should be disabled
    _disable_adapters: bool = False

    # the currently active adapter(s)
    _active_adapter: Union[str, List[str]]

    # List all merged adapters
    merged_adapters: "List[str]" = []

    def __init__(self, base_layer: nn.Module) -> None:
        r"""
        This method initializes an instance of the LoKrLayer class.
        
        Args:
            self: The instance of the LoKrLayer class.
            base_layer (nn.Module): The base layer used for the LoKrLayer.
            
        Returns:
            None: This method does not return any value.
        
        Raises:
            None.
        """
        super().__init__()
        # LoKr info
        self.lokr_w1 = ParameterDict({})
        self.lokr_w1_a = ParameterDict({})
        self.lokr_w1_b = ParameterDict({})
        self.lokr_w2 = ParameterDict({})
        self.lokr_w2_a = ParameterDict({})
        self.lokr_w2_b = ParameterDict({})
        self.lokr_t2 = ParameterDict({})
        self.base_layer = base_layer
        self.r = {}
        self.alpha = {}
        self.scaling = {}
        self.rank_dropout = {}
        self.cell_dropout = {}

        # Tuner info
        self._disable_adapters = False
        self.merged_adapters = []

    @abstractmethod
    def _get_delta_activations(
        self, adapter_name: str, x: ms.Tensor, *args: Any, **kwargs: Any
    ) -> ms.Tensor:
        """Activations added on top of the base layer output (i.e. after the base layer forward pass)"""
    @property
    def _available_adapters(self) -> Set[str]:
        r"""
        Method to retrieve the set of available adapters.
        
        Args:
            self (LoKrLayer): The instance of the LoKrLayer class.
                This parameter represents the current instance of the LoKrLayer class.
                
        Returns:
            Set[str]: A set containing strings representing available adapters.
                The set includes available adapters from different sources within the LoKrLayer instance.
                
        Raises:
            None
        """
        return {
            *self.lokr_w1,
            *self.lokr_w1_a,
            *self.lokr_w1_b,
            *self.lokr_w2,
            *self.lokr_w2_a,
            *self.lokr_w2_b,
            *self.lokr_t2,
        }

    @property
    def active_adapter(self) -> str:
        r"""
        This method returns the active adapter.
        
        Args:
            self: Instance of the LoKrLayer class.
        
        Returns:
            str: The active adapter as a string.
        
        Raises:
            None
        """
        # use a property to ensure that active_adapter is not set directly, instead use the set_adapter method
        return self._active_adapter

    @property
    def disable_adapters(self) -> bool:
        r"""
        Disables the adapters in the LoKrLayer.
        
        Args:
            self: An instance of the LoKrLayer class.
        
        Returns:
            bool: True if the adapters are successfully disabled, False otherwise.
        
        Raises:
            None.
        """
        # use a property to ensure that disable_adapters is not set directly, instead use the enable_adapters method
        return self._disable_adapters

    @property
    def merged(self) -> bool:
        r"""
        Returns a boolean value indicating whether the 'LoKrLayer' instance has any merged adapters.
        
        Args:
            self: The current instance of 'LoKrLayer'.
        
        Returns:
            bool: True if the 'LoKrLayer' instance has merged adapters, False otherwise.
        
        Raises:
            None.
        """
        return bool(self.merged_adapters)

    @property
    def active_adapters(self):
        r"""
        This method 'active_adapters' in the class 'LoKrLayer' retrieves the active adapters.
        
        Args:
            self: The instance of the 'LoKrLayer' class.
            
        Returns:
            If the 'active_adapter' attribute of the instance is a string, this method returns a list containing that string.
            If the 'active_adapter' attribute of the instance is not a string, the method returns the 'active_adapter' attribute itself.
        
        Raises:
            No specific exceptions are raised by this method.
        """
        if isinstance(self.active_adapter, str):
            return [self.active_adapter]
        # is already a list of str
        return self.active_adapter

    def get_base_layer(self) -> nn.Module:
        """
        (Recursively) get the base_layer.

        This is necessary for the case that the tuner layer wraps another tuner layer.

        """
        base_layer = self
        while hasattr(base_layer, "base_layer"):
            base_layer = base_layer.base_layer
        return base_layer

    def create_adapter_parameters(
        self,
        adapter_name: str,
        r: int,
        shape,
        use_w1: bool,
        use_w2: bool,
        use_effective_conv2d: bool,
    ):
        r"""Create adapter parameters for the LoKrLayer class.
        
        This method creates and initializes adapter parameters based on the provided arguments. The adapter parameters are used for the LoKrLayer class.
        
        Args:
            self (LoKrLayer): The instance of the LoKrLayer class.
            adapter_name (str): The name of the adapter.
            r (int): The value of 'r' used for parameter initialization.
            shape: The shape of the parameters. It can be a tuple or a list of tuples, depending on the number of dimensions.
            use_w1 (bool): A flag indicating whether to use the 'w1' parameter.
            use_w2 (bool): A flag indicating whether to use the 'w2' parameter.
            use_effective_conv2d (bool): A flag indicating whether to use the 'effective_conv2d' parameter.
        
        Returns:
            None: This method does not return any value.
        
        Raises:
            None: This method does not raise any exceptions.
        """
        if use_w1:
            self.lokr_w1[adapter_name] = ms.Parameter(
                ops.zeros((shape[0][0], shape[1][0]))
            )
        else:
            self.lokr_w1_a[adapter_name] = ms.Parameter(ops.zeros((shape[0][0], r)))
            self.lokr_w1_b[adapter_name] = ms.Parameter(ops.zeros((r, shape[1][0])))

        if len(shape) == 4:
            # Conv2d
            if use_w2:
                self.lokr_w2[adapter_name] = ms.Parameter(
                    ops.zeros((shape[0][1], shape[1][1], *shape[2:]))
                )
            elif use_effective_conv2d:
                self.lokr_t2[adapter_name] = ms.Parameter(
                    ops.zeros((r, r, shape[2], shape[3]))
                )
                self.lokr_w2_a[adapter_name] = ms.Parameter(
                    ops.zeros((r, shape[0][1]))
                )  # b, 1-mode
                self.lokr_w2_b[adapter_name] = ms.Parameter(
                    ops.zeros((r, shape[1][1]))
                )  # d, 2-mode
            else:
                self.lokr_w2_a[adapter_name] = ms.Parameter(ops.zeros((shape[0][1], r)))
                self.lokr_w2_b[adapter_name] = ms.Parameter(
                    ops.zeros((r, shape[1][1] * shape[2] * shape[3]))
                )
        else:
            # Linear
            if use_w2:
                self.lokr_w2[adapter_name] = ms.Parameter(
                    ops.zeros((shape[0][1], shape[1][1]))
                )
            else:
                self.lokr_w2_a[adapter_name] = ms.Parameter(ops.zeros((shape[0][1], r)))
                self.lokr_w2_b[adapter_name] = ms.Parameter(ops.zeros((r, shape[1][1])))

    def reset_adapter_parameters(self, adapter_name: str):
        r"""
        Reset the parameters of the specified adapter within the LoKrLayer.
        
        Args:
            self (LoKrLayer): The instance of the LoKrLayer class.
            adapter_name (str): The name of the adapter whose parameters are to be reset.
        
        Returns:
            None: This method does not return any value.
        
        Raises:
            KeyError: If the specified adapter_name is not found within the LoKrLayer attributes.
            ValueError: If the specified adapter_name is not found within the LoKrLayer attributes and no corresponding backup attributes exist.
        """
        if adapter_name in self.lokr_w1:
            # nn.init.zeros_(self.lokr_w1[adapter_name])
            self.lokr_w1[adapter_name].assign_value(
                initializer(
                    Zero(),
                    self.lokr_w1[adapter_name].shape,
                    self.lokr_w1[adapter_name].dtype,
                )
            )

        else:
            # nn.init.zeros_(self.lokr_w1_a[adapter_name])
            self.lokr_w1_a[adapter_name].assign_value(
                initializer(
                    Zero(),
                    self.lokr_w1[adapter_name].shape,
                    self.lokr_w1[adapter_name].dtype,
                )
            )
            # nn.init.kaiming_uniform_(self.lokr_w1_b[adapter_name], a=math.sqrt(5))
            self.lokr_w1_b[adapter_name].assign_value(
                initializer(
                    HeUniform(negative_slope=math.sqrt(5)),
                    self.lokr_w1_b[adapter_name].shape,
                    self.lokr_w1_b[adapter_name].dtype,
                )
            )
        if adapter_name in self.lokr_w2:
            # nn.init.kaiming_uniform_(self.lokr_w2[adapter_name], a=math.sqrt(5))
            self.lokr_w2[adapter_name].assign_value(
                initializer(
                    HeUniform(negative_slope=math.sqrt(5)),
                    self.lokr_w2[adapter_name].shape,
                    self.lokr_w2[adapter_name].dtype,
                )
            )
        else:
            # nn.init.kaiming_uniform_(self.lokr_w2_a[adapter_name], a=math.sqrt(5))
            self.lokr_w2_a[adapter_name].assign_value(
                initializer(
                    HeUniform(negative_slope=math.sqrt(5)),
                    self.lokr_w2_a[adapter_name].shape,
                    self.lokr_w2_a[adapter_name].dtype,
                )
            )
            # nn.init.kaiming_uniform_(self.lokr_w2_b[adapter_name], a=math.sqrt(5))
            self.lokr_w2_b[adapter_name].assign_value(
                initializer(
                    HeUniform(negative_slope=math.sqrt(5)),
                    self.lokr_w2_b[adapter_name].shape,
                    self.lokr_w2_b[adapter_name].dtype,
                )
            )

        if adapter_name in self.lokr_t2:
            # nn.init.kaiming_uniform_(self.lokr_t2[adapter_name], a=math.sqrt(5))
            self.lokr_t2[adapter_name].assign_value(
                initializer(
                    HeUniform(negative_slope=math.sqrt(5)),
                    self.lokr_t2[adapter_name].shape,
                    self.lokr_t2[adapter_name].dtype,
                )
            )

    def reset_adapter_parameters_random(self, adapter_name: str):
        r"""
        Resets the adapter parameters randomly for the specified adapter in the LoKrLayer class.
        
        Args:
            self: The instance of the LoKrLayer class.
            adapter_name (str): The name of the adapter to reset.
        
        Returns:
            None. This method does not return any value.
        
        Raises:
            None.
        
        This method resets the adapter parameters randomly based on the adapter name provided. If the adapter name is found in the self.lokr_w1 dictionary, the self.lokr_w1[adapter_name] parameter is reset
using HeUniform initialization with a negative slope of the square root of 5. If the adapter name is not found in the self.lokr_w1 dictionary, the self.lokr_w1_a[adapter_name] and self.lokr_w1_b[adapter_name]
parameters are reset using the same initialization.
        
        Similarly, the self.lokr_w2 and self.lokr_t2 parameters are reset based on the adapter name. If the adapter name is found in the self.lokr_w2 dictionary, the self.lokr_w2[adapter_name] parameter is
reset using HeUniform initialization. If the adapter name is not found in the self.lokr_w2 dictionary, the self.lokr_w2_a[adapter_name] and self.lokr_w2_b[adapter_name] parameters are reset using the same
initialization.
        
        Note: This method assumes that the initializer and HeUniform functions are defined and available.
        """
        if adapter_name in self.lokr_w1:
            # nn.init.kaiming_uniform_(self.lokr_w1[adapter_name], a=math.sqrt(5))
            self.lokr_w1[adapter_name].assign_value(
                initializer(
                    HeUniform(negative_slope=math.sqrt(5)),
                    self.lokr_w1[adapter_name].shape,
                    self.lokr_w1[adapter_name].dtype,
                )
            )
        else:
            # nn.init.kaiming_uniform_(self.lokr_w1_a[adapter_name], a=math.sqrt(5))
            self.lokr_w1_a[adapter_name].assign_value(
                initializer(
                    HeUniform(negative_slope=math.sqrt(5)),
                    self.lokr_w1_a[adapter_name].shape,
                    self.lokr_w1_a[adapter_name].dtype,
                )
            )
            # nn.init.kaiming_uniform_(self.lokr_w1_b[adapter_name], a=math.sqrt(5))
            self.lokr_w1_b[adapter_name].assign_value(
                initializer(
                    HeUniform(negative_slope=math.sqrt(5)),
                    self.lokr_w1_b[adapter_name].shape,
                    self.lokr_w1_b[adapter_name].dtype,
                )
            )

        if adapter_name in self.lokr_w2:
            # nn.init.kaiming_uniform_(self.lokr_w2[adapter_name], a=math.sqrt(5))
            self.lokr_w2[adapter_name].assign_value(
                initializer(
                    HeUniform(negative_slope=math.sqrt(5)),
                    self.lokr_w2[adapter_name].shape,
                    self.lokr_w2[adapter_name].dtype,
                )
            )
        else:
            # nn.init.kaiming_uniform_(self.lokr_w2_a[adapter_name], a=math.sqrt(5))
            self.lokr_w2_a[adapter_name].assign_value(
                initializer(
                    HeUniform(negative_slope=math.sqrt(5)),
                    self.lokr_w2_a[adapter_name].shape,
                    self.lokr_w2_a[adapter_name].dtype,
                )
            )
            # nn.init.kaiming_uniform_(self.lokr_w2_b[adapter_name], a=math.sqrt(5))
            self.lokr_w2_b[adapter_name].assign_value(
                initializer(
                    HeUniform(negative_slope=math.sqrt(5)),
                    self.lokr_w2_b[adapter_name].shape,
                    self.lokr_w2_b[adapter_name].dtype,
                )
            )

        if adapter_name in self.lokr_t2:
            # nn.init.kaiming_uniform_(self.lokr_t2[adapter_name], a=math.sqrt(5))
            self.lokr_t2[adapter_name].assign_value(
                initializer(
                    HeUniform(negative_slope=math.sqrt(5)),
                    self.lokr_t2[adapter_name].shape,
                    self.lokr_t2[adapter_name].dtype,
                )
            )

    def update_layer(
        self,
        adapter_name: str,
        r: int,
        alpha: float,
        rank_dropout: float,
        cell_dropout: float,
        init_weights: bool,
        use_effective_conv2d: bool,
        decompose_both: bool,
        decompose_factor: int,
        **kwargs,
    ) -> None:
        """Internal function to create lokr adapter

        Args:
            adapter_name (`str`): Name for the adapter to add.
            r (`int`): Rank for the added adapter.
            alpha (`float`): Alpha for the added adapter.
            rank_dropout (`float`): The dropout probability for rank dimension during training
            cell_dropout (`float`): The dropout probability for disabling adapter during training.
            init_weights (`bool`): Whether to initialize adapter weights.
            use_effective_conv2d (`bool`): Use parameter effective decomposition for Conv2d with ksize > 1.
            decompose_both (`bool`): Perform rank decomposition of left kronecker product matrix.
            decompose_factor (`int`): Kronecker product decomposition factor.
        """
        if r <= 0:
            raise ValueError(
                f"`r` should be a positive integer value but the value passed is {r}"
            )

        self.r[adapter_name] = r
        self.alpha[adapter_name] = alpha
        self.scaling[adapter_name] = alpha / r
        self.rank_dropout[adapter_name] = rank_dropout
        self.cell_dropout[adapter_name] = cell_dropout
        base_layer = self.get_base_layer()

        # Determine shape of LoKr weights
        if isinstance(base_layer, nn.Linear):
            in_dim, out_dim = base_layer.in_channels, base_layer.out_channels

            in_m, in_n = factorization(in_dim, decompose_factor)
            out_l, out_k = factorization(out_dim, decompose_factor)
            shape = (
                (out_l, out_k),
                (in_m, in_n),
            )  # ((a, b), (c, d)), out_dim = a*c, in_dim = b*d

            use_w1 = not (decompose_both and r < max(shape[0][0], shape[1][0]) / 2)
            use_w2 = not (r < max(shape[0][1], shape[1][1]) / 2)
            use_effective_conv2d = False
        elif isinstance(base_layer, nn.Conv2d):
            in_dim, out_dim = base_layer.in_channels, base_layer.out_channels
            k_size = base_layer.kernel_size

            in_m, in_n = factorization(in_dim, decompose_factor)
            out_l, out_k = factorization(out_dim, decompose_factor)
            shape = ((out_l, out_k), (in_m, in_n), *k_size)  # ((a, b), (c, d), *k_size)

            use_w1 = not (decompose_both and r < max(shape[0][0], shape[1][0]) / 2)
            use_w2 = r >= max(shape[0][1], shape[1][1]) / 2
            use_effective_conv2d = use_effective_conv2d and base_layer.kernel_size != (
                1,
                1,
            )
        else:
            raise TypeError(
                f"LoKr is not implemented for base layers of type {type(base_layer).__name__}"
            )

        # Create weights with provided shape
        self.create_adapter_parameters(
            adapter_name, r, shape, use_w1, use_w2, use_effective_conv2d
        )

        # Initialize weights
        if init_weights:
            self.reset_adapter_parameters(adapter_name)
        else:
            self.reset_adapter_parameters_random(adapter_name)

        self.set_adapter(self.active_adapters)

    def set_adapter(self, adapter_names) -> None:
        """Set the active adapter(s).

        Additionally, this function will set the specified adapters to trainable (i.e., requires_grad=True). If this is
        not desired, use the following code.

        ```py
        >>> for name, param in model_peft.named_parameters():
        ...     if ...:  # some check on name (ex. if 'lora' in name)
        ...         param.requires_grad = False
        ```

        Args:
            adapter_name (`str` or `List[str]`): Name of the adapter(s) to be activated.
        """
        if isinstance(adapter_names, str):
            adapter_names = [adapter_names]

        # Deactivate grads on the inactive adapter and activate grads on the active adapter
        for layer_name in self.adapter_layer_names:
            cell_dict = getattr(self, layer_name)
            for key, layer in cell_dict.items():
                if key in adapter_names:
                    # Note: It is possible that not a single layer is called with requires_grad_(True) here. This may
                    # happen if a completely different adapter layer is being activated.
                    layer.requires_grad = True
                else:
                    layer.requires_grad = False

        self._active_adapter = adapter_names

    def merge(
        self, safe_merge: bool = False, adapter_names: Optional[List[str]] = None
    ) -> None:
        """
        Merge the active adapter weights into the base weights

        Args:
            safe_merge (`bool`, *optional*):
                If `True`, the merge operation will be performed in a copy of the original weights and check for NaNs
                before merging the weights. This is useful if you want to check if the merge operation will produce
                NaNs. Defaults to `False`.
            adapter_names (`List[str]`, *optional*):
                The list of adapter names that should be merged. If `None`, all active adapters will be merged.
                Defaults to `None`.
        """
        adapter_names = check_adapters_to_merge(self, adapter_names)
        if not adapter_names:
            # no adapter to merge
            return

        for active_adapter in adapter_names:
            if active_adapter in self._available_adapters:
                base_layer = self.get_base_layer()
                if safe_merge:
                    orig_weights = base_layer.weight.data
                    orig_weights += self.get_delta_weight(active_adapter)

                    if not ops.isfinite(orig_weights).all():
                        raise ValueError(
                            f"NaNs detected in the merged weights. The adapter {active_adapter} seems to be broken"
                        )

                    base_layer.weight.data = orig_weights
                else:
                    base_layer.weight.data += self.get_delta_weight(active_adapter)
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
            if active_adapter in self._available_adapters:
                self.get_base_layer().weight.data -= self.get_delta_weight(
                    active_adapter
                )

    def get_delta_weight(self, adapter_name: str) -> ms.Tensor:
        r"""
        This method calculates the delta weight for a given adapter.
        
        Args:
            self: The instance of the LoKrLayer class.
            adapter_name (str): The name of the adapter for which the delta weight is to be calculated. It is a required parameter.
        
        Returns:
            ms.Tensor: Returns a tensor representing the delta weight calculated for the specified adapter.
        
        Raises:
            ValueError: If the adapter_name is not found in the internal data structures.
            RuntimeError: If an error occurs during the calculation of the delta weight.
            TypeError: If the input data types are incorrect or incompatible.
        """
        # https://github.com/KohakuBlueleaf/LyCORIS/blob/e4259b870d3354a9615a96be61cb5d07455c58ea/lycoris/cells/lokr.py#L224
        if adapter_name in self.lokr_w1:
            w1 = self.lokr_w1[adapter_name]
        else:
            w1 = self.lokr_w1_a[adapter_name] @ self.lokr_w1_b[adapter_name]

        if adapter_name in self.lokr_w2:
            w2 = self.lokr_w2[adapter_name]
        elif adapter_name in self.lokr_t2:
            w2 = make_weight_cp(
                self.lokr_t2[adapter_name],
                self.lokr_w2_a[adapter_name],
                self.lokr_w2_b[adapter_name],
            )
        else:
            w2 = self.lokr_w2_a[adapter_name] @ self.lokr_w2_b[adapter_name]

        # Make weights with Kronecker product
        weight = make_kron(w1, w2)
        weight = weight.reshape(self.get_base_layer().weight.shape)

        # Perform rank dropout during training - drop rows of addition weights
        rank_dropout = self.rank_dropout[adapter_name]
        if self.training and rank_dropout:
            drop = (ops.rand(weight.size(0)) > rank_dropout).float()
            drop = drop.view(-1, *[1] * len(weight.shape[1:]))
            drop /= drop.mean()
            weight *= drop

        return weight

    def forward(self, x: ms.Tensor, *args, **kwargs) -> ms.Tensor:
        """
        Constructs the output tensor using the specified input tensor and additional arguments.
        
        Args:
            self (LoKrLayer): The instance of the LoKrLayer class.
            x (ms.Tensor): The input tensor to be processed.
            
        Returns:
            ms.Tensor: The output tensor forwarded based on the input tensor and additional arguments.
        
        Raises:
            TypeError: If the input tensor x is not of type ms.Tensor.
            ValueError: If the input tensor x has an unsupported dtype.
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

                cell_dropout = self.cell_dropout[active_adapter]

                # Modify current execution weights
                if (not self.training) or (
                    self.training and ops.rand(1) > cell_dropout
                ):
                    result = result + self._get_delta_activations(
                        active_adapter, x, *args, **kwargs
                    )

        result = result.to(previous_dtype)
        return result


class Dense(LoKrLayer):
    """LoKr implemented in Dense layer"""
    def __init__(
        self,
        base_layer: nn.Module,
        adapter_name: str = "default",
        r: int = 0,
        alpha: float = 0.0,
        rank_dropout: float = 0.0,
        cell_dropout: float = 0.0,
        init_weights: bool = True,
        **kwargs,
    ):
        """
        Initializes a new instance of the Dense class.
        
        Args:
            self: The object itself.
            base_layer (nn.Module): The base layer for the adapter.
            adapter_name (str): The name of the adapter. Defaults to 'default'.
            r (int): The value of r for adapter update. Defaults to 0.
            alpha (float): The value of alpha for adapter update. Defaults to 0.0.
            rank_dropout (float): The dropout value for rank. Defaults to 0.0.
            cell_dropout (float): The dropout value for cell. Defaults to 0.0.
            init_weights (bool): A flag to initialize weights. Defaults to True.
            **kwargs: Additional keyword arguments.
        
        Returns:
            None. This method does not return any value.
        
        Raises:
            <List of exceptions that the function may raise, if any>
        """
        super().__init__(base_layer)

        # Create adapter and set it active
        self._active_adapter = adapter_name
        self.update_layer(
            adapter_name, r, alpha, rank_dropout, cell_dropout, init_weights, **kwargs
        )

    def _get_delta_activations(
        self, adapter_name: str, input: ms.Tensor, *args: Any, **kwargs: Any
    ) -> ms.Tensor:
        """
        Method to calculate the delta activations for a given adapter.
        
        Args:
            self: The instance of the Dense class.
            adapter_name (str): The name of the adapter to retrieve delta weight for.
            input (ms.Tensor): The input tensor for which delta activations are calculated.
            *args: Additional positional arguments.
            **kwargs: Additional keyword arguments.
        
        Returns:
            ms.Tensor: The calculated delta activations as a tensor.
        
        Raises:
            (Exception): If there is an error in retrieving the delta weight or in performing the dense operation.
        """
        delta_weight = self.get_delta_weight(
            adapter_name
        )  # Forced synchronization of parameter types, dangerous operation
        # don't add bias here, because the bias is already included in the output of the base_layer
        return ops.dense(input, delta_weight)

    def __repr__(self) -> str:
        r"""
        This method returns a string representation of the object.
        
        Args:
            self (Dense): The instance of the Dense class.
            
        Returns:
            str: A string representation of the object prefixed with 'lokr.'.
        
        Raises:
            This method does not raise any exceptions.
        """
        rep = super().__repr__()
        return "lokr." + rep


class Conv2d(LoKrLayer):
    """LoKr implemented in Conv2d layer"""
    def __init__(
        self,
        base_layer: nn.Module,
        adapter_name: str = "default",
        r: int = 0,
        alpha: float = 0.0,
        rank_dropout: float = 0.0,
        cell_dropout: float = 0.0,
        use_effective_conv2d: bool = False,
        init_weights: bool = True,
        **kwargs,
    ):
        r"""
        Initializes an instance of the Conv2d class.
        
        Args:
            self: The instance of the Conv2d class.
            base_layer (nn.Module): The base layer that the adapter will be added on top of.
            adapter_name (str): The name of the adapter. Defaults to 'default'.
            r (int): The value of parameter 'r'.
            alpha (float): The value of parameter 'alpha'.
            rank_dropout (float): The value of rank dropout.
            cell_dropout (float): The value of cell dropout.
            use_effective_conv2d (bool): Flag indicating whether to use effective Conv2d.
            init_weights (bool): Flag indicating whether to initialize weights.
            **kwargs: Additional keyword arguments.
        
        Returns:
            None. This method does not return any value.
        
        Raises:
            None specified.
        """
        super().__init__(base_layer)

        # Create adapter and set it active
        self._active_adapter = adapter_name
        self.update_layer(
            adapter_name,
            r,
            alpha,
            rank_dropout,
            cell_dropout,
            init_weights,
            use_effective_conv2d,
            **kwargs,
        )

    def _get_delta_activations(
        self, adapter_name: str, input: ms.Tensor, *args: Any, **kwargs: Any
    ) -> ms.Tensor:
        r"""
        Method to calculate delta activations for Conv2d layer.
        
        Args:
            self (Conv2d): The instance of the Conv2d class.
            adapter_name (str): The name of the adapter used for getting delta weight.
            input (ms.Tensor): The input tensor for the convolution operation.
        
        Returns:
            ms.Tensor: Returns the delta activations tensor calculated based on the input, delta weight, and base layer parameters.
        
        Raises:
            - KeyError: If the provided adapter_name does not exist.
            - ValueError: If there are issues with the input tensor shape or data.
            - RuntimeError: If there are runtime issues during the convolution operation.
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
            groups=base_layer.group,
        )

    def __repr__(self) -> str:
        r"""
        Return a string representation of the 'Conv2d' object.
        
        Args:
            self: The 'Conv2d' object itself.
        
        Returns:
            A string representation of the 'Conv2d' object, prefixed with 'lokr.'.
        
        Raises:
            None
        
        Example:
            >>> conv = Conv2d()
            >>> repr(conv)
            'lokr.Conv2d()'
        """
        rep = super().__repr__()
        return "lokr." + rep


def factorization(dimension: int, factor: int = -1) -> Tuple[int, int]:
    """Factorizes the provided number into the product of two numbers

    Args:
        dimension (`int`): The number that needs to be factorized.
        factor (`int`, optional):
            Factorization divider. The algorithm will try to output two numbers, one of each will be as close to the
            factor as possible. If -1 is provided, the decomposition algorithm would try to search dividers near the
            square root of the dimension. Defaults to -1.

    Returns:
        Tuple[`int`, `int`]: A tuple of two numbers, whose product is equal to the provided number. The first number is
        always less than or equal to the second.

    Example:
        ```py
        >>> factorization(256, factor=-1)
        (16, 16)

        >>> factorization(128, factor=-1)
        (8, 16)

        >>> factorization(127, factor=-1)
        (1, 127)

        >>> factorization(128, factor=4)
        (4, 32)
        ```
    """
    if factor > 0 and (dimension % factor) == 0:
        m = factor
        n = dimension // factor
        return m, n
    if factor == -1:
        factor = dimension
    m, n = 1, dimension
    length = m + n
    while m < n:
        new_m = m + 1
        while dimension % new_m != 0:
            new_m += 1
        new_n = dimension // new_m
        if new_m + new_n > length or new_m > factor:
            break
        else:
            m, n = new_m, new_n
    if m > n:
        n, m = m, n
    return m, n


def make_weight_cp(t, wa, wb):
    r"""
    This function creates a weight tensor by performing the contraction of four-dimensional tensor 't' with two matrices 'wa' and 'wb' along specific dimensions.
    
    Args:
        t (ndarray): A four-dimensional tensor with shape (i, j, k, l), where i, j, k, l represent the dimensions of the tensor. The tensor serves as the base for the contraction operation.
        wa (ndarray): A matrix with shape (i, p), where i represents the dimension matching the first dimension of 't', and p represents the desired dimension of the resulting tensor along the first axis.
        wb (ndarray): A matrix with shape (j, r), where j represents the dimension matching the second dimension of 't', and r represents the desired dimension of the resulting tensor along the second axis.
    
    Returns:
        ndarray: The resulting weight tensor after performing the contraction operation. The shape of the output tensor is (p, r, k, l), where p and r represent the dimensions specified by 'wa' and 'wb',
respectively, and k, l represent the remaining dimensions inherited from 't'.
    
    Raises:
        None: This function does not raise any exceptions.
    """
    rebuild2 = ops.einsum("i j k l, i p, j r -> p r k l", t, wa, wb)  # [c, d, k1, k2]
    return rebuild2


def make_kron(w1, w2, scale=1.0):
    r"""
    This function creates a Kronecker product of two input tensors w1 and w2, and then scales the result by the specified scale factor.
    
    Args:
        w1 (tensor): The first input tensor.
        w2 (tensor): The second input tensor. For 4-dimensional tensors, w1 will be modified with unsqueeze operations before computing the Kronecker product.
        scale (float, optional): The scaling factor applied to the Kronecker product. Defaults to 1.0.
    
    Returns:
        None: The function returns None.
    
    Raises:
        None.
    """
    if len(w2.shape) == 4:
        w1 = w1.unsqueeze(2).unsqueeze(2)
    # w2 = w2
    rebuild = ops.kron(w1, w2)

    return rebuild * scale
