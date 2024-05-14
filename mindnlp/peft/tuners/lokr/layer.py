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
from mindspore import nn, ops
from mindspore.common.initializer import initializer, HeUniform, Zero

# import mindnlp._legacy.functional as F
from mindnlp._legacy.abc import ParameterDict

# from ..import_utils import is_bnb_4bit_available, is_bnb_available

from ..tuners_utils import (
    BaseTunerLayer,
    check_adapters_to_merge,
)


class LoKrLayer(nn.Cell, BaseTunerLayer):
    other_param_names = ("r", "alpha", "scaling", "rank_dropout", "module_dropout")
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
            Whether the adapter layer can be plugged to any pytorch module
        active_adapters (Union[List[`str`], `str`], *optional*):
            The name of the active adapter.
    """

    # indicates whether all adapters should be disabled
    _disable_adapters: bool = False

    # the currently active adapter(s)
    _active_adapter: Union[str, List[str]]

    # List all merged adapters
    merged_adapters: "list[str]" = []

    def __init__(self, base_layer: nn.Cell) -> None:
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
        self.module_dropout = {}

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
        # use a property to ensure that active_adapter is not set directly, instead use the set_adapter method
        return self._active_adapter

    @property
    def disable_adapters(self) -> bool:
        # use a property to ensure that disable_adapters is not set directly, instead use the enable_adapters method
        return self._disable_adapters

    @property
    def merged(self) -> bool:
        return bool(self.merged_adapters)

    @property
    def active_adapters(self):
        if isinstance(self.active_adapter, str):
            return [self.active_adapter]
        # is already a list of str
        return self.active_adapter

    def get_base_layer(self) -> nn.Cell:
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
        if adapter_name in self.lokr_w1:
            # nn.init.zeros_(self.lokr_w1[adapter_name])
            self.lokr_w1[adapter_name].set_data(
                initializer(
                    Zero(),
                    self.lokr_w1[adapter_name].shape,
                    self.lokr_w1[adapter_name].dtype,
                )
            )

        else:
            # nn.init.zeros_(self.lokr_w1_a[adapter_name])
            self.lokr_w1_a[adapter_name].set_data(
                initializer(
                    Zero(),
                    self.lokr_w1[adapter_name].shape,
                    self.lokr_w1[adapter_name].dtype,
                )
            )
            # nn.init.kaiming_uniform_(self.lokr_w1_b[adapter_name], a=math.sqrt(5))
            self.lokr_w1_b[adapter_name].set_data(
                initializer(
                    HeUniform(negative_slope=math.sqrt(5)),
                    self.lokr_w1_b[adapter_name].shape,
                    self.lokr_w1_b[adapter_name].dtype,
                )
            )
        if adapter_name in self.lokr_w2:
            # nn.init.kaiming_uniform_(self.lokr_w2[adapter_name], a=math.sqrt(5))
            self.lokr_w2[adapter_name].set_data(
                initializer(
                    HeUniform(negative_slope=math.sqrt(5)),
                    self.lokr_w2[adapter_name].shape,
                    self.lokr_w2[adapter_name].dtype,
                )
            )
        else:
            # nn.init.kaiming_uniform_(self.lokr_w2_a[adapter_name], a=math.sqrt(5))
            self.lokr_w2_a[adapter_name].set_data(
                initializer(
                    HeUniform(negative_slope=math.sqrt(5)),
                    self.lokr_w2_a[adapter_name].shape,
                    self.lokr_w2_a[adapter_name].dtype,
                )
            )
            # nn.init.kaiming_uniform_(self.lokr_w2_b[adapter_name], a=math.sqrt(5))
            self.lokr_w2_b[adapter_name].set_data(
                initializer(
                    HeUniform(negative_slope=math.sqrt(5)),
                    self.lokr_w2_b[adapter_name].shape,
                    self.lokr_w2_b[adapter_name].dtype,
                )
            )

        if adapter_name in self.lokr_t2:
            # nn.init.kaiming_uniform_(self.lokr_t2[adapter_name], a=math.sqrt(5))
            self.lokr_t2[adapter_name].set_data(
                initializer(
                    HeUniform(negative_slope=math.sqrt(5)),
                    self.lokr_t2[adapter_name].shape,
                    self.lokr_t2[adapter_name].dtype,
                )
            )

    def reset_adapter_parameters_random(self, adapter_name: str):
        if adapter_name in self.lokr_w1:
            # nn.init.kaiming_uniform_(self.lokr_w1[adapter_name], a=math.sqrt(5))
            self.lokr_w1[adapter_name].set_data(
                initializer(
                    HeUniform(negative_slope=math.sqrt(5)),
                    self.lokr_w1[adapter_name].shape,
                    self.lokr_w1[adapter_name].dtype,
                )
            )
        else:
            # nn.init.kaiming_uniform_(self.lokr_w1_a[adapter_name], a=math.sqrt(5))
            self.lokr_w1_a[adapter_name].set_data(
                initializer(
                    HeUniform(negative_slope=math.sqrt(5)),
                    self.lokr_w1_a[adapter_name].shape,
                    self.lokr_w1_a[adapter_name].dtype,
                )
            )
            # nn.init.kaiming_uniform_(self.lokr_w1_b[adapter_name], a=math.sqrt(5))
            self.lokr_w1_b[adapter_name].set_data(
                initializer(
                    HeUniform(negative_slope=math.sqrt(5)),
                    self.lokr_w1_b[adapter_name].shape,
                    self.lokr_w1_b[adapter_name].dtype,
                )
            )

        if adapter_name in self.lokr_w2:
            # nn.init.kaiming_uniform_(self.lokr_w2[adapter_name], a=math.sqrt(5))
            self.lokr_w2[adapter_name].set_data(
                initializer(
                    HeUniform(negative_slope=math.sqrt(5)),
                    self.lokr_w2[adapter_name].shape,
                    self.lokr_w2[adapter_name].dtype,
                )
            )
        else:
            # nn.init.kaiming_uniform_(self.lokr_w2_a[adapter_name], a=math.sqrt(5))
            self.lokr_w2_a[adapter_name].set_data(
                initializer(
                    HeUniform(negative_slope=math.sqrt(5)),
                    self.lokr_w2_a[adapter_name].shape,
                    self.lokr_w2_a[adapter_name].dtype,
                )
            )
            # nn.init.kaiming_uniform_(self.lokr_w2_b[adapter_name], a=math.sqrt(5))
            self.lokr_w2_b[adapter_name].set_data(
                initializer(
                    HeUniform(negative_slope=math.sqrt(5)),
                    self.lokr_w2_b[adapter_name].shape,
                    self.lokr_w2_b[adapter_name].dtype,
                )
            )

        if adapter_name in self.lokr_t2:
            # nn.init.kaiming_uniform_(self.lokr_t2[adapter_name], a=math.sqrt(5))
            self.lokr_t2[adapter_name].set_data(
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
        module_dropout: float,
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
            module_dropout (`float`): The dropout probability for disabling adapter during training.
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
        self.module_dropout[adapter_name] = module_dropout
        base_layer = self.get_base_layer()

        # Determine shape of LoKr weights
        if isinstance(base_layer, nn.Dense):
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
        """
        # Move new weights to device
        weight = getattr(self.get_base_layer(), "weight", None)
        
        if weight is not None:
            # the layer is already completely initialized, this is an update
            if weight.dtype.is_floating_point or weight.dtype.is_complex:
                self.to(weight.device, dtype=weight.dtype)
            else:
                self.to(weight.device)
        """
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
            module_dict = getattr(self, layer_name)
            for key, layer in module_dict.items():
                if key in adapter_names:
                    # Note: It is possible that not a single layer is called with requires_grad_(True) here. This may
                    # happen if a completely different adapter layer is being activated.
                    layer.requires_grad = True
                else:
                    layer.requires_grad = False

        self._active_adapter = adapter_names

    def merge(
        self, safe_merge: bool = False, adapter_names: Optional[list[str]] = None
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
        # https://github.com/KohakuBlueleaf/LyCORIS/blob/e4259b870d3354a9615a96be61cb5d07455c58ea/lycoris/modules/lokr.py#L224
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

    def construct(self, x: ms.Tensor, *args, **kwargs) -> ms.Tensor:
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


class Dense(LoKrLayer):
    """LoKr implemented in Dense layer"""

    def __init__(
        self,
        base_layer: nn.Cell,
        adapter_name: str = "default",
        r: int = 0,
        alpha: float = 0.0,
        rank_dropout: float = 0.0,
        module_dropout: float = 0.0,
        init_weights: bool = True,
        **kwargs,
    ):
        super().__init__(base_layer)

        # Create adapter and set it active
        self._active_adapter = adapter_name
        self.update_layer(
            adapter_name, r, alpha, rank_dropout, module_dropout, init_weights, **kwargs
        )

    def _get_delta_activations(
        self, adapter_name: str, input: ms.Tensor, *args: Any, **kwargs: Any
    ) -> ms.Tensor:
        delta_weight = self.get_delta_weight(
            adapter_name
        )  # Forced synchronization of parameter types, dangerous operation
        # don't add bias here, because the bias is already included in the output of the base_layer
        return ops.dense(input, delta_weight)

    def __repr__(self) -> str:
        rep = super().__repr__()
        return "lokr." + rep


class Conv2d(LoKrLayer):
    """LoKr implemented in Conv2d layer"""

    def __init__(
        self,
        base_layer: nn.Cell,
        adapter_name: str = "default",
        r: int = 0,
        alpha: float = 0.0,
        rank_dropout: float = 0.0,
        module_dropout: float = 0.0,
        use_effective_conv2d: bool = False,
        init_weights: bool = True,
        **kwargs,
    ):
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
        self, adapter_name: str, input: ms.Tensor, *args: Any, **kwargs: Any
    ) -> ms.Tensor:
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
    rebuild2 = ops.einsum("i j k l, i p, j r -> p r k l", t, wa, wb)  # [c, d, k1, k2]
    return rebuild2


def make_kron(w1, w2, scale=1.0):
    if len(w2.shape) == 4:
        w1 = w1.unsqueeze(2).unsqueeze(2)
    # w2 = w2.contiguous()
    rebuild = ops.kron(w1, w2)

    return rebuild * scale
