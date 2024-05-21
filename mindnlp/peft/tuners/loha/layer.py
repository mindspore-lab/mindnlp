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
"""LoHa Layer"""
import math
import mindspore

from typing import Any, Set, Tuple
from mindspore import nn, ops
from mindnlp.abc import ParameterDict
from mindnlp.peft.tuners.lycoris_utils import LycorisLayer
from mindspore.common.initializer import HeUniform, initializer


class LoHaLayer(nn.Cell, LycorisLayer):
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

    def __init__(self, base_layer: nn.Cell):
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
        # https://github.com/KohakuBlueleaf/LyCORIS/blob/eb460098187f752a5d66406d3affade6f0a07ece/lycoris/modules/loha.py#L130C9-L143C75
        if len(shape) == 4:
            self.hada_t1[adapter_name] = mindspore.Parameter(
                ops.zeros((r, r, shape[2], shape[3]))
            )
            self.hada_w1_a[adapter_name] = mindspore.Parameter(
                ops.zeros((r, shape[0]))
            )  # out_dim, 1-mode
            self.hada_w1_b[adapter_name] = mindspore.Parameter(
                ops.zeros((r, shape[1]))
            )  # in_dim , 2-mode

            self.hada_t2[adapter_name] = mindspore.Parameter(
                ops.zeros((r, r, shape[2], shape[3]))
            )
            self.hada_w2_a[adapter_name] = mindspore.Parameter(
                ops.zeros((r, shape[0]))
            )  # out_dim, 1-mode

            self.hada_w2_b[adapter_name] = mindspore.Parameter(
                ops.zeros((r, shape[1]))
            )  # in_dim , 2-mode

        else:
            self.hada_w1_a[adapter_name] = mindspore.Parameter(ops.zeros((shape[0], r)))
            self.hada_w1_b[adapter_name] = mindspore.Parameter(ops.zeros((r, shape[1])))

            self.hada_w2_a[adapter_name] = mindspore.Parameter(ops.zeros((shape[0], r)))
            self.hada_w2_b[adapter_name] = mindspore.Parameter(ops.zeros((r, shape[1])))

    # TODO
    def reset_adapter_parameters(self, adapter_name: str):
        # Original implementation performs initialization with normal distribution
        # https://github.com/KohakuBlueleaf/LyCORIS/blob/3549fdef8f564761d68b695a08ef88b1122fdedc/lycoris/modules/loha.py#L158

        # FedPara paper proposes to perform He initialization, let's stick with it
        # It is enough to initialize only single matrix with zeros to make adapter do nothing after initialization
        if adapter_name in self.hada_w1_a.keys():
            self.hada_w1_a[adapter_name].set_data(
                initializer(
                    HeUniform(math.sqrt(5)), shape=self.hada_w1_a[adapter_name].shape
                )
            )
            self.hada_w1_b[adapter_name].set_data(
                initializer(
                    HeUniform(math.sqrt(5)), shape=self.hada_w1_b[adapter_name].shape
                )
            )
            self.hada_w2_a[adapter_name].set_data(
                initializer(
                    HeUniform(math.sqrt(5)), shape=self.hada_w2_a[adapter_name].shape
                )
            )
            self.hada_w2_b[adapter_name].set_data(
                initializer(
                    HeUniform(math.sqrt(5)), shape=self.hada_w2_b[adapter_name].shape
                )
            )
        if adapter_name in self.hada_t1.keys():
            self.hada_t1[adapter_name].set_data(
                initializer(
                    HeUniform(math.sqrt(5)), shape=self.hada_t1[adapter_name].shape
                )
            )
            self.hada_t2[adapter_name].set_data(
                initializer(
                    HeUniform(math.sqrt(5)), shape=self.hada_t2[adapter_name].shape
                )
            )

    def reset_adapter_parameters_random(self, adapter_name: str):
        # Original implementation performs initialization with normal distribution
        # https://github.com/KohakuBlueleaf/LyCORIS/blob/3549fdef8f564761d68b695a08ef88b1122fdedc/lycoris/modules/loha.py#L158

        # FedPara paper proposes to perform He initialization, let's stick with it
        # It is enough to initialize only single matrix with zeros to make adapter do nothing after initialization
        if adapter_name in self.hada_w1_a.keys():
            self.hada_w1_a[adapter_name].set_data(
                initializer(
                    HeUniform(math.sqrt(5)), shape=self.hada_w1_a[adapter_name].shape
                )
            )
            self.hada_w1_b[adapter_name].set_data(
                initializer(
                    HeUniform(math.sqrt(5)), shape=self.hada_w1_b[adapter_name].shape
                )
            )
            self.hada_w2_a[adapter_name].set_data(
                initializer(
                    HeUniform(math.sqrt(5)), shape=self.hada_w2_a[adapter_name].shape
                )
            )
            self.hada_w2_b[adapter_name].set_data(
                initializer(
                    HeUniform(math.sqrt(5)), shape=self.hada_w2_b[adapter_name].shape
                )
            )
        if adapter_name in self.hada_t1.keys():
            self.hada_t1[adapter_name].set_data(
                initializer(
                    HeUniform(math.sqrt(5)), shape=self.hada_t1[adapter_name].shape
                )
            )
            self.hada_t2[adapter_name].set_data(
                initializer(
                    HeUniform(math.sqrt(5)), shape=self.hada_t2[adapter_name].shape
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
        if isinstance(base_layer, nn.Dense):
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
        # Move new weights to device
        # weight = getattr(self.get_base_layer(), "weight", None)
        # if weight is not None:
        #     # the layer is already completely initialized, this is an update
        #     if weight.is_floating_point or weight.is_complex:
        #         self.to(weight.device, dtype=weight.dtype)
        #     else:
        #         self.to(weight.device)
        self.set_adapter(self.active_adapters)

    def get_delta_weight(self, adapter_name: str) -> mindspore.Tensor:
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

    def construct(self, x: mindspore.Tensor, *args, **kwargs) -> mindspore.Tensor:
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
        self, adapter_name: str, input: mindspore.Tensor, *args: Any, **kwargs: Any
    ) -> mindspore.Tensor:
        delta_weight = self.get_delta_weight(adapter_name)
        # don't add bias here, because the bias is already included in the output of the base_layer
        return ops.dense(input, delta_weight)

    def __repr__(self) -> str:
        rep = super().__repr__()
        return "loha." + rep


class Conv2d(LoHaLayer):
    """LoHa implemented in Conv2d layer"""

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
        self, adapter_name: str, input: mindspore.Tensor, *args: Any, **kwargs: Any
    ) -> mindspore.Tensor:
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
        rep = super().__repr__()
        return "loha." + rep


# Below code is a direct copy from https://github.com/KohakuBlueleaf/LyCORIS/blob/eb460098187f752a5d66406d3affade6f0a07ece/lycoris/modules/loha.py#L9


class HadaWeight(nn.Cell):

    def __init__(self):
        super().__init__()

    def construct(self, w1a, w1b, w2a, w2b, scale=mindspore.tensor(1)):
        diff_weight = ((w1a @ w1b) * (w2a @ w2b)) * scale
        return diff_weight

    def bprop(self, w1a, w1b, w2a, w2b, scale, out, dout):
        dout = dout * scale
        temp = dout * (w2a @ w2b)
        grad_w1a = temp @ w1b.T
        grad_w1b = w1a.T @ temp

        temp = dout * (w1a @ w1b)
        grad_w2a = temp @ w2b.T
        grad_w2b = w2a.T @ temp

        return grad_w1a, grad_w1b, grad_w2a, grad_w2b, scale


class HadaWeightCP(nn.Cell):
    def __init__(self):
        super().__init__()

    def construct(self, t1, w1a, w1b, t2, w2a, w2b, scale=mindspore.tensor(1)):
        rebuild1 = ops.einsum("i j k l, j r, i p -> p r k l", t1, w1b, w1a)
        rebuild2 = ops.einsum("i j k l, j r, i p -> p r k l", t2, w2b, w2a)
        return rebuild1 * rebuild2 * scale

    def bprop(self, t1, w1a, w1b, t2, w2a, w2b, scale, out, dout):
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
    hadaweight = HadaWeight()
    return hadaweight(w1a, w1b, w2a, w2b, scale)


def make_weight_cp(t1, w1a, w1b, t2, w2a, w2b, scale):
    hadaweightcp = HadaWeightCP()
    return hadaweightcp(t1, w1a, w1b, t2, w2a, w2b, scale)
