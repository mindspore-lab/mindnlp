# Copyright 2024 Huawei Technologies Co., Ltd
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
'''
    mindbnb modules
'''
# pylint: disable=E0611, E0401, W1113
import copy
from typing import TypeVar

import mindspore

from mindspore._c_expression import (
    Tensor as CTensor,
)  # pylint: disable=no-name-in-module, import-error

import bitsandbytes as bnb
from bitsandbytes.autograd._functions import get_tile_inds, undo_layout
from bitsandbytes.utils import (
    INVERSE_LINEAR_8BIT_WEIGHTS_FORMAT_MAPPING,
    LINEAR_8BIT_WEIGHTS_FORMAT_MAPPING,
)

from mindnlp.core import nn
from mindnlp.core.nn import Parameter


def empty(*size, dtype=None):
    if isinstance(size[0], (tuple, list)):
        size = size[0]
    out = CTensor(dtype, size)
    return mindspore.Tensor(out)


T = TypeVar("T", bound="mindspore.nn.Module")


class Int8Params(Parameter):

    def __new__(
        cls,
        data=None,
        requires_grad=True,
        has_fp16_weights=False,
        CB=None,
        SCB=None,
        *args,
        **kwargs,
    ):
        if data is None:
            data = empty(0, dtype=mindspore.float16)

        obj = super(Int8Params, cls).__new__(cls, data)
        obj.__init__(data, requires_grad=requires_grad, *args, **kwargs)
        # parent_class = Int8Params.__bases__[0]
        # super(parent_class, obj).__init__(data, requires_grad, *args, **kwargs)

        # 初始化子类属性
        obj.has_fp16_weights = has_fp16_weights
        obj.CB = CB
        obj.SCB = SCB

        return obj

    def __deepcopy__(self, memo):
        # Perform deep copy of the instance
        new_instance = type(self).__new__(
            type(self),
            data=copy.deepcopy(self.data, memo),
            requires_grad=self.requires_grad,
            has_fp16_weights=self.has_fp16_weights,
            CB=copy.deepcopy(self.CB, memo),
            SCB=copy.deepcopy(self.SCB, memo),
        )
        return new_instance


def maybe_rearrange_weight(
    state_dict,
    prefix,
    local_metadata,
    strict,
    missing_keys,
    unexpected_keys,
    error_msgs,
):
    weight = state_dict.get(f"{prefix}weight")
    if weight is None:
        # if the state dict has no weights for this layer (e.g., LoRA finetuning), do nothing
        return
    weight_format = state_dict.pop(f"{prefix}weight_format", "row")

    if isinstance(weight_format, mindspore.Tensor):
        weight_format = weight_format.item()

    # For new weights format storage type, we explicitly check
    # if weights_format is on the mapping
    if (
        isinstance(weight_format, int)
        and weight_format not in INVERSE_LINEAR_8BIT_WEIGHTS_FORMAT_MAPPING
    ):
        raise ValueError(f"Expected supported weight format - got {weight_format}")
    elif (
        isinstance(weight_format, int)
        and weight_format in INVERSE_LINEAR_8BIT_WEIGHTS_FORMAT_MAPPING
    ):
        weight_format = INVERSE_LINEAR_8BIT_WEIGHTS_FORMAT_MAPPING[weight_format]

    if weight_format != "row":
        tile_indices = get_tile_inds(weight_format, weight.device)
        state_dict[f"{prefix}weight"] = undo_layout(weight, tile_indices)


class Linear8bitLt(nn.Linear):
    """
    This class is the base module for the [LLM.int8()](https://arxiv.org/abs/2208.07339) algorithm.
    To read more about it, have a look at the paper.

    In order to quantize a linear layer one should first load the original fp16 / bf16 weights into
    the Linear8bitLt module, then call `int8_module.to("cuda")` to quantize the fp16 weights.

    Example:

    ```python
    import torch
    import torch.nn as nn

    import bitsandbytes as bnb
    from bnb.nn import Linear8bitLt

    fp16_model = nn.Sequential(
        nn.Linear(64, 64),
        nn.Linear(64, 64)
    )

    int8_model = nn.Sequential(
        Linear8bitLt(64, 64, has_fp16_weights=False),
        Linear8bitLt(64, 64, has_fp16_weights=False)
    )

    int8_model.load_state_dict(fp16_model.state_dict())
    int8_model = int8_model.to(0) # Quantization happens here
    ```
    """

    def __init__(
        self,
        input_features: int,
        output_features: int,
        bias=True,
        has_fp16_weights=True,
        memory_efficient_backward=False,
        threshold=0.0,
        index=None,
    ):
        """
        Initialize Linear8bitLt class.

        Args:
            input_features (`int`):
                Number of input features of the linear layer.
            output_features (`int`):
                Number of output features of the linear layer.
            bias (`bool`, defaults to `True`):
                Whether the linear class uses the bias term as well.
        """
        super().__init__(input_features, output_features, bias)
        assert (
            not memory_efficient_backward
        ), "memory_efficient_backward is no longer required and the argument is deprecated in 0.37.0 and will be removed in 0.39.0"
        self.state = bnb.MatmulLtState()
        self.index = index

        self.state.threshold = threshold
        self.state.has_fp16_weights = has_fp16_weights
        self.state.memory_efficient_backward = memory_efficient_backward
        if threshold > 0.0 and not has_fp16_weights:
            self.state.use_pool = True

        self.weight = Int8Params(
            self.weight.data,
            requires_grad=has_fp16_weights,
            has_fp16_weights=has_fp16_weights,
        )

    def _save_to_state_dict(self, destination, prefix, keep_vars):
        super()._save_to_state_dict(destination, prefix, keep_vars)

        # we only need to save SCB as extra data, because CB for quantized weights is already stored in weight.data
        scb_name = "SCB"

        # case 1: .cuda was called, SCB is in self.weight
        param_from_weight = getattr(self.weight, scb_name)
        # case 2: self.init_8bit_state was called, SCB is in self.state
        param_from_state = getattr(self.state, scb_name)
        # case 3: SCB is in self.state, weight layout reordered after first forward()
        layout_reordered = self.state.CxB is not None

        key_name = prefix + f"{scb_name}"
        format_name = prefix + "weight_format"

        if not self.state.has_fp16_weights:
            if param_from_weight is not None:
                destination[key_name] = (
                    param_from_weight if keep_vars else param_from_weight.detach()
                )
                destination[format_name] = mindspore.tensor(0, dtype=mindspore.uint8)
            elif param_from_state is not None and not layout_reordered:
                destination[key_name] = (
                    param_from_state if keep_vars else param_from_state.detach()
                )
                destination[format_name] = mindspore.tensor(0, dtype=mindspore.uint8)
            elif param_from_state is not None:
                destination[key_name] = (
                    param_from_state if keep_vars else param_from_state.detach()
                )
                weights_format = self.state.formatB
                # At this point `weights_format` is an str
                if weights_format not in LINEAR_8BIT_WEIGHTS_FORMAT_MAPPING:
                    raise ValueError(f"Unrecognized weights format {weights_format}")

                weights_format = LINEAR_8BIT_WEIGHTS_FORMAT_MAPPING[weights_format]

                destination[format_name] = mindspore.tensor(
                    weights_format, dtype=mindspore.uint8
                )

    def _load_from_state_dict(
        self,
        state_dict,
        prefix,
        local_metadata,
        strict,
        missing_keys,
        unexpected_keys,
        error_msgs,
    ):
        super()._load_from_state_dict(
            state_dict,
            prefix,
            local_metadata,
            strict,
            missing_keys,
            unexpected_keys,
            error_msgs,
        )
        unexpected_copy = list(unexpected_keys)

        for key in unexpected_copy:
            input_name = key[len(prefix) :]
            if input_name == "SCB":
                if self.weight.SCB is None:
                    # buffers not yet initialized, can't access them directly without quantizing first
                    raise RuntimeError(
                        "Loading a quantized checkpoint into non-quantized Linear8bitLt is "
                        "not supported. Please call module.cuda() before module.load_state_dict()",
                    )

                input_param = state_dict[key]
                self.weight.SCB.copy_(input_param)

                if self.state.SCB is not None:
                    self.state.SCB = self.weight.SCB

                unexpected_keys.remove(key)

    def init_8bit_state(self):
        self.state.CB = self.weight.CB
        self.state.SCB = self.weight.SCB
        self.weight.CB = None
        self.weight.SCB = None

    def quant(
        self,
    ):
        for key, param in self.parameters_dict().items():
            if param is None:
                continue
            if key == "weight":
                self.cuda(self.weight)
        return self

    def cuda(self, param):
        if param.has_fp16_weights:
            param.data.astype(mindspore.float16)
            return self
        else:
            # we store the 8-bit rows-major weight
            # we convert this weight to the turning/ampere weight during the first inference pass
            B = param.data.astype(mindspore.float16)
            CB, CBt, SCB, SCBt, coo_tensorB = bnb.functional.double_quant(B)
            del CBt
            del SCBt
            param.assign_value(CB)
            param.CB = CB
            param.SCB = SCB

        return self

    def forward(self, x: mindspore.Tensor):
        self.state.is_training = self.training
        if self.weight.CB is not None:
            self.init_8bit_state()

        # weights are cast automatically as Int8Params, but the bias has to be cast manually
        if self.bias is not None and self.bias.dtype != x.dtype:
            self.bias = Parameter(
                self.bias.astype(x.dtype), requires_grad=self.bias.requires_grad
            )

        out = bnb.matmul(x, self.weight, bias=self.bias, state=self.state)

        if not self.state.has_fp16_weights:
            if self.state.CB is not None and self.state.CxB is not None:
                # we converted 8-bit row major to turing/ampere format in the first inference pass
                # we no longer need the row-major weight
                del self.state.CB
                self.weight.assign_value(self.state.CxB)
        return out
