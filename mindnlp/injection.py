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
# pylint: disable=global-variable-not-assigned
# pylint: disable=redefined-builtin
# pylint: disable=invalid-name
"""
Injection mindspore.nn for MindNLP
"""
import math
from functools import partial
from packaging import version
import mindspore
import mindspore.common.dtype as mstype
from mindspore import nn, ops, Tensor, Parameter
from mindspore.nn.layer.conv import _Conv
from mindspore.common._stub_tensor import StubTensor
from mindspore.common.initializer import initializer, Normal, HeUniform, Uniform, _calculate_fan_in_and_fan_out
from mindspore import _checkparam as Validator
from mindspore.ops._primitive_cache import _get_cache_prim
from mindnlp._legacy.functional import einsum

DEVICE_TARGET = mindspore.get_context('device_target')
GLOBAL_FP16_PATCH = False

if DEVICE_TARGET == 'Ascend':
    GLOBAL_FP16_PATCH = True

def fp16_patch_decorator(func):
    """fp16 patch on ascend"""
    def wrapper(*args, **kwargs):
        global GLOBAL_FP16_PATCH
        if GLOBAL_FP16_PATCH:
            args = [arg.astype(mstype.float16) if arg is not None and isinstance(arg, Tensor) \
                    else arg for arg in args]
            kwargs = {k: (v.astype(mstype.float16) if v is not None and isinstance(v, Tensor) else v) \
                      for k, v in kwargs.items()}
            result = func(*args, **kwargs)
            result = result.astype(mstype.float32)
            return result
        return func(*args, **kwargs)

    return wrapper

def int32_patch_decorator(func):
    """int32 patch on ascend"""
    def wrapper(*args, **kwargs):
        args = [arg.astype(mstype.int32) if isinstance(arg, Tensor) and arg.dtype in (mstype.int64, mstype.bool_) \
                else arg for arg in args]
        has_int64 = any(bool(isinstance(arg, Tensor) and arg.dtype in (mstype.int64, mstype.bool_)) for arg in args)
        kwargs = {k: (v.astype(mstype.int32) if isinstance(v, Tensor) and v.dtype in (mstype.int64, mstype.bool_) else v) \
                  for k, v in kwargs.items()}
        result = func(*args, **kwargs)
        if has_int64:
            result = result.astype(mstype.int64)
        return result

    return wrapper

def bool_patch_decorator(func):
    """bool patch on ascend"""
    def wrapper(*args, **kwargs):
        args = [arg.astype(mstype.int32) if isinstance(arg, Tensor) and arg.dtype == mstype.bool_ \
                else arg for arg in args]
        if isinstance(args[0], (list, tuple)):
            # for concat
            args[0] = [arg.astype(mstype.int32) if isinstance(arg, Tensor) and arg.dtype == mstype.bool_ \
                else arg for arg in args[0]]
        kwargs = {k: (v.astype(mstype.int32) if isinstance(v, Tensor) and v.dtype == mstype.bool_ else v) \
                  for k, v in kwargs.items()}
        result = func(*args, **kwargs)
        return result

    return wrapper

# matmul
origin_matmul = ops.matmul
ops.matmul = fp16_patch_decorator(origin_matmul)
# mm
ops.mm = fp16_patch_decorator(origin_matmul)
# addbmm
origin_addbmm = ops.addbmm
ops.addbmm = fp16_patch_decorator(origin_addbmm)
# addmm
origin_addmm = ops.addmm
ops.addmm = fp16_patch_decorator(origin_addmm)
# addmv
origin_addmv = ops.addmv
ops.addmv = fp16_patch_decorator(origin_addmv)
# addr
origin_addr = ops.addr
ops.addr = fp16_patch_decorator(origin_addr)
# baddbmm
origin_baddbmm = ops.baddbmm
ops.baddbmm = fp16_patch_decorator(origin_baddbmm)
# bmm
origin_bmm = ops.bmm
ops.bmm = fp16_patch_decorator(origin_bmm)
# dense
def dense(input, weight, bias=None):
    """patched dense"""
    dense_ = _get_cache_prim(ops.Dense)()
    return dense_(input, weight, bias)

ops.dense = fp16_patch_decorator(dense)
# einsum
ops.einsum = einsum
# conv1d
ops.conv1d = fp16_patch_decorator(ops.conv1d)

# unfold
def _get_unfold_indices(input_shape, dimension, size, step):
    if dimension < 0:
        dimension += len(input_shape)
    indices = []
    for i in range(0, input_shape[dimension] - size + 1, step):
        indices.append(list(range(i, i + size)))

    return indices, dimension

def unfold(self, dimension, size, step):
    """torch-like unfold"""
    _indices, _dimension = _get_unfold_indices(self.shape, dimension, size, step)
    indices = mindspore.Tensor(_indices).astype(mindspore.int32)
    output = ops.gather(self, indices, axis=_dimension)
    output = ops.moveaxis(output, _dimension + 1, -1)
    return output

Tensor.unfold = unfold
StubTensor.unfold = unfold

# var_mean
def var_mean(input, axis=None, *, correction=1, keepdims=False):
    """torch-like var_mean"""
    axis = Validator.check_and_canonicalize_axes(axis, input.ndim)
    x_mean = ops.mean(input, axis, True)
    x_sub = ops.sub(input, x_mean)
    x_pow = ops.pow(x_sub, 2)
    x_sum = ops.sum(x_pow, axis, keepdims)
    res_mean = ops.mean(input, axis, keepdims)
    nums = 1
    if not axis:
        nums = input.size
    else:
        for ax in axis:
            nums *= input.shape[ax]
    return ops.true_divide(x_sum, nums - correction), res_mean

ops.var_mean = var_mean

# std_mean
def std_mean(input, axis=None, *, correction=1, keepdims=False):
    """torch-like std_mean"""
    output = var_mean(input, axis, correction=correction, keepdims=keepdims)
    return ops.pow(output[0], 0.5), output[1]

ops.std_mean = std_mean

# masked_fill
def masked_fill(inputs, mask, value):
    """patched masked_fill"""
    masked_value = ops.fill(inputs.dtype, inputs.shape, value)
    return ops.select(mask, masked_value, inputs)

def _masked_fill(self, mask, value):
    return masked_fill(self, mask, value)

ops.masked_fill = masked_fill
Tensor.masked_fill = _masked_fill
StubTensor.masked_fill = _masked_fill

# ops.std
def std(input, axis=None, ddof=0, keepdims=False):
    """patched std"""
    # Calculate mean
    mean = ops.mean(input, axis=axis, keep_dims=keepdims)

    # Squared differences from the mean
    squared_diff = (input - mean)**2

    # Sum along the specified dimension
    if axis is not None:
        sum_along_dim = ops.sum(squared_diff, dim=axis, keepdim=keepdims)
    else:
        sum_along_dim = squared_diff.sum()

    # Calculate the correction factor
    factor = 1.0 / (input.shape[axis] - ddof) if axis is not None else 1.0 / (input.size - ddof)

    # Calculate the standard deviation
    out = ops.sqrt(factor * sum_along_dim)

    return out

def _std(self, axis=None, ddof=0, keepdims=False):
    return std(self, axis, ddof, keepdims)

ops.std = std
Tensor.std = _std
StubTensor.std = _std

# Tensor.__contains__
def _contains(self, key):
    eq_res = ops.equal(self, key)
    res = ops.any(eq_res)
    return bool(res)

Tensor.__contains__ = _contains
StubTensor.__contains__ = _contains

if DEVICE_TARGET == 'Ascend':
    # cumsum
    ops.cumsum = int32_patch_decorator(ops.cumsum)
    def _cumsum(self, axis):
        return ops.cumsum(self, axis)
    Tensor.cumsum = _cumsum
    StubTensor.cumsum = _cumsum
    # prod
    ops.prod = bool_patch_decorator(ops.prod)
    def prod(self, axis=None, keep_dims=False):
        """patched prod on Ascend"""
        return bool_patch_decorator(ops.prod)(self, axis, keep_dims)
    Tensor.prod = prod
    StubTensor.prod = prod
    # bitwise_or
    ops.bitwise_or = bool_patch_decorator(ops.bitwise_or)
    def bitwise_or(self, other):
        """patched bitwise_or on Ascend"""
        return bool_patch_decorator(ops.bitwise_or)(self, other)
    Tensor.bitwise_or = bitwise_or
    Tensor.__or__ = bitwise_or
    StubTensor.bitwise_or = bitwise_or
    StubTensor.__or__ = bitwise_or
    # bitwise_xor
    ops.bitwise_xor = bool_patch_decorator(ops.bitwise_xor)
    def bitwise_xor(self, other):
        """patched bitwise_xor on Ascend"""
        return bool_patch_decorator(ops.bitwise_xor)(self, other)
    Tensor.bitwise_xor = bitwise_xor
    Tensor.__xor__ = bitwise_xor
    StubTensor.bitwise_xor = bitwise_xor
    StubTensor.__xor__ = bitwise_xor
    # bitwise_and
    ops.bitwise_and = bool_patch_decorator(ops.bitwise_and)
    def bitwise_and(self, other):
        """patched bitwise_and on Ascend"""
        return bool_patch_decorator(ops.bitwise_and)(self, other)
    Tensor.bitwise_and = bitwise_and
    Tensor.__and__ = bitwise_and
    StubTensor.bitwise_and = bitwise_and
    StubTensor.__and__ = bitwise_and
    # isclose
    ops.isclose = partial(ops.isclose, equal_nan=True)
    # concat
    ops.cat = bool_patch_decorator(ops.cat)
    ops.concat = bool_patch_decorator(ops.concat)



if version.parse(mindspore.__version__) < version.parse('2.2.0'):
    def eq(self, other):
        """patched eq"""
        return ops.equal(self, other)
    Tensor.eq = eq
    StubTensor.eq = eq

class Dense(nn.Cell):
    """patched Dense"""
    def __init__(self,
                 in_channels,
                 out_channels,
                 has_bias=True,
                 dtype=mstype.float32):
        """Initialize Dense."""
        super().__init__()
        self.in_channels = Validator.check_positive_int(
            in_channels, "in_channels", self.cls_name)
        self.out_channels = Validator.check_positive_int(
            out_channels, "out_channels", self.cls_name)
        self.has_bias = Validator.check_bool(
            has_bias, "has_bias", self.cls_name)

        self.weight = Parameter(initializer(
            'zeros', [out_channels, in_channels], dtype=dtype), name="weight")

        self.bias = None
        if self.has_bias:
            self.bias = Parameter(initializer(
                'zeros', [out_channels], dtype=dtype), name="bias")
        self.reset_parameters()

    def reset_parameters(self):
        """reset_embedding_params"""
        self.weight.set_data(initializer(HeUniform(math.sqrt(5)), self.weight.shape))
        if self.has_bias:
            fan_in, _ = _calculate_fan_in_and_fan_out(self.weight.shape)
            bound = 1 / math.sqrt(fan_in)
            self.bias.set_data(initializer(Uniform(bound), [self.out_channels]))

    def construct(self, x):
        x_shape = x.shape
        if len(x_shape) != 2:
            x = x.reshape(-1, x.shape[-1])
        x = ops.matmul(x, self.weight.T)
        if self.has_bias:
            x = ops.add(x, self.bias)
        if len(x_shape) != 2:
            out_shape = x_shape[:-1] + (x.shape[-1],)
            x = x.reshape(out_shape)
        # return ops.dense(x, self.weight, self.bias)
        return x

class Embedding(nn.Cell):
    """patched Embedding"""
    def __init__(self, vocab_size, embedding_size, use_one_hot=False, dtype=mstype.float32, padding_idx=None):
        """Initialize Embedding."""
        super().__init__()
        self.vocab_size = Validator.check_value_type('vocab_size', vocab_size, [int], self.cls_name)
        self.embedding_size = Validator.check_value_type('embedding_size', embedding_size, [int], self.cls_name)
        Validator.check_value_type('use_one_hot', use_one_hot, [bool], self.cls_name)
        Validator.check_subclass("dtype", dtype, mstype.number_type, self.cls_name)
        self.use_one_hot = use_one_hot
        self.dtype = dtype
        self.padding_idx = padding_idx
        self.embedding_table = Parameter(initializer('zeros', [vocab_size, embedding_size]), name='embedding_table')
        self.reset_parameters()

    def reset_parameters(self):
        """reset_embedding_params"""
        init_tensor = initializer(Normal(1.0), self.embedding_table.shape)
        init_tensor = init_tensor.init_data()
        if self.padding_idx:
            init_tensor = init_tensor.asnumpy()
            init_tensor[self.padding_idx] = 0
            init_tensor = Tensor(init_tensor)
        self.embedding_table.assign_value(init_tensor)

    def construct(self, ids):
        out_shape = ids.shape + (self.embedding_size,)
        flat_ids = ids.reshape((-1,))

        if self.use_one_hot:
            one_hot_ids = ops.one_hot(flat_ids, self.vocab_size)
            output_for_reshape = ops.matmul(one_hot_ids, self.embedding_table)
        else:
            output_for_reshape = ops.gather(self.embedding_table, flat_ids, 0)

        output = output_for_reshape.reshape(out_shape)
        return output

    def extend_repr(self):
        return f'vocab_size={self.vocab_size}, embedding_size={self.embedding_size}, use_one_hot={self.use_one_hot}, ' \
            f'embedding_table={self.embedding_table}, dtype={self.dtype}, padding_idx={self.padding_idx}'

class Conv1d(_Conv):
    """patched Conv1d"""
    def __init__(self,
                 in_channels,
                 out_channels,
                 kernel_size,
                 stride=1,
                 pad_mode='same',
                 padding=0,
                 dilation=1,
                 group=1,
                 has_bias=False):
        """Initialize Conv1d."""
        Validator.check_value_type("kernel_size", kernel_size, [int], self.cls_name)
        Validator.check_value_type("stride", stride, [int], self.cls_name)
        Validator.check_value_type("padding", padding, [int], self.cls_name)
        Validator.check_value_type("dilation", dilation, [int], self.cls_name)
        Validator.check_int(kernel_size, 1, Validator.GE, 'kernel_size', self.cls_name)
        Validator.check_int(stride, 1, Validator.GE, 'stride', self.cls_name)
        Validator.check_non_negative_int(padding, 'padding', self.cls_name)
        Validator.check_int(dilation, 1, Validator.GE, 'dilation', self.cls_name)
        Validator.check_positive_int(group, 'group', self.cls_name)
        if not (in_channels % group == 0 and out_channels % group == 0):
            raise ValueError(f"The argument 'group' should be divisible by 'in_channels' " \
                             f"and 'out_channels', but got group:{group}, in_channels:{in_channels}, " \
                             f"out_channels:{out_channels}.")
        kernel_size = (kernel_size,)
        if mindspore.__version__ == '2.0.0':
            stride = (1, stride,)
        else:
            stride = (stride,)

        dilation = (dilation,)

        super().__init__(
            in_channels,
            out_channels,
            kernel_size,
            stride,
            pad_mode,
            padding,
            dilation,
            group,
            has_bias,
            'zeros',
            'zeros')
        self.padding = padding

    def construct(self, x):
        return ops.conv1d(x, self.weight, self.bias, stride=self.stride, pad_mode=self.pad_mode,
                          padding=self.padding, dilation=self.dilation, groups=self.group)

class LayerNorm(nn.Cell):
    r"""
    Applies Layer Normalization over a mini-batch of inputs.
    """

    def __init__(self,
                 normalized_shape,
                 begin_norm_axis=-1,
                 begin_params_axis=-1,
                 gamma_init='ones',
                 beta_init='zeros',
                 epsilon=1e-7,
                 dtype=mstype.float32,
                 elementwise_affine=True
                 ):
        """Initialize LayerNorm."""
        super().__init__()
        if not isinstance(normalized_shape, (tuple, list)):
            raise TypeError(f"For '{self.cls_name}', the type of 'normalized_shape' must be tuple[int] or list[int], "
                            f"but got {normalized_shape} and the type is {type(normalized_shape)}.")
        if not normalized_shape:
            raise ValueError(
                f"Expected normalized_shape to be at least 1-dimensional, i.e., containing at "
                f"least one element, but got normalized_shape = {normalized_shape}"
            )
        self.normalized_shape = normalized_shape
        self.begin_norm_axis = begin_norm_axis
        self.begin_params_axis = begin_params_axis
        self.epsilon = epsilon
        self.gamma = Parameter(initializer(
            gamma_init, normalized_shape, dtype=dtype), name="gamma")
        self.beta = Parameter(initializer(
            beta_init, normalized_shape, dtype=dtype), name="beta")
        self.layer_norm = ops.LayerNorm(begin_norm_axis=self.begin_norm_axis,
                                      begin_params_axis=self.begin_params_axis,
                                      epsilon=self.epsilon)
        self.elementwise_affine = elementwise_affine

    def construct(self, input_x):
        if self.elementwise_affine:
            y, _, _ = self.layer_norm(input_x, self.gamma.astype(input_x.dtype), self.beta.astype(input_x.dtype))
        else:
            y, _, _ = self.layer_norm(input_x, ops.ones(self.normalized_shape, input_x.dtype),
                                      ops.zeros(self.normalized_shape, input_x.dtype),)
        return y

    def extend_repr(self):
        return f'normalized_shape={self.normalized_shape}, begin_norm_axis={self.begin_norm_axis}, ' \
               f'begin_params_axis={self.begin_params_axis}, gamma={self.gamma}, beta={self.beta}'


def half(self):
    """patched nn.Cell.half"""
    for param in self.get_parameters():
        if param.dtype in (mindspore.float32, mindspore.float16):
            param.set_dtype(mindspore.float16)

nn.Cell.half = half

nn.LayerNorm = LayerNorm
nn.Conv1d = Conv1d
nn.Embedding = Embedding
nn.Dense = Dense
