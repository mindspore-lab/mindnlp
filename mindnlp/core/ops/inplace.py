import numbers
import numpy as np
import mindspore
from mindspore import ops
from mindspore._c_expression import typing
from mindspore.ops._primitive_cache import _get_cache_prim
from mindspore.common.generator import default_generator
from mindspore.ops.auto_generate.gen_ops_prim import inplace_normal_op, inplace_scatter_value_op, inplace_scatter_src_reduce_op, \
    inplace_scatter_src_op, inplace_fill_tensor_op, inplace_fill_scalar_op, inplace_zero_op, inplace_uniform_op, \
    inplace_masked_fill_scalar_op, inplace_masked_fill_tensor_op, inplace_random_op, inplace_clamp_scalar_op, \
    inplace_clamp_tensor_op, inplace_copy_op, inplace_index_add_op, inplace_erfinv_op

from mindnlp import core
from ..configs import use_pyboost
from ._inner import assign

generator_step_ = 12

def inplace_copy(self, other):
    if self.device.type == 'npu':
        inplace_copy_op(self, other)
    else:
        self.data = other
    return self

def inplace_zero(input):
    if input.device == 'npu':
        inplace_zero_op(input)
    else:
        input.data = ops.zeros(input.shape, dtype=input.dtype)
    return input

def inplace_fill(input, value):
    if input.device.type == 'npu':
        if isinstance(value, (int, float, bool)):
            inplace_fill_scalar_op(input, value)
        else:
            inplace_fill_tensor_op(input, value)
    else:
        input.data = ops.full(input.shape, value, dtype=input.dtype)
    return input

def inplace_normal(input, mean=0, std=1, *, generator=None):
    if generator is None:
        generator = default_generator
    seed, offset = generator._step(generator_step_)
    if isinstance(mean, core.Tensor):
        mean = mean.item()
    if isinstance(std, core.Tensor):
        std = std.item()
    if input.device.type == 'npu':
        inplace_normal_op(input, mean, std, seed, offset)
    else:
        input.data = core.tensor(np.random.normal(mean, std, input.shape), dtype=input.dtype)
    return input

# uniform_
def inplace_uniform(input, *args, **kwargs):
    if len(args) == 1:
        from_ = args[0]
        to_ = None
    elif len(args) == 2:
        from_ = args[0]
        to_ = args[1]
    elif len(args) == 3:
        from_ = args[0]
        to_ = args[1]
    else:
        from_ = 0
        to_ = 1

    from_ = kwargs.get("from", 0) if from_ is None else from_
    # to_ = kwargs.get("to", 1)
    generator_ = kwargs.get("generator", None)
    if generator_ is None:
        generator_ = default_generator
    seed, offset = generator_._step(generator_step_)
    if input.device.type == 'npu':
        inplace_uniform_op(input, from_, to_, seed, offset)
    else:
        input.data = core.tensor(np.random.uniform(from_, to_, input.shape), dtype=input.dtype)
        # core.rand(input.shape, generator=generator_, dtype=input.dtype) * (to_ - from_) + from_
    return input

def inplace_add(input, other, alpha):
    execute('inplace_add_ext', input, other, alpha)
    return input

def inplace_scatter(input, dim, index, src):
    if not isinstance(src, numbers.Number):
        return inplace_scatter_src_op(input, dim, index, src)
    return inplace_scatter_value_op(input, dim, index, src)

def inplace_index_copy(input, dim, index, tensor):
    selected = input.index_select(dim, index)
    input.index_add_(dim, index, -selected)
    input.index_add_(dim, index, tensor)
    return input

def inplace_index_add(input, dim, index, source):
    if input.device == 'npu':
        inplace_index_add_op(input, dim, index, source)
    else:
        _inplace = _get_cache_prim(ops.IndexAdd)(dim)
        input.data = _inplace(input, index.int(), source)
    return input

has_squeeze = hasattr(mindspore.mint, 'squeeze')
def inplace_squeeze(input, *dim, **kwargs):
    dim = kwargs.get('dim', dim)
    if use_pyboost() and has_squeeze:
        out = mindspore.mint.squeeze(input, dim)
    else:
        out = ops.squeeze(input, dim)
    input.assign_value(out)
    return input


has_unsqueeze = hasattr(mindspore.mint, 'unsqueeze')
def inplace_unsqueeze(input, dim=None):
    if use_pyboost() and has_unsqueeze:
        out = mindspore.mint.unsqueeze(input, dim)
    out = ops.expand_dims(input, dim)
    input.assign_value(out)
    return input

def inplace_fill_diagonal(input, fill_value, wrap=False):
    fill_diagnoal_ = _get_cache_prim(ops.FillDiagonal)(float(fill_value), wrap)
    out = fill_diagnoal_(input)
    input.assign_value(out)
    return input

def inplace_triu(input, diagonal=0):
    out = ops.triu(input, diagonal)
    input.assign_value(out)
    return input

def inplace_round(input, decimals=0):
    out = ops.round(input, decimals=decimals)
    input.assign_value(out)
    return input

def inplace_scatter_reduce(input, dim, index, src, reduce, *, include_self=True):
    if reduce == 'sum':
        reduce = "add"
    return inplace_scatter_src_reduce_op(input, dim, index, src, reduce)

def inplace_exponential(tensor, lambd=1.0):
    """
    原地操作的指数分布采样 (类似Tensor.exponential_)
    :param tensor: 要填充的目标张量
    :param lambd: 率参数 (λ > 0)
    :return: 修改后的张量 (原张量被覆盖)
    """
    assert lambd > 0, "lambd 必须大于0"
    
    # 生成与目标张量形状相同的均匀分布随机数
    u = core.rand_like(tensor)
    
    # 数值保护
    u = u.clamp(min=core.finfo(u.dtype).eps, max=1.0)
    
    # 逆变换法赋值
    tensor.data = -core.log(1 - u) / lambd

    return tensor

def inplace_log(self):
    self.data = core.log(self)
    return self

def inplace_mul(self, other):
    self.data = core.mul(self, other)
    return self

def inplace_neg(self):
    self.data = core.neg(self)
    return self

def inplace_exp(self):
    self.data = core.exp(self)
    return self

def inplace_sub(self, other):
    self.data = core.sub(self, other)
    return self

def inplace_bernoulli(self, p=0.5, *, generator=None):
    self.data = core.bernoulli(self, generator=generator, p=p)
    return self

def inplace_tril(self, diagonal=0):
    self.data = core.tril(self, diagonal)
    return self

def inplace_masked_fill(self, mask, value):
    if self.device.type == 'npu':
        if isinstance(value, (int, float, bool)):
            inplace_masked_fill_scalar_op(self, mask, value)
        else:
            inplace_masked_fill_tensor_op(self, mask, value)
    else:
        self.data = ops.masked_fill(self, mask, value)
    return self

def inplace_random(self, from_=0, to=None, *, generator=None):
    if self.device.type == 'npu':
        if not generator:
            generator = default_generator
        seed, offset = generator._step(  # pylint: disable=protected-access
            generator_step_)
        return inplace_random_op(input, from_, to, seed, offset)
    else:
        if isinstance(self.dtype, typing.Float):
            self.uniform_(from_, to, generator=generator)
        elif isinstance(self.dtype, typing.Int):
            if to is None:
                to = core.iinfo(mindspore.int32).max
            self.data = core.randint(from_, to, size=self.shape, dtype=self.dtype)
    return self

def inplace_clamp(self, min=None, max=None):
    if self.device.type == 'npu':
        if isinstance(min, (int, float, bool)) or isinstance(max, (int, float, bool)):
            inplace_clamp_scalar_op(self, min, max)
        else:
            inplace_clamp_tensor_op(self, min, max)
    else:
        self.data = ops.clamp(self, min, max)
    return self

def inplace_erfinv(self):
    if self.device.type == 'npu':
        inplace_erfinv_op(self)
    else:
        self.data = core.erfinv(self)
    return self

__all__ = [
    'inplace_copy',
    'inplace_zero',
    'inplace_normal',
    'inplace_fill',
    'inplace_uniform',
    'inplace_add',
    'inplace_scatter',
    'inplace_index_copy',
    'inplace_index_add',
    'inplace_squeeze',
    'inplace_unsqueeze',
    'inplace_fill_diagonal',
    'inplace_triu',
    'inplace_round',
    'inplace_scatter_reduce',
    'inplace_exponential',
    'inplace_log',
    'inplace_mul',
    'inplace_neg',
    'inplace_exp',
    'inplace_sub',
    'inplace_bernoulli',
    'inplace_tril',
    'inplace_masked_fill',
    'inplace_random',
    'inplace_clamp',
    'inplace_erfinv'
]
