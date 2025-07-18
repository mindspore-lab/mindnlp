import mindspore
from mindspore import ops
from mindspore.ops._primitive_cache import _get_cache_prim
from mindspore.common.generator import default_generator
from mindspore.ops.auto_generate.gen_ops_prim import inplace_normal_op

from mindnlp import core
from ..configs import use_pyboost
from ._inner import assign

generator_step_ = 12

def inplace_copy(self, other):
    if self.device != other.device:
        other = other.to(self.device)
    if self.device.type == 'cpu':
        # execute('assign', self, other)
        # # self._data.assign_value_cpp(other._data)
        self.data = other
    else:
        execute('inplace_copy', self, other)
    return self

def inplace_zero(input):
    device = input.device
    if input.device == 'npu':
        execute('inplace_zero', input)
    elif input.device.type == 'cpu':
        out = execute('zeros', input.shape, input.dtype, device=device)
        input.data = out
    return input

def inplace_fill(input, value):
    device = input.device
    if input.device == 'npu':
        if isinstance(value, (int, float, bool)):
            execute('inplace_fill_scalar', input, value)
        execute('inplace_fill_tensor', input, value)
    elif input.device.type == 'cpu':
        out = execute('full', input.shape, value, device=device)
        input.data = out
    return input

def inplace_normal(input, mean=0, std=1, *, generator=None):
    if generator is None:
        generator = default_generator
    seed, offset = generator._step(generator_step_)
    if isinstance(mean, core.Tensor):
        mean = mean.item()
    if isinstance(std, core.Tensor):
        std = std.item()
    inplace_normal_op(input, mean, std, seed, offset)

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
        execute("inplace_uniform", input, from_, to_, seed, offset)
    elif input.device.type == 'cpu':
        input.data = core.rand(input.shape, generator=generator_, dtype=input.dtype) * (to_ - from_) + from_
    return input

def inplace_add(input, other, alpha):
    execute('inplace_add_ext', input, other, alpha)
    return input

def inplace_scatter(input, dim, index, src):
    if not isinstance(src, core.Tensor):
        return execute('inplace_scatter_value', input, dim, index, src)
    return execute('inplace_scatter', input, dim, index, src)

def inplace_index_copy(input, dim, index, tensor):
    selected = input.index_select(dim, index)
    input.index_add_(dim, index, -selected)
    input.index_add_(dim, index, tensor)
    return input

def inplace_index_add(input, dim, index, source):
    _inplace = _get_cache_prim(ops.InplaceIndexAdd)(dim)
    return _inplace(input, index, source)

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
    'inplace_unsqueeze'
]
