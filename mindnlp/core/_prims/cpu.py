import numbers
from mindspore.ops.auto_generate import gen_ops_prim
from mindspore.ops._primitive_cache import _get_cache_prim
from mindspore._c_expression import _empty_instance
from mindspore.ops.operations._grad_ops import StridedSliceGrad

import mindspore
from mindspore import ops

from mindnlp import core

__all__ = []
op_list = list(filter(lambda s: s.endswith("_op"), dir(gen_ops_prim)))

for op_name in op_list:
    func_name = op_name.replace('_op', '')
    __all__.append(func_name)
    globals()[func_name] = getattr(gen_ops_prim, op_name).__class__().set_device('CPU')

def empty(*args, **kwargs):
    return _empty_instance(*args, **kwargs, device='CPU')

normal_op = ops.StandardNormal().set_device('CPU')
def normal(*args, **kwargs):
    return normal_op(*args, **kwargs)

__all__.append('normal')

full_op = ops.FillV2().set_device('CPU')
def full(*args):
    return full_op(*args)

__all__.append('full')

range_op = ops.Range().set_device('CPU')
def arange(start, end, step, dtype):
    return cast(range_op(start, end, step), dtype)

__all__.append('arange')


broadcast_to_op = ops.Primitive('BroadcastTo').set_device('CPU')
def broadcast_to(*args):
    return broadcast_to_op(*args)

__all__.append('broadcast_to')

def concat(tensors, dim):
    concat_op = ops.Concat(dim).set_device('CPU')
    return concat_op(tensors)

__all__.append('concat')

zeros_op = ops.Zeros().set_device('CPU')
def zeros(*args):
    return zeros_op(*args)

__all__.append('zeros')

ones_op = ops.Ones().set_device('CPU')
def ones(*args):
    return ones_op(*args)

__all__.append('ones')

uniform_real_op = ops.UniformReal().set_device('CPU')
def uniform_real(*args):
    return uniform_real_op(*args)

__all__.append('uniform_real')

def pad_v3(input_x, padding, mode='constant', value=None):
    pad_op = ops.PadV3(mode=mode, paddings_contiguous=True).set_device('CPU')
    if isinstance(value, (float, int)):
        value = core.tensor(value, dtype=input_x.dtype)
    return pad_op(input_x, padding, value)

__all__.append('pad_v3')

reduce_any_op = ops.ReduceAny().set_device('CPU')
reduce_any_keepdim_op = ops.ReduceAny(True).set_device('CPU')
def reduce_any(input, dim, keepdim):
    if keepdim:
        return reduce_any_keepdim_op(input, dim)
    return reduce_any_op(input, dim)

__all__.append('reduce_any')

reduce_all_op = ops.ReduceAll().set_device('CPU')
reduce_all_keepdim_op = ops.ReduceAll(True).set_device('CPU')
def reduce_all(input, dim, keepdim):
    if keepdim:
        return reduce_all_keepdim_op(input, dim)
    return reduce_all_op(input, dim)

__all__.append('reduce_all')

def isclose(input, other, rtol=1e-05, atol=1e-08, equal_nan=False):
    is_close = _get_cache_prim(ops.IsClose)(rtol=rtol, atol=atol, equal_nan=equal_nan).set_device('CPU')
    return is_close(input, other)

__all__.append('isclose')

tile_op = ops.Primitive('Tile').set_device('CPU')
def tile(*args):
    return tile_op(*args)

__all__.append('tile')

def randint(low, high, shape, dtype, generator):
    rand_op = ops.UniformInt().set_device('CPU')
    output = rand_op(shape, mindspore.Tensor(low, mindspore.int32), mindspore.Tensor(high, mindspore.int32))
    return cast(output, dtype)
    # return mindspore.Tensor(np.random.randint(low, high, shape))

cast_op = ops.Cast().set_device('CPU')
def cast(input, dtype):
    return cast_op(input, dtype)

__all__.append('cast')

def tril_ext(input, diagonal):
    tril_op = ops.Tril(diagonal).set_device('CPU')
    return tril_op(input)

def strided_slice(input, begin, end, strides, begin_mask, end_mask, ellipsis_mask, new_axis_mask, shrink_axis_mask):
    strided_slice_op = _get_cache_prim(ops.StridedSlice)(begin_mask, end_mask, ellipsis_mask, new_axis_mask, shrink_axis_mask).set_device('CPU')
    return strided_slice_op(input, begin, end, strides)

__all__.append('strided_slice')

def strided_slice_grad(input, begin, end, strides, update, begin_mask=0, end_mask=0, ellipsis_mask=0, new_axis_mask=0, shrink_axis_mask=0):
    strided_slice_grad = _get_cache_prim(StridedSliceGrad)(begin_mask, end_mask, ellipsis_mask, new_axis_mask, shrink_axis_mask).set_device('CPU')
    return strided_slice_grad(update, input.shape, begin, end, strides)

__all__.append('strided_slice_grad')

def squeeze(input, dim):
    squeeze_op = ops.Squeeze(dim).set_device('CPU')
    return squeeze_op(input)

__all__.append('squeeze')

def sort_ext(input, dim, descending, stable):
    sort_op = ops.Sort(dim, descending).set_device('CPU')
    return sort_op(input)

__all__.append('sort_ext')

def stack(tensors, dim):
    stack_op = ops.Stack(dim).set_device('CPU')
    return stack_op(tensors)

__all__.append('stack')

def gather(input_params, input_indices, axis, batch_dims=0):
    gather_op = _get_cache_prim(ops.Gather)(batch_dims).set_device('CPU')
    return gather_op(input_params, input_indices, axis)

__all__.append('gather')

def softmax(input, dim):
    softmax_op = ops.Softmax(dim).set_device('CPU')
    return softmax_op(input)

__all__.append('softmax')

def topk(input, k, sorted=True):
    topk_op = ops.TopK(sorted).set_device('CPU')
    return topk_op(input, k)

__all__.append('topk')

dyn_shape_op = ops.TensorShape().set_device('CPU')
def dyn_shape(self):
    return dyn_shape_op(self)

__all__.append('dyn_shape')

bitwise_and_op = ops.BitwiseAnd().set_device('CPU')
def bitwise_and_scalar(input, other):
    return bitwise_and_op(input, other)

bitwise_right_shift_op = ops.RightShift().set_device('CPU')
def bitwise_right_shift(input, other):
    if isinstance(input, numbers.Number):
        if not isinstance(input, int):
            raise TypeError(f"For 'bitwise_left_shift', 'input' must be an integer, but got input:{type(input)}.")
        input = cast(input, other.dtype)
    elif isinstance(other, numbers.Number):
        if not isinstance(other, int):
            raise TypeError(f"For 'bitwise_left_shift', 'other' must be an integer, but got other:{type(other)}.")
        other = cast(other, input.dtype)
    return bitwise_right_shift_op(input, other)

__all__.append('bitwise_right_shift')

embedding_op = ops.Gather().set_device('CPU')
def embedding(input, weight, padding_idx, max_norm, norm_type, scale_grad_by_freq):
    return embedding_op(weight, input, 0)

__all__.append('embedding')


def randn(size, seed, offset, dtype):
    rand_op = ops.StandardNormal()
    output = rand_op(size)
    return output

__all__.append('randn')

