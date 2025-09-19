import numbers
import numpy as np
from mindspore import ops
from mindspore.ops._primitive_cache import _get_cache_prim
from mindspore.ops.auto_generate import gen_ops_prim
from mindspore.ops.auto_generate import pyboost_inner_prim
from mindspore._c_expression import _empty_instance

import mindtorch
from mindtorch._C import default_generator

op_list = list(filter(lambda s: s.endswith("_op"), dir(gen_ops_prim)))

__all__ = []

for op_name in op_list:
    func_name = op_name.replace('_op', '')
    __all__.append(func_name)
    globals()[func_name] = getattr(gen_ops_prim, op_name).__class__().set_device('Ascend')

def empty(*args, **kwargs):
    return _empty_instance(*args, **kwargs, device='Ascend')

def reduce_any(input, dim, keepdim):
    if dim is None:
        dim = ()
    return pyboost_inner_prim.reduce_any_impl(input, dim, keepdim)

__all__.append('reduce_any')

def reduce_all(input, dim, keepdim):
    if dim is None:
        dim = ()
    return pyboost_inner_prim.reduce_all_impl(input, dim, keepdim)

__all__.append('reduce_all')

broadcast_to_op = ops.Primitive('BroadcastTo').set_device('Ascend')
def broadcast_to(*args):
    return broadcast_to_op(*args)

__all__.append('broadcast_to')

cast_op = ops.Cast().set_device('Ascend')
def cast(*args):
    return cast_op(*args)

__all__.append('cast')

zeros_op = ops.Zeros().set_device('Ascend')
def zeros(*args):
    return zeros_op(*args)

__all__.append('zeros')

def softmax(*args):
    return pyboost_inner_prim.softmax_impl(*args)

__all__.append('softmax')

def dropout_ext(input, p):
    seed, offset = default_generator._step(12)  # pylint: disable=protected-access
    return gen_ops_prim.dropout_ext_op(input, p, seed, offset)

def squeeze(*args):
    return pyboost_inner_prim.squeeze_impl(*args)

__all__.append('squeeze')

ones_op = ops.Ones().set_device('Ascend')
def ones(*args):
    return ones_op(*args)

__all__.append('ones')

def nllloss(*args):
    return pyboost_inner_prim.nllloss_impl(*args)

__all__.append('nllloss')

def repeat_elements(*args):
    return ops.repeat_elements(*args)

__all__.append('repeat_elements')

def concat(*args):
    return pyboost_inner_prim.concat_impl(*args)

__all__.append('concat')

def multinomial_ext(input, num_samples, replacement, generator):
    seed, offset = generator._step(12)  # pylint: disable=protected-access
    return gen_ops_prim.multinomial_ext_op(input, num_samples, replacement, seed, offset)

def isclose(*args):
    return pyboost_inner_prim.isclose_impl(*args)

__all__.append('isclose')

tile_op = ops.Primitive('Tile').set_device('Ascend')
def tile(*args):
    return tile_op(*args)

__all__.append('tile')

def pad_v3(input_x, padding, mode='constant', value=None):
    pad_op = ops.PadV3(mode=mode, paddings_contiguous=True).set_device('CPU')
    if isinstance(value, (float, int)):
        value = mindtorch.tensor(value, dtype=input_x.dtype)
    return pad_op(input_x, padding, value)

__all__.append('pad_v3')

def inplace_uniform(input, from_, to_, generator_):
    seed, offset = generator_._step(12)
    return gen_ops_prim.inplace_uniform_op(input, from_, to_, seed, offset)

def binary_cross_entropy_with_logits(*args):
    return pyboost_inner_prim.binary_cross_entropy_with_logits_impl(*args)

__all__.append('binary_cross_entropy_with_logits')

def gather(input_params, input_indices, axis, batch_dims=0):
    return ops.gather(input_params, input_indices, axis, batch_dims)

__all__.append('gather')

def randint(low, high, shape, dtype, generator):
    seed, offset = generator._step(12)  # pylint: disable=protected-access
    return gen_ops_prim.randint_op(low, high, shape, seed, offset, dtype)

def stack_ext(*args):
    return pyboost_inner_prim.stack_ext_impl(*args)

__all__.append('stack_ext')

def argmax_with_value(*args):
    return pyboost_inner_prim.argmax_with_value_impl(*args)

__all__.append('argmax_with_value')

def argmin_with_value(*args):
    return pyboost_inner_prim.argmin_with_value_impl(*args)

__all__.append('argmin_with_value')


right_shift_op = ops.RightShift().set_device('Ascend')
def right_shift(input, other):
    if isinstance(other, numbers.Number):
        other = mindtorch.Tensor(other, input.dtype)
    return right_shift_op(input, other)

tensor_mul = ops.Mul().set_device('Ascend')
tensor_pow = ops.Pow().set_device('Ascend')
def ldexp(input, other):
    out = tensor_mul(input, tensor_pow(2.0, other))
    return out

__all__.append('ldexp')

def reverse_v2(input, dims):
    if isinstance(dims, int):
        dims = (dims,)
    return pyboost_inner_prim.reverse_v2_impl(input, dims)

__all__.append('reverse_v2')

range_op = ops.Range().set_device('Ascend')
def arange(start, end, step, dtype):
    return cast(range_op(start, end, step), dtype)

matmul_op = gen_ops_prim.matmul_ext_op.set_device('Ascend')
def matmul_ext(input, other):
    input_dtype = input.dtype
    out = matmul_op(cast(input, mindtorch.float16), cast(other, mindtorch.float16))
    return cast(out, input_dtype)

def dropout_ext(input, p):
    keep_prob = 1 - p
    dropout_op = ops.Dropout(keep_prob=keep_prob).set_device('Ascend')
    return dropout_op(input)

def isclose(input, other, rtol, atol, equal_nan):
    out = np.isclose(input.asnumpy(), other.asnumpy(), rtol, atol, equal_nan)
    if not isinstance(out, np.ndarray):
        out = np.array(out)
    return mindtorch.Tensor.from_numpy(out)

stop_gradient_op = ops.StopGradient().set_device('Ascend')
def stop_gradient(*args):
    return stop_gradient_op(*args)

__all__.append('stop_gradient')

def sort_ext(input, dim, descending, stable):
    ops.sort
    _sort = _get_cache_prim(ops.Sort)(dim, descending).set_device('Ascend')
    return _sort(input)

def tensor_scatter_elements(input, index, src, dim):
    scatter_op = gen_ops_prim.TensorScatterElements(dim).set_device('Ascend')
    return scatter_op(input, index, src)

__all__.append('tensor_scatter_elements')

def topk_ext(input, k, dim, largest, sorted):
    top_k_ = _get_cache_prim(ops.TopK)(sorted).set_device('Ascend')
    if not largest:
        input = -input
    if dim is None or dim == input.ndim - 1:
        if not largest:
            res = top_k_(input, k)
            values, indices = -res[0], res[1]
            return values, indices
        return top_k_(input, k)
    input = input.swapaxes(dim, input.ndim - 1)
    output = top_k_(input, k)
    values = transpose_ext_view(output[0], dim, input.ndim - 1)
    indices = transpose_ext_view(output[1], dim, input.ndim - 1)
    if not largest:
        res = (-values, indices)
    else:
        res = (values, indices)
    return res

def std(input, dim, correction, keepdim):
    std_op = _get_cache_prim(ops.ReduceStd)(axis=dim, unbiased=bool(correction), keep_dims=keepdim)
    std_op.set_device('Ascend')
    return std_op(input)[0]
