import re
import inspect
import ctypes
import numpy as np
import mindtorch
from mindspore import ops
from mindspore.ops.auto_generate import gen_ops_prim
from mindspore._c_expression import _empty_instance
from mindspore.ops._primitive_cache import _get_cache_prim
from mindspore.ops.operations._grad_ops import StridedSliceGrad

gen_ops_list = list(filter(lambda s: s.startswith("pyboost"), dir(gen_ops_prim)))

__all__ = []

def camel_to_snake_case(camel_case_str):
    snake_case_str = re.sub(r'(?<!^)(?=[A-Z])', '_', camel_case_str).lower()
    return snake_case_str

op_func_no_init = '''
def {name}(*args):
    op = _get_cache_prim(ops.{op})().set_device('CPU')
    return op(*args)
'''

op_func_with_init = '''
def {name}(*args):
    op = _get_cache_prim(ops.{op})(*args[-{idx}:]).set_device('CPU')
    return op(*args[:-{idx}])
'''


for pyboost_op_name in gen_ops_list:
    op_name = pyboost_op_name.replace('pyboost_', '') + '_op'
    func_name = op_name.replace('_op', '')
    op_instance = getattr(gen_ops_prim, op_name, None)
    if op_instance is not None:
        __all__.append(func_name)
        globals()[func_name] = getattr(gen_ops_prim, op_name).__class__().set_device('CPU')

def empty(*args, **kwargs):
    return _empty_instance(*args, **kwargs, device='CPU')

__all__.append('empty')

old_op_list = list(filter(lambda s: s[0].isupper(), dir(ops)))
for old_op_name in old_op_list:
    if old_op_name in ['P', 'Print', 'Assert', 'Custom', 'CustomOpBuilder', 'DataType', 'ReduceOp', 'TBERegOp', 'Tensor']:
        continue
    # print(old_op_name)
    ops_class = getattr(ops, old_op_name, None)
    init_signature = inspect.signature(ops_class.__init__)
    if len(init_signature.parameters) > 1:
        name = camel_to_snake_case(old_op_name)
        init_args = list(init_signature.parameters.keys())
        init_args.pop(0)
        exec(op_func_with_init.format(name=name, op=old_op_name, idx=len(init_args)), globals())

    else:
        name = camel_to_snake_case(old_op_name)
        exec(op_func_no_init.format(name=name, op=old_op_name), globals())

    __all__.append(name)
    # print(old_op_name, init_signature.parameters, call_signature.parameters)
    # print(old_op_name, len(init_signature.parameters), len(call_signature.parameters))
    # break

# normal_op = ops.StandardNormal().set_device('CPU')
# def normal(size):
#     return normal_op(size)

# __all__.append('normal')
dyn_shape_op = ops.TensorShape().set_device('CPU')
def dyn_shape(self):
    return dyn_shape_op(self)

__all__.append('dyn_shape')

# def strided_slice(input, begin, end, strides, begin_mask, end_mask, ellipsis_mask, new_axis_mask, shrink_axis_mask):
#     strided_slice_op = ops.StridedSlice(begin_mask, end_mask, ellipsis_mask, new_axis_mask, shrink_axis_mask).set_device('CPU')
#     return strided_slice_op(input, begin, end, strides)

# __all__.append('strided_slice')

# def broadcast_to(input, shape):
#     broadcast_to_op = ops.BroadcastTo(shape).set_device('CPU')
#     return broadcast_to_op(input)

# __all__.append('broadcast_to')

def strided_slice_grad(input, begin, end, strides, update, begin_mask=0, end_mask=0, ellipsis_mask=0, new_axis_mask=0, shrink_axis_mask=0):
    strided_slice_grad = _get_cache_prim(StridedSliceGrad)(begin_mask, end_mask, ellipsis_mask, new_axis_mask, shrink_axis_mask).set_device('CPU')
    return strided_slice_grad(update, input.shape, begin, end, strides)

__all__.append('strided_slice_grad')

# full_op = ops.FillV2().set_device('CPU')
# def full(shape, value):
#     return full_op(shape, value)

# __all__.append('full')

def numpy_to_tensor_overwrite(np_array, torch_tensor):
    if not np_array.flags.c_contiguous:
        np_array = np.ascontiguousarray(np_array)

    tensor_ptr = torch_tensor.data_ptr()
        
    ctypes.memmove(tensor_ptr, np_array.ctypes.data, torch_tensor.nbytes)
    
    return torch_tensor

def inplace_uniform(input, from_, to_, seed, offset):
    np.random.seed(seed.item())
    out = np.random.uniform(from_, to_, input.shape).astype(mindtorch.dtype2np[input.dtype])
    numpy_to_tensor_overwrite(out, input)
    return input

__all__.append('inplace_uniform')

def inplace_normal(input, mean, std, seed, offset):
    np.random.seed(seed.item())
    out = np.random.normal(mean, std, input.shape).astype(mindtorch.dtype2np[input.dtype])
    numpy_to_tensor_overwrite(out, input)

    return input

__all__.append('inplace_normal')

# class GetItem(mindtorch.autograd.Function):
#     @staticmethod
#     def forward(ctx, input, slice):
#         if isinstance(slice, tuple):
#             new_slice = ()
#             for s in slice:
#                 if isinstance(s, mindtorch.Tensor):
#                     s = s.numpy()
#                 new_slice += (s,)
#         else:
#             new_slice = slice
#         out = input.asnumpy()[new_slice]

#         ctx.save_for_backward(input)
#         ctx.slice = slice
#         if not isinstance(out, np.ndarray):
#             out = np.array(out)
#         return mindtorch.Tensor.from_numpy(out)

#     @staticmethod
#     def backward(ctx, grad_output):
#         input, = ctx.saved_tensors
#         slice = ctx.slice
#         grad_input = mindtorch.zeros_like(input)
#         grad_input[slice] = grad_output
#         return grad_input
