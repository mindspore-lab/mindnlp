import re
import inspect
from mindspore import ops
from mindspore.ops._primitive_cache import _get_cache_prim
from mindspore.ops.operations._grad_ops import StridedSliceGrad

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
