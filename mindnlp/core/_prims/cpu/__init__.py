from mindspore.common.api import _pynative_executor
from mindspore.ops.auto_generate import gen_ops_prim
from mindspore.ops.auto_generate.gen_ops_prim import *
from mindspore._c_expression import Tensor as MSTensor
from mindspore._c_expression import pyboost_cast, pyboost_empty, pyboost_zeros, pyboost_ones
from mindspore.ops.operations.manually_defined.ops_def import Cast, Zeros, Ones
from mindspore.ops._primitive_cache import _get_cache_prim
from mindspore.ops import StopGradient, Primitive, ApplyAdadelta, Adam, ApplyAdamWithAmsgradV2, SGD
from mindspore.ops import FillV2, UniformReal, Stack, StandardNormal, TensorScatterUpdate
from mindspore.ops.operations import identity, TensorShape
from mindspore.ops.operations._grad_ops import StridedSliceGrad


pyboost_list = list(filter(lambda s: s.startswith("pyboost"), dir(gen_ops_prim)))
pyboost_op_list = [op.replace('pyboost_', '') + '_op' for op in pyboost_list]
aclop_list = list(filter(lambda s: s.endswith("_op") and not s in pyboost_op_list, dir(gen_ops_prim)))


pyboost_func = '''
def {name}(*args):
    return {pyboost}({op}, args)
'''

aclop_func = '''
def {name}(*args):
    return _pynative_executor.run_op_async({obj}, {obj}.name, args)
'''

__all__ = []

for op_name in pyboost_list:
    op = getattr(gen_ops_prim, op_name)
    func_name = op_name.replace('pyboost_', '') + '_cpu'
    prim_op = func_name.replace('_cpu', '_op')
    if not hasattr(gen_ops_prim, prim_op):
        continue
    __all__.append(func_name)
    globals()[prim_op] = getattr(gen_ops_prim, prim_op).__class__().set_device('CPU')
    exec(pyboost_func.format(name=func_name, pyboost=op_name, op=prim_op), globals())


for op_name in aclop_list:
    func_name = op_name.replace('_op', '_cpu')
    __all__.append(func_name)
    prim_op = func_name + '_prim'
    globals()[prim_op] = getattr(gen_ops_prim, op_name).__class__().set_device('CPU')
    exec(aclop_func.format(name=func_name, obj=prim_op), globals())

cast_op = Cast().set_device('CPU')
def cast_cpu(*args):
    return pyboost_cast(cast_op, args)

__all__.append('cast_cpu')

def empty_cpu(size, dtype):
    return pyboost_empty([size, dtype, 'CPU'])

__all__.append('empty_cpu')

zeros_op = Zeros().set_device('CPU')
def zeros_cpu(*args):
    return pyboost_zeros(zeros_op, args)

__all__.append('zeros_cpu')

ones_op = Ones().set_device('CPU')
def ones_cpu(*args):
    return pyboost_ones(ones_op, args)

__all__.append('ones_cpu')


squeeze_op = Squeeze().set_device('CPU')
def squeeze_cpu(*args):
    return pyboost_squeeze(squeeze_op, args)

__all__.append('squeeze_cpu')

stack_ext_op = StackExt().set_device('CPU')
def stack_ext_cpu(*args):
    return pyboost_stack_ext(stack_ext_op, args)

__all__.append('stack_ext_cpu')

tile_op = Primitive('Tile').set_device('CPU')
def tile_cpu(*args):
    return pyboost_tile(tile_op, args)

__all__.append('tile_cpu')

greater_equal_op = GreaterEqual().set_device('CPU')
def greater_equal_cpu(*args):
    return pyboost_greater_equal(greater_equal_op, args)

__all__.append('greater_equal_cpu')

isclose_op = IsClose().set_device('CPU')
def isclose_cpu(*args):
    return pyboost_isclose(isclose_op, args)

__all__.append('isclose_cpu')

range_op = Range().set_device('CPU')
def range_cpu(*args):
    return _pynative_executor.run_op_async(range_op, range_op.name, args)

__all__.append('range_cpu')

linspace_op = LinSpace().set_device('CPU')
def linspace_cpu(*args):
    return _pynative_executor.run_op_async(linspace_op, linspace_op.name, args)

__all__.append('linspace_cpu')

full_op = FillV2().set_device('CPU')
def full_cpu(shape, value):
    return _pynative_executor.run_op_async(full_op, full_op.name, [shape, MSTensor(value)])

__all__.append('full_cpu')

stop_gradient_op = StopGradient().set_device('CPU')
def stop_gradient_cpu(*args):
    return _pynative_executor.run_op_async(stop_gradient_op, stop_gradient_op.name, args)

__all__.append('stop_gradient_cpu')

identity_op = identity().set_device('CPU')
def identity_cpu(*args):
    return _pynative_executor.run_op_async(identity_op, identity_op.name, args)

__all__.append('identity_cpu')


tensor_shape_op = TensorShape().set_device('CPU')
def tensor_shape_cpu(*args):
    return _pynative_executor.run_op_async(tensor_shape_op, tensor_shape_op.name, args)

__all__.append('stop_gradient_cpu')

adadelta_op = ApplyAdadelta().set_device('CPU')
def raw_adadelta_cpu(param, square_avg, acc_delta, lr, rho, eps, grad):
    args = (param, square_avg, acc_delta, lr, rho, eps, grad)
    return _pynative_executor.run_op_async(adadelta_op, adadelta_op.name, args)

adam_op = Adam().set_device('CPU')
def raw_adam_cpu(param, exp_avg, exp_avg_sq, beta1_power, beta2_power, lr, beta1, beta2, epsilon, grad):
    # var, m, v, beta1_power, beta2_power, lr, beta1, beta2, epsilon, grad
    args = (param, exp_avg, exp_avg_sq, beta1_power, beta2_power, lr, beta1, beta2, epsilon, grad)
    return _pynative_executor.run_op_async(adam_op, adam_op.name, args)

adam_amsgrad_op = ApplyAdamWithAmsgradV2().set_device('CPU')
def raw_adam_amsgrad_cpu(param, exp_avg, exp_avg_sq, max_exp_avg_sq, beta1_power, beta2_power, lr, beta1, beta2, epsilon, grad):
    # var, m, v, beta1_power, beta2_power, lr, beta1, beta2, epsilon, grad
    args = (param, exp_avg, exp_avg_sq, max_exp_avg_sq,
                         beta1_power, beta2_power, lr, beta1, beta2, epsilon, grad)
    return _pynative_executor.run_op_async(adam_amsgrad_op, adam_amsgrad_op.name, args)


def raw_sgd_cpu(param, grad, lr, dampening, weight_decay, nesterov, accum, momentum, stat):
    sgd_op = _get_cache_prim(SGD)(dampening, weight_decay, nesterov).set_device('CPU')
    args = (param, grad, lr, accum, momentum, stat)
    return _pynative_executor.run_op_async(sgd_op, sgd_op.name, args)

__all__.extend(
    [
        'raw_adadelta_cpu',
        'raw_adam_cpu',
        'raw_adam_amsgrad_cpu',
        'raw_sgd_cpu'
    ]
)

uniform_real_op = UniformReal().set_device('CPU')
def uniform_real_cpu(*args):
    return _pynative_executor.run_op_async(uniform_real_op, uniform_real_op.name, args)

__all__.append('uniform_real_cpu')

def stack_cpu(tensors, dim):
    stack_op = _get_cache_prim(Stack)(dim).set_device('CPU')
    return _pynative_executor.run_op_async(stack_op, stack_op.name, tensors)

__all__.append('stack_cpu')

argmax_with_value_op = ArgMaxWithValue().set_device('CPU')
def argmax_with_value_cpu(*args):
    return pyboost_argmax_with_value(argmax_with_value_op, args)

__all__.append('argmax_with_value_cpu')

argmin_with_value_op = ArgMinWithValue().set_device('CPU')
def argmin_with_value_cpu(*args):
    return pyboost_argmin_with_value(argmin_with_value_op, args)

__all__.append('argmin_with_value_cpu')

log_softmax_op = LogSoftmax().set_device('CPU')
def log_softmax_cpu(*args):
    return pyboost_log_softmax(log_softmax_op, args)

__all__.append('log_softmax_cpu')

strided_slice_op = StridedSlice().set_device('CPU')
def strided_slice_cpu(*args):
    return _pynative_executor.run_op_async(strided_slice_op, strided_slice_op.name, args)

__all__.append('strided_slice_cpu')

hard_shrink_op = HShrink().set_device('CPU')
def hard_shrink_cpu(*args):
    return pyboost_hshrink(hard_shrink_op, args)

__all__.append('hard_shrink_cpu')

normal_op = StandardNormal().set_device('CPU')
def normal_cpu(*args):
    return _pynative_executor.run_op_async(normal_op, normal_op.name, args)

__all__.append('normal_cpu')

reduce_any_op = ReduceAny().set_device('CPU')
def reduce_any_cpu(*args):
    return pyboost_reduce_any(reduce_any_op, args)

__all__.append('reduce_any_cpu')

def strided_slice_grad_cpu(input, begin, end, strides, update, begin_mask=0, end_mask=0, ellipsis_mask=0, new_axis_mask=0, shrink_axis_mask=0):
    strided_slice_grad = _get_cache_prim(StridedSliceGrad)(begin_mask, end_mask, ellipsis_mask, new_axis_mask, shrink_axis_mask).set_device('CPU')
    return _pynative_executor.run_op_async(strided_slice_grad, strided_slice_grad.name, [update, input.shape, begin, end, strides])

__all__.append('strided_slice_grad_cpu')

tensor_scatter_update_op = TensorScatterUpdate().set_device('CPU')
def tensor_scatter_update_cpu(*args):
    return _pynative_executor.run_op_async(tensor_scatter_update_op, tensor_scatter_update_op.name, args)

__all__.append('tensor_scatter_update_cpu')

broadcast_to_op = Primitive('BroadcastTo').set_device('CPU')
def broadcast_to_cpu(*args):
    return pyboost_broadcast_to(broadcast_to_op, args)

__all__.append('broadcast_to_cpu')

concat_op = Concat().set_device('CPU')
def concat_cpu(*args):
    return pyboost_concat(concat_op, args)

__all__.append('concat_cpu')
