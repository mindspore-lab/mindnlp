from mindspore.ops.auto_generate import gen_ops_prim
from mindspore.common.api import _pynative_executor
from mindspore.ops._primitive_cache import _get_cache_prim
from mindspore.ops.auto_generate.gen_ops_prim import Range, Cdist
from mindspore.ops import StopGradient, Primitive, ApplyAdadelta, Adam, ApplyAdamWithAmsgradV2, SGD, Imag

pyboost_list = list(filter(lambda s: s.startswith("pyboost"), dir(gen_ops_prim)))
pyboost_op_list = [op.replace('pyboost_', '') + '_op' for op in pyboost_list]
aclop_list = list(filter(lambda s: s.endswith("_op") and not s in pyboost_op_list, dir(gen_ops_prim)))

aclop_func = '''
def {name}(*args):
    return _pynative_executor.run_op_async({obj}, {obj}.name, args)
'''

__all__ = []

for op_name in aclop_list:
    func_name = op_name.replace('_op', '_npu')
    __all__.append(func_name)
    prim_op = func_name + '_prim'
    globals()[prim_op] = getattr(gen_ops_prim, op_name).__class__().set_device('Ascend')
    exec(aclop_func.format(name=func_name, obj=prim_op), globals())

imag_op = Imag().set_device('Ascend')
def imag_npu(*args):
    return _pynative_executor.run_op_async(imag_op, range_op.name, args)

__all__.append('imag_npu')

range_op = Range().set_device('Ascend')
def range_npu(*args):
    return _pynative_executor.run_op_async(range_op, range_op.name, args)

__all__.append('range_npu')

cdist_op = Cdist().set_device('Ascend')
def cdist_npu(*args):
    return _pynative_executor.run_op_async(cdist_op, cdist_op.name, args)

__all__.append('cdist_npu')


stop_gradient_op = StopGradient().set_device('Ascend')
def stop_gradient_npu(*args):
    return _pynative_executor.run_op_async(stop_gradient_op, stop_gradient_op.name, args)

__all__.append('stop_gradient_npu')

diagonal_op = Primitive('Diagonal').set_device('Ascend')
def diagonal_npu(*args):
    return _pynative_executor.run_op_async(diagonal_op, diagonal_op.name, args)

__all__.append('diagonal_npu')

adadelta_op = ApplyAdadelta().set_device('Ascend')
def raw_adadelta_npu(param, square_avg, acc_delta, lr, rho, eps, grad):
    args = (param, square_avg, acc_delta, lr, rho, eps, grad)
    return _pynative_executor.run_op_async(adadelta_op, adadelta_op.name, args)

adam_op = Adam().set_device('Ascend')
def raw_adam_npu(param, exp_avg, exp_avg_sq, beta1_power, beta2_power, lr, beta1, beta2, epsilon, grad):
    # var, m, v, beta1_power, beta2_power, lr, beta1, beta2, epsilon, grad
    args = (param, exp_avg, exp_avg_sq, beta1_power, beta2_power, lr, beta1, beta2, epsilon, grad)
    return _pynative_executor.run_op_async(adam_op, adam_op.name, args)

adam_amsgrad_op = ApplyAdamWithAmsgradV2().set_device('Ascend')
def raw_adam_amsgrad_npu(param, exp_avg, exp_avg_sq, max_exp_avg_sq, beta1_power, beta2_power, lr, beta1, beta2, epsilon, grad):
    # var, m, v, beta1_power, beta2_power, lr, beta1, beta2, epsilon, grad
    args = (param, exp_avg, exp_avg_sq, max_exp_avg_sq,
                         beta1_power, beta2_power, lr, beta1, beta2, epsilon, grad)
    return _pynative_executor.run_op_async(adam_amsgrad_op, adam_amsgrad_op.name, args)


def raw_sgd_npu(param, grad, lr, dampening, weight_decay, nesterov, accum, momentum, stat):
    sgd_op = _get_cache_prim(SGD)(dampening, weight_decay, nesterov).set_device('Ascend')
    args = (param, grad, lr, accum, momentum, stat)
    return _pynative_executor.run_op_async(sgd_op, sgd_op.name, args)

__all__.extend(
    [
        'raw_adadelta_npu',
        'raw_adam_npu',
        'raw_adam_amsgrad_npu',
        'raw_sgd_npu'
    ]
)
