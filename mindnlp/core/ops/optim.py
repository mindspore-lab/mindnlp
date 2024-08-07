"""optim op"""
from mindspore import ops
from mindspore.ops._primitive_cache import _get_cache_prim

_adadelta = ops.ApplyAdadelta()
def raw_adadelta(param, square_avg, acc_delta, lr, rho, eps, grad):
    return _adadelta(param, square_avg, acc_delta, lr, rho, eps, grad)

_adam = ops.Adam()
def raw_adam(param, exp_avg, exp_avg_sq, beta1_power, beta2_power, lr, beta1, beta2, epsilon, grad):
    # var, m, v, beta1_power, beta2_power, lr, beta1, beta2, epsilon, grad
    return _adam(param, exp_avg, exp_avg_sq, beta1_power, beta2_power, lr, beta1, beta2, epsilon, grad)

_adam_amsgrad = ops.ApplyAdamWithAmsgradV2()
def raw_adam_amsgrad(param, exp_avg, exp_avg_sq, max_exp_avg_sq, beta1_power, beta2_power, lr, beta1, beta2, epsilon, grad):
    # var, m, v, beta1_power, beta2_power, lr, beta1, beta2, epsilon, grad
    return _adam_amsgrad(param, exp_avg, exp_avg_sq, max_exp_avg_sq,
                         beta1_power, beta2_power, lr, beta1, beta2, epsilon, grad)


def raw_sgd(param, grad, lr, dampening, weight_decay, nesterov, accum, momentum, stat):
    _sgd = _get_cache_prim(ops.SGD)(dampening, weight_decay, nesterov)
    return _sgd(param, grad, lr, accum, momentum, stat)
