"""optim op"""
from mindtorch.executor import execute

def raw_adadelta(param, square_avg, acc_delta, lr, rho, eps, grad):
    return execute('raw_adadelta', param, square_avg, acc_delta, lr, rho, eps, grad)

def raw_adam(param, exp_avg, exp_avg_sq, beta1_power, beta2_power, lr, beta1, beta2, epsilon, grad):
    # var, m, v, beta1_power, beta2_power, lr, beta1, beta2, epsilon, grad
    return execute('raw_adam', param, exp_avg, exp_avg_sq, beta1_power, beta2_power, lr, beta1, beta2, epsilon, grad)

def raw_adam_amsgrad(param, exp_avg, exp_avg_sq, max_exp_avg_sq, beta1_power, beta2_power, lr, beta1, beta2, epsilon, grad):
    # var, m, v, beta1_power, beta2_power, lr, beta1, beta2, epsilon, grad
    return execute('raw_adam_amsgrad', param, exp_avg, exp_avg_sq, max_exp_avg_sq, beta1_power, beta2_power, lr, beta1, beta2, epsilon, grad)

def raw_sgd(param, grad, lr, dampening, weight_decay, nesterov, accum, momentum, stat):
    return execute('raw_sgd', param, grad, lr, dampening, weight_decay, nesterov, accum, momentum, stat)

__all__ = ['raw_adadelta', 'raw_adam', 'raw_adam_amsgrad', 'raw_sgd']
