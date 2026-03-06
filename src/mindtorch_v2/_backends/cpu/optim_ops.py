"""CPU kernels for optimizer step operations.

Each function implements a single per-parameter optimizer step using numpy.
These are registered as dispatch ops so NPU backends can override them.
"""
import math

import numpy as np

from ..._tensor import Tensor
from ..._storage import typed_storage_from_numpy


def _to_numpy(t):
    return t._numpy_view()


def _write_back(t, arr):
    """Write numpy array back into a Tensor's storage."""
    t.storage()._data[:] = arr


def _sgd_step(param, grad, buf, lr, momentum, dampening, weight_decay, nesterov, maximize):
    p_data = _to_numpy(param)
    g_data = _to_numpy(grad)
    if maximize:
        g_data = -g_data
    else:
        g_data = g_data.copy()

    if weight_decay != 0:
        g_data = g_data + weight_decay * p_data

    if momentum != 0:
        buf_data = _to_numpy(buf)
        buf_data[:] = momentum * buf_data + (1 - dampening) * g_data
        if nesterov:
            g_data = g_data + momentum * buf_data
        else:
            g_data = buf_data

    _write_back(param, p_data - lr * g_data)
    return param


def _adam_step(param, grad, exp_avg, exp_avg_sq, max_exp_avg_sq,
               step, lr, beta1, beta2, eps, weight_decay, amsgrad, maximize):
    p_data = _to_numpy(param)
    g_data = _to_numpy(grad)
    ea = _to_numpy(exp_avg)
    eas = _to_numpy(exp_avg_sq)

    if maximize:
        g_data = -g_data
    else:
        g_data = g_data.copy()

    if weight_decay != 0:
        g_data = g_data + weight_decay * p_data

    ea[:] = beta1 * ea + (1 - beta1) * g_data
    eas[:] = beta2 * eas + (1 - beta2) * g_data * g_data

    bc1 = 1 - beta1 ** step
    bc2 = 1 - beta2 ** step

    if amsgrad:
        meas = _to_numpy(max_exp_avg_sq)
        np.maximum(meas, eas, out=meas)
        denom = meas ** 0.5 / math.sqrt(bc2) + eps
    else:
        denom = eas ** 0.5 / math.sqrt(bc2) + eps

    step_size = lr / bc1
    _write_back(param, p_data - step_size * ea / denom)
    return param


def _adamw_step(param, grad, exp_avg, exp_avg_sq, max_exp_avg_sq,
                step, lr, beta1, beta2, eps, weight_decay, amsgrad, maximize):
    p_data = _to_numpy(param)
    g_data = _to_numpy(grad)
    ea = _to_numpy(exp_avg)
    eas = _to_numpy(exp_avg_sq)

    if maximize:
        g_data = -g_data
    else:
        g_data = g_data.copy()

    # Decoupled weight decay
    if weight_decay != 0:
        p_data[:] = p_data * (1 - lr * weight_decay)

    ea[:] = beta1 * ea + (1 - beta1) * g_data
    eas[:] = beta2 * eas + (1 - beta2) * g_data * g_data

    bc1 = 1 - beta1 ** step
    bc2 = 1 - beta2 ** step

    if amsgrad:
        meas = _to_numpy(max_exp_avg_sq)
        np.maximum(meas, eas, out=meas)
        denom = meas ** 0.5 / math.sqrt(bc2) + eps
    else:
        denom = eas ** 0.5 / math.sqrt(bc2) + eps

    step_size = lr / bc1
    _write_back(param, p_data - step_size * ea / denom)
    return param


def _adagrad_step(param, grad, state_sum, step, lr, lr_decay, weight_decay, eps, maximize):
    p_data = _to_numpy(param)
    g_data = _to_numpy(grad)
    s = _to_numpy(state_sum)

    if maximize:
        g_data = -g_data
    else:
        g_data = g_data.copy()

    if weight_decay != 0:
        g_data = g_data + weight_decay * p_data

    s[:] = s + g_data * g_data
    clr = lr / (1 + (step - 1) * lr_decay)
    _write_back(param, p_data - clr * g_data / (s ** 0.5 + eps))
    return param


def _rmsprop_step(param, grad, square_avg, grad_avg, buf,
                  step, lr, alpha, eps, weight_decay, momentum, centered, maximize):
    p_data = _to_numpy(param)
    g_data = _to_numpy(grad)
    sq_avg = _to_numpy(square_avg)

    if maximize:
        g_data = -g_data
    else:
        g_data = g_data.copy()

    if weight_decay != 0:
        g_data = g_data + weight_decay * p_data

    sq_avg[:] = alpha * sq_avg + (1 - alpha) * g_data * g_data

    if centered:
        ga = _to_numpy(grad_avg)
        ga[:] = alpha * ga + (1 - alpha) * g_data
        avg = sq_avg - ga * ga
        denom = avg ** 0.5 + eps
    else:
        denom = sq_avg ** 0.5 + eps

    if momentum > 0:
        b = _to_numpy(buf)
        b[:] = momentum * b + g_data / denom
        _write_back(param, p_data - lr * b)
    else:
        _write_back(param, p_data - lr * g_data / denom)
    return param


def _adadelta_step(param, grad, square_avg, acc_delta, lr, rho, eps, weight_decay, maximize):
    p_data = _to_numpy(param)
    g_data = _to_numpy(grad)
    sq_avg = _to_numpy(square_avg)
    ad = _to_numpy(acc_delta)

    if maximize:
        g_data = -g_data
    else:
        g_data = g_data.copy()

    if weight_decay != 0:
        g_data = g_data + weight_decay * p_data

    sq_avg[:] = rho * sq_avg + (1 - rho) * g_data * g_data
    std = (ad + eps) ** 0.5
    delta = std / (sq_avg + eps) ** 0.5 * g_data
    ad[:] = rho * ad + (1 - rho) * delta * delta
    _write_back(param, p_data - lr * delta)
    return param


def _adamax_step(param, grad, exp_avg, exp_inf, step, lr, beta1, beta2, eps, weight_decay, maximize):
    p_data = _to_numpy(param)
    g_data = _to_numpy(grad)
    ea = _to_numpy(exp_avg)
    ei = _to_numpy(exp_inf)

    if maximize:
        g_data = -g_data
    else:
        g_data = g_data.copy()

    if weight_decay != 0:
        g_data = g_data + weight_decay * p_data

    ea[:] = beta1 * ea + (1 - beta1) * g_data
    np.maximum(beta2 * ei, np.abs(g_data) + eps, out=ei)

    bc1 = 1 - beta1 ** step
    step_size = lr / bc1
    _write_back(param, p_data - step_size * ea / ei)
    return param


def _nadam_step(param, grad, exp_avg, exp_avg_sq, step,
                lr, beta1, beta2, eps, weight_decay,
                mu, mu_next, mu_product, mu_product_next, maximize):
    p_data = _to_numpy(param)
    g_data = _to_numpy(grad)
    ea = _to_numpy(exp_avg)
    eas = _to_numpy(exp_avg_sq)

    if maximize:
        g_data = -g_data
    else:
        g_data = g_data.copy()

    if weight_decay != 0:
        g_data = g_data + weight_decay * p_data

    ea[:] = beta1 * ea + (1 - beta1) * g_data
    eas[:] = beta2 * eas + (1 - beta2) * g_data * g_data

    bc2 = 1 - beta2 ** step

    # Nesterov-corrected first moment
    ea_hat = mu_next / (1 - mu_product_next) * ea + mu / (1 - mu_product) * g_data
    eas_hat = eas / bc2

    _write_back(param, p_data - lr * ea_hat / (eas_hat ** 0.5 + eps))
    return param


def _radam_step(param, grad, exp_avg, exp_avg_sq, step, lr, beta1, beta2, eps, weight_decay, maximize):
    p_data = _to_numpy(param)
    g_data = _to_numpy(grad)
    ea = _to_numpy(exp_avg)
    eas = _to_numpy(exp_avg_sq)

    if maximize:
        g_data = -g_data
    else:
        g_data = g_data.copy()

    if weight_decay != 0:
        g_data = g_data + weight_decay * p_data

    ea[:] = beta1 * ea + (1 - beta1) * g_data
    eas[:] = beta2 * eas + (1 - beta2) * g_data * g_data

    bc1 = 1 - beta1 ** step
    bc2 = 1 - beta2 ** step

    ea_corrected = ea / bc1

    # Compute max length of approximated SMA
    rho_inf = 2.0 / (1 - beta2) - 1.0
    rho_t = rho_inf - 2.0 * step * (beta2 ** step) / bc2

    if rho_t > 5:
        eas_corrected = eas / bc2
        rect = math.sqrt(
            (rho_t - 4) * (rho_t - 2) * rho_inf
            / ((rho_inf - 4) * (rho_inf - 2) * rho_t)
        )
        _write_back(param, p_data - lr * rect * ea_corrected / (eas_corrected ** 0.5 + eps))
    else:
        _write_back(param, p_data - lr * ea_corrected)
    return param


def _asgd_step(param, grad, ax, step, lr, lambd, alpha, t0, weight_decay, maximize):
    p_data = _to_numpy(param)
    g_data = _to_numpy(grad)
    ax_data = _to_numpy(ax)

    if maximize:
        g_data = -g_data
    else:
        g_data = g_data.copy()

    if weight_decay != 0:
        g_data = g_data + weight_decay * p_data

    # Decay term
    eta = lr / ((1 + lambd * lr * step) ** alpha)
    p_new = p_data - eta * g_data
    _write_back(param, p_new)

    # Update Polyak averaging
    if step >= t0:
        mu_t = 1.0 / max(1, step - t0 + 1)
        ax_data[:] = ax_data + mu_t * (p_new - ax_data)
    else:
        ax_data[:] = p_new
    return param


def _rprop_step(param, grad, prev, step_sizes, lr, etaminus, etaplus, step_size_min, step_size_max, maximize):
    p_data = _to_numpy(param)
    g_data = _to_numpy(grad)
    prev_data = _to_numpy(prev)
    ss = _to_numpy(step_sizes)

    if maximize:
        g_data = -g_data
    else:
        g_data = g_data.copy()

    sign = g_data * prev_data
    # Where sign > 0, increase step size
    pos_mask = sign > 0
    neg_mask = sign < 0
    zero_mask = ~pos_mask & ~neg_mask

    ss[pos_mask] = np.minimum(ss[pos_mask] * etaplus, step_size_max)
    ss[neg_mask] = np.maximum(ss[neg_mask] * etaminus, step_size_min)
    # step_sizes unchanged for zero_mask

    # Update parameters
    p_data[pos_mask] -= np.sign(g_data[pos_mask]) * ss[pos_mask]
    p_data[neg_mask] -= np.sign(g_data[neg_mask]) * ss[neg_mask]
    p_data[zero_mask] -= np.sign(g_data[zero_mask]) * ss[zero_mask]

    # Update prev: zero out grad where sign was negative
    prev_data[:] = g_data
    prev_data[neg_mask] = 0

    _write_back(param, p_data)
    return param


def _sparse_adam_step(param, grad, exp_avg, exp_avg_sq, step, lr, beta1, beta2, eps):
    """Row-sparse Adam: only updates rows with non-zero gradients."""
    p_data = _to_numpy(param)
    g_data = _to_numpy(grad).copy()
    ea = _to_numpy(exp_avg)
    eas = _to_numpy(exp_avg_sq)

    # Identify non-zero rows
    if g_data.ndim >= 2:
        row_norms = np.abs(g_data.reshape(g_data.shape[0], -1)).sum(axis=1)
        active = np.where(row_norms != 0)[0]
    else:
        active = np.where(g_data != 0)[0]

    if len(active) == 0:
        return param

    if g_data.ndim >= 2:
        g = g_data[active]
        ea[active] = beta1 * ea[active] + (1 - beta1) * g
        eas[active] = beta2 * eas[active] + (1 - beta2) * g * g
    else:
        g = g_data[active]
        ea[active] = beta1 * ea[active] + (1 - beta1) * g
        eas[active] = beta2 * eas[active] + (1 - beta2) * g * g

    bc1 = 1 - beta1 ** step
    bc2 = 1 - beta2 ** step

    m_hat = ea[active] / bc1
    v_hat = eas[active] / bc2
    denom = v_hat ** 0.5 + eps

    p_data[active] = p_data[active] - lr * m_hat / denom
    _write_back(param, p_data)
    return param
