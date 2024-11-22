"""functional autograd"""
from collections.abc import Generator

import mindspore
from mindspore.ops.composite import GradOperation
from mindspore.ops import stop_gradient
from mindspore.common.api import _pynative_executor
from mindnlp.configs import GENERATOR_SEED

grad_ = GradOperation(False, True, False)

def value_and_grad(fn, params_or_argnums, has_aux=False, attach_grads=True):
    use_argnums = False
    if isinstance(params_or_argnums, Generator):
        params_or_argnums = tuple(params_or_argnums)

    if isinstance(params_or_argnums[0], int):
        use_argnums = True

    def fn_aux(*args):
        outputs = fn(*args)
        no_grad_outputs = ()
        for out in outputs[1:]:
            no_grad_outputs += (stop_gradient(out),)
        return outputs[0], no_grad_outputs

    if has_aux:
        fn_ = fn_aux
    else:
        fn_ = fn

    def value_and_grad_f(*args, **kwargs):
        _pynative_executor.set_grad_flag(True)
        _pynative_executor.new_graph(fn, *args, **kwargs)
        values = fn_(*args, **kwargs)
        _pynative_executor.end_graph(fn, values, *args, **kwargs)

        run_args = args
        if kwargs:
            run_args = args + tuple(kwargs.values())

        if GENERATOR_SEED:
            grads = _pynative_executor.grad(fn_, grad_, params_or_argnums, None, *run_args)
            # grads = grad_(fn_, params)(*args, *params)
        else:
            _pynative_executor.grad(fn_, grad_, params_or_argnums, None, *run_args)
            grads = _pynative_executor() # pylint: disable=not-callable
        grads = tuple(mindspore.Tensor(grad) for grad in grads)
        if attach_grads:
            for param, grad in zip(params_or_argnums, grads):
                if param.grad is None:
                    param.grad = grad
                else:
                    param.grad += grad
            return values
        return values, grads
    return value_and_grad_f

def grad(fn, params_or_argnums=None, has_aux=False):
    value_and_grad_f = value_and_grad(fn, params_or_argnums, has_aux)
    def grad_f(*args):
        _, g = value_and_grad_f(*args)
        return g
    return grad_f
