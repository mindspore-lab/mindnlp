"""functional autograd"""
from collections.abc import Generator

import mindspore
from mindspore.ops.composite import GradOperation
from mindspore.common.api import _pynative_executor
from mindspore._c_expression import Cell_
from .grad_mode import no_grad

try:
    from mindspore import _Function as Function
except:
    Function = None

import mindtorch

grad_ = GradOperation(False, True, False)
grad_sens_ = GradOperation(False, True, True)
grad_input_sens_ = GradOperation(True, True, True)

def value_and_grad(fn, params_or_argnums, has_aux=False, attach_grads=True):
    grad_fn = mindspore.value_and_grad(fn, None, tuple(params_or_argnums), has_aux)
    if attach_grads:
        def new_grad_fn(*args, **kwargs):
            values, grads = grad_fn(*args, **kwargs)
            for param, grad in zip(params_or_argnums, grads):
                grad = mindtorch.tensor(grad, device=param.device)
                if param.grad is None:
                    param.grad = grad
                else:
                    param.grad += grad
            return values
        return new_grad_fn
    return grad_fn
    # use_argnums = False
    # if isinstance(params_or_argnums, Generator):
    #     params_or_argnums = tuple(params_or_argnums)

    # if isinstance(params_or_argnums[0], int):
    #     use_argnums = True

    # def fn_aux(*args):
    #     outputs = fn(*args)
    #     no_grad_outputs = ()
    #     for out in outputs[1:]:
    #         no_grad_outputs += (out.detach(),)
    #     return outputs[0], no_grad_outputs

    # if has_aux:
    #     fn_ = fn_aux
    # else:
    #     fn_ = fn

    # def value_and_grad_f(*args, **kwargs):
    #     _pynative_executor.set_grad_flag(True)
    #     _pynative_executor.new_graph(fn_, *args, **kwargs)
    #     values = fn_(*args, **kwargs)
    #     _pynative_executor.end_graph(fn_, values, *args, **kwargs)

    #     run_args = args
    #     if kwargs:
    #         run_args = args + tuple(kwargs.values())

    #     grads = _pynative_executor.check_run(grad_, fn_, params_or_argnums, None, *run_args)
    #     grads = _pynative_executor.grad(fn_, grad_, params_or_argnums, None, *run_args)

    #     if attach_grads:
    #         for param, grad in zip(params_or_argnums, grads):
    #             grad = mindtorch.tensor(grad, device=param.device)
    #             if param.grad is None:
    #                 param.grad = grad
    #             else:
    #                 param.grad += grad
    #         return values
    #     return values, grads
    # return value_and_grad_f

def grad(fn, params_or_argnums=None, has_aux=False):
    value_and_grad_f = value_and_grad(fn, params_or_argnums, has_aux)
    def grad_f(*args):
        _, g = value_and_grad_f(*args)
        return g
    return grad_f


if Function is None:
    class Function(Cell_):
        def __init__(self):
            super().__init__(str(self.__class__)[8:-2])
            self.saved_tensors = []
            self.used_bprop_inputs = []

        def save_for_backward(self, *args):
            if isinstance(args, tuple):
                self.saved_tensors.extend(list(args))
            else:
                self.saved_tensors.append(args)

        @staticmethod
        def forward(ctx, *args, **kwargs):
            raise NotImplementedError

        @staticmethod
        def backward(ctx, *args, **kwargs):
            raise NotImplementedError

        def construct(self, *args, **kwargs):
            self.needs_input_grad = [input_.requires_grad if hasattr(input_, 'requires_grad') else False for input_ in args]
            args = (self,) + args
            return self.forward(*args, **kwargs)

        def bprop(self, *args, **kwargs):
            args = (args[-1],)
            args = (self,) + args
            return self.backward(*args, **kwargs)

        def __call__(self, *args, **kwargs):
            with no_grad():
                output = self.construct(*args, **kwargs)
            if _pynative_executor.requires_grad():
                _pynative_executor.call_custom_bprop(self, output, *args, **kwargs)
            return output

        @classmethod
        def apply(cls, *args, **kwargs):
            return cls()(*args, **kwargs)
