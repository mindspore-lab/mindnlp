# Copyright 2022 Huawei Technologies Co., Ltd
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ============================================================================
# pylint: disable=E0611
# pylint: disable=C0103
"""
inner Grad
"""
from __future__ import absolute_import

from types import FunctionType, MethodType
from mindspore import context
from mindspore.parallel._utils import _grads_divided_by_device_num_if_recomputation
from mindspore._c_expression import GradOperation_
from mindspore.common.api import jit, _pynative_executor, _wrap_func
from mindspore import nn, ops
from mindspore.ops import constexpr
from mindspore.ops.composite.base import _get_grad_weights_id, _combine_with_ids
from mindspore.ops.function.grad.grad_func import _convert_grad_position_type


class _Grad(GradOperation_):
    """
    A higher-order function which is used to generate the gradient function by position for the input function.
    """

    def __init__(self, get_all=False, get_by_list=False, sens_param=False, get_by_position=False, has_aux=False,
                 get_value=False, return_ids=False, merge_forward=False):
        """Initialize _Grad."""
        if not isinstance(get_by_position, bool):
            raise TypeError(f"For '_Grad', the 'get_by_position' should be bool, "
                            f"but got {type(get_by_position).__name__}")
        if not isinstance(get_by_list, bool):
            raise TypeError(f"For '_Grad', the 'get_by_list' should be bool, "
                            f"but got {type(get_by_list).__name__}")
        if not isinstance(sens_param, bool):
            raise TypeError(f"For '_Grad', the 'sens_param' should be bool, "
                            f"but got {type(sens_param).__name__}")
        if not isinstance(has_aux, bool):
            raise TypeError(f"For '_Grad', the 'has_aux' should be bool, "
                            f"but got {type(has_aux).__name__}")
        if not isinstance(get_value, bool):
            raise TypeError(f"For '_Grad', the 'get_value' should be bool, "
                            f"but got {type(get_value).__name__}")
        if not isinstance(return_ids, bool):
            raise TypeError(f"For '_Grad', the 'return_ids' should be bool, "
                            f"but got {type(return_ids).__name__}")
        self.get_all = get_all
        self.get_by_position = get_by_position
        self.get_by_list = get_by_list
        self.sens_param = sens_param
        self.has_aux = has_aux
        self.get_value = get_value
        self.return_ids = return_ids
        self.merge_forward = merge_forward
        GradOperation_.__init__(self, 'grad', get_all, get_by_list, sens_param, get_by_position, has_aux, get_value,
                                return_ids, merge_forward)
        self.grad_fn = None
        self.fn = None
        self.pynative_ = False
        self.grad_position = None
        self.weights_id = None

    def __call__(self, fn, weights=None, grad_position=0):
        weights_id = _get_grad_weights_id(weights)
        if self.grad_fn is not None and self.fn == fn and self.grad_position == grad_position and \
                self.weights_id == weights_id:
            return self.grad_fn

        def aux_fn(*args, **kwargs):
            outputs = fn(*args, **kwargs)
            if not isinstance(outputs, tuple) or len(outputs) < 2:
                raise ValueError("When has_aux is True, origin fn requires more than one outputs.")
            res = (outputs[0],)
            for item in outputs[1:]:
                res += (ops.stop_gradient(item),)
            return res

        grad_ = _Grad(self.get_all, self.get_by_list, self.sens_param, self.get_by_position, self.has_aux,
                      self.get_value, self.return_ids, self.merge_forward)
        # If calling Grad in GRAPH_MODE or calling Grad in functions decorated with 'jit', do grad in GRAPH_MODE
        # If calling Grad in pure PYNATIVE_MODE do grad in PYNATIVE_MODE
        #   In pure PYNATIVE_MODE the out layer after_grad just used to set pynative flag for inner GradOperation.
        #   In PYNATIVE_MODE calling Grad from functions decorated with 'jit', use the out layer after_grad do
        #   grad in GRAPH_MODE.
        if context.get_context("mode") == context.GRAPH_MODE:
            dynamic_shape_inputs = None
            if isinstance(fn, nn.Cell):
                dynamic_shape_inputs = fn.get_inputs()
            if self.get_by_position:
                @jit(input_signature=dynamic_shape_inputs)
                def after_grad(*args):
                    return grad_(fn, weights, grad_position)(*args)
            else:
                if self.get_by_list:
                    @jit(input_signature=dynamic_shape_inputs)
                    def after_grad(*args):
                        return grad_(fn, weights)(*args)
                else:
                    @jit(input_signature=dynamic_shape_inputs)
                    def after_grad(*args):
                        return grad_(fn)(*args)
        elif self.pynative_:
            if not _pynative_executor.enable_grad():
                raise RuntimeError("In no_grad context, you can not calculate gradient")

            @_wrap_func
            def after_grad(*args, **kwargs):
                res = self._pynative_forward_run(fn, grad_, weights, args, kwargs)
                _pynative_executor.grad(fn, grad_, weights, grad_position, *args, **kwargs)
                out = _pynative_executor()
                out = _grads_divided_by_device_num_if_recomputation(out)
                if self.return_ids and out:
                    out = _combine_with_ids(grad_position, weights, out)
                if self.get_value:
                    return res, out
                if self.has_aux:
                    return out, res[1:]
                return out
        else:
            if not _pynative_executor.enable_grad():
                raise RuntimeError("In no_grad context, you can not calculate gradient")
            grad_.pynative_ = True
            fn_ = fn
            if self.has_aux:
                fn_ = aux_fn
            # after_grad of this branch can't use @jit, just directly call grad_
            if self.get_by_position:
                def after_grad(*args, **kwargs):
                    return grad_(fn_, weights, grad_position)(*args, **kwargs)
            else:
                if self.get_by_list:
                    def after_grad(*args, **kwargs):
                        return grad_(fn_, weights)(*args, **kwargs)
                else:
                    def after_grad(*args, **kwargs):
                        return grad_(fn_)(*args, **kwargs)

        self.grad_fn = after_grad
        self.fn = fn
        self.grad_position = grad_position
        self.weights_id = weights_id
        return self.grad_fn

    def _pynative_forward_run(self, fn, grad, weights, args, kwargs):
        """ Pynative forward runs to build grad graph. """
        new_kwargs = kwargs
        outputs = ()
        if self.sens_param:
            if 'sens' in kwargs.keys():
                new_kwargs = kwargs.copy()
                new_kwargs.pop('sens')
            else:
                args = args[:-1]
        if isinstance(fn, (FunctionType, MethodType)):
            if not _pynative_executor.check_run(grad, fn, weights, self.grad_position, *args, **new_kwargs):
                _pynative_executor.set_grad_flag(True)
                _pynative_executor.new_graph(fn, *args, **new_kwargs)
                outputs = fn(*args, **new_kwargs)
                _pynative_executor.end_graph(fn, outputs, *args, **new_kwargs)
                return outputs
        else:
            # Check if fn has run already.
            if not _pynative_executor.check_run(grad, fn, weights, self.grad_position, *args, **new_kwargs):
                fn.set_grad()
                outputs = fn(*args, **new_kwargs)
                fn.set_grad(False)
                return outputs
        if (self.get_value or self.has_aux) and not outputs:
            outputs = fn(*args, **new_kwargs)
        return outputs


@constexpr
def _get_grad_op(get_by_list, get_by_position, has_aux, get_value=False, return_ids=False):
    return _Grad(get_by_list=get_by_list, get_by_position=get_by_position, has_aux=has_aux, get_value=get_value,
                 return_ids=return_ids)


def value_and_grad(fn, grad_position=0, weights=None, has_aux=False):
    """
    A wrapper function to generate the function to calculate forward output and gradient for the input function.

    As for gradient, three typical cases are included:

    1. gradient with respect to inputs. In this case, `grad_position` is not None while `weights` is None.
    2. gradient with respect to weights. In this case, `grad_position` is None while `weights` is not None.
    3. gradient with respect to inputs and weights. In this case, `grad_position` and `weights` are not None.

    Args:
        fn (Union[Cell, Function]): Function to do GradOperation.
        grad_position (Union[NoneType, int, tuple[int]]): Index to specify which inputs to be differentiated.
            If int, get the gradient with respect to single input.
            If tuple, get the gradients with respect to selected inputs. `grad_position` begins with 0.
            If None, none derivative of any input will be solved, and in this case, `weights` is required.
            Default: ``0`` .
        weights (Union[ParameterTuple, Parameter, list[Parameter]]): The parameters of the training network that need to
            calculate the gradient. `weights` can be got through `weights = net.trainable_params()` .
            Default: ``None`` .
        has_aux (bool): If ``True`` , only the first output of `fn` contributes the gradient of `fn`, while the other
            outputs will be returned straightly. It means the `fn` must return more than one outputs in this case.
            Default: ``False`` .

    Returns:
        Function, returns the gradient function to calculate forward output and gradient for the input function or cell.
        For example, as for `out1, out2 = fn(*args)` , gradient function will return outputs like
        `((out1, out2), gradient)` . When `has_aux` is set True, only `out1` contributes to the differentiation.

    Raises:
        ValueError: If both `grad_position` and `weights` are None.
        TypeError: If type of Args does not belong to required ones.

    Supported Platforms:
        ``Ascend`` ``GPU`` ``CPU``

    Examples:
        >>> import numpy as np
        >>> import mindspore
        >>> from mindspore import Tensor, ops, nn
        >>> from mindspore import value_and_grad
        >>>
        >>> # Cell object to be differentiated
        >>> class Net(nn.Cell):
        ...     def construct(self, x, y, z):
        ...         return x * y * z
        >>> x = Tensor([1, 2], mindspore.float32)
        >>> y = Tensor([-2, 3], mindspore.float32)
        >>> z = Tensor([0, 3], mindspore.float32)
        >>> net = Net()
        >>> grad_fn = value_and_grad(net, grad_position=1)
        >>> output, inputs_gradient = grad_fn(x, y, z)
        >>> print(output)
        [-0. 18.]
        >>> print(inputs_gradient)
        [0. 6.]
        >>>
        >>> # Function object to be differentiated
        >>> def fn(x, y, z):
        ...     res = x * ops.exp(y) * ops.pow(z, 2)
        ...     return res, z
        >>> x = Tensor(np.array([3, 3]).astype(np.float32))
        >>> y = Tensor(np.array([0, 0]).astype(np.float32))
        >>> z = Tensor(np.array([5, 5]).astype(np.float32))
        >>> output, inputs_gradient = value_and_grad(fn, grad_position=(1, 2), weights=None, has_aux=True)(x, y, z)
        >>> print(output)
        (Tensor(shape=[2], dtype=Float32, value= [ 7.50000000e+01,  7.50000000e+01]),
         Tensor(shape=[2], dtype=Float32, value= [ 5.00000000e+00,  5.00000000e+00]))
        >>> print(inputs_gradient)
        (Tensor(shape=[2], dtype=Float32, value= [ 7.50000000e+01,  7.50000000e+01]),
         Tensor(shape=[2], dtype=Float32, value= [ 3.00000000e+01,  3.00000000e+01]))
        >>>
        >>> # For given network to be differentiated with both inputs and weights, there are 3 cases.
        >>> net = nn.Dense(10, 1)
        >>> loss_fn = nn.MSELoss()
        >>> def forward(inputs, labels):
        ...     logits = net(inputs)
        ...     loss = loss_fn(logits, labels)
        ...     return loss, logits
        >>> inputs = Tensor(np.random.randn(16, 10).astype(np.float32))
        >>> labels = Tensor(np.random.randn(16, 1).astype(np.float32))
        >>> weights = net.trainable_params()
        >>>
        >>> # Case 1: gradient with respect to inputs.
        >>> # For has_aux is set True, only loss contributes to the gradient.
        >>> grad_fn = value_and_grad(forward, grad_position=0, weights=None, has_aux=True)
        >>> (loss, logits), inputs_gradient = grad_fn(inputs, labels)
        >>> print(logits.shape)
        (16, 1)
        >>> print(inputs.shape, inputs_gradient.shape)
        (16, 10) (16, 10)
        >>>
        >>> # Case 2: gradient with respect to weights.
        >>> # For has_aux is set True, only loss contributes to the gradient.
        >>> grad_fn = value_and_grad(forward, grad_position=None, weights=weights, has_aux=True)
        >>> (loss, logits), params_gradient = grad_fn(inputs, labels)
        >>> print(logits.shape)
        (16, 1)
        >>> print(len(weights), len(params_gradient))
        2 2
        >>>
        >>> # Case 3: gradient with respect to inputs and weights.
        >>> # For has_aux is set False, both loss and logits contribute to the gradient.
        >>> grad_fn = value_and_grad(forward, grad_position=0, weights=weights, has_aux=False)
        >>> (loss, logits), (inputs_gradient, params_gradient) = grad_fn(inputs, labels)
        >>> print(logits.shape)
        (16, 1)
        >>> print(inputs.shape, inputs_gradient.shape)
        (16, 10) (16, 10)
        >>> print(len(weights), len(params_gradient))
        2 2
    """
    if grad_position is None and weights is None:
        raise ValueError("`grad_position` and `weight` can not be None at the same time.")

    if grad_position is None:
        return _get_grad_op(True, False, has_aux, True)(fn, weights)

    grad_position = _convert_grad_position_type(grad_position)
    if weights is None:
        return _get_grad_op(False, True, has_aux, True)(fn, None, grad_position)
    return _get_grad_op(True, True, has_aux, True)(fn, weights, grad_position)
