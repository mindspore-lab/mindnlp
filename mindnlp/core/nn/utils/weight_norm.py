# Copyright 2024 Huawei Technologies Co., Ltd
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ============================================================================
r"""Weight Normalization from https://arxiv.org/abs/1602.07868."""
from typing import Any, TypeVar
from mindspore import Parameter
from ..modules import Module
from ... import ops

__all__ = ['WeightNorm', 'weight_norm', 'remove_weight_norm']

def norm_except_dim(weight_v, pows, dim):
    r"""
    calculte g/||weight_v|| * weight_v method 
    """
    if dim == -1:
        return ops.norm(weight_v, pows)
    if dim == 0:
        w_shape_v = weight_v.shape[0] # avoid macOS error
        output_size = (w_shape_v,) + (1,) * (weight_v.ndim - 1)
        return ops.norm(weight_v.view((w_shape_v, -1)), pows, 1).view(output_size)
    if dim == (weight_v.ndim - 1):
        output_size = (1,) * (weight_v.ndim - 1) + (weight_v.shape[weight_v.ndim - 1],)
        return ops.norm(weight_v.view((-1, weight_v.shape[weight_v.ndim - 1])), pows, 0).view(output_size)
    return norm_except_dim(weight_v.swapaxes(0, dim), pows, dim).swapaxes(0, dim)

def _weight_norm(weight_v, weight_g, dim):
    r"""
    calculte weight_g/||weight_v|| * weight_v method 
    """
    return weight_v * (weight_g / norm_except_dim(weight_v, 2, dim))


class WeightNorm:

    r"""
    The 'WeightNorm' class implements weight normalization for neural network modules. It provides methods to compute normalized weights, apply weight normalization to a cell, wrap a function, and remove
    weight bias from a cell. The class also contains an initializer for the name and dimension of the weight parameters, as well as a method to compute the weight using the normalized parameters. Additionally, it
    includes a method to remove the weight bias and a wrapper function for transposing cell_id to cell. 
    """
    name: str
    dim: int

    def __init__(self, name: str, dim: int) -> None:
        if dim is None:
            dim = -1
        self.name = name
        self.dim = dim

    # TODO Make return type more specific
    def compute_weight(self, module: Module) -> Any:
        g = getattr(module, self.name + '_g')
        v = getattr(module, self.name + '_v')
        return Parameter(_weight_norm(v, g, self.dim))

    @staticmethod
    def apply(module, name: str, dim: int) -> 'WeightNorm':
        for k, hook in module._forward_pre_hooks.items():
            if isinstance(hook, WeightNorm) and hook.name == name:
                raise RuntimeError("Cannot register two weight_norm hooks on "
                                   "the same parameter {}".format(name))

        if dim is None:
            dim = -1

        fn = WeightNorm(name, dim)

        weight = getattr(module, name)
        # if isinstance(weight, UninitializedParameter):
        #     raise ValueError(
        #         'The module passed to `WeightNorm` can\'t have uninitialized parameters. '
        #         'Make sure to run the dummy forward before applying weight normalization')
        # remove w from parameter list
        del module._parameters[name]

        # add g and v as new parameters and express w as g/||v|| * v
        module.register_parameter(name + '_g', Parameter(norm_except_dim(weight, 2, dim)))
        module.register_parameter(name + '_v', Parameter(weight))
        setattr(module, name, fn.compute_weight(module))

        # recompute weight before every forward()
        module.register_forward_pre_hook(fn)

        return fn

    def wrapper_func(self, cell, func):
        r"""
        wrapper_func where used to transpose cell_id to cell
        """
        def new_func(_, inputs):
            nonlocal cell
            return func(cell, inputs)
        return new_func

    def remove(self, module: Module) -> None:
        weight = self.compute_weight(module)
        delattr(module, self.name)
        del module._parameters[self.name + '_g']
        del module._parameters[self.name + '_v']
        setattr(module, self.name, weight)

    def __call__(self, module: Module, inputs: Any) -> None:
        setattr(module, self.name, self.compute_weight(module))


T_module = TypeVar('T_module', bound=Module)

def weight_norm(module: T_module, name: str = 'weight', dim: int = 0) -> T_module:
    r"""Apply weight normalization to a parameter in the given module.

    .. math::
         \mathbf{w} = g \dfrac{\mathbf{v}}{\|\mathbf{v}\|}

    Weight normalization is a reparameterization that decouples the magnitude
    of a weight tensor from its direction. This replaces the parameter specified
    by :attr:`name` (e.g. ``'weight'``) with two parameters: one specifying the magnitude
    (e.g. ``'weight_g'``) and one specifying the direction (e.g. ``'weight_v'``).
    Weight normalization is implemented via a hook that recomputes the weight
    tensor from the magnitude and direction before every :meth:`~Module.forward`
    call.

    By default, with ``dim=0``, the norm is computed independently per output
    channel/plane. To compute a norm over the entire weight tensor, use
    ``dim=None``.

    See https://arxiv.org/abs/1602.07868

    .. warning::

        This function is deprecated.  Use :func:`torch.nn.utils.parametrizations.weight_norm`
        which uses the modern parametrization API.  The new ``weight_norm`` is compatible
        with ``state_dict`` generated from old ``weight_norm``.

        Migration guide:

        * The magnitude (``weight_g``) and direction (``weight_v``) are now expressed
          as ``parametrizations.weight.original0`` and ``parametrizations.weight.original1``
          respectively.  If this is bothering you, please comment on
          https://github.com/pytorch/pytorch/issues/102999

        * To remove the weight normalization reparametrization, use
          :func:`torch.nn.utils.parametrize.remove_parametrizations`.

        * The weight is no longer recomputed once at module forward; instead, it will
          be recomputed on every access.  To restore the old behavior, use
          :func:`torch.nn.utils.parametrize.cached` before invoking the module
          in question.

    Args:
        module (Module): containing module
        name (str, optional): name of weight parameter
        dim (int, optional): dimension over which to compute the norm

    Returns:
        The original module with the weight norm hook

    Example::

        >>> m = weight_norm(nn.Linear(20, 40), name='weight')
        >>> m
        Linear(in_features=20, out_features=40, bias=True)
        >>> m.weight_g.size()
        torch.Size([40, 1])
        >>> m.weight_v.size()
        torch.Size([40, 20])

    """
    WeightNorm.apply(module, name, dim)
    return module

def remove_weight_norm(module: T_module, name: str = 'weight') -> T_module:
    r"""Removes the weight normalization reparameterization from a module.

    Args:
        module (Module): containing module
        name (str, optional): name of weight parameter

    Example:
        >>> m = weight_norm(nn.Linear(20, 40))
        >>> remove_weight_norm(m)
    """
    for k, hook in module._forward_pre_hooks.items():
        if isinstance(hook, WeightNorm) and hook.name == name:
            hook.remove(module)
            del module._forward_pre_hooks[k]
            return module

    raise ValueError("weight_norm of '{}' not found in {}"
                     .format(name, module))
