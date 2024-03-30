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
from mindspore import Parameter, nn, ops
from mindspore import numpy as mnp

__all__ = ['WeightNorm', 'weight_norm', 'remove_weight_norm']

def norm_except_dim(weight_v, pows, dim):
    r"""
    calculte g/||weight_v|| * weight_v method 
    """
    if dim == -1:
        return mnp.norm(weight_v, pows)
    if dim == 0:
        w_shape_v = ops.shape(weight_v)[0] # avoid macOS error
        output_size = (w_shape_v,) + (1,) * (weight_v.ndim - 1)
        return mnp.norm(weight_v.view((w_shape_v, -1)), pows, 1).view(output_size)
    if dim == (weight_v.ndim - 1):
        output_size = (1,) * (weight_v.ndim - 1) + (weight_v.shape[weight_v.ndim - 1])
        return mnp.norm(weight_v.view((-1, weight_v.shape[weight_v.ndim - 1])), pows, 0).view(output_size)
    return norm_except_dim(weight_v.swapaxes(0, dim), pows, dim).swapaxes(0, dim)

def _weight_norm(weight_v, weight_g, dim):
    r"""
    calculte weight_g/||weight_v|| * weight_v method 
    """
    return weight_v * (weight_g / norm_except_dim(weight_v, 2, dim))


class WeightNorm:
    name: str
    dim: int

    def __init__(self, name: str, dim: int) -> None:
        if dim is None:
            dim = -1
        self.name = name
        self.dim = dim

    # TODO Make return type more specific
    def compute_weight(self, module: nn.Cell) -> Any:
        g = getattr(module, self.name + '_g')
        v = getattr(module, self.name + '_v')
        return Parameter(_weight_norm(v, g, self.dim), 'weight')

    @staticmethod
    def apply(cell: nn.Cell, name: str, dim: int) -> 'WeightNorm':
        r"""
        construct methods
        """
        # warnings.warn("torch.nn.utils.weight_norm is deprecated in favor of torch.nn.utils.parametrizations.weight_norm.")
        for hook in cell._forward_pre_hook.values():
            if isinstance(hook, WeightNorm) and hook.name == name:
                raise RuntimeError(f"Cannot register two weight_norm hooks on the same parameter {name}")

        if dim is None:
            dim = -1

        weight_fn = WeightNorm(name, dim)

        weight = getattr(cell, name)
        del cell._params[name]
        setattr(cell, name + '_g', Parameter(norm_except_dim(weight, 2 ,dim)))
        setattr(cell, name + '_v', Parameter(weight.data))
        setattr(cell, name, Parameter(weight_fn.compute_weight(cell)))
        cell.register_forward_pre_hook(weight_fn.wrapper_func(cell, weight_fn.__call__))
        return weight_fn

    def wrapper_func(self, cell, func):
        r"""
        wrapper_func where used to transpose cell_id to cell
        """
        def new_func(_, inputs):
            nonlocal cell
            return func(cell, inputs)
        return new_func

    def remove(self, cell: nn.Cell) -> None:
        r"""
        remove weight bias
        """
        weight = self.compute_weight(cell)
        delattr(cell, self.name)
        del cell._params[self.name + '_g']
        del cell._params[self.name + '_v']
        setattr(cell, self.name, Parameter(weight.data))

    def __call__(self, module: nn.Cell, inputs: Any) -> None:
        setattr(module, self.name, self.compute_weight(module))


T_module = TypeVar('T_module', bound=nn.Cell)

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
    r"""Remove the weight normalization reparameterization from a module.

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

    raise ValueError(f"weight_norm of '{name}' not found in {module}")
