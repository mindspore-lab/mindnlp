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
"""Weight Norm"""

from typing import Any
from types import MethodType

import numpy as np
from mindspore import nn
from mindspore import Tensor, Parameter

__all__ = [
    'weight_norm',
    'remove_weight_norm'
]


# ref: https://zhuanlan.zhihu.com/p/425483093
def norm_except_dim(weight_v:Tensor, p:float, dim:int=-1) -> Tensor:
    ''' ||weight_v|| '''
    weight_v = weight_v.asnumpy()
    if dim == -1:
        return np.linalg.norm(weight_v, p)
    if dim == 0:
        output_size = (weight_v.shape[0],) + (1,) * (weight_v.ndim - 1)
        return np.linalg.norm(weight_v.reshape((weight_v.shape[0], -1)), p, 1).reshape(output_size)
    if dim == (weight_v.ndim - 1):
        output_size = (1,) * (weight_v.ndim - 1) + (weight_v.shape[weight_v.ndim - 1],)
        return np.linalg.norm(weight_v.reshape((-1, weight_v.shape[weight_v.ndim - 1])), p, 0).reshape(output_size)
    return norm_except_dim(weight_v.swapaxes(0, dim), p, dim).swapaxes(0, dim)


def _weight_norm(weight_v:Tensor, weight_g:Tensor, dim:int=-1) -> Tensor:
    ''' weight = weight_g * weight_v / ||weight_v|| '''
    return weight_g.asnumpy() * weight_v.asnumpy() / norm_except_dim(weight_v, 2, dim)


def recompute_weight(cell:nn.Module):
    """
    Recomputes the weight of a neural network cell.
    
    Args:
        cell (nn.Module): The neural network cell for which the weight needs to be recomputed.
    
    Returns:
        None. The function updates the weight of the input cell in-place.
    
    Raises:
        AssertionError: If the weight property specified by the cell's wn_name is not found.
    """
    name: str = cell.wn_name
    g = getattr(cell, f'{name}_g')
    v = getattr(cell, f'{name}_v')
    new_weight = _weight_norm(v, g, cell.wn_dim)
    weight: Parameter = getattr(cell, name, None)
    assert weight is not None, f'property {name!r} not found'
    weight.set_data(Tensor(new_weight))


def weight_norm(cell:nn.Module, name:str='weight', dim:int=-1, axis:int=None) -> nn.Module:
    r"""
    Applies weight normalization to a neural network cell.
    
    Args:
        cell (nn.Module): The neural network cell to apply weight normalization to.
        name (str, optional): The name of the weight property. Defaults to 'weight'.
        dim (int, optional): The dimension along which the normalization is computed. Defaults to -1.
        axis (int, optional): An alternative way to specify the dimension along which the normalization is computed. 
            If provided, it overrides the value of 'dim'. Defaults to None.
    
    Returns:
        nn.Module: The neural network cell with weight normalization applied.
    
    Raises:
        AssertionError: If the weight property with the specified name is not found.
    
    Note:
        The 'cell' parameter is modified in-place by adding additional properties and methods to enable weight normalization.
    """
    if axis is not None:
        dim = axis     # compat fix
    weight: Parameter = getattr(cell, name, None)
    assert weight is not None, f'property {name!r} not found'
    dtype = weight.data.dtype
    setattr(cell, f'{name}_g', Parameter(Tensor(norm_except_dim(weight.data, 2, dim), dtype)))
    setattr(cell, f'{name}_v', Parameter(Tensor(weight.data, dtype)))

    cell.wn_forward = cell.forward
    def forward_hijack(self:nn.Module, *args, **kwargs) -> Any:
        recompute_weight(self)
        return self.wn_forward(*args, **kwargs)
    cell.forward = MethodType(forward_hijack, cell)
    cell.wn_name = name
    cell.wn_dim = dim
    return cell


def remove_weight_norm(cell:nn.Module) -> nn.Module:
    r"""
    Removes weight normalization from the given neural network cell.
    
    Args:
        cell (nn.Module): The neural network cell from which weight normalization will be removed.
    
    Returns:
        nn.Module: The modified neural network cell with weight normalization removed.
    
    Raises:
        None.
    
    Note:
        This function assumes that the input cell has weight normalization applied. If the cell does not have any weight normalization, it returns the cell as is.
    
    Example:
        >>> cell = MyNeuralNetworkCell()
        >>> cell = remove_weight_norm(cell)
    """
    if getattr(cell, 'wn_name', None) is None:
        return cell

    recompute_weight(cell)
    name = cell.wn_name
    delattr(cell, f'{name}_g')
    delattr(cell, f'{name}_v')
    cell.forward = cell.wn_forward
    del cell.wn_forward
    del cell.wn_name
    del cell.wn_dim
    return cell
