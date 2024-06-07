#################################################################################################
# Copyright (c) 2022-2024 Ali Hassani.
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.
#
#################################################################################################

from typing import Optional

import torch
from torch.cuda import _device_t

from ..types import DimensionType


def get_device_cc(device_index: Optional[_device_t] = None) -> int:
    major, minor = torch.cuda.get_device_capability(device_index)
    return major * 10 + minor


def _ceil_div(x: int, y: int) -> int:
    return (x + y - 1) // y


def get_min_splits(na_dim: int) -> DimensionType:
    # accommodating type checker
    # return tuple(1 for _ in range(na_dim))
    assert 0 < na_dim < 4
    if na_dim == 1:
        return (1,)
    if na_dim == 2:
        return (1, 1)
    return (1, 1, 1)


def get_max_splits(
    spatial_extent: DimensionType, dilation: DimensionType, kv_tile_shape: DimensionType
) -> DimensionType:
    # Again, accommodating type checker!
    # extent_post_partitioning = _ceil_div_dim(spatial_extent, dilation)
    # return tuple(
    #    _ceil_div(extent_, kv_tile)
    #    for extent_, kv_tile in zip(extent_post_partitioning, kv_tile_shape)
    # )
    assert len(spatial_extent) == len(dilation) == len(kv_tile_shape)
    na_dim = len(spatial_extent)
    assert 0 < na_dim < 4
    if na_dim == 2:
        assert len(spatial_extent) == len(dilation) == len(kv_tile_shape) == 2
        return (
            _ceil_div(_ceil_div(spatial_extent[0], dilation[0]), kv_tile_shape[0]),
            _ceil_div(_ceil_div(spatial_extent[1], dilation[1]), kv_tile_shape[1]),
        )
    if na_dim == 3:
        assert len(spatial_extent) == len(dilation) == len(kv_tile_shape) == 3
        return (
            _ceil_div(_ceil_div(spatial_extent[0], dilation[0]), kv_tile_shape[0]),
            _ceil_div(_ceil_div(spatial_extent[1], dilation[1]), kv_tile_shape[1]),
            _ceil_div(_ceil_div(spatial_extent[2], dilation[2]), kv_tile_shape[2]),
        )
    assert len(spatial_extent) == len(dilation) == len(kv_tile_shape) == 1
    return (_ceil_div(_ceil_div(spatial_extent[0], dilation[0]), kv_tile_shape[0]),)
