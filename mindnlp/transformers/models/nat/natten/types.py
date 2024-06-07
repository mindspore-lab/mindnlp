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

from typing import List, Tuple, Union

from torch import Tensor

NoneType = type(None)

Dimension1DType = Tuple[int]
Dimension2DType = Tuple[int, int]
Dimension3DType = Tuple[int, int, int]

CausalArg1DType = Tuple[bool]
CausalArg2DType = Tuple[bool, bool]
CausalArg3DType = Tuple[bool, bool, bool]

# NOTE: switch to | when < 3.10 support is dropped
Dimension1DTypeOrDed = Union[int, Dimension1DType]
Dimension2DTypeOrDed = Union[int, Dimension2DType]
Dimension3DTypeOrDed = Union[int, Dimension3DType]

CausalArg1DTypeOrDed = Union[bool, CausalArg1DType]
CausalArg2DTypeOrDed = Union[bool, CausalArg2DType]
CausalArg3DTypeOrDed = Union[bool, CausalArg3DType]

DimensionType = Union[Dimension1DType, Dimension2DType, Dimension3DType]
CausalArgType = Union[CausalArg1DType, CausalArg2DType, CausalArg3DType]

# (query_tile_shape, kv_tile_shape)
FnaTileShapeType = Union[
    Tuple[Dimension1DType, Dimension1DType],
    Tuple[Dimension2DType, Dimension2DType],
    Tuple[Dimension3DType, Dimension3DType],
]

# (query_tile_shape, kv_tile_shape)
FnaForwardConfigType = FnaTileShapeType

# (query_tile_shape, kv_tile_shape, num_kv_splits, use_torch_to_compute_delta)
FnaBackwardConfigType = Union[
    Tuple[Dimension1DType, Dimension1DType, Dimension1DType, bool],
    Tuple[Dimension2DType, Dimension2DType, Dimension2DType, bool],
    Tuple[Dimension3DType, Dimension3DType, Dimension3DType, bool],
]

ListOrNestedTensor = Union[List, Tensor]


# Redundant, but here to accommodate the type checker.
def create_dim_from_int(na_dim: int, value: int) -> DimensionType:
    assert 0 < na_dim < 4
    if na_dim == 2:
        return (value, value)
    if na_dim == 3:
        return (value, value, value)
    return (value,)


def create_causal_arg_from_bool(na_dim: int, value: bool) -> CausalArgType:
    assert 0 < na_dim < 4
    if na_dim == 2:
        return (value, value)
    if na_dim == 3:
        return (value, value, value)
    return (value,)
