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

from typing import Any, List, Tuple

import torch
from torch import Tensor

try:
    from natten import libnatten  # type: ignore
except ImportError:
    raise ImportError(
        "Failed to import NATTEN's CPP backend. "
        "This could be due to an invalid/incomplete install. "
        "Please uninstall NATTEN (pip uninstall natten) and re-install with the"
        " correct torch build: shi-labs.com/natten ."
    )

from ..types import CausalArgType, DimensionType, FnaForwardConfigType
from .configs import (
    _FNA_FORWARD_32x128_TILE_SIZES,
    _FNA_FORWARD_64x128_TILE_SIZES,
    _FNA_FORWARD_64x64_TILE_SIZES,
)
from .misc import get_device_cc

FNA_FORWARD_CPP_FUNC = {
    1: libnatten.na1d_forward,
    2: libnatten.na2d_forward,
    3: libnatten.na3d_forward,
}

FnaForwardInputsType = Tuple[Tensor, Tensor, Tensor, Tensor]


def get_default_tiling_config_for_fna_forward(
    na_dim: int,
    input_tensor: Tensor,  # ignored
    dilation: DimensionType,  # ignored
) -> FnaForwardConfigType:
    assert na_dim > 0 and na_dim < 4
    if na_dim == 2:
        return ((8, 8), (8, 8))
    if na_dim == 3:
        return ((4, 4, 4), (4, 4, 4))
    return ((64,), (64,))


def _get_all_tiling_configs_for_fna_forward(
    na_dim: int,
    device: Any,
) -> List[FnaForwardConfigType]:
    assert na_dim > 0 and na_dim < 4
    # SM80 and SM90 have more shared memory than SM86 and SM89
    # and their tensor core GEMMs can therefore target larger
    # tile shapes.
    # SM80 and 86 have been tested, but I don't have an SM89.
    # However, I suspect SM89 is to SM90 what SM86 was to SM80
    # in terms of shared memory (and only that).
    # Better to disable the larger tile configs for SM89 as well
    # as 86 until we can test it.
    if get_device_cc(device) in [86, 89]:
        return (
            _FNA_FORWARD_32x128_TILE_SIZES[na_dim]
            + _FNA_FORWARD_64x64_TILE_SIZES[na_dim]
        )

    return (
        _FNA_FORWARD_32x128_TILE_SIZES[na_dim]
        + _FNA_FORWARD_64x64_TILE_SIZES[na_dim]
        + _FNA_FORWARD_64x128_TILE_SIZES[na_dim]
    )


def get_all_tiling_configs_for_fna_forward(
    na_dim: int,
    input_tensor: Tensor,
    dilation: DimensionType,  # ignored
) -> List[FnaForwardConfigType]:
    return _get_all_tiling_configs_for_fna_forward(
        na_dim=na_dim,
        device=input_tensor.device,
    )


def _initialize_tensors_for_fna_forward(
    na_dim: int,
    tensor_shape: torch.Size,
    dtype: Any,
    device: Any,
) -> FnaForwardInputsType:
    q = torch.empty(tensor_shape, dtype=dtype, device=device)
    k = torch.empty_like(q)
    v = torch.empty_like(q)
    out = torch.empty_like(q)
    return (out, q, k, v)


def initialize_tensors_for_fna_forward(
    na_dim: int, input_tensor: Tensor
) -> FnaForwardInputsType:
    assert input_tensor.dim() == na_dim + 3
    return _initialize_tensors_for_fna_forward(
        na_dim=na_dim,
        tensor_shape=input_tensor.shape,
        dtype=input_tensor.dtype,
        device=input_tensor.device,
    )


def run_fna_forward(
    na_dim: int,
    inputs: FnaForwardInputsType,
    kernel_size: DimensionType,
    dilation: DimensionType,
    is_causal: CausalArgType,
    tile_config: FnaForwardConfigType,
):
    assert na_dim in FNA_FORWARD_CPP_FUNC.keys()
    assert isinstance(inputs, tuple) and len(inputs) == 4
    out, q, k, v = inputs
    FNA_FORWARD_CPP_FUNC[na_dim](
        out,
        q,
        k,
        v,
        None,  # rpb
        None,  # logsumexp
        kernel_size,
        dilation,
        is_causal,
        1.0,  # attn_scale
        *tile_config
    )
