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

import math
from itertools import product
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

from ..context import (
    is_autotuner_thorough_for_backward,
    is_kv_parallelism_in_fused_na_enabled,
    is_memory_usage_strict,
    is_memory_usage_unrestricted,
)
from ..types import (
    CausalArgType,
    DimensionType,
    FnaBackwardConfigType,
    FnaTileShapeType,
)
from .configs import (
    _FNA_BACKWARD_128x128_TILE_SIZES,
    _FNA_BACKWARD_128x64_TILE_SIZES,
    _FNA_BACKWARD_64x64_TILE_SIZES,
)
from .misc import get_device_cc, get_max_splits, get_min_splits


def _get_max_grid_size_allowed() -> int:
    if is_memory_usage_unrestricted():
        return 65535
    if is_memory_usage_strict():
        return 1024

    return 4096


def _reduce_max_kv_splits(
    na_dim: int,
    kv_splits: DimensionType,
    max_splits: int,
) -> DimensionType:
    assert isinstance(kv_splits, tuple)
    if na_dim == 1:
        assert len(kv_splits) == 1
        return (min(kv_splits[0], max_splits),)

    if na_dim == 2:
        assert len(kv_splits) == 2
        splits_x = max(min(max_splits // 2, kv_splits[0]), 1)
        splits_y = max(min(max_splits // splits_x, kv_splits[1]), 1)
        assert (
            0 < splits_x * splits_y <= max_splits
        ), f"{splits_x=} * {splits_y=} does not fall in range [0, {max_splits}]"
        return (splits_x, splits_y)

    if na_dim == 3:
        assert len(kv_splits) == 3
        splits_x = max(min(max_splits // 3, kv_splits[0]), 1)
        splits_y = max(min(max_splits // splits_x, kv_splits[1]), 1)
        splits_z = max(min(max_splits // (splits_x * splits_y), kv_splits[2]), 1)
        assert (
            0 < splits_x * splits_y * splits_z <= max_splits
        ), f"{splits_x=} * {splits_y=} * {splits_z=} does not fall in range [0, {max_splits}]"
        return (splits_x, splits_y, splits_z)

    raise NotImplementedError()


def _get_possible_kv_splits(
    min_splits: DimensionType,
    max_splits: DimensionType,
):
    assert 0 < len(min_splits) == len(max_splits) < 4
    na_dim = len(max_splits)
    if na_dim == 1:
        return product(
            range(min_splits[0], max_splits[0] + 1),
        )
    if na_dim == 2:
        assert len(min_splits) == len(max_splits) == 2
        return product(
            range(min_splits[0], max_splits[0] + 1),
            range(min_splits[1], max_splits[1] + 1),
        )
    if na_dim == 3:
        assert len(min_splits) == len(max_splits) == 3
        return product(
            range(min_splits[0], max_splits[0] + 1),
            range(min_splits[1], max_splits[1] + 1),
            range(min_splits[2], max_splits[2] + 1),
        )

    raise NotImplementedError()


FNA_BACKWARD_CPP_FUNC = {
    1: libnatten.na1d_backward,
    2: libnatten.na2d_backward,
    3: libnatten.na3d_backward,
}


FnaBackwardInputsType = Tuple[
    Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor
]


def get_default_tiling_config_for_fna_backward(
    na_dim: int,
    input_tensor: Tensor,
    dilation: DimensionType,
) -> FnaBackwardConfigType:
    assert na_dim > 0 and na_dim < 4
    assert input_tensor.dim() == na_dim + 3
    spatial_extent = tuple(x for x in input_tensor.shape[1 : na_dim + 1])
    assert len(spatial_extent) == na_dim and all(
        isinstance(x, int) for x in spatial_extent
    )

    default_q_tile: DimensionType = (64,)
    default_k_tile: DimensionType = (64,)
    kv_splits: DimensionType = (1,)

    if na_dim == 2:
        default_q_tile = (8, 8)
        default_k_tile = (8, 8)
        kv_splits = (1, 1)

    elif na_dim == 3:
        default_q_tile = (4, 4, 4)
        default_k_tile = (4, 4, 4)
        kv_splits = (1, 1, 1)

    if (
        is_kv_parallelism_in_fused_na_enabled()
        and not torch.are_deterministic_algorithms_enabled()
    ):
        kv_splits = get_max_splits(spatial_extent, dilation, default_k_tile)  # type: ignore
        total_kv_splits = math.prod(kv_splits)

        batch_size = input_tensor.shape[0]
        num_heads = input_tensor.shape[-2]
        num_dilation_splits = math.prod(dilation)
        max_kv_splits_allowed = max(
            1,
            _get_max_grid_size_allowed()
            // (batch_size * num_heads * num_dilation_splits),
        )

        if total_kv_splits > max_kv_splits_allowed:
            kv_splits = _reduce_max_kv_splits(
                na_dim=na_dim, kv_splits=kv_splits, max_splits=max_kv_splits_allowed
            )
    return (default_q_tile, default_k_tile, kv_splits, False)  # type: ignore


def _get_all_tile_shapes_for_fna_backward(
    na_dim: int,
    dim_per_head: int,
    device: Any,
    dtype: Any,
) -> List[FnaTileShapeType]:
    assert na_dim > 0 and na_dim < 4
    assert dtype in [torch.float32, torch.float16, torch.bfloat16]
    compute_cap = get_device_cc(device)

    if dtype == torch.float32 and compute_cap not in [80, 90]:
        return _FNA_BACKWARD_64x64_TILE_SIZES[na_dim]
    elif dtype == torch.float32:
        return (
            _FNA_BACKWARD_64x64_TILE_SIZES[na_dim]
            + _FNA_BACKWARD_128x64_TILE_SIZES[na_dim]
        )

    if compute_cap == 70:
        return (
            _FNA_BACKWARD_64x64_TILE_SIZES[na_dim]
            + _FNA_BACKWARD_128x64_TILE_SIZES[na_dim]
        )

    if compute_cap in [80, 90] and dim_per_head <= 128:
        return (
            _FNA_BACKWARD_64x64_TILE_SIZES[na_dim]
            + _FNA_BACKWARD_128x128_TILE_SIZES[na_dim]
        )
    elif compute_cap in [80, 90]:
        return (
            _FNA_BACKWARD_64x64_TILE_SIZES[na_dim]
            + _FNA_BACKWARD_128x64_TILE_SIZES[na_dim]
        )

    return _FNA_BACKWARD_64x64_TILE_SIZES[na_dim]


def _get_all_tiling_configs_for_fna_backward(
    na_dim: int,
    batch_size: int,
    num_heads: int,
    spatial_extent: DimensionType,
    dilation: DimensionType,
    dim_per_head: int,
    device: Any,
    dtype: Any,
) -> List[FnaBackwardConfigType]:
    possible_tile_shapes = _get_all_tile_shapes_for_fna_backward(
        na_dim=na_dim, dim_per_head=dim_per_head, device=device, dtype=dtype
    )
    possible_configs = []
    assert (
        isinstance(spatial_extent, tuple)
        and isinstance(dilation, tuple)
        and len(spatial_extent) == len(dilation) == na_dim
    )
    num_dilation_splits = math.prod(dilation)
    max_kv_splits_allowed = (
        1
        if not is_kv_parallelism_in_fused_na_enabled()
        else max(
            1,
            _get_max_grid_size_allowed()
            // (batch_size * num_heads * num_dilation_splits),
        )
    )
    for query_tile_shape, kv_tile_shape in possible_tile_shapes:
        min_kv_splits = get_min_splits(na_dim)
        max_kv_splits = get_max_splits(spatial_extent, dilation, kv_tile_shape)
        max_kv_splits_total = math.prod(max_kv_splits)
        if max_kv_splits_total > max_kv_splits_allowed:
            max_kv_splits = _reduce_max_kv_splits(
                na_dim=na_dim, kv_splits=max_kv_splits, max_splits=max_kv_splits_allowed
            )

        # Potential duplicates
        if math.prod(max_kv_splits) > 1:
            if is_autotuner_thorough_for_backward():
                for kv_splits in _get_possible_kv_splits(min_kv_splits, max_kv_splits):
                    for use_pt_reduction in [False, True]:
                        possible_configs.append(
                            (
                                query_tile_shape,
                                kv_tile_shape,
                                kv_splits,
                                use_pt_reduction,
                            )
                        )
            else:
                # for use_pt_reduction in [False, True]:
                possible_configs.append(
                    (
                        query_tile_shape,
                        kv_tile_shape,
                        min_kv_splits,
                        False,
                    )
                )
                possible_configs.append(
                    (
                        query_tile_shape,
                        kv_tile_shape,
                        max_kv_splits,
                        False,
                    )
                )
        else:
            # min_kv_splits == max_kv_splits
            # for use_pt_reduction in [False, True]:
            possible_configs.append(
                (query_tile_shape, kv_tile_shape, min_kv_splits, False)
            )

    return possible_configs  # type: ignore


def get_all_tiling_configs_for_fna_backward(
    na_dim: int,
    input_tensor: Tensor,
    dilation: DimensionType,
) -> List[FnaBackwardConfigType]:
    assert input_tensor.dim() == na_dim + 3
    spatial_extent = tuple(x for x in input_tensor.shape[1 : na_dim + 1])
    assert len(spatial_extent) == na_dim and all(
        isinstance(x, int) for x in spatial_extent
    )
    batch_size = input_tensor.shape[0]
    num_heads = input_tensor.shape[-2]
    dim_per_head = input_tensor.shape[-1]
    return _get_all_tiling_configs_for_fna_backward(
        na_dim=na_dim,
        batch_size=batch_size,
        num_heads=num_heads,
        spatial_extent=spatial_extent,  # type: ignore
        dilation=dilation,
        dim_per_head=dim_per_head,
        device=input_tensor.device,
        dtype=input_tensor.dtype,
    )


def _initialize_tensors_for_fna_backward(
    na_dim: int,
    tensor_shape: torch.Size,
    dtype: Any,
    device: Any,
) -> FnaBackwardInputsType:
    lse_shape = tensor_shape[:-1]  # batch, *, heads
    q = torch.empty(tensor_shape, dtype=dtype, device=device)
    lse = torch.empty(lse_shape, dtype=torch.float32, device=device)
    k = torch.empty_like(q)
    v = torch.empty_like(q)
    out = torch.empty_like(q)
    d_q = torch.empty_like(q)
    d_k = torch.empty_like(q)
    d_v = torch.empty_like(q)
    d_out = torch.empty_like(q)
    return (d_q, d_k, d_v, q, k, v, out, d_out, lse)


def initialize_tensors_for_fna_backward(
    na_dim: int, input_tensor: Tensor
) -> FnaBackwardInputsType:
    return _initialize_tensors_for_fna_backward(
        na_dim=na_dim,
        tensor_shape=input_tensor.shape,
        dtype=input_tensor.dtype,
        device=input_tensor.device,
    )


def run_fna_backward(
    na_dim: int,
    inputs: FnaBackwardInputsType,
    kernel_size: DimensionType,
    dilation: DimensionType,
    is_causal: CausalArgType,
    tile_config: FnaBackwardConfigType,
):
    assert na_dim in FNA_BACKWARD_CPP_FUNC.keys()
    assert isinstance(inputs, tuple) and len(inputs) == 9
    (d_q, d_k, d_v, q, k, v, out, d_out, lse) = inputs
    FNA_BACKWARD_CPP_FUNC[na_dim](
        d_q,
        d_k,
        d_v,
        q,
        k,
        v,
        out,
        d_out,
        lse,
        kernel_size,
        dilation,
        is_causal,
        1.0,  # attn_scale
        *tile_config,
    )
