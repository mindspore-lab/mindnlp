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
from numbers import Number
from typing import Any, List

from fvcore.nn import FlopCountAnalysis  # type: ignore
from fvcore.nn.jit_handles import get_shape  # type: ignore


def fna_generic_flops(inputs: List[Any], outputs: List[Any]) -> Number:
    """
    Counts flops for generic fused neighborhood attention.
    """
    assert (
        len(inputs) >= 3
    ), f"Expected at least 3 inputs (query, key, value), got {len(inputs)}"
    has_bias = len(inputs) == 4 and inputs[-1] is not None
    assert len(outputs) >= 1, f"Expected at least 1 output, got {len(outputs)}"
    input_shapes = [get_shape(v) for v in inputs]
    output_shapes = [get_shape(v) for v in outputs]
    assert len(input_shapes[0]) in [
        4,
        5,
        6,
    ], f"Input tensors must be of rank 4, 5, or 6, got {len(input_shapes[0])}"
    assert len(input_shapes[1]) == len(input_shapes[1]) == len(input_shapes[2]), (
        f"All input tensors must be of the same rank, got {len(input_shapes[0])}, "
        + f"{len(input_shapes[1])} and {len(input_shapes[2])}"
    )
    assert len(output_shapes[0]) == len(
        input_shapes[0]
    ), f"Output tensor must match the rank of input tensors, got {len(output_shapes[0])} != {len(input_shapes[0])}"
    assert input_shapes[0] == input_shapes[1] == input_shapes[2] == output_shapes[0], (
        "Query, key, value, and output must match in shape, got q.shape="
        + f"{input_shapes[0]}, k.shape={input_shapes[1]}, v.shape={input_shapes[1]}."
    )
    batch_size, heads, dim = (
        input_shapes[0][0],
        input_shapes[0][-2],
        input_shapes[0][-1],
    )

    # NOTE: really hacky way to extract non-tensor args, but gets the job done.
    # The jit trace only picks up tensor operands in inputs and outputs, but
    # it's impossible to compute FLOPs without knowing kernel size.
    assert hasattr(inputs[0], "uses") and callable(inputs[0].uses)
    _uses = inputs[0].uses()
    assert (
        hasattr(_uses, "__len__")
        and len(_uses) == 1
        and hasattr(_uses[0], "user")
        and hasattr(_uses[0].user, "scalar_args")
        and callable(_uses[0].user.scalar_args)
    )
    scalar_args = _uses[0].user.scalar_args()
    assert hasattr(scalar_args, "__len__") and len(scalar_args) == 5
    kernel_size, dilation, is_causal, attn_scale, tiling_config = scalar_args
    # TODO: it's very easy to hit this assertion. We must make sure
    # arguments like kernel size are checked before calling the autograd function,
    # not inside it.
    assert isinstance(
        kernel_size, tuple
    ), f"Expected kernel_size to be a tuple, got {type(kernel_size)=}."

    assert len(kernel_size) + 3 == len(input_shapes[0]), (
        "Tensor rank must be equal to len(kernel_size) + 3 = "
        + f"{len(kernel_size) + 3}, got {len(input_shapes[0])}"
    )

    spatial_extent = input_shapes[0][1 : len(kernel_size) + 1]

    assert len(spatial_extent) == len(kernel_size)

    spatial_extent_int = math.prod(spatial_extent)
    kernel_size_int = math.prod(kernel_size)

    flops = batch_size * heads * spatial_extent_int * dim * kernel_size_int  # QK

    flops += batch_size * heads * spatial_extent_int * kernel_size_int  # softmax
    flops += batch_size * heads * spatial_extent_int * dim * kernel_size_int  # AV

    if has_bias:
        flops += batch_size * heads * spatial_extent_int * kernel_size_int  # RPB
    return flops


def qk_1d_rpb_flop(inputs: List[Any], outputs: List[Any]) -> Number:
    """
    Counts flops for the 1D QK operation.
    """
    assert (
        len(inputs) >= 2
    ), f"Expected at least 2 inputs (query, key), got {len(inputs)}"
    assert len(outputs) == 1, f"Expected 1 output (attn), got {len(outputs)}"
    input_shapes = [get_shape(v) for v in inputs]
    output_shapes = [get_shape(v) for v in outputs]
    assert (
        len(input_shapes[0]) == 4
    ), f"Query must be a 4-dim tensor, got {len(input_shapes[0])}"
    assert (
        len(input_shapes[1]) == 4
    ), f"Key must be a 4-dim tensor, got {len(input_shapes[1])}"
    assert (
        len(output_shapes[0]) == 4
    ), f"Output must be a 4-dim tensor, got {len(output_shapes[0])}"
    assert (
        input_shapes[0] == input_shapes[1]
    ), f"Query and Key shapes did not match! Q: {input_shapes[0]}, K: {input_shapes[1]}"
    batch_size, heads, length, dim = input_shapes[0]
    batch_size, heads, length, kernel_size = output_shapes[0]

    flops = batch_size * heads * length * dim * kernel_size
    flops += batch_size * heads * length * kernel_size
    return flops


def av_1d_flop(inputs: List[Any], outputs: List[Any]) -> Number:
    """
    Counts flops for the 1D AV operation.
    """
    assert len(inputs) == 2, f"Expected 2 inputs (attn and value), got {len(inputs)}"
    assert len(outputs) == 1, f"Expected 1 output (out), got {len(outputs)}"
    input_shapes = [get_shape(v) for v in inputs]
    output_shapes = [get_shape(v) for v in outputs]
    assert (
        len(input_shapes[0]) == 4
    ), f"Attn must be a 4-dim tensor, got {len(input_shapes[0])}"
    assert (
        len(input_shapes[1]) == 4
    ), f"Value must be a 4-dim tensor, got {len(input_shapes[1])}"
    assert (
        len(output_shapes[0]) == 4
    ), f"Output must be a 4-dim tensor, got {len(output_shapes[0])}"
    assert output_shapes[0] == input_shapes[1], (
        f"Out and Value shapes did not match! O: {output_shapes[0]}, V:"
        f" {input_shapes[1]}"
    )
    batch_size, heads, length, kernel_size = input_shapes[0]
    batch_size, heads, length, dim = output_shapes[0]
    flops = batch_size * heads * length * dim * kernel_size
    return flops


def qk_2d_rpb_flop(inputs: List[Any], outputs: List[Any]) -> Number:
    """
    Counts flops for the 2D QK operation.
    """
    assert (
        len(inputs) >= 2
    ), f"Expected at least 2 inputs (query, key), got {len(inputs)}"
    assert len(outputs) == 1, f"Expected 1 output (attn), got {len(outputs)}"
    input_shapes = [get_shape(v) for v in inputs]
    output_shapes = [get_shape(v) for v in outputs]
    assert (
        len(input_shapes[0]) == 5
    ), f"Query must be a 5-dim tensor, got {len(input_shapes[0])}"
    assert (
        len(input_shapes[1]) == 5
    ), f"Key must be a 5-dim tensor, got {len(input_shapes[1])}"
    assert (
        len(output_shapes[0]) == 5
    ), f"Output must be a 5-dim tensor, got {len(output_shapes[0])}"
    assert (
        input_shapes[0] == input_shapes[1]
    ), f"Query and Key shapes did not match! Q: {input_shapes[0]}, K: {input_shapes[1]}"
    batch_size, heads, height, width, dim = input_shapes[0]
    batch_size, heads, height, width, kernel_size_sq = output_shapes[0]

    flops = batch_size * heads * height * width * dim * kernel_size_sq
    flops += batch_size * heads * height * width * kernel_size_sq
    return flops


def av_2d_flop(inputs: List[Any], outputs: List[Any]) -> Number:
    """
    Counts flops for the 2D AV operation.
    """
    assert len(inputs) == 2, f"Expected 2 inputs (attn and value), got {len(inputs)}"
    assert len(outputs) == 1, f"Expected 1 output (out), got {len(outputs)}"
    input_shapes = [get_shape(v) for v in inputs]
    output_shapes = [get_shape(v) for v in outputs]
    assert (
        len(input_shapes[0]) == 5
    ), f"Attn must be a 5-dim tensor, got {len(input_shapes[0])}"
    assert (
        len(input_shapes[1]) == 5
    ), f"Value must be a 5-dim tensor, got {len(input_shapes[1])}"
    assert (
        len(output_shapes[0]) == 5
    ), f"Output must be a 5-dim tensor, got {len(output_shapes[0])}"
    assert output_shapes[0] == input_shapes[1], (
        f"Out and Value shapes did not match! O: {output_shapes[0]}, V:"
        f" {input_shapes[1]}"
    )
    batch_size, heads, height, width, kernel_size_sq = input_shapes[0]
    batch_size, heads, height, width, dim = output_shapes[0]
    flops = batch_size * heads * height * width * dim * kernel_size_sq
    return flops


def qk_3d_rpb_flop(inputs: List[Any], outputs: List[Any]) -> Number:
    """
    Counts flops for the 3D QK operation.
    """
    assert (
        len(inputs) >= 2
    ), f"Expected at least 2 inputs (query, key), got {len(inputs)}"
    assert len(outputs) == 1, f"Expected 1 output (attn), got {len(outputs)}"
    input_shapes = [get_shape(v) for v in inputs]
    output_shapes = [get_shape(v) for v in outputs]
    assert (
        len(input_shapes[0]) == 6
    ), f"Query must be a 6-dim tensor, got {len(input_shapes[0])}"
    assert (
        len(input_shapes[1]) == 6
    ), f"Key must be a 6-dim tensor, got {len(input_shapes[1])}"
    assert (
        len(output_shapes[0]) == 6
    ), f"Output must be a 6-dim tensor, got {len(output_shapes[0])}"
    assert (
        input_shapes[0] == input_shapes[1]
    ), f"Query and Key shapes did not match! Q: {input_shapes[0]}, K: {input_shapes[1]}"
    batch_size, heads, depth, height, width, dim = input_shapes[0]
    batch_size, heads, depth, height, width, kernel_size_cu = output_shapes[0]

    flops = batch_size * heads * depth * height * width * dim * kernel_size_cu
    flops += batch_size * heads * depth * height * width * kernel_size_cu
    return flops


def av_3d_flop(inputs: List[Any], outputs: List[Any]) -> Number:
    """
    Counts flops for the 3D AV operation.
    """
    assert len(inputs) == 2, f"Expected 2 inputs (attn and value), got {len(inputs)}"
    assert len(outputs) == 1, f"Expected 1 output (out), got {len(outputs)}"
    input_shapes = [get_shape(v) for v in inputs]
    output_shapes = [get_shape(v) for v in outputs]
    assert (
        len(input_shapes[0]) == 6
    ), f"Attn must be a 6-dim tensor, got {len(input_shapes[0])}"
    assert (
        len(input_shapes[1]) == 6
    ), f"Value must be a 6-dim tensor, got {len(input_shapes[1])}"
    assert (
        len(output_shapes[0]) == 6
    ), f"Output must be a 6-dim tensor, got {len(output_shapes[0])}"
    assert output_shapes[0] == input_shapes[1], (
        f"Out and Value shapes did not match! O: {output_shapes[0]}, V:"
        f" {input_shapes[1]}"
    )
    batch_size, heads, depth, height, width, kernel_size_cu = input_shapes[0]
    batch_size, heads, depth, height, width, dim = output_shapes[0]
    flops = batch_size * heads * depth * height * width * dim * kernel_size_cu
    return flops


def add_natten_handle(flop_ctr):
    return flop_ctr.set_op_handle(
        **{
            "prim::PythonOp.NATTEN1DQKRPBFunction": qk_1d_rpb_flop,
            "prim::PythonOp.NATTEN1DAVFunction": av_1d_flop,
            "prim::PythonOp.NATTEN2DQKRPBFunction": qk_2d_rpb_flop,
            "prim::PythonOp.NATTEN2DAVFunction": av_2d_flop,
            "prim::PythonOp.NATTEN3DQKRPBFunction": qk_3d_rpb_flop,
            "prim::PythonOp.NATTEN3DAVFunction": av_3d_flop,
            "prim::PythonOp.NeighborhoodAttention1DQKAutogradFunction": qk_1d_rpb_flop,
            "prim::PythonOp.NeighborhoodAttention1DAVAutogradFunction": av_1d_flop,
            "prim::PythonOp.NeighborhoodAttention2DQKAutogradFunction": qk_2d_rpb_flop,
            "prim::PythonOp.NeighborhoodAttention2DAVAutogradFunction": av_2d_flop,
            "prim::PythonOp.NeighborhoodAttention3DQKAutogradFunction": qk_3d_rpb_flop,
            "prim::PythonOp.NeighborhoodAttention3DAVAutogradFunction": av_3d_flop,
            # Fused ops
            "prim::PythonOp.FusedNeighborhoodAttention1D": fna_generic_flops,
            "prim::PythonOp.FusedNeighborhoodAttention2D": fna_generic_flops,
            "prim::PythonOp.FusedNeighborhoodAttention3D": fna_generic_flops,
        }
    )


def get_flops(model, input, disable_warnings=False):
    flop_ctr = FlopCountAnalysis(model, input)
    flop_ctr = add_natten_handle(flop_ctr)
    if disable_warnings:
        flop_ctr = flop_ctr.unsupported_ops_warnings(False)
    return flop_ctr.total()
