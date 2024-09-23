# Copyright 2024 Huawei Technologies Co., Ltd
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
"""
    _functions
"""
# pylint: disable=E0401, E0611
from dataclasses import dataclass
from functools import reduce  # Required in Python 3
import operator
from typing import Callable, Optional, Tuple
import warnings
import mindspore
from mindspore import ops, Tensor, nn
import bitsandbytes.functional as F
from mindspore._c_expression import (
    Tensor as CTensor,
)  # pylint: disable=no-name-in-module, import-error


def empty(*size, dtype=None):
    if isinstance(size[0], (tuple, list)):
        size = size[0]
    out = CTensor(dtype, size)
    return mindspore.Tensor(out)


# math.prod not compatible with python < 3.8
def prod(iterable):
    return reduce(operator.mul, iterable, 1)


def clone(tensor):
    return tensor.copy()


# The inverse transformation for the colTuring and colAmpere format were contributed by Alex Borzunov:
# https://github.com/bigscience-workshop/petals/blob/main/src/petals/utils/linear8bitlt_patch.py


"""
    This class pools outlier dimensions across layers.
    This is particularly important for small models where outlier features
    are less systematic and occur with low frequency.
"""


class GlobalOutlierPooler:
    _instance = None

    def __init__(self):
        raise RuntimeError("Call get_instance() instead")

    def initialize(self):
        self.outliers = set()
        self.model_dim = None

    @classmethod
    def get_instance(cls):
        if cls._instance is None:
            cls._instance = cls.__new__(cls)
            cls._instance.initialize()
        return cls._instance

    def add_outliers(self, outlier_idx, feature_dim):
        if self.model_dim is None:
            self.model_dim = feature_dim
        if feature_dim != self.model_dim:
            return  # we do not encode outliers for the 2nd FFN layer

        self.outliers.update(outlier_idx.tolist())

    def get_current_outlier_idx(self):
        return mindspore.Tensor(list(self.outliers)).to(mindspore.int64)


def get_inverse_transform_indices(
    transform_tile: Callable[[mindspore.Tensor], mindspore.Tensor],
    tile_size: Tuple[int, int],
):
    """
    Compute a permutation of indices that invert the specified (tiled) matrix transformation

    :param transform_tile: a function that applies forward transform to a tensor of shape [dim1, dim2]
    :param tile_size: higher-level tile dimensions, i.e. (8, 32) for Turing and (32, 32) for Ampere
    :note: we assume that tile_transform applies to a cpu-based int8 tensor of shape tile_size
    :example: transform_tile function for the turing layout (bitsandbytes.functional as F)
    :returns: indices
    """
    d1, d2 = tile_size
    assert 0 < d1 * d2 < 2**64
    tile_indices = ops.arange(d1 * d2, dtype=mindspore.int64).view(d1, d2)
    # encode each position in tile as a tuple of <= 8 unique bytes
    permuted_tile_indices = ops.zeros_like(tile_indices)
    for i in range(8):
        # select i-th byte, apply transformation and trace where each index ended up
        ith_dim_indices = ops.div(tile_indices, 256**i, rounding_mode="trunc") % 256
        sample_tile_i = (ith_dim_indices - 128).to(mindspore.int8).contiguous()
        assert ops.all(sample_tile_i.int() + 128 == ith_dim_indices), "int overflow"
        permuted_tile_i = transform_tile(sample_tile_i)
        ith_permuted_indices = permuted_tile_i.to(tile_indices.dtype) + 128
        permuted_tile_indices += ith_permuted_indices * (256**i)
        if d1 * d2 < 256**i:
            break  # if all indices fit in i bytes, stop early
    return permuted_tile_indices


def undo_layout(
    permuted_tensor: mindspore.Tensor, tile_indices: mindspore.Tensor
) -> mindspore.Tensor:
    """
    Undo a tiled permutation such as turing or ampere layout

    :param permuted_tensor: mindspore tensor in a permuted layout
    :param tile_indices: reverse transformation indices, from get_inverse_transform_indices
    :return: contiguous row-major tensor
    """
    (rows, cols), (tile_rows, tile_cols) = permuted_tensor.shape, tile_indices.shape
    assert (
        rows % tile_rows == cols % tile_cols == 0
    ), "tensor must contain a whole number of tiles"
    tensor = permuted_tensor.reshape(-1, tile_indices.numel()).t()
    outputs = Tensor(
        shape=tensor.shape, dtype=tensor.dtype
    )  # note: not using .index_copy because it was slower on cuda
    outputs[tile_indices.flatten()] = tensor
    outputs = outputs.reshape(
        tile_rows, tile_cols, cols // tile_cols, rows // tile_rows
    )
    outputs = outputs.permute(
        3, 0, 2, 1
    )  # (rows // tile_rows, tile_rows), (cols // tile_cols, tile_cols)
    return outputs.reshape(rows, cols).contiguous()


class MatMul8bit:
    @staticmethod
    def construct(ctx, A, B, out=None, quant_type="vector", precision=None):
        if precision is None:
            precision = [8, 8, 8]
        if precision[0] != 8:
            output = ops.matmul(A, B)
        else:
            if len(B.shape) == 2:
                dim = 0
            else:
                dim = 1
            qA, SA = F.vectorwise_quant(A, dim=-1, quant_type=quant_type)
            qB, SB = F.vectorwise_quant(B, dim=dim, quant_type=quant_type)
            iout = F.igemm(qA, qB)
            output = F.vectorwise_mm_dequant(iout, SA, SB, A.dtype, quant_type)

        if A.requires_grad or B.requires_grad:
            ctx.save_for_backward(A, B)

        ctx.quant_type = quant_type
        ctx.precision = precision

        return output


def supports_igemmlt() -> bool:
    """检查当前设备是否支持优化的 int8 内核"""
    device_name = F.GPU_NAME
    if device_name not in F.gpus_compute_capability_over_7_5:
        return False
    else:
        nvidia16_models = (
            "NVIDIA GeForce GTX 1630",
            "NVIDIA GeForce GTX 1650",
            "NVIDIA GeForce GTX 1660",
        )  # https://en.wikipedia.org/wiki/GeForce_16_series
        if any(model_name in device_name for model_name in nvidia16_models):
            return False  # 这些设备在技术上是 cuda 7.5 兼容的，但缺少张量核心

    return True


def _get_tile_size(format):
    assert format in (
        "col_turing",
        "col_ampere",
    ), f"please find this assert and manually enter tile size for {format}"
    return (8, 32) if format == "col_turing" else (32, 32)


def get_tile_inds(format, device):
    def transform(x):
        return F.transform(x, from_order="row", to_order=format)[0].to(x.device)

    return get_inverse_transform_indices(transform, _get_tile_size(format))


@dataclass
class MatmulLtState:

    _tile_indices: Optional[mindspore.Tensor] = None
    force_no_igemmlt: bool = False
    CB = None
    CxB = None
    SB = None
    SCB = None

    CxBt = None
    SBt = None
    CBt = None

    subB = None

    outlier_pool = None
    has_accumulated_gradients = False
    threshold = 0.0
    idx = None
    is_training = True
    has_fp16_weights = True
    memory_efficient_backward = False
    use_pool = False
    formatB = F.get_special_format_str()

    def reset_grads(
        self,
    ):
        self.CB = None
        self.CxB = None
        self.SB = None
        self.SCB = None

        self.CxBt = None
        self.SBt = None
        self.CBt = None

    @property
    def tile_indices(self):
        if self._tile_indices is None:
            self._tile_indices = get_tile_inds(self.formatB, self.CxB.device)
        return self._tile_indices


class MatMul8bitLt(nn.Cell):
    # forward is the same, but we added the fallback for pre-turing GPUs
    # backward is mostly the same, but adds one extra clause (see "elif state.CxB is not None")
    def __init__(
        self,
    ):
        super().__init__()
        self.needs_input_grad = [False, False, False, False, False]

    def construct(self, A, B, out=None, bias=None, state=MatmulLtState):
        using_igemmlt = supports_igemmlt() and not state.force_no_igemmlt
        # default of pymindspore behavior if inputs are empty
        self.is_empty = False
        if prod(A.shape) == 0:
            self.is_empty = True
            self.A = A
            self.B = B
            self.bias = bias
            if A.shape[-1] == B.shape[0]:
                return empty(A.shape[:-1] + B.shape[1:], dtype=A.dtype)
            else:
                return empty(A.shape[:-1] + B.shape[:1], dtype=A.dtype)

        # 1. Quantize A
        # 2. Quantize B
        # 3. Matmul
        # 4. Mixed-precision decomposition matmul
        # 5. Save state
        formatB = state.formatB
        input_shape = A.shape
        if state.outlier_pool is None:
            state.outlier_pool = GlobalOutlierPooler.get_instance()

        # Cast A to fp16
        if A.dtype != mindspore.float16:
            warnings.warn(
                f"MatMul8bitLt: inputs will be cast from {A.dtype} to float16 during quantization"
            )
        # 1. Quantize A
        if len(A.shape) == 3:
            A = A.reshape(-1, A.shape[-1])
        CA, CAt, SCA, SCAt, coo_tensorA = F.double_quant(
            A.astype(mindspore.float16), threshold=state.threshold
        )

        if state.threshold > 0.0 and coo_tensorA is not None:
            if state.has_fp16_weights:
                _, idx = ops.unique(coo_tensorA.colidx)
                idx.astype(mindspore.int64)
                CA[:, idx] = 0
                CAt[:, idx] = 0
                subA = A[:, idx]
                state.subB = B[:, idx].t()
                state.idx = idx
            else:
                if state.CxB is None and using_igemmlt:
                    # B in in 8-bit row-major, we can transform it back to 16-bit to extract outlier dimensions
                    # we also need to convert it to the turing/ampere format
                    state.CxB, state.SB = F.transform(state.CB, to_order=formatB)
        else:
            if not state.has_fp16_weights and state.CxB is None and using_igemmlt:
                state.CxB, state.SB = F.transform(state.CB, to_order=formatB)
            subA = None
        # 2. Quantize B
        if state.has_fp16_weights:
            has_grad = getattr(B, "grad", None) is not None
            if (state.is_training and not has_grad) or state.CxB is None:
                state.reset_grads()
                (
                    CB,
                    state.CBt,
                    state.SCB,
                    state.SCBt,
                    coo_tensorB,
                ) = F.double_quant(B.to(mindspore.float16))
                if using_igemmlt:
                    state.CxB, state.SB = F.transform(CB, to_order=formatB)
                else:
                    state.CB = CB
        else:
            has_grad = False

        if coo_tensorA is not None and not state.has_fp16_weights:
            # extract outliers

            outlier_idx, _ = ops.unique(coo_tensorA.colidx)
            state.idx = outlier_idx
            # state.outlier_pool.add_outliers(outlier_idx, A.shape[-1])
            # if state.use_pool and state.outlier_pool.model_dim == A.shape[-1]:
            #    # do not use pool for 2nd FFN layer
            #    state.idx = state.outlier_pool.get_current_outlier_idx().to(A.device)
            # else:
            #    state.idx = outlier_idx
            if state.CxB is not None:
                outliers = F.extract_outliers(state.CxB, state.SB, state.idx.int())
            else:
                outliers = state.CB[:, state.idx.long()].clone()

            state.subB = (outliers * state.SCB.view(-1, 1) / 127.0).t().to(A.dtype)
            CA[:, state.idx.long()] = 0
            CAt[:, state.idx.long()] = 0
            subA = A[:, state.idx.long()]

        shapeB = state.SB[0] if state.SB else B.shape

        if len(input_shape) == 3:
            output_shape = (input_shape[0], input_shape[1], shapeB[0])
        else:
            output_shape = (input_shape[0], shapeB[0])
        # 3. Matmul
        if using_igemmlt:
            C32A, SA = F.transform(CA, "col32")
            out32, Sout32 = F.igemmlt(C32A, state.CxB, SA, state.SB)
            if bias is None or bias.dtype == mindspore.float16:
                # we apply the fused bias here
                output = F.mm_dequant(out32, Sout32, SCA, state.SCB, bias=bias)
                output = output.to(A.dtype)
            else:  # apply bias separately
                output = F.mm_dequant(out32, Sout32, SCA, state.SCB, bias=None)
                output = output.to(A.dtype) + bias

        else:
            A_wo_outliers = A.copy()
            if state.idx is not None:
                A_wo_outliers[:, state.idx.long()] = 0
            output = ops.dense(A_wo_outliers, state.CB.to(A.dtype))
            scb = state.SCB.unsqueeze(0)
            scb = scb * (1.0 / 127.0)
            output = output * scb
            if bias is not None:
                output = output + bias
        # 4. Mixed-precision decomposition matmul
        if coo_tensorA is not None and subA is not None:
            output += ops.matmul(subA, state.subB)
        # 5. Save state
        self.state = state

        self.formatB = formatB
        self.grad_shape = input_shape
        self.dtype_A, self.dtype_B, self.dtype_bias = (
            A.dtype,
            B.dtype,
            None if bias is None else bias.dtype,
        )

        if any(self.needs_input_grad[:2]):
            self.tensors = (CAt, subA, A)
            self.tensor_states = (SCAt, state.idx)
        else:
            self.tensors = [None, None, A]
            self.tensor_states = (None, None)

        clone_func = clone if len(output_shape) == 3 else lambda x: x

        return clone_func(output.view(output_shape))


matmul8bitlt = MatMul8bitLt()


def matmul(
    A: mindspore.Tensor,
    B: mindspore.Tensor,
    out: Optional[mindspore.Tensor] = None,
    state: Optional[MatmulLtState] = None,
    threshold=0.0,
    bias=None,
):
    state = state or MatmulLtState()
    if threshold > 0.0:
        state.threshold = threshold
    # return MatMul8bitLt(A, B, out, bias, state)
    return matmul8bitlt(A, B, out, bias, state)
