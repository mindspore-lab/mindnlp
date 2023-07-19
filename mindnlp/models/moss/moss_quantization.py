# Copyright 2023 Huawei Technologies Co., Ltd
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
# pylint: disable=E1121
# pylint: disable=R0913
# pylint: disable=R0914
# pylint: disable=C0321
# pylint: disable=W0401
# pylint: disable=C0103

"""
moss quantization
"""

import math
import numpy as np
import triton.language as tl
import mindspore
from mindspore import nn, ops, Tensor
from .moss_custom_autotune import *


def find_layers(module, layers=None, name=''):
    """
    Recursively finds layers of specific types within a module.

    Args:
        - module: The module to search for layers.
        - layers: List of layer types to search for. Defaults to [nn.Conv2d, nn.Dense].
        - name: Name of the current module. Used for hierarchical naming of layers. Defaults to ''.

    Returns:
        - res: Dictionary of found layers with their corresponding names.

    """
    if layers is None:
        layers = [nn.Conv2d, nn.Dense]
    if type(module) in layers:
        return {name: module}
    res = {}
    for name1, child in module.named_children():
        res.update(find_layers(
            child, layers=layers, name=name + '.' + name1 if name != '' else name1
        ))
    return res


# code based https://github.com/fpgaminer/GPTQ-triton
@autotune(
    configs=[
        triton.Config({'BLOCK_SIZE_M': 256, 'BLOCK_SIZE_N': 64, 'BLOCK_SIZE_K': 32, 'GROUP_SIZE_M': 8},
                      num_stages=4, num_warps=4),
        triton.Config({'BLOCK_SIZE_M': 64, 'BLOCK_SIZE_N': 256, 'BLOCK_SIZE_K': 32, 'GROUP_SIZE_M': 8},
                      num_stages=4, num_warps=4),
        triton.Config({'BLOCK_SIZE_M': 128, 'BLOCK_SIZE_N': 128, 'BLOCK_SIZE_K': 32, 'GROUP_SIZE_M': 8},
                      num_stages=4, num_warps=4),
        triton.Config({'BLOCK_SIZE_M': 128, 'BLOCK_SIZE_N': 64, 'BLOCK_SIZE_K': 32, 'GROUP_SIZE_M': 8},
                      num_stages=4, num_warps=4),
        triton.Config({'BLOCK_SIZE_M': 64, 'BLOCK_SIZE_N': 128, 'BLOCK_SIZE_K': 32, 'GROUP_SIZE_M': 8},
                      num_stages=4, num_warps=4),
        triton.Config({'BLOCK_SIZE_M': 128, 'BLOCK_SIZE_N': 32, 'BLOCK_SIZE_K': 32, 'GROUP_SIZE_M': 8},
                      num_stages=4, num_warps=4),
        # These provided a benefit on a 3090
        triton.Config({'BLOCK_SIZE_M': 64, 'BLOCK_SIZE_N': 64, 'BLOCK_SIZE_K': 32, 'GROUP_SIZE_M': 8}, num_stages=4,
                      num_warps=4),
        triton.Config({'BLOCK_SIZE_M': 32, 'BLOCK_SIZE_N': 64, 'BLOCK_SIZE_K': 32, 'GROUP_SIZE_M': 8}, num_stages=4,
                      num_warps=4),
        triton.Config({'BLOCK_SIZE_M': 64, 'BLOCK_SIZE_N': 32, 'BLOCK_SIZE_K': 32, 'GROUP_SIZE_M': 8}, num_stages=4,
                      num_warps=4),
        triton.Config({'BLOCK_SIZE_M': 32, 'BLOCK_SIZE_N': 64, 'BLOCK_SIZE_K': 64, 'GROUP_SIZE_M': 8}, num_stages=4,
                      num_warps=4),
        triton.Config({'BLOCK_SIZE_M': 64, 'BLOCK_SIZE_N': 64, 'BLOCK_SIZE_K': 64, 'GROUP_SIZE_M': 8}, num_stages=4,
                      num_warps=4),
        triton.Config({'BLOCK_SIZE_M': 64, 'BLOCK_SIZE_N': 32, 'BLOCK_SIZE_K': 64, 'GROUP_SIZE_M': 8}, num_stages=4,
                      num_warps=4),
        triton.Config({'BLOCK_SIZE_M': 32, 'BLOCK_SIZE_N': 64, 'BLOCK_SIZE_K': 128, 'GROUP_SIZE_M': 8},
                      num_stages=4, num_warps=4),
    ],
    key=['M', 'N'],
    nearest_power_of_two=True,
)
@triton.jit
def matmul_248_kernel(a_ptr, b_ptr, c_ptr,
                      scales_ptr, zeros_ptr, g_ptr,
                      M, N, K, bits, maxq,
                      stride_am, stride_ak,
                      stride_bk, stride_bn,
                      stride_cm, stride_cn,
                      stride_scales, stride_zeros,
                      BLOCK_SIZE_M: tl.constexpr, BLOCK_SIZE_N: tl.constexpr, BLOCK_SIZE_K: tl.constexpr,
                      GROUP_SIZE_M: tl.constexpr):
    """
    Compute the matrix multiplication C = A x B.
    A is of shape (M, K) float16
    B is of shape (K//8, N) int32
    C is of shape (M, N) float16
    scales is of shape (G, N) float16
    zeros is of shape (G, N) float16
    g_ptr is of shape (K) int32
    """
    infearure_per_bits = 32 // bits

    pid = tl.program_id(axis=0)
    num_pid_m = tl.cdiv(M, BLOCK_SIZE_M)
    num_pid_n = tl.cdiv(N, BLOCK_SIZE_N)
    num_pid_k = tl.cdiv(K, BLOCK_SIZE_K)
    num_pid_in_group = GROUP_SIZE_M * num_pid_n
    group_id = pid // num_pid_in_group
    first_pid_m = group_id * GROUP_SIZE_M
    GROUP_SIZE_M = min(num_pid_m - first_pid_m, GROUP_SIZE_M)
    pid_m = first_pid_m + (pid % GROUP_SIZE_M)
    pid_n = (pid % num_pid_in_group) // GROUP_SIZE_M

    offs_am = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    offs_bn = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    offs_k = tl.arange(0, BLOCK_SIZE_K)
    # (BLOCK_SIZE_M, BLOCK_SIZE_K)
    a_ptrs = a_ptr + (offs_am[:, None] * stride_am +
                      offs_k[None, :] * stride_ak)
    a_mask = (offs_am[:, None] < M)
    # b_ptrs is set up such that it repeats elements along the K axis 8 times
    b_ptrs = b_ptr + ((offs_k[:, None] // infearure_per_bits) * stride_bk +
                      offs_bn[None, :] * stride_bn)  # (BLOCK_SIZE_K, BLOCK_SIZE_N)
    g_ptrs = g_ptr + offs_k
    # shifter is used to extract the N bits of each element in the 32-bit word from B
    scales_ptrs = scales_ptr + offs_bn[None, :]
    zeros_ptrs = zeros_ptr + (offs_bn[None, :] // infearure_per_bits)

    shifter = (offs_k % infearure_per_bits) * bits
    zeros_shifter = (offs_bn % infearure_per_bits) * bits
    accumulator = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)

    for _ in range(0, num_pid_k):
        g_idx = tl.load(g_ptrs)

        # Fetch scales and zeros; these are per-outfeature and thus reused in the inner loop
        # (BLOCK_SIZE_K, BLOCK_SIZE_N,)
        scales = tl.load(scales_ptrs + g_idx[:, None] * stride_scales)
        # (BLOCK_SIZE_K, BLOCK_SIZE_N,)
        zeros = tl.load(zeros_ptrs + g_idx[:, None] * stride_zeros)

        zeros = (zeros >> zeros_shifter[None, :]) & maxq
        zeros = zeros + 1

        # (BLOCK_SIZE_M, BLOCK_SIZE_K)
        load_1 = tl.load(a_ptrs, mask=a_mask, other=0.)
        load_2 = tl.load(b_ptrs)  # (BLOCK_SIZE_K, BLOCK_SIZE_N), but repeated

        # Now we need to unpack load_2 (which is N-bit values) into 32-bit values
        load_2 = (load_2 >> shifter[:, None]) & maxq  # Extract the N-bit values
        load_2 = (load_2 - zeros) * scales  # Scale and shift

        accumulator += tl.dot(load_1, load_2)
        a_ptrs += BLOCK_SIZE_K
        b_ptrs += (BLOCK_SIZE_K // infearure_per_bits) * stride_bk
        g_ptrs += BLOCK_SIZE_K

    accumulator.to(tl.float16)
    c_ptrs = c_ptr + stride_cm * \
             offs_am[:, None] + stride_cn * offs_bn[None, :]
    c_mask = (offs_am[:, None] < M) & (offs_bn[None, :] < N)
    tl.store(c_ptrs, accumulator, mask=c_mask)


# code based https://github.com/fpgaminer/GPTQ-triton
@autotune(
    configs=[
        triton.Config({'BLOCK_SIZE_M': 256, 'BLOCK_SIZE_K': 64, 'BLOCK_SIZE_N': 32, 'GROUP_SIZE_M': 8},
                      num_stages=4, num_warps=4),
        triton.Config({'BLOCK_SIZE_M': 64, 'BLOCK_SIZE_K': 256, 'BLOCK_SIZE_N': 32, 'GROUP_SIZE_M': 8},
                      num_stages=4, num_warps=4),
        triton.Config({'BLOCK_SIZE_M': 128, 'BLOCK_SIZE_K': 128, 'BLOCK_SIZE_N': 32, 'GROUP_SIZE_M': 8},
                      num_stages=4, num_warps=4),
        triton.Config({'BLOCK_SIZE_M': 128, 'BLOCK_SIZE_K': 64, 'BLOCK_SIZE_N': 32, 'GROUP_SIZE_M': 8},
                      num_stages=4, num_warps=4),
        triton.Config({'BLOCK_SIZE_M': 64, 'BLOCK_SIZE_K': 128, 'BLOCK_SIZE_N': 32, 'GROUP_SIZE_M': 8},
                      num_stages=4, num_warps=4),
        triton.Config({'BLOCK_SIZE_M': 128, 'BLOCK_SIZE_K': 32, 'BLOCK_SIZE_N': 32, 'GROUP_SIZE_M': 8},
                      num_stages=4, num_warps=4),
        # These provided a benefit on a 3090
        triton.Config({'BLOCK_SIZE_M': 64, 'BLOCK_SIZE_K': 64, 'BLOCK_SIZE_N': 32, 'GROUP_SIZE_M': 8}, num_stages=4,
                      num_warps=4),
        triton.Config({'BLOCK_SIZE_M': 32, 'BLOCK_SIZE_K': 64, 'BLOCK_SIZE_N': 32, 'GROUP_SIZE_M': 8}, num_stages=4,
                      num_warps=4),
        triton.Config({'BLOCK_SIZE_M': 64, 'BLOCK_SIZE_K': 32, 'BLOCK_SIZE_N': 32, 'GROUP_SIZE_M': 8}, num_stages=4,
                      num_warps=4),
        triton.Config({'BLOCK_SIZE_M': 32, 'BLOCK_SIZE_K': 64, 'BLOCK_SIZE_N': 64, 'GROUP_SIZE_M': 8}, num_stages=4,
                      num_warps=4),
        triton.Config({'BLOCK_SIZE_M': 64, 'BLOCK_SIZE_K': 64, 'BLOCK_SIZE_N': 64, 'GROUP_SIZE_M': 8}, num_stages=4,
                      num_warps=4),
        triton.Config({'BLOCK_SIZE_M': 64, 'BLOCK_SIZE_K': 32, 'BLOCK_SIZE_N': 64, 'GROUP_SIZE_M': 8}, num_stages=4,
                      num_warps=4),
        triton.Config({'BLOCK_SIZE_M': 32, 'BLOCK_SIZE_K': 64, 'BLOCK_SIZE_N': 128, 'GROUP_SIZE_M': 8},
                      num_stages=4, num_warps=4),
    ],
    key=['M', 'K'],
    nearest_power_of_two=True,
)
@triton.jit
def trans_matmul_248_kernel(a_ptr, b_ptr, c_ptr,
                            scales_ptr, zeros_ptr, g_ptr,
                            M, N, K, bits, maxq,
                            stride_am, stride_ak,
                            stride_bk, stride_bn,
                            stride_cm, stride_cn,
                            stride_scales, stride_zeros,
                            BLOCK_SIZE_M: tl.constexpr, BLOCK_SIZE_N: tl.constexpr, BLOCK_SIZE_K: tl.constexpr,
                            GROUP_SIZE_M: tl.constexpr):
    """
    Compute the matrix multiplication C = A x B.
    A is of shape (M, N) float16
    B is of shape (K//8, N) int32
    C is of shape (M, K) float16
    scales is of shape (G, N) float16
    zeros is of shape (G, N) float16
    g_ptr is of shape (K) int32
    """
    infearure_per_bits = 32 // bits

    pid = tl.program_id(axis=0)
    num_pid_m = tl.cdiv(M, BLOCK_SIZE_M)
    num_pid_k = tl.cdiv(K, BLOCK_SIZE_K)
    num_pid_n = tl.cdiv(N, BLOCK_SIZE_N)
    num_pid_in_group = GROUP_SIZE_M * num_pid_k
    group_id = pid // num_pid_in_group
    first_pid_m = group_id * GROUP_SIZE_M
    GROUP_SIZE_M = min(num_pid_m - first_pid_m, GROUP_SIZE_M)
    pid_m = first_pid_m + (pid % GROUP_SIZE_M)
    pid_k = (pid % num_pid_in_group) // GROUP_SIZE_M

    offs_am = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    offs_bk = pid_k * BLOCK_SIZE_K + tl.arange(0, BLOCK_SIZE_K)
    offs_n = tl.arange(0, BLOCK_SIZE_N)
    # (BLOCK_SIZE_M, BLOCK_SIZE_N)
    a_ptrs = a_ptr + (offs_am[:, None] * stride_am +
                      offs_n[None, :] * stride_ak)
    a_mask = (offs_am[:, None] < M)
    # b_ptrs is set up such that it repeats elements along the K axis 8 times
    b_ptrs = b_ptr + ((offs_bk[:, None] // infearure_per_bits) * stride_bk + offs_n[None,
                                                                             :] * stride_bn)  # (BLOCK_SIZE_K, BLOCK_SIZE_N)
    g_ptrs = g_ptr + offs_bk
    g_idx = tl.load(g_ptrs)

    # shifter is used to extract the N bits of each element in the 32-bit word from B
    scales_ptrs = scales_ptr + offs_n[None, :] + g_idx[:, None] * stride_scales
    zeros_ptrs = zeros_ptr + \
                 (offs_n[None, :] // infearure_per_bits) + g_idx[:, None] * stride_zeros

    shifter = (offs_bk % infearure_per_bits) * bits
    zeros_shifter = (offs_n % infearure_per_bits) * bits
    accumulator = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_K), dtype=tl.float32)

    for _ in range(0, num_pid_n):
        # Fetch scales and zeros; these are per-outfeature and thus reused in the inner loop
        scales = tl.load(scales_ptrs)  # (BLOCK_SIZE_K, BLOCK_SIZE_N,)
        zeros = tl.load(zeros_ptrs)  # (BLOCK_SIZE_K, BLOCK_SIZE_N,)

        zeros = (zeros >> zeros_shifter[None, :]) & maxq
        zeros = zeros + 1

        # (BLOCK_SIZE_M, BLOCK_SIZE_N)
        load_1 = tl.load(a_ptrs, mask=a_mask, other=0.)
        load_2 = tl.load(b_ptrs)  # (BLOCK_SIZE_K, BLOCK_SIZE_N), but repeated

        # Now we need to unpack load_2 (which is N-bit values) into 32-bit values
        load_2 = (load_2 >> shifter[:, None]) & maxq  # Extract the N-bit values
        load_2 = (load_2 - zeros) * scales  # Scale and shift
        load_2 = tl.trans(load_2)

        accumulator += tl.dot(load_1, load_2)
        a_ptrs += BLOCK_SIZE_N
        b_ptrs += BLOCK_SIZE_N
        scales_ptrs += BLOCK_SIZE_N
        zeros_ptrs += (BLOCK_SIZE_N // infearure_per_bits)

    accumulator.to(tl.float16)
    c_ptrs = c_ptr + stride_cm * offs_am[:, None] + stride_cn * offs_bk[None, :]
    c_mask = (offs_am[:, None] < M) & (offs_bk[None, :] < K)
    tl.store(c_ptrs, accumulator, mask=c_mask)


def matmul248(input_tensor, qweight, scales, qzeros, g_idx, bits, maxq):
    """
    Perform 248-bit matrix multiplication operation with quantization.
    """
    output = mindspore.numpy.empty(
        (input_tensor.shape[0], qweight.shape[1]), dtype=mindspore.float16)

    def grid(meta): return (
        triton.cdiv(input_tensor.shape[0], meta['BLOCK_SIZE_M']) * triton.cdiv(qweight.shape[1], meta['BLOCK_SIZE_N']),)

    matmul_248_kernel[grid](input_tensor, qweight, output,
                            scales, qzeros, g_idx,
                            input_tensor.shape[0], qweight.shape[1], input_tensor.shape[1], bits, maxq,
                            input_tensor.stride(0), input_tensor.stride(1),
                            qweight.stride(0), qweight.stride(1),
                            output.stride(0), output.stride(1),
                            scales.stride(0), qzeros.stride(0))
    return output


def transpose_matmul248(input_tensor, qweight, scales, qzeros, g_idx, bits, maxq):
    """
    Calculate the output dimension based on the shape of 'qweight' and the number of bits.
    """
    output_dim = (qweight.shape[0] * 32) // bits
    output = mindspore.numpy.empty((input_tensor.shape[0], output_dim), dtype=mindspore.float16)

    def grid(meta):
        return (
            triton.cdiv(input_tensor.shape[0], meta['BLOCK_SIZE_M']) * triton.cdiv(output_dim, meta['BLOCK_SIZE_K']),)

    trans_matmul_248_kernel[grid](input_tensor, qweight, output,
                                  scales, qzeros, g_idx,
                                  input_tensor.shape[0], qweight.shape[1], output_dim, bits, maxq,
                                  input_tensor.stride(0), input_tensor.stride(1),
                                  qweight.stride(0), qweight.stride(1),
                                  output.stride(0), output.stride(1),
                                  scales.stride(0), qzeros.stride(0))
    return output


class QuantLinearFunction(nn.Cell):
    # @staticmethod
    # @custom_fwd(cast_inputs=mindspore.float16)
    """
    Quantized linear function implementation as a subclass of `nn.Cell`.
    """

    def __init__(self, keep_prob):
        super().__init__()
        self.keep_prob = keep_prob
        self.dropout2d = ops.Dropout2D(keep_prob)

    def construct(self, ctx, input_tensor, qweight, scales, qzeros, g_idx, bits, maxq):
        output = matmul248(input_tensor, qweight, scales, qzeros, g_idx, bits, maxq)
        ctx.save_for_backward(qweight, scales, qzeros, g_idx)
        ctx.bits, ctx.maxq = bits, maxq
        return output

    def backward(self, ctx, grad_output):
        """
        Compute the backward pass for the custom autograd function.
        """
        qweight, scales, qzeros, g_idx = ctx.saved_tensors
        bits, maxq = ctx.bits, ctx.maxq
        grad_input = None

        if ctx.needs_input_grad[0]:
            grad_input = transpose_matmul248(
                grad_output, qweight, scales, qzeros, g_idx, bits, maxq)
        return grad_input, None, None, None, None, None, None


class QuantLinear(nn.Cell):
    """
    Quantized linear layer implementation as a subclass of `nn.Cell`.
    """

    def __init__(self, bits, groupsize, infeatures, outfeatures, bias):
        super().__init__()
        if bits not in [2, 4, 8]:
            raise NotImplementedError("Only 2,4,8 bits are supported.")
        self.infeatures = infeatures
        self.outfeatures = outfeatures
        self.bits = bits
        self.maxq = 2 ** self.bits - 1
        self.groupsize = groupsize if groupsize != -1 else infeatures

        self.register_buffer('qweight', ops.zeros(
            (infeatures // 32 * self.bits, outfeatures), dtype=mindspore.int32))
        self.register_buffer('qzeros',
                             ops.zeros((math.ceil(infeatures / self.groupsize), outfeatures // 32 * self.bits),
                                       dtype=mindspore.int32))
        self.register_buffer('scales',
                             ops.zeros((math.ceil(infeatures / self.groupsize), outfeatures), dtype=mindspore.float16))
        self.register_buffer('g_idx',
                             Tensor([i // self.groupsize for i in range(infeatures)], dtype=mindspore.int32))
        if bias:
            self.register_buffer('bias', ops.zeros(
                outfeatures, dtype=mindspore.float16))
        else:
            self.bias = None
        self.g_idx = None
        self.scales = None
        self.qweight = None
        self.qzeros = None

    def pack(self, linear, scales, zeros, g_idx=None):
        """
        Pack the given 'linear', 'scales', and 'zeros' data into the class.
        """
        self.g_idx = g_idx.clone() if g_idx is not None else self.g_idx

        scales = scales.t().contiguous()
        zeros = zeros.t().contiguous()
        scale_zeros = zeros * scales
        self.scales = scales.clone().half()
        if linear.bias is not None:
            self.bias = linear.bias.clone().half()

        intweight = []
        for idx in range(self.infeatures):
            intweight.append(ops.round(
                (linear.weight.data[:, idx] + scale_zeros[self.g_idx[idx]]) / self.scales[self.g_idx[idx]]).to(
                mindspore.int32)[:, None])
        intweight = ops.cat(intweight, axis=1)
        intweight = intweight.t().contiguous()
        intweight = intweight.numpy().astype(np.uint32)
        qweight = np.zeros(
            (intweight.shape[0] // 32 * self.bits, intweight.shape[1]), dtype=np.uint32)
        i = 0
        row = 0
        while row < qweight.shape[0]:
            if self.bits in [2, 4, 8]:
                for j in range(i, i + (32 // self.bits)):
                    qweight[row] |= intweight[j] << (self.bits * (j - i))
                i += 32 // self.bits
                row += 1
            else:
                raise NotImplementedError("Only 2,4,8 bits are supported.")

        qweight = qweight.astype(np.int32)
        self.qweight = Tensor.from_numpy(qweight)

        zeros -= 1
        zeros = zeros.numpy().astype(np.uint32)
        qzeros = np.zeros(
            (zeros.shape[0], zeros.shape[1] // 32 * self.bits), dtype=np.uint32)
        i = 0
        col = 0
        while col < qzeros.shape[1]:
            if self.bits in [2, 4, 8]:
                for j in range(i, i + (32 // self.bits)):
                    qzeros[:, col] |= zeros[:, j] << (self.bits * (j - i))
                i += 32 // self.bits
                col += 1
            else:
                raise NotImplementedError("Only 2,4,8 bits are supported.")

        qzeros = qzeros.astype(np.int32)
        self.qzeros = Tensor.from_numpy(qzeros)

    def construct(self, x):
        out_shape = x.shape[:-1] + (self.outfeatures,)
        out = QuantLinearFunction.apply(x.reshape(-1, x.shape[-1]), self.qweight, self.scales,
                                        self.qzeros, self.g_idx, self.bits, self.maxq)
        out = out + self.bias if self.bias is not None else out
        return out.reshape(out_shape)


def make_quant(module, names, bits, groupsize, name=''):
    """
    Recursively applies quantization to specific layers within a module.

    Args:
        - module: The module to apply quantization.
        - names: Dictionary of layer names to be quantized.
        - bits: Number of bits for weight quantization.
        - groupsize: Group size for quantization.
        - name: Name of the current module. Used for hierarchical naming of layers. Defaults to ''.

    Returns:
        - None

    """
    if isinstance(module, QuantLinear):
        return
    for attr in dir(module):
        tmp = getattr(module, attr)
        name1 = name + '.' + attr if name != '' else attr
        if name1 in names:
            delattr(module, attr)
            setattr(module, attr, QuantLinear(bits, groupsize,
                                              tmp.in_features, tmp.out_features, tmp.bias is not None))
    for name1, child in module.named_children():
        make_quant(child, names, bits, groupsize, name +
                                                  '.' + name1 if name != '' else name1)


def quantize_with_gptq(model, wbits, groupsize):
    """
    Function to quantize a model using GPTQ.

    Args:
        - model: The model to be quantized.
        - wbits: Number of bits for weight quantization, specifying the quantization precision for weights.
        - groupsize: Group size for quantization, specifying the grouping size for weight quantization.

    Returns:
        - model: Quantized model.

    """
    model = model.eval()
    layers = find_layers(model)
    for name in ['lm_head']:
        if name in layers:
            del layers[name]
    make_quant(model, layers, wbits, groupsize)
    return model
