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
    mindbnb functions
"""
# pylint: disable=E0611, E0401
from functools import reduce  # Required in Python 3
import itertools
import operator
import subprocess
from typing import Any, Dict
import ctypes as ct
import numpy as np
import mindspore
from mindspore import Tensor
from mindspore import ops
from mindspore import context
from mindspore._c_expression import (
    Tensor as CTensor,
)  # pylint: disable=no-name-in-module, import-error

from bitsandbytes.utils import pack_dict_to_tensor, unpack_tensor_to_dict

from bitsandbytes import bnbop

gpus_compute_capability_over_7_5 = [
    # Compute Capability 7.5
    "Tesla T4",
    "Quadro T1000",
    "Quadro T2000",
    "Quadro RTX 3000",
    "Quadro RTX 4000",
    "Quadro RTX 5000",
    "Quadro RTX 6000",
    "Quadro RTX 8000",
    "NVIDIA GeForce GTX 1650",
    "NVIDIA GeForce GTX 1660",
    "NVIDIA GeForce GTX 1660 Ti",
    # Compute Capability 8.0
    "NVIDIA A100-SXM4-40GB",
    "NVIDIA A100-SXM4-80GB",
    "NVIDIA A100-PCIe-40GB",
    "NVIDIA A100-PCIe-80GB",
    "NVIDIA GeForce RTX 3070",
    "NVIDIA GeForce RTX 3080",
    "NVIDIA GeForce RTX 3090",
    "NVIDIA GeForce RTX 3080 Ti",
    "NVIDIA GeForce RTX 3090 Ti",
    "NVIDIA RTX A40",
    "NVIDIA RTX A10",
    # Compute Capability 8.6
    "NVIDIA GeForce RTX 3050",
    "NVIDIA GeForce RTX 3060",
    "NVIDIA GeForce RTX 3060 Ti",
    "NVIDIA GeForce RTX 3070 Ti",
    # Compute Capability 8.7
    "NVIDIA GeForce RTX 4080",
    "NVIDIA GeForce RTX 4090",
    "NVIDIA RTX A4500",
    "NVIDIA RTX A5500",
    "NVIDIA RTX A6000",
    # Compute Capability 9.0
    "NVIDIA H100-SXM5-80GB",
    "NVIDIA H100-PCIe-80GB",
    # Compute Capability 9.1
    "NVIDIA RTX 6000 Ada Generation",
    # Compute Capability 9.2
    "NVIDIA RTX 4000 Ada Generation",
    "NVIDIA RTX 5000 Ada Generation",
]

turing_gpus = [
    # GeForce RTX 20 Series
    "NVIDIA GeForce RTX 2080 Ti",
    "NVIDIA GeForce RTX 2080 Super",
    "NVIDIA GeForce RTX 2080",
    "NVIDIA GeForce RTX 2070 Super",
    "NVIDIA GeForce RTX 2070",
    "NVIDIA GeForce RTX 2060 Super",
    "NVIDIA GeForce RTX 2060",
    # GeForce GTX 16 Series
    "NVIDIA GeForce GTX 1660 Ti",
    "NVIDIA GeForce GTX 1660 Super",
    "NVIDIA GeForce GTX 1660",
    "NVIDIA GeForce GTX 1650 Super",
    "NVIDIA GeForce GTX 1650",
    # Quadro RTX Series
    "Quadro RTX 8000",
    "Quadro RTX 6000",
    "Quadro RTX 5000",
    "Quadro RTX 4000",
    # Titan RTX
    "Titan RTX",
]

ampere_gpus = [
    # GeForce RTX 30 Series
    "NVIDIA GeForce RTX 3090",
    "NVIDIA GeForce RTX 3080 Ti",
    "NVIDIA GeForce RTX 3080",
    "NVIDIA GeForce RTX 3070 Ti",
    "NVIDIA GeForce RTX 3070",
    "NVIDIA GeForce RTX 3060 Ti",
    "NVIDIA GeForce RTX 3060",
    # Quadro RTX Series
    "Quadro RTX A6000",
    "Quadro RTX A5000",
    "Quadro RTX A4000",
]


def empty(*size, dtype):
    if isinstance(size[0], (tuple, list)):
        size = size[0]
    out = CTensor(dtype, size)
    return mindspore.Tensor(out)


def get_gpu_name(device: int):

    try:
        result = subprocess.run(
            ["nvidia-smi", "--query-gpu=name", "--format=csv,noheader"],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            check=True,
        )
        if result.returncode != 0:
            raise RuntimeError(f"nvidia-smi error: {result.stderr}")
        gpu_name = result.stdout.strip()
        return gpu_name.split("\n")[device]
    except FileNotFoundError:
        return (
            "nvidia-smi command not found. Make sure you have NVIDIA drivers installed."
        )


GPU_NAME = get_gpu_name(context.get_context("device_id"))


def get_special_format_str():
    device_target = context.get_context("device_target")
    if device_target == "CPU":
        return "col_turing"

    device_name = GPU_NAME
    if device_name in turing_gpus:
        return "col_turing"
    if device_name in ampere_gpus:
        return "col_ampere"

    return "col_turing"


# math.prod not compatible with python < 3.8
def prod(iterable):
    return reduce(operator.mul, iterable, 1)


name2qmap = {}


class GlobalPageManager:
    _instance = None

    def __init__(self):
        raise RuntimeError("Call get_instance() instead")

    def initialize(self):
        self.paged_tensors = []

    @classmethod
    def get_instance(cls):
        if cls._instance is None:
            cls._instance = cls.__new__(cls)
            cls._instance.initialize()
        return cls._instance


class Cusparse_Context:
    _instance = None

    def __init__(self):
        raise RuntimeError("Call get_instance() instead")

    def initialize(self):
        self.context = ct.c_void_p(bnbop.get_cusparse())

    @classmethod
    def get_instance(cls):
        if cls._instance is None:
            cls._instance = cls.__new__(cls)
            cls._instance.initialize()
        return cls._instance


dtype2bytes = {}
dtype2bytes[mindspore.float32] = 4
dtype2bytes[mindspore.float16] = 2
dtype2bytes[mindspore.bfloat16] = 2
dtype2bytes[mindspore.uint8] = 1
dtype2bytes[mindspore.int8] = 1


def frombuffer(buffer, dtype, count, shape):
    # 使用 numpy.frombuffer 创建一个 NumPy 数组
    np_array = np.frombuffer(buffer, dtype=dtype, count=count)
    # 将 NumPy 数组转换为 MindSpore Tensor
    tensor = Tensor(np_array).reshape(shape)
    return tensor


def get_paged(*shape, dtype=mindspore.float32):
    num_bytes = dtype2bytes[dtype] * prod(shape)
    cuda_ptr = bnbop.cget_managed_ptr(ct.c_size_t(num_bytes))
    c_ptr = ct.cast(cuda_ptr, ct.POINTER(ct.c_int))
    new_array = np.ctypeslib.as_array(c_ptr, shape=shape)
    out = frombuffer(new_array, dtype=dtype, count=prod(shape), shape=shape)
    out.is_paged = True
    return out


def create_linear_map(signed=True, total_bits=8, add_zero=True):
    sign = -1.0 if signed else 0.0
    total_values = 2**total_bits
    if add_zero or total_bits < 8:
        # add a zero
        # since we simulate less bits by having zeros in the data type, we
        # we need to center the quantization around zero and as such lose
        # a single value
        total_values = 2**total_bits if not signed else 2**total_bits - 1

    values = ops.linspace(sign, 1.0, total_values)
    gap = 256 - values.numel()
    if gap == 0:
        return values
    else:
        l = values.numel() // 2  # noqa: E741
        return mindspore.Tensor(values[:l].tolist() + [0] * gap + values[l:].tolist())


def create_normal_map(offset=0.9677083, use_extra_value=True):
    try:
        from scipy.stats import norm
    except ImportError as ie:
        raise ImportError(
            "Scipy is required for `create_normal_map`. Install `bitsandbytes` with the `[test]` extra.",
        ) from ie

    if use_extra_value:
        # one more positive value, this is an asymmetric type
        v1 = norm.ppf(ops.linspace(offset, 0.5, 9)[:-1]).tolist()
        v2 = [0] * (256 - 15)  ## we have 15 non-zero values in this data type
        v3 = (-norm.ppf(ops.linspace(offset, 0.5, 8)[:-1])).tolist()
    else:
        v1 = norm.ppf(ops.linspace(offset, 0.5, 8)[:-1]).tolist()
        v2 = [0] * (256 - 14)  ## we have 14 non-zero values in this data type
        v3 = (-norm.ppf(ops.linspace(offset, 0.5, 8)[:-1])).tolist()

    v = v1 + v2 + v3

    values = mindspore.Tensor(v)
    values = values.sort().values
    values /= values.max()

    assert values.numel() == 256

    return values


def create_fp8_map(signed=True, exponent_bits=5, precision_bits=2, total_bits=8):
    e = exponent_bits
    p = precision_bits
    has_sign = 1 if signed else 0
    assert e + p == total_bits - has_sign
    # the exponent is biased to 2^(e-1) -1 == 0
    evalues = []
    pvalues = []
    for i, val in enumerate(
        range(-(2 ** (exponent_bits - has_sign)), 2 ** (exponent_bits - has_sign), 1)
    ):
        evalues.append(2**val)

    values = []
    lst = list(itertools.product([0, 1], repeat=precision_bits))
    # for ev in evalues:
    bias = 2 ** (exponent_bits - 1)
    for evalue in range(2 ** (exponent_bits)):
        for bit_pattern in lst:
            value = 1 if evalue != 0 else 0
            for i, pval in enumerate(list(bit_pattern)):
                value += pval * (2 ** -(i + 1))
            if evalue == 0:
                # subnormals
                value = value * 2 ** -(bias)
            else:
                # normals
                value = value * 2 ** -(evalue - bias - 1)
            values.append(value)
            if signed:
                values.append(-value)

    assert len(values) == 2**total_bits
    values.sort()
    if total_bits < 8:
        gap = 256 - len(values)
        for i in range(gap):
            values.append(0)
    values.sort()
    code = mindspore.Tensor(values)
    code /= code.max()

    return code


def create_dynamic_map(signed=True, max_exponent_bits=7, total_bits=8):
    """
    Creates the dynamic quantiztion map.

    The dynamic data type is made up of a dynamic exponent and
    fraction. As the exponent increase from 0 to -7 the number
    of bits available for the fraction shrinks.

    This is a generalization of the dynamic type where a certain
    number of the bits and be reserved for the linear quantization
    region (the fraction). n determines the maximum number of
    exponent bits.

    For more details see
    (8-Bit Approximations for Parallelism in Deep Learning)[https://arxiv.org/abs/1511.04561]
    """

    data = []
    # these are additional items that come from the case
    # where all the exponent bits are zero and no
    # indicator bit is present
    non_sign_bits = total_bits - (1 if signed else 1)
    additional_items = 2 ** (non_sign_bits - max_exponent_bits) - 1
    for i in range(max_exponent_bits):
        fraction_items = int(
            (
                2 ** (i + non_sign_bits - max_exponent_bits) + 1
                if signed
                else 2 ** (i + non_sign_bits - max_exponent_bits + 1) + 1
            ),
        )
        boundaries = ops.linspace(0.1, 1, fraction_items)
        means = (boundaries[:-1] + boundaries[1:]) / 2.0
        data += ((10 ** (-(max_exponent_bits - 1) + i)) * means).tolist()
        if signed:
            data += (-(10 ** (-(max_exponent_bits - 1) + i)) * means).tolist()

    if additional_items > 0:
        boundaries = ops.linspace(0.1, 1, additional_items + 1)
        means = (boundaries[:-1] + boundaries[1:]) / 2.0
        data += ((10 ** (-(max_exponent_bits - 1) + i)) * means).tolist()
        if signed:
            data += (-(10 ** (-(max_exponent_bits - 1) + i)) * means).tolist()

    data.append(0)
    data.append(1.0)

    assert len(data) == 2**total_bits

    gap = 256 - len(data)
    for i in range(gap):
        data.append(0)

    data.sort()
    return Tensor(data)


def get_transform_func(dtype, orderA, orderOut, transpose=False):
    name = f'ctransform_{(8 if dtype == mindspore.int8 else 32)}_{orderA}_to_{orderOut}_{"t" if transpose else "n"}'
    if not hasattr(bnbop, name):
        print(name)
        raise ValueError(
            f"Transform function not supported: {orderA} to {orderOut} for data type {dtype} and transpose={transpose}",
        )
    else:
        return getattr(bnbop, name)


def get_transform_buffer(shape, dtype, to_order, from_order="row", transpose=False):
    init_func = ops.zeros
    dims = len(shape)

    rows = shape[0]
    if dims == 3:
        rows = rows * shape[1]
    cols = shape[-1]

    state = (shape, to_order)
    if transpose:
        # swap dims
        rows, cols = cols, rows
        state = (shape[::-1], to_order)

    if to_order in ("row", "col"):
        return (
            init_func(
                shape,
                dtype=dtype,
            ),
            state,
        )
    elif to_order == "col32":
        # blocks of 32 columns (padded)
        cols = 32 * ((cols + 31) // 32)
        return (
            init_func(
                (rows, cols),
                dtype=dtype,
            ),
            state,
        )
    elif to_order == "col_turing":
        # blocks of 32 columns and 8 rows
        cols = 32 * ((cols + 31) // 32)
        rows = 8 * ((rows + 7) // 8)
        return (
            init_func(
                (rows, cols),
                dtype=dtype,
            ),
            state,
        )
    elif to_order == "col_ampere":
        # blocks of 32 columns and 32 rows
        cols = 32 * ((cols + 31) // 32)
        rows = 32 * ((rows + 31) // 32)
        return (
            init_func(
                (rows, cols),
                dtype=dtype,
            ),
            state,
        )
    else:
        raise NotImplementedError(f"To_order not supported: {to_order}")


def nvidia_transform(
    A,
    to_order,
    from_order="row",
    out=None,
    transpose=False,
    state=None,
    ld=None,
):
    if state is None:
        state = (A.shape, from_order)
    else:
        from_order = state[1]
    if out is None:
        out, new_state = get_transform_buffer(state[0], A.dtype, to_order, state[1])
    else:
        new_state = (state[1], to_order)
    func = get_transform_func(A.dtype, from_order, to_order, transpose)

    shape = state[0]
    if len(shape) == 2:
        dim1 = ct.c_int32(shape[0])
        dim2 = ct.c_int32(shape[1])
    elif ld is not None:
        n = prod(shape)
        dim1 = prod([shape[i] for i in ld])
        dim2 = ct.c_int32(n // dim1)
        dim1 = ct.c_int32(dim1)
    else:
        dim1 = ct.c_int32(shape[0] * shape[1])
        dim2 = ct.c_int32(shape[2])

    return out, new_state


class QuantState:
    """container for quantization state components to work with Params4bit and similar classes"""

    valid_quant_types = ("fp4", "nf4")
    valid_qs_type_keys = [f"bitsandbytes__{x}" for x in valid_quant_types]
    valid_qs_keys = [
        "absmax",
        "quant_map",
        "nested_absmax",
        "nested_quant_map",
        "quant_state",
        "quant_type",
        "blocksize",
        "dtype",
        "shape",
        "nested_blocksize",
        "nested_dtype",
        "nested_offset",
    ]

    def __init__(
        self,
        absmax,
        shape=None,
        code=None,
        blocksize=None,
        quant_type=None,
        dtype=None,
        offset=None,
        state2=None,
    ):
        self.absmax = absmax
        self.shape = shape
        self.code = code
        self.dtype = dtype
        self.blocksize = blocksize
        self.quant_type = quant_type
        self.offset = offset
        self.state2 = state2
        self.nested = state2 is not None

    def __get_item__(self, idx):
        """
        ensures compatibility with older quant state scheme with nested lists.
        assumes the following layout:
        state = [qabsmax, input_shape, A.dtype, blocksize, [offset, state2], quant_type]
        state2 = [absmax, input_shape, A.dtype, blocksize, None, quant_type]
        """
        if self.nested:
            list_repr = [
                self.absmax,
                self.shape,
                self.dtype,
                self.blocksize,
                [self.offset, self.state2],
                self.quant_type,
            ]
        else:
            list_repr = [
                self.absmax,
                self.shape,
                self.dtype,
                self.blocksize,
                None,
                self.quant_type,
            ]
        return list_repr[idx]

    @classmethod
    def from_dict(cls, qs_dict: Dict[str, Any]) -> "QuantState":
        """
        unpacks components of state_dict into QuantState
        where necessary, convert into strings, mindspore.dtype, ints, etc.

        qs_dict: based on state_dict, with only relevant keys, striped of prefixes.

        item with key `quant_state.bitsandbytes__[nf4/fp4]` may contain minor and non-tensor quant state items.
        """

        # unpacking tensor with non-tensor components
        qs_key = [
            k
            for k, v in qs_dict.items()
            if "quant_state" in k and isinstance(v, mindspore.Tensor)
        ]
        if len(qs_key) == 0 and "quant_type" not in qs_dict:
            raise ValueError(
                "Expected packed or unpacked quant_state items, found neither"
            )
        elif len(qs_key) != 1 or qs_key[0].split(".")[-1] not in cls.valid_qs_type_keys:
            raise ValueError(
                f"There should be exactly one `quant_state` item with ending from {cls.valid_qs_type_keys}.\nDetected {qs_key}.",
            )

        # unpacking minor and non-tensor quant state items if necessary
        if len(qs_key) == 1:
            first_qs_key = qs_key[0]
            qs_dict.update(unpack_tensor_to_dict(qs_dict.pop(first_qs_key)))

        qs_dict = {k.split(".")[-1]: v for k, v in qs_dict.items()}  # strip prefixes
        assert set(qs_dict.keys()).issubset(cls.valid_qs_keys)

        if "nested_absmax" in qs_dict:
            offset = mindspore.tensor(float(qs_dict["nested_offset"]))
            state2 = cls(
                absmax=qs_dict["nested_absmax"],
                blocksize=qs_dict["nested_blocksize"],
                code=qs_dict["nested_quant_map"],
                dtype=getattr(mindspore, qs_dict["nested_dtype"]),
            )
        else:
            offset, state2 = None, None

        quant_state = cls(
            quant_type=qs_dict["quant_type"],
            absmax=qs_dict["absmax"],
            blocksize=qs_dict["blocksize"],
            code=qs_dict["quant_map"],
            dtype=getattr(mindspore, qs_dict["dtype"]),
            shape=(
                mindspore.Size(qs_dict["shape"])
                if qs_dict["shape"] is not None
                else None
            ),
            offset=offset,
            state2=state2,
        )
        return quant_state

    def as_dict(self, packed=False):
        """
        returns dict of tensors and strings to use in serialization via _save_to_state_dict()
        param: packed -- returns dict[str, mindspore.Tensor] for state_dict fit for safetensors saving
        """
        qs_dict = {
            "quant_type": self.quant_type,
            "absmax": self.absmax,
            "blocksize": self.blocksize,
            "quant_map": self.code,
            "dtype": str(self.dtype).strip("mindspore."),
            "shape": tuple(self.shape),
        }
        if self.nested:
            qs_dict.update(
                {
                    "nested_absmax": self.state2.absmax,
                    "nested_blocksize": self.state2.blocksize,
                    "nested_quant_map": self.state2.code.clone(),  # un-shared to avoid restoring it after shared tensors are removed by safetensors
                    "nested_dtype": str(self.state2.dtype).strip("mindspore."),
                    "nested_offset": self.offset.item(),
                },
            )
        if not packed:
            return qs_dict

        # packed format allows serialization of non-tensor components, critical for saving in safetensors format
        qs_packed_dict = {
            k: v for k, v in qs_dict.items() if isinstance(v, mindspore.Tensor)
        }
        non_tensor_dict = {
            k: v for k, v in qs_dict.items() if not isinstance(v, mindspore.Tensor)
        }
        qs_packed_dict["quant_state." + "bitsandbytes__" + self.quant_type] = (
            pack_dict_to_tensor(non_tensor_dict)
        )
        return qs_packed_dict

    def __eq__(self, other):
        if not isinstance(other, QuantState):
            return False

        return (
            np.allclose(self.absmax, other.absmax, atol=1e-6)
            and self.shape == other.shape
            and np.allclose(self.code, other.code, atol=1e-6)
            and self.dtype == other.dtype
            and self.blocksize == other.blocksize
            and self.quant_type == other.quant_type
            and (
                self.offset == other.offset
                if self.offset is not None and other.offset is not None
                else self.offset is other.offset
            )
            and (
                self.state2 == other.state2
                if self.state2 is not None and other.state2 is not None
                else self.state2 is other.state2
            )
        )


def check_matmul(A, B, out, transposed_A, transposed_B, expected_type=mindspore.int8):
    if context.get_context("device_target") != "GPU":
        context.set_context(device_target="GPU")

    # 检查数据类型
    if A.dtype != expected_type or B.dtype != expected_type:
        raise TypeError(
            f"Expected {expected_type} input tensors A and B, but got {A.dtype} and {B.dtype}"
        )

    sA = A.shape
    sB = B.shape
    tA = transposed_A
    tB = transposed_B

    correct = True

    if len(sA) == 2 and len(sB) == 2:
        if not tA and not tB and A.shape[1] != B.shape[0]:
            correct = False
        elif tA and not tB and A.shape[0] != B.shape[0]:
            correct = False
        elif tA and tB and A.shape[0] != B.shape[1]:
            correct = False
        elif not tA and tB and A.shape[1] != B.shape[1]:
            correct = False
    elif len(sA) == 3 and len(sB) == 2:
        if not tA and not tB and A.shape[2] != B.shape[0]:
            correct = False
        elif tA and not tB and A.shape[1] != B.shape[0]:
            correct = False
        elif tA and tB and A.shape[1] != B.shape[1]:
            correct = False
        elif not tA and tB and A.shape[2] != B.shape[1]:
            correct = False
    elif len(sA) == 3 and len(sB) == 3:
        if not tA and not tB and A.shape[2] != B.shape[1]:
            correct = False
        elif tA and not tB and A.shape[1] != B.shape[1]:
            correct = False
        elif tA and tB and A.shape[1] != B.shape[2]:
            correct = False
        elif not tA and tB and A.shape[2] != B.shape[2]:
            correct = False

    if out is not None:
        sout = out.shape
        # special case common in backprop
        if not correct and len(sA) == 3 and len(sB) == 3:
            if (
                sout[0] == sA[2]
                and sout[1] == sB[2]
                and sA[0] == sB[0]
                and sA[1] == sB[1]
            ):
                correct = True
    else:
        if len(sA) == 2 and len(sB) == 2:
            if not tA and not tB:
                sout = (sA[0], sB[1])
            elif tA and tB:
                sout = (sA[1], sB[0])
            elif tA and not tB:
                sout = (sA[1], sB[1])
            elif not tA and tB:
                sout = (sA[0], sB[0])
        elif len(sA) == 3 and len(sB) == 2:
            if not tA and not tB:
                sout = (sA[0], sA[1], sB[1])
            elif tA and tB:
                sout = (sA[0], sA[2], sB[0])
            elif tA and not tB:
                sout = (sA[0], sA[2], sB[1])
            elif not tA and tB:
                sout = (sA[0], sA[1], sB[0])
        elif len(sA) == 3 and len(sB) == 3:
            if not tA and not tB:
                sout = (sA[0], sA[1], sB[2])
            elif tA and tB:
                sout = (sA[0], sA[2], sB[1])
            elif tA and not tB:
                sout = (sA[0], sA[2], sB[2])
            elif not tA and tB:
                sout = (sA[0], sA[1], sB[1])

    if not correct:
        raise ValueError(
            f"Tensor dimensions incorrect for matrix mulitiplication: A x B: {sA} x {sB} with transpose for A x B: {tA} x {tB}.",
        )

    return sout


def igemmlt(A, B, SA, SB, out=None, Sout=None, dtype=mindspore.int32):
    shapeA = SA[0]
    shapeB = SB[0]
    dimsA = len(shapeA)
    dimsB = len(shapeB)
    assert dimsB == 2, "Only two dimensional matrices are supported for argument B"

    m = shapeA[0]

    if dimsA == 3:
        m = m * shapeA[1]

    rows = n = shapeB[0]
    assert prod(list(shapeA)) > 0, f"Input tensor dimensions need to be > 0: {shapeA}"

    # if the tensor is empty, return a transformed empty tensor with the right dimensions
    if shapeA[0] == 0 and dimsA == 2:
        return empty((0, shapeB[0]), dtype=mindspore.float16)
    elif shapeA[1] == 0 and dimsA == 3:
        return empty(tuple(shapeA[:2] + [shapeB[0]]), dtype=mindspore.float16)

    if dimsA == 2 and out is None:
        out, Sout = get_transform_buffer((shapeA[0], shapeB[0]), dtype, "col32", "row")
    elif dimsA == 3 and out is None:
        out, Sout = get_transform_buffer(
            (shapeA[0], shapeA[1], shapeB[0]), dtype, "col32", "row"
        )

    assert dimsB != 3, "len(B.shape)==3 not supported"
    assert context.get_context("device_target") == "GPU"
    assert A.dtype == mindspore.int8
    assert B.dtype == mindspore.int8
    assert out.dtype == dtype
    assert SA[1] == "col32"
    assert SB[1] in ["col_turing", "col_ampere"]
    assert Sout[1] == "col32"
    assert (
        shapeA[-1] == shapeB[-1]
    ), f"Matmullt only supports A @ B^T. Inner matrix dimensions do not match: A @ B = {shapeA} @ {shapeB}"
    formatB = SB[1]

    # ptr = CUBLAS_Context.get_instance().get_context()

    k = shapeA[-1]
    lda = m * 32
    if formatB == "col_turing":
        # turing: tiles with rows filled up to multiple of 8 rows by 32 columns
        # n = rows
        ldb = ((rows + 7) // 8) * 8 * 32
    else:
        # ampere: tiles with rows filled up to multiple of 32 rows by 32 columns
        # n = rows
        ldb = ((rows + 31) // 32) * 32 * 32

    ldc = m * 32

    has_error = 0
    ptrRowScale = None
    if formatB == "col_turing":
        if dtype == mindspore.int32:
            bnbop.cigemmlt_turing_32(
                m, n, k, A, B, out, ptrRowScale, lda, ldb, ldc, has_error
            )
        else:
            bnbop.cigemmlt_turing_8(
                m, n, k, A, B, out, ptrRowScale, lda, ldb, ldc, has_error
            )
    elif formatB == "col_ampere":
        if dtype == mindspore.int32:
            bnbop.cigemmlt_ampere_32(
                m, n, k, A, B, out, ptrRowScale, lda, ldb, ldc, has_error
            )
        else:
            bnbop.cigemmlt_ampere_8(
                m, n, k, A, B, out, ptrRowScale, lda, ldb, ldc, has_error
            )

    if has_error == 100:  # `ERR_NOT_IMPLEMENTED` is defined as 100 in `ops.cu`
        raise NotImplementedError(
            "igemmlt not available (probably built with NO_CUBLASLT)"
        )

    if has_error:
        print(
            f"A: {shapeA}, B: {shapeB}, C: {Sout[0]}; (lda, ldb, ldc): {(lda, ldb, ldc)}; (m, n, k): {(m, n, k)}"
        )
        raise Exception("cublasLt ran into an error!")

    return out, Sout


def mm_dequant(
    A,
    quant_state,
    row_stats,
    col_stats,
    out=None,
    new_row_stats=None,
    new_col_stats=None,
    bias=None,
):
    assert A.dtype == mindspore.int32
    if bias is not None:
        assert bias.dtype == mindspore.float16
    out_shape = quant_state[0]
    if len(out_shape) == 3:
        out_shape = (out_shape[0] * out_shape[1], out_shape[2])
    if out is None:
        out = empty(out_shape, dtype=mindspore.float16)
    if new_row_stats is None:
        new_row_stats = empty(
            out_shape[0],
            dtype=mindspore.float32,
        )
    if new_col_stats is None:
        new_col_stats = empty(
            out_shape[1],
            dtype=mindspore.float32,
        )
    assert (
        new_row_stats.shape[0] == row_stats.shape[0]
    ), f"{new_row_stats.shape} vs {row_stats.shape}"
    assert (
        new_col_stats.shape[0] == col_stats.shape[0]
    ), f"{new_col_stats.shape} vs {col_stats.shape}"

    numRows = out_shape[0]
    numCols = out_shape[1]
    bnbop.cdequant_mm_int32_fp16(
        A,
        row_stats,
        col_stats,
        out,
        new_row_stats,
        new_col_stats,
        bias,
        numRows,
        numCols,
    )

    return out


def get_colrow_absmax(
    A, row_stats=None, col_stats=None, nnz_block_ptr=None, threshold=0.0
):
    assert A.dtype == mindspore.float16

    cols = A.shape[-1]
    if len(A.shape) == 3:
        rows = A.shape[0] * A.shape[1]
    else:
        rows = A.shape[0]

    col_tiles = (cols + 255) // 256
    tiled_rows = ((rows + 15) // 16) * 16

    if row_stats is None:
        row_stats = empty(
            (rows,),
            dtype=mindspore.float32,
        ).fill(-50000.0)
    if col_stats is None:
        col_stats = empty(
            (cols,),
            dtype=mindspore.float32,
        ).fill(-50000.0)
    # if nnz_block_ptr is None and threshold > 0.0:
    if nnz_block_ptr is None and threshold > 0.0:
        nnz_block_ptr = ops.zeros(
            ((tiled_rows * col_tiles) + 1,),
            dtype=mindspore.int32,
        )

    bnbop.cget_col_row_stats(
        A, row_stats, col_stats, nnz_block_ptr, threshold, rows, cols
    )

    if threshold > 0.0:
        nnz_block_ptr = nnz_block_ptr.cumsum(axis=0)

    return row_stats, col_stats, nnz_block_ptr


class COOSparseTensor:
    def __init__(self, rows, cols, nnz, rowidx, colidx, values):
        assert rowidx.dtype == mindspore.int32
        assert colidx.dtype == mindspore.int32
        assert values.dtype == mindspore.float16
        assert values.numel() == nnz
        assert rowidx.numel() == nnz
        assert colidx.numel() == nnz

        self.rows = rows
        self.cols = cols
        self.nnz = nnz
        self.rowidx = rowidx
        self.colidx = colidx
        self.values = values


class CSRSparseTensor:
    def __init__(self, rows, cols, nnz, rowptr, colidx, values):
        assert rowptr.dtype == mindspore.int32
        assert colidx.dtype == mindspore.int32
        assert values.dtype == mindspore.float16
        assert values.numel() == nnz
        assert colidx.numel() == nnz
        assert rowptr.numel() == rows + 1

        self.rows = rows
        self.cols = cols
        self.nnz = nnz
        self.rowptr = rowptr
        self.colidx = colidx
        self.values = values


class CSCSparseTensor:
    def __init__(self, rows, cols, nnz, colptr, rowidx, values):
        assert colptr.dtype == mindspore.int32
        assert rowidx.dtype == mindspore.int32
        assert values.dtype == mindspore.float16
        assert values.numel() == nnz
        assert rowidx.numel() == nnz
        assert colptr.numel() == cols + 1

        self.rows = rows
        self.cols = cols
        self.nnz = nnz
        self.colptr = colptr
        self.rowidx = rowidx
        self.values = values


def coo_zeros(rows, cols, nnz, dtype=mindspore.half):
    rowidx = ops.zeros(
        (nnz,),
        dtype=mindspore.int32,
    )
    colidx = ops.zeros(
        (nnz,),
        dtype=mindspore.int32,
    )
    values = ops.zeros(
        (nnz,),
        dtype=dtype,
    )
    return COOSparseTensor(rows, cols, nnz, rowidx, colidx, values)


def double_quant(
    A, col_stats=None, row_stats=None, out_col=None, out_row=None, threshold=0.0
):
    assert A.dtype == mindspore.half

    cols = A.shape[-1]
    if len(A.shape) == 3:
        rows = A.shape[0] * A.shape[1]
    else:
        rows = A.shape[0]

    if row_stats is None or col_stats is None:
        row_stats, col_stats, nnz_row_ptr = get_colrow_absmax(A, threshold=threshold)

    if out_col is None:
        out_col = ops.zeros(A.shape, dtype=mindspore.int8)
    if out_row is None:
        out_row = ops.zeros(A.shape, dtype=mindspore.int8)

    coo_tensor = None

    if threshold > 0.0:
        nnz = nnz_row_ptr[-1].item()
        if nnz > 0:
            coo_tensor = coo_zeros(
                A.shape[0],
                A.shape[1],
                nnz_row_ptr[-1].item(),
            )
            row_idx = coo_tensor.rowidx
            col_idx = coo_tensor.colidx
            val = coo_tensor.values

            bnbop.cdouble_rowcol_quant(
                A,
                row_stats,
                col_stats,
                out_col,
                out_row,
                row_idx,
                col_idx,
                val,
                nnz_row_ptr,
                threshold,
                rows,
                cols,
            )
            val, idx = ops.sort(coo_tensor.rowidx)
            coo_tensor.rowidx = val
            coo_tensor.colidx = coo_tensor.colidx[idx]
            coo_tensor.values = coo_tensor.values[idx]
        else:
            bnbop.cdouble_rowcol_quant(
                A,
                row_stats,
                col_stats,
                out_col,
                out_row,
                None,
                None,
                None,
                None,
                0.0,
                rows,
                cols,
            )
    else:
        bnbop.cdouble_rowcol_quant(
            A,
            row_stats,
            col_stats,
            out_col,
            out_row,
            None,
            None,
            None,
            None,
            threshold,
            rows,
            cols,
        )

    return out_row, out_col, row_stats, col_stats, coo_tensor


def transform(
    A, to_order, from_order="row", out=None, transpose=False, state=None, ld=None
):
    if state is None:
        state = (A.shape, from_order)
    else:
        from_order = state[1]
    if out is None:
        out, new_state = get_transform_buffer(
            state[0], A.dtype, to_order, state[1], transpose
        )
    else:
        new_state = (state[0], to_order)  # (shape, order)

    shape = state[0]
    if len(shape) == 2:
        dim1 = shape[0]
        dim2 = shape[1]
    else:
        dim1 = shape[0] * shape[1]
        dim2 = shape[2]

    if to_order == "col32":
        if transpose:
            bnbop.ctransform_row2col32T(A, out, dim1, dim2)
        else:
            bnbop.ctransform_row2col32(A, out, dim1, dim2)
    elif to_order == "col_turing":
        if transpose:
            bnbop.ctransform_row2turingT(A, out, dim1, dim2)
        else:
            bnbop.ctransform_row2turing(A, out, dim1, dim2)
    elif to_order == "col_ampere":
        if transpose:
            bnbop.ctransform_row2ampereT(A, out, dim1, dim2)
        else:
            bnbop.ctransform_row2ampere(A, out, dim1, dim2)
    elif to_order == "row":
        if from_order == "col_turing":
            bnbop.ctransform_turing2row(A, out, dim1, dim2)
        elif from_order == "col_ampere":
            bnbop.ctransform_ampere2row(A, out, dim1, dim2)
    else:
        raise NotImplementedError(
            f"Transform function not implemented: From {from_order} to {to_order}"
        )

    return out, new_state


C = 127.0


def vectorwise_quant(x, axis=1, quant_type="vector"):
    if quant_type == "linear":
        max1 = ops.abs(x).max().float()
        xq = ops.round(x / max1 * 127).astype(mindspore.int8)
        return xq, max1
    elif quant_type in ["vector", "row"]:
        max1 = ops.amax(ops.abs(x), axis=axis, keepdims=True)
        xq = ops.round(x * (C / max1)).astype(mindspore.int8)
        return xq, max1
    elif quant_type == "zeropoint":
        dtype = x.dtype
        x = x.float()
        dyna = x.max() - x.min()
        if dyna == 0:
            dyna = 1
        qx = 255.0 / dyna
        minx = x.min()
        zpx = ops.round(minx * qx)
        x = ops.round(qx * x - zpx) + zpx
        return x, qx
    elif quant_type in ["vector-zeropoint", "row-zeropoint"]:
        dtype = x.dtype
        x = x.float()
        dyna = ops.amax(x, axis=axis, keepdims=True) - ops.amin(x, axis=axis, keepdims=True)
        dyna[dyna == 0] = 1
        qx = 255.0 / dyna
        minx = ops.amin(x, axis=axis, keepdims=True)
        zpx = ops.round(minx * qx)
        x = ops.round(qx * x - zpx) + zpx
        return x, qx
    elif quant_type == "truncated-vector":
        absx = ops.abs(x)
        max1 = ops.amax(absx, axis=axis, keepdims=True)
        max1 = max1 * 0.7
        idx = absx > max1.expand_as(absx)
        sign = ops.sign(x[idx])
        x[idx] = max1.expand_as(absx)[idx] * sign
        xq = ops.round(x / max1 * C).astype(mindspore.int8)
        return xq, max1
    else:
        return None


def vectorwise_dequant(xq, max1, quant_type="vector"):
    if quant_type == "vector":
        x = (xq / C * max1).astype(mindspore.float32)
        return x
    else:
        return None


def vectorwise_mm_dequant(xq, S1, S2, dtype=mindspore.half, quant_type="vector"):
    if quant_type == "linear":
        norm = S1 * S2 / (C * C)
        # double cast needed to prevent overflows
        return (xq.float() * norm).to(dtype)
    elif quant_type == "zeropoint":
        norm = 1.0 / (S1 * S2)
        return (xq.float() * norm).to(dtype)
    elif quant_type == "row-zeropoint":
        norm = 1.0 / (S1 * S2)
        x = xq.float()
        if len(S1.shape) == 3 and len(x.shape) == 2:
            S1 = S1.squeeze(0)
        if len(S2.shape) == 3 and len(x.shape) == 2:
            S2 = S2.squeeze(0)
        if len(S1.shape) == 2:
            x *= norm
        else:
            x *= norm
        return x.to(dtype)
    elif quant_type == "vector-zeropoint":
        x = xq.float()
        if len(S1.shape) == 3 and len(x.shape) == 2:
            S1 = S1.squeeze(0)
        if len(S2.shape) == 3 and len(x.shape) == 2:
            S2 = S2.squeeze(0)
        if len(S1.shape) == 2:
            x *= 1.0 / S1
        else:
            x *= 1.0 / S1
        x *= 1.0 / S2.t()
        return x.to(dtype)
    elif quant_type == "row":
        x = xq.float()
        if len(S1.shape) == 3 and len(x.shape) == 2:
            S1 = S1.squeeze(0)
        if len(S2.shape) == 3 and len(x.shape) == 2:
            S2 = S2.squeeze(0)
        if len(S1.shape) == 2:
            x *= S1 * S2 / (C * C)
        else:
            x *= S1 * S2 / (C * C)
        return x.to(dtype)
    elif quant_type in ["truncated-vector", "vector"]:
        x = xq.float()
        if len(S1.shape) == 3 and len(x.shape) == 2:
            S1 = S1.squeeze(0)
        if len(S2.shape) == 3 and len(x.shape) == 2:
            S2 = S2.squeeze(0)
        if len(S1.shape) == 2:
            x *= S1 / C
        else:
            x *= S1 / C
        x *= S2 / C
        return x.to(dtype)
    else:
        return None


def dequant_min_max(xq, A, B, SA, SB, dtype=mindspore.half):
    offset = B.float().t().sum(0) * (SA[0] + SA[1])
    x = xq.float()
    if len(xq.shape) == 2 and len(SB.shape) == 3:
        SB = SB.squeeze(0)
    if len(SB.shape) == 2:
        x *= SB.t() / 127
    else:
        x *= SB / 127
    x *= SA[1] / 127
    x += offset
    return x.to(dtype)


def extract_outliers(A, SA, idx):
    shapeA = SA[0]
    formatA = SA[1]
    assert formatA in ["col_turing", "col_ampere"]

    out = ops.zeros((shapeA[0], idx.numel()), dtype=mindspore.int8)

    idx_size = idx.numel()
    rows = shapeA[0]
    cols = shapeA[1]

    if formatA == "col_turing":
        bnbop.cextractOutliers_turing(A, idx, out, idx_size, rows, cols)
    elif formatA == "col_ampere":
        bnbop.cextractOutliers_ampere(A, idx, out, idx_size, rows, cols)

    return out
