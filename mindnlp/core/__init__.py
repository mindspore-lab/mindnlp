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
"""core module"""
import os
import math
from typing import (
    Any as _Any,
    Callable as _Callable,
    get_origin as _get_origin,
    Optional as _Optional,
    overload as _overload,
    TYPE_CHECKING,
    TypeVar as _TypeVar,
    Union as _Union,
)

import mindspore
from mindspore.runtime import Stream
from mindspore.common.api import _pynative_executor

pi = math.pi
strided = None
contiguous_format = None
preserve_format = None
legacy_contiguous_format = None
channels_last_3d = None

inf = float("inf")
nan = float("nan")

from ._C import *
from ._dtype import *
from ._tensor import Tensor, tensor, is_tensor, \
    LongTensor, FloatTensor, BoolTensor, HalfTensor, BFloat16Tensor, IntTensor
from .types import device
from ._C.size import Size
from .types import device
from .autograd import *
from .ops import *
from .serialization import load, save
from ._bind import get_default_dtype, set_default_dtype, get_default_device
from .amp import autocast, GradScaler
from .func import vmap
from .configs import set_pyboost

from . import profiler, cuda, amp, compiler, jit, version, __future__, overrides, \
    return_types, linalg, fx, backends, testing, nn, fft, _jit_internal, utils, optim
from ._lowrank import svd_lowrank
from .random import get_rng_state, initial_seed, manual_seed, seed, set_rng_state


def _has_compatible_shallow_copy_type(tensor, other):
    """
    Mimics the behavior of mindtorch._has_compatible_shallow_copy_type.

    Args:
        tensor (mindtorch.Tensor): The source tensor.
        other (mindtorch.Tensor): The target tensor to check compatibility.

    Returns:
        bool: True if `tensor` and `other` have compatible types for shallow copy.
    """
    # Check if both tensors have the same type
    if not is_tensor(tensor) or not is_tensor(other):
        return False

    # Check if both tensors are on the same device
    if tensor.shape != other.shape:
        return False

    # Compatibility confirmed
    return True

def compile(fn=None, *args, **kwargs):
    def wrap_func(fn):
        return fn
    if fn is not None:
        return wrap_func(fn)
    return wrap_func

AUTO_CAST_DTYE = {
    'cuda': float16,
    'cpu': float16,
    'npu': float16,
    'Ascend': float16
}

def set_autocast_dtype(device_type, dtype):
    assert device_type in AUTO_CAST_DTYE.keys(), f'{device_type} is not in {AUTO_CAST_DTYE.keys()}'
    AUTO_CAST_DTYE[device_type] = dtype

def get_autocast_dtype(device_type):
    return AUTO_CAST_DTYE[device_type]

def get_autocast_gpu_dtype():
    return AUTO_CAST_DTYE['cuda']

def is_autocast_enabled():
    return True

def use_deterministic_algorithms(mode, *, warn_only=False):
    mindspore.set_context(deterministic='ON' if mode else 'OFF')

def is_grad_enabled():
    return _pynative_executor.enable_grad()

def set_grad_enabled(enable_grad):
    return _pynative_executor.set_enable_grad(enable_grad)

def typename(obj: _Any, /) -> str:
    """
    String representation of the type of an object.

    This function returns a fully qualified string representation of an object's type.
    Args:
        obj (object): The object whose type to represent
    Returns:
        str: the type of the object `o`
    Example:
        >>> x = torch.tensor([1, 2, 3])
        >>> torch.typename(x)
        'torch.LongTensor'
        >>> torch.typename(torch.nn.Parameter)
        'torch.nn.parameter.Parameter'
    """
    if isinstance(obj, Tensor):
        return obj.type()

    module = getattr(obj, "__module__", "") or ""
    qualname = ""

    if hasattr(obj, "__qualname__"):
        qualname = obj.__qualname__
    elif hasattr(obj, "__name__"):
        qualname = obj.__name__
    else:
        module = obj.__class__.__module__ or ""
        qualname = obj.__class__.__qualname__

    if module in {"", "builtins"}:
        return qualname
    return f"{module}.{qualname}"


__version__ = 'test_version_no_value'