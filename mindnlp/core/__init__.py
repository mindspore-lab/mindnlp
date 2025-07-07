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
import platform
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

strided = None
contiguous_format = None
preserve_format = None

inf = float("inf")
nan = float("nan")

from ._C import *
from ._dtype import *
from ._tensor import Tensor, tensor, is_tensor, \
    LongTensor, FloatTensor, BoolTensor, HalfTensor, BFloat16Tensor
from .types import device
from ._C.size import Size
from .types import device
from .autograd import *
from .ops import *
from .serialization import load, save
from ._bind import get_default_dtype, set_default_dtype

from . import profiler, cuda, optim, amp, compiler, jit, version, __future__, overrides, \
    return_types

