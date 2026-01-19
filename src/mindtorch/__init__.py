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
from mindspore._c_expression import MSContext # pylint: disable=no-name-in-module, import-error

_pynative_executor.set_grad_flag(True)

# for huawei cloud modelarts
if 'RANK_TABLE_FILE' in os.environ:
    del os.environ['RANK_TABLE_FILE']

try:
    from mindspore._c_expression import disable_multi_thread
except:
    disable_multi_thread = None

if os.environ.get('DEVICE_TARGET', None) is not None:
    mindspore.set_device(os.environ.get('DEVICE_TARGET'))

# for different ascend devices
if platform.system().lower() == 'linux' and mindspore.get_context('device_target') == 'Ascend':
    SOC = MSContext.get_instance().get_ascend_soc_version()
    # enable vmm since only vmm can release device memory when del tensor.
    if SOC != 'ascend310b':
        os.environ["MS_ALLOC_CONF"] = 'enable_vmm:True,vmm_align_size:2MB'

    if SOC in ('ascend910', 'ascend310b'):
        # context.set_context(ascend_config={"precision_mode": "allow_mix_precision"})
        mindspore.device_context.ascend.op_precision.precision_mode('allow_mix_precision')
    # if SOC == 'ascend310b' and disable_multi_thread is not None:
    #     disable_multi_thread()
    if SOC == 'ascend310b':
        # export MAX_COMPILE_CORE_NUMBER=1
        # export TE_PARALLEL_COMPILER=1
        os.environ["MAX_COMPILE_CORE_NUMBER"] = '1'
        os.environ["TE_PARALLEL_COMPILER"] = '1'
        mindspore.device_context.ascend.op_debug.execute_timeout(200)
        mindspore.runtime.dispatch_threads_num(1)
        mindspore.device_context.cpu.op_tuning.threads_num(1)


pi = math.pi
layout = object
strided = None
contiguous_format = None
preserve_format = None
legacy_contiguous_format = None
channels_last_3d = None
channels_last = None
memory_format = None

inf = float("inf")
nan = float("nan")

class OutOfMemoryError(RuntimeError):
    """Compatibility alias for torch.OutOfMemoryError."""
    pass

from . import _C
from ._dtype import *
from ._tensor import Tensor, tensor, scalar_tensor, is_tensor, \
    LongTensor, FloatTensor, BoolTensor, HalfTensor, BFloat16Tensor, IntTensor, ByteTensor

from ._C import *
from ._C.size import Size
from .ops import *
from ._tensor import enable_mindspore_patch
enable_mindspore_patch()

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


def use_deterministic_algorithms(mode, *, warn_only=False):
    mindspore.set_context(deterministic='ON')

def is_grad_enabled():
    return _pynative_executor.enable_grad()

def set_grad_enabled(enable_grad):
    return _pynative_executor.set_enable_grad(enable_grad)

def is_same_size(tensor1, tensor2):
    """
    Check if two tensors have the same size.
    
    Args:
        tensor1: First tensor
        tensor2: Second tensor
    
    Returns:
        bool: True if both tensors have the same shape, False otherwise
    """
    if not isinstance(tensor1, Tensor) or not isinstance(tensor2, Tensor):
        return False
    return tensor1.shape == tensor2.shape

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


def _nnpack_available():
    return False

def _running_with_deploy():
    return False

def ms_run_check():
    try:
        x = mindspore.mint.empty(1)
        x.__str__()
    except:
        pass

from .autograd import *
from .serialization import load, save
from ._bind import get_default_dtype, set_default_dtype, get_default_device, is_autocast_enabled, set_autocast_enabled, \
    set_autocast_dtype, get_autocast_dtype, asarray as _mt_asarray

from .amp import autocast, GradScaler
from .func import vmap
from .storage import UntypedStorage, Storage, TypedStorage

# Provide torch-compatible asarray API at top-level
# Delegate to _bind.asarray with sensible defaults
def asarray(obj, *, dtype=None, device=None, copy=None, requires_grad=False):
    if dtype is None:
        dtype = get_default_dtype()
    return _mt_asarray(obj, dtype=dtype, device=device, copy=copy, requires_grad=requires_grad)

from . import _dynamo, library
from . import profiler, cuda, npu, xpu, mps, amp, compiler, jit, version, __future__, overrides, \
    return_types, linalg, fx, backends, nn, fft, _jit_internal, utils, optim, testing, _ops, accelerator, special
from ._lowrank import svd_lowrank
from .random import get_rng_state, initial_seed, manual_seed, seed, set_rng_state

if mindspore.get_context('device_target') == 'Ascend':
    cuda = npu

__version__ = 'test_version_no_value'


from .torch_proxy import initialize_torch_proxy, setup_metadata_patch
initialize_torch_proxy()
setup_metadata_patch()

# Patch diffusers' AutoencoderKLAllegro to enable tiled decoding by default
# This avoids NotImplementedError when decoding without tiling.
try:
    from diffusers.models.autoencoders.autoencoder_kl_allegro import AutoencoderKLAllegro  # type: ignore
    _orig_decode = AutoencoderKLAllegro.decode
    def _patched_decode(self, *args, **kwargs):
        try:
            # Prefer official API if available
            if hasattr(self, "enable_tiling") and callable(getattr(self, "enable_tiling")):
                # Only enable once
                if not getattr(self, "use_tiling", False):
                    try:
                        self.enable_tiling()
                    except Exception:
                        # Fallback: set flag directly if method fails
                        setattr(self, "use_tiling", True)
            else:
                # Fallback if API not present
                if not getattr(self, "use_tiling", False) and hasattr(self, "tiled_decode"):
                    setattr(self, "use_tiling", True)
        except Exception:
            # Best-effort patch; never break user code
            pass
        return _orig_decode(self, *args, **kwargs)
    AutoencoderKLAllegro.decode = _patched_decode
except Exception:
    # diffusers might be absent; ignore
    pass

# Patch diffusers' get_timestep_embedding to accept 2D inputs by flattening
try:
    import diffusers.models.embeddings as _emb_mod  # type: ignore
    _orig_get_timestep_embedding = _emb_mod.get_timestep_embedding
    def _patched_get_timestep_embedding(timesteps, embedding_dim, flip_sin_to_cos=False,
                                        downscale_freq_shift=1, scale=1, max_period=10000):
        try:
            # Ensure 1D timesteps as expected by diffusers implementation
            if hasattr(timesteps, 'shape') and len(timesteps.shape) != 1:
                # Prefer squeezing singleton last dim, else flatten
                try:
                    if timesteps.shape[-1] == 1:
                        timesteps = timesteps.squeeze(-1)
                    else:
                        timesteps = timesteps.reshape(-1)
                except Exception:
                    timesteps = timesteps.reshape(-1)
        except Exception:
            # Best-effort: leave as is if any issue
            pass
        return _orig_get_timestep_embedding(
            timesteps, embedding_dim, flip_sin_to_cos, downscale_freq_shift, scale, max_period
        )
    _emb_mod.get_timestep_embedding = _patched_get_timestep_embedding
except Exception:
    # diffusers might be absent; ignore
    pass
