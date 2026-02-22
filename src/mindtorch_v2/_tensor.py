import numpy as np

from ._storage import (
    Storage,
    empty_cpu_typed_storage,
    meta_typed_storage_from_shape,
    npu_typed_storage_from_ptr,
    pinned_cpu_typed_storage_from_numpy,
    typed_storage_from_numpy,
)
from ._device import _default_device, device as Device
from ._dtype import float32, float16, float64, bfloat16, int8, int16, int32, int64, uint8
from ._dtype import bool as dtype_bool
from ._dtype import to_numpy_dtype
from ._functional import add, mul, matmul, relu, sum, abs as abs_dispatch, neg as neg_dispatch
from ._functional import exp as exp_dispatch, log as log_dispatch, sqrt as sqrt_dispatch
from ._functional import sin as sin_dispatch, cos as cos_dispatch, tan as tan_dispatch
from ._functional import tanh as tanh_dispatch, sigmoid as sigmoid_dispatch
from ._functional import floor as floor_dispatch, ceil as ceil_dispatch, round as round_dispatch
from ._functional import trunc as trunc_dispatch, frac as frac_dispatch
from ._functional import pow as pow_dispatch, log2 as log2_dispatch, log10 as log10_dispatch
from ._functional import exp2 as exp2_dispatch, rsqrt as rsqrt_dispatch
from ._functional import sign as sign_dispatch, signbit as signbit_dispatch
from ._functional import isnan as isnan_dispatch, isinf as isinf_dispatch, isfinite as isfinite_dispatch
from ._functional import sinh as sinh_dispatch, cosh as cosh_dispatch
from ._functional import asinh as asinh_dispatch, acosh as acosh_dispatch, atanh as atanh_dispatch
from ._functional import erf as erf_dispatch, erfc as erfc_dispatch, softplus as softplus_dispatch
from ._functional import clamp as clamp_dispatch, clamp_min as clamp_min_dispatch, clamp_max as clamp_max_dispatch
from ._functional import relu6 as relu6_dispatch, hardtanh as hardtanh_dispatch
from ._functional import min as min_dispatch, max as max_dispatch
from ._functional import amin as amin_dispatch, amax as amax_dispatch
from ._functional import fmin as fmin_dispatch, fmax as fmax_dispatch
from ._functional import where as where_dispatch
from ._functional import atan as atan_dispatch, atan2 as atan2_dispatch
from ._functional import asin as asin_dispatch, acos as acos_dispatch
from ._functional import lerp as lerp_dispatch
from ._functional import addcmul as addcmul_dispatch, addcdiv as addcdiv_dispatch
from ._functional import logaddexp as logaddexp_dispatch, logaddexp2 as logaddexp2_dispatch
from ._functional import hypot as hypot_dispatch, remainder as remainder_dispatch, fmod as fmod_dispatch
from ._functional import all as all_dispatch, any as any_dispatch, argmax as argmax_dispatch
from ._functional import argmin as argmin_dispatch, count_nonzero as count_nonzero_dispatch
from ._functional import allclose as allclose_dispatch, isclose as isclose_dispatch, equal as equal_dispatch
from ._functional import cumsum as cumsum_dispatch, cumprod as cumprod_dispatch, cummax as cummax_dispatch
from ._functional import reshape as reshape_dispatch
from ._functional import transpose as transpose_dispatch, view as view_dispatch, to as to_dispatch
from ._autograd.engine import backward as _backward
from ._autograd.version_counter import VersionCounter
from ._printing import format_tensor


def _compute_strides(shape):
    stride = []
    acc = 1
    for d in reversed(shape):
        stride.append(acc)
        acc *= d
    return tuple(reversed(stride))


def _bf16_to_f32(arr):
    """Convert bfloat16 (stored as uint16) to float32."""
    u32 = arr.astype(np.uint32) << 16
    return u32.view(np.float32)


def _f32_to_bf16(arr):
    """Convert float32 to bfloat16 (stored as uint16), round-to-nearest-even."""
    u32 = arr.view(np.uint32)
    # Round to nearest even: add bias + (lsb of result)
    rounding_bias = (u32 >> 16) & 1
    u32 = u32 + 0x7FFF + rounding_bias
    return (u32 >> 16).astype(np.uint16)


class _HookHandle:
    _next_id = 0

    def __init__(self, hooks):
        self._hooks = hooks
        self.id = _HookHandle._next_id
        _HookHandle._next_id += 1

    def remove(self):
        if self._hooks is None:
            return
        self._hooks.pop(self.id, None)
        self._hooks = None

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.remove()


class Tensor:
    def __init__(self, storage, shape, stride, offset=0, requires_grad=False):
        self._storage = storage
        self.shape = tuple(shape)
        self.stride = tuple(stride)
        self.offset = int(offset)
        self.requires_grad = requires_grad
        self.grad = None
        self.grad_fn = None
        self._pending = False
        self._retain_grad = False
        self._backward_hooks = None
        self._version_counter = VersionCounter()
        self._base = None
        self._view_meta = None

    @property
    def dtype(self):
        return self._storage.dtype

    @property
    def device(self):
        return self._storage.device

    def storage(self):
        return self._storage

    def untyped_storage(self):
        """Return the underlying untyped storage.

        This is needed for compatibility with safetensors which calls
        tensor.untyped_storage().nbytes() to determine storage size.
        """
        return self._storage.untyped_storage()

    def dim(self):
        return len(self.shape)

    def numel(self):
        result = 1
        for s in self.shape:
            result *= s
        return result

    # Alias for numel (PyTorch compatibility)
    nelement = numel

    def element_size(self):
        return self.dtype.itemsize

    def is_floating_point(self):
        return self.dtype.is_floating_point

    def is_complex(self):
        return self.dtype.is_complex

    def is_contiguous(self, memory_format=None):
        """Check if tensor is contiguous in row-major order."""
        expected = _compute_strides(self.shape)
        return self.stride == expected

    def contiguous(self, memory_format=None):
        """Return contiguous tensor (copy if not already contiguous)."""
        if self.is_contiguous():
            return self

        # Use dispatch to stay on device (avoid numpy round-trip)
        from ._dispatch import dispatch
        return dispatch("contiguous", self.device.type, self)

    def _numpy_view(self):
        if self.device.type == "meta":
            raise RuntimeError("meta tensor has no data")
        base = self._storage.data.ravel()
        itemsize = base.itemsize
        strides = tuple(s * itemsize for s in self.stride)
        return np.lib.stride_tricks.as_strided(
            base[self.offset:], shape=self.shape, strides=strides
        )

    def reshape(self, new_shape):
        return reshape_dispatch(self, new_shape)

    def view(self, new_shape):
        return view_dispatch(self, new_shape)

    def transpose(self, dim0, dim1):
        return transpose_dispatch(self, dim0, dim1)

    def _ones_like(self):
        if self.device.type == "meta":
            storage = meta_typed_storage_from_shape(self.shape, self.dtype, device=self.device)
            return Tensor(storage, self.shape, self.stride)
        arr = np.ones(self.shape, dtype=to_numpy_dtype(self.dtype))
        storage = typed_storage_from_numpy(arr, self.dtype, device=self.device if self.device.type == "cpu" else None)
        stride = tuple(np.array(arr.strides) // arr.itemsize)
        tensor = Tensor(storage, arr.shape, stride)
        if self.device.type != "cpu":
            return tensor.to(self.device)
        return tensor

    def record_stream(self, stream):
        if self.device.type != "npu":
            return
        from ._backends.npu import allocator as npu_allocator

        alloc = npu_allocator.get_allocator(self.device.index or 0)
        alloc.record_stream(self.storage().data_ptr(), stream.stream)

    def numpy(self):
        if self._pending:
            from ._dispatch.pipeline import current_pipeline

            pipe = current_pipeline()
            if pipe is not None:
                pipe.flush()
        if self.device.type == "meta":
            raise RuntimeError("meta tensor has no data")
        if self.device.type != "cpu":
            raise RuntimeError("numpy() is only available for CPU tensors")
        return self._numpy_view()

    def item(self):
        if self.numel() != 1:
            raise ValueError("only one element tensors can be converted to Python scalars")
        if self.device.type != "cpu":
            return self.to("cpu").item()
        return self._numpy_view().flat[0].item()

    def tolist(self):
        if self.device.type != "cpu":
            return self.to("cpu").tolist()
        return self._numpy_view().tolist()

    def __int__(self):
        return int(self.item())

    def __float__(self):
        return float(self.item())

    def __bool__(self):
        return bool(self.item())

    def backward(self, gradient=None, retain_graph=False, create_graph=False):
        if self._pending:
            from ._dispatch.pipeline import current_pipeline

            pipe = current_pipeline()
            if pipe is not None:
                pipe.flush()
        _backward(self, gradient, retain_graph=retain_graph, create_graph=create_graph)

    def pin_memory(self):
        if self.device.type != "cpu":
            raise RuntimeError("pin_memory only supports CPU tensors")
        from . import npu as npu_api

        if not npu_api.is_available():
            raise RuntimeError("Cannot access accelerator device when none is available.")
        if self.is_pinned():
            return self
        storage = pinned_cpu_typed_storage_from_numpy(self._numpy_view(), self.dtype, device=self.device)
        return Tensor(storage, self.shape, self.stride, self.offset, self.requires_grad)

    def is_pinned(self):
        return self._storage.is_pinned()

    def retain_grad(self):
        self._retain_grad = True

    def requires_grad_(self, requires_grad=True):
        self.requires_grad = bool(requires_grad)
        if not self.requires_grad:
            self.grad_fn = None
        return self

    def detach(self):
        out = Tensor(self._storage, self.shape, self.stride, self.offset, requires_grad=False)
        out.grad_fn = None
        out.grad = None
        out._pending = self._pending
        out._version_counter = self._version_counter
        return out

    def detach_(self):
        self.requires_grad = False
        self.grad_fn = None
        self._retain_grad = False
        return self

    def register_hook(self, hook):
        if not callable(hook):
            raise TypeError("hook must be callable")
        if self._backward_hooks is None:
            self._backward_hooks = {}
        handle = _HookHandle(self._backward_hooks)
        self._backward_hooks[handle.id] = hook
        return handle

    def _bump_version(self):
        self._version_counter.bump()

    def _is_view(self):
        return self._base is not None

    def _check_inplace(self):
        from ._autograd.grad_mode import is_grad_enabled

        if not is_grad_enabled():
            return
        if not self.requires_grad:
            return
        if self.grad_fn is None and not self._is_view():
            raise RuntimeError("a leaf Variable that requires grad is being used in an in-place operation.")
        if self._is_view() and self._base is not None and self._base.grad_fn is None and self._base.requires_grad:
            raise RuntimeError("a view of a leaf Variable that requires grad is being used in an in-place operation.")

    def add_(self, other):
        from ._dispatch.dispatcher import dispatch

        self._check_inplace()
        out = dispatch("add_", self.device.type, self, other)
        self._bump_version()
        return out

    def mul_(self, other):
        from ._dispatch.dispatcher import dispatch

        self._check_inplace()
        out = dispatch("mul_", self.device.type, self, other)
        self._bump_version()
        return out

    def relu_(self):
        from ._dispatch.dispatcher import dispatch

        self._check_inplace()
        out = dispatch("relu_", self.device.type, self)
        self._bump_version()
        return out

    def zero_(self):
        from ._dispatch.dispatcher import dispatch

        self._check_inplace()
        out = dispatch("zero_", self.device.type, self)
        self._bump_version()
        return out

    def to(self, *args, **kwargs):
        if self._pending:
            from ._dispatch.pipeline import current_pipeline

            pipe = current_pipeline()
            if pipe is not None:
                pipe.flush()
        # Parse arguments: to(device), to(dtype), to(device, dtype), to(dtype=, device=)
        device = None
        dtype = None
        non_blocking = kwargs.get("non_blocking", False)
        for arg in args:
            if isinstance(arg, Device):
                device = arg
            elif isinstance(arg, str):
                from ._dtype import from_name
                dt = from_name(arg)
                if dt is not None:
                    dtype = dt
                else:
                    device = Device(arg)
            elif hasattr(arg, 'name') and hasattr(arg, 'itemsize'):
                dtype = arg
            else:
                device = Device(str(arg))
        if "device" in kwargs:
            device = kwargs["device"]
            if isinstance(device, str):
                device = Device(device)
        if "dtype" in kwargs:
            dtype = kwargs["dtype"]
        result = self
        if dtype is not None and dtype != self.dtype:
            result = result._to_dtype(dtype)
        if device is not None:
            result = to_dispatch(result, device, non_blocking=non_blocking)
        if result is self and dtype is None and device is None:
            return self
        return result

    def _to_dtype(self, dtype):
        if self.device.type == "cpu":
            arr = self._numpy_view()
            src_dtype = self.dtype
            target_np = to_numpy_dtype(dtype)
            if src_dtype == bfloat16:
                # bfloat16 -> target: first convert uint16 bits to float32
                arr = _bf16_to_f32(arr)
            if dtype == bfloat16:
                # source -> bfloat16: convert to float32 then to uint16 bits
                arr = arr.astype(np.float32)
                arr = _f32_to_bf16(arr)
            else:
                arr = arr.astype(target_np)
            storage = typed_storage_from_numpy(arr, dtype, device=self.device)
            stride = tuple(np.array(arr.strides) // arr.itemsize)
            return Tensor(storage, arr.shape, stride)
        elif self.device.type == "meta":
            storage = meta_typed_storage_from_shape(self.shape, dtype, device=self.device)
            return Tensor(storage, self.shape, _compute_strides(self.shape))
        else:
            raise RuntimeError(
                f"dtype conversion not yet supported on device {self.device.type}"
            )

    def float(self):
        return self._to_dtype(float32) if self.dtype != float32 else self

    def half(self):
        return self._to_dtype(float16) if self.dtype != float16 else self

    def double(self):
        return self._to_dtype(float64) if self.dtype != float64 else self

    def bfloat16(self):
        return self._to_dtype(bfloat16) if self.dtype != bfloat16 else self

    def long(self):
        return self._to_dtype(int64) if self.dtype != int64 else self

    def int(self):
        return self._to_dtype(int32) if self.dtype != int32 else self

    def short(self):
        return self._to_dtype(int16) if self.dtype != int16 else self

    def char(self):
        return self._to_dtype(int8) if self.dtype != int8 else self

    def byte(self):
        return self._to_dtype(uint8) if self.dtype != uint8 else self

    def bool(self):
        return self._to_dtype(dtype_bool) if self.dtype != dtype_bool else self

    def type(self, dtype=None):
        if dtype is None:
            return f"torch.{self.dtype.name.capitalize()}Tensor"
        if isinstance(dtype, str):
            from ._dtype import from_name
            _type_map = {
                "torch.FloatTensor": float32,
                "torch.DoubleTensor": float64,
                "torch.HalfTensor": float16,
                "torch.BFloat16Tensor": bfloat16,
                "torch.LongTensor": int64,
                "torch.IntTensor": int32,
                "torch.ShortTensor": int16,
                "torch.CharTensor": int8,
                "torch.ByteTensor": uint8,
                "torch.BoolTensor": dtype_bool,
            }
            dt = _type_map.get(dtype) or from_name(dtype)
            if dt is None:
                raise RuntimeError(f"Unknown type: {dtype}")
            return self._to_dtype(dt)
        return self._to_dtype(dtype)

    def __add__(self, other):
        return add(self, other)

    def __mul__(self, other):
        return mul(self, other)

    def __neg__(self):
        return neg_dispatch(self)

    def matmul(self, other):
        return matmul(self, other)

    def relu(self):
        return relu(self)

    def abs(self):
        return abs_dispatch(self)

    def neg(self):
        return neg_dispatch(self)

    def exp(self):
        return exp_dispatch(self)

    def log(self):
        return log_dispatch(self)

    def sqrt(self):
        return sqrt_dispatch(self)

    def sin(self):
        return sin_dispatch(self)

    def cos(self):
        return cos_dispatch(self)

    def tan(self):
        return tan_dispatch(self)

    def tanh(self):
        return tanh_dispatch(self)

    def sigmoid(self):
        return sigmoid_dispatch(self)

    def floor(self):
        return floor_dispatch(self)

    def ceil(self):
        return ceil_dispatch(self)

    def round(self):
        return round_dispatch(self)

    def trunc(self):
        return trunc_dispatch(self)

    def frac(self):
        return frac_dispatch(self)

    def pow(self, exponent):
        return pow_dispatch(self, exponent)

    def log2(self):
        return log2_dispatch(self)

    def log10(self):
        return log10_dispatch(self)

    def exp2(self):
        return exp2_dispatch(self)

    def rsqrt(self):
        return rsqrt_dispatch(self)

    def sign(self):
        return sign_dispatch(self)

    def signbit(self):
        return signbit_dispatch(self)

    def isnan(self):
        return isnan_dispatch(self)

    def isinf(self):
        return isinf_dispatch(self)

    def isfinite(self):
        return isfinite_dispatch(self)

    def sinh(self):
        return sinh_dispatch(self)

    def cosh(self):
        return cosh_dispatch(self)

    def asinh(self):
        return asinh_dispatch(self)

    def acosh(self):
        return acosh_dispatch(self)

    def atanh(self):
        return atanh_dispatch(self)

    def erf(self):
        return erf_dispatch(self)

    def erfc(self):
        return erfc_dispatch(self)

    def softplus(self):
        return softplus_dispatch(self)

    def clamp(self, min_val=None, max_val=None):
        return clamp_dispatch(self, min_val, max_val)

    def clamp_min(self, min_val):
        return clamp_min_dispatch(self, min_val)

    def clamp_max(self, max_val):
        return clamp_max_dispatch(self, max_val)

    def relu6(self):
        return relu6_dispatch(self)

    def hardtanh(self, min_val=-1.0, max_val=1.0):
        return hardtanh_dispatch(self, min_val, max_val)

    def min(self, other):
        return min_dispatch(self, other)

    def max(self, other):
        return max_dispatch(self, other)

    def amin(self, dim=None, keepdim=False):
        return amin_dispatch(self, dim=dim, keepdim=keepdim)

    def amax(self, dim=None, keepdim=False):
        return amax_dispatch(self, dim=dim, keepdim=keepdim)

    def fmin(self, other):
        return fmin_dispatch(self, other)

    def fmax(self, other):
        return fmax_dispatch(self, other)

    def where(self, condition, other):
        return where_dispatch(condition, self, other)

    def atan(self):
        return atan_dispatch(self)

    def atan2(self, other):
        return atan2_dispatch(self, other)

    def asin(self):
        return asin_dispatch(self)

    def acos(self):
        return acos_dispatch(self)

    def lerp(self, other, weight):
        return lerp_dispatch(self, other, weight)

    def addcmul(self, tensor1, tensor2, value=1.0):
        return addcmul_dispatch(self, tensor1, tensor2, value=value)

    def addcdiv(self, tensor1, tensor2, value=1.0):
        return addcdiv_dispatch(self, tensor1, tensor2, value=value)

    def logaddexp(self, other):
        return logaddexp_dispatch(self, other)

    def logaddexp2(self, other):
        return logaddexp2_dispatch(self, other)

    def hypot(self, other):
        return hypot_dispatch(self, other)

    def remainder(self, other):
        return remainder_dispatch(self, other)

    def fmod(self, other):
        return fmod_dispatch(self, other)
    def sum(self, dim=None, keepdim=False):
        return sum(self, dim=dim, keepdim=keepdim)

    def all(self, dim=None, keepdim=False):
        return all_dispatch(self, dim=dim, keepdim=keepdim)

    def any(self, dim=None, keepdim=False):
        return any_dispatch(self, dim=dim, keepdim=keepdim)

    def argmax(self, dim=None, keepdim=False):
        return argmax_dispatch(self, dim=dim, keepdim=keepdim)

    def argmin(self, dim=None, keepdim=False):
        return argmin_dispatch(self, dim=dim, keepdim=keepdim)

    def count_nonzero(self, dim=None, keepdim=False):
        return count_nonzero_dispatch(self, dim=dim, keepdim=keepdim)

    def cumsum(self, dim=0):
        return cumsum_dispatch(self, dim)

    def cumprod(self, dim=0):
        return cumprod_dispatch(self, dim)

    def cummax(self, dim=0):
        return cummax_dispatch(self, dim)

    def allclose(self, other, rtol=1e-05, atol=1e-08, equal_nan=False):
        return allclose_dispatch(self, other, rtol=rtol, atol=atol, equal_nan=equal_nan)

    def isclose(self, other, rtol=1e-05, atol=1e-08, equal_nan=False):
        return isclose_dispatch(self, other, rtol=rtol, atol=atol, equal_nan=equal_nan)

    def equal(self, other):
        return equal_dispatch(self, other)

    def __getitem__(self, key):
        from ._dispatch.dispatcher import dispatch

        return dispatch("getitem", self.device.type, self, key)

    def __setitem__(self, key, value):
        from ._dispatch.dispatcher import dispatch

        self._check_inplace()
        dispatch("setitem", self.device.type, self, key, value)
        self._bump_version()

    def __repr__(self):
        return format_tensor(self)

    def __str__(self):
        return format_tensor(self)

    def __len__(self):
        if self.dim() == 0:
            raise TypeError("len() of a 0-d tensor")
        return self.shape[0]

    def __iter__(self):
        if self.dim() == 0:
            raise TypeError("iteration over a 0-d tensor")
        for i in range(len(self)):
            yield self[i]

    def __gt__(self, other):
        if isinstance(other, Tensor):
            if self.numel() == 1 and other.numel() == 1:
                return self.item() > other.item()
        return self.item() > other

    def __lt__(self, other):
        if isinstance(other, Tensor):
            if self.numel() == 1 and other.numel() == 1:
                return self.item() < other.item()
        return self.item() < other

    def __ge__(self, other):
        if isinstance(other, Tensor):
            if self.numel() == 1 and other.numel() == 1:
                return self.item() >= other.item()
        return self.item() >= other

    def __le__(self, other):
        if isinstance(other, Tensor):
            if self.numel() == 1 and other.numel() == 1:
                return self.item() <= other.item()
        return self.item() <= other

    def __eq__(self, other):
        if isinstance(other, Tensor):
            if self.numel() == 1 and other.numel() == 1:
                return self.item() == other.item()
        return self.item() == other

    def __ne__(self, other):
        if isinstance(other, Tensor):
            if self.numel() == 1 and other.numel() == 1:
                return self.item() != other.item()
        return self.item() != other
