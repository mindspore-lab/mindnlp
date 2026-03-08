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
from ._functional import add, mul, matmul, relu, sum, mean as mean_dispatch, std as std_dispatch, true_divide as true_divide_dispatch, repeat as repeat_dispatch, chunk as chunk_dispatch, split as split_dispatch, abs as abs_dispatch, neg as neg_dispatch
from ._functional import sub as sub_dispatch, div as div_dispatch
from ._functional import exp as exp_dispatch, log as log_dispatch, sqrt as sqrt_dispatch
from ._functional import sin as sin_dispatch, cos as cos_dispatch, tan as tan_dispatch
from ._functional import tanh as tanh_dispatch, sigmoid as sigmoid_dispatch
from ._functional import floor as floor_dispatch, ceil as ceil_dispatch, round as round_dispatch
from ._functional import trunc as trunc_dispatch, frac as frac_dispatch
from ._functional import pow as pow_dispatch, log2 as log2_dispatch, log10 as log10_dispatch
from ._functional import exp2 as exp2_dispatch, rsqrt as rsqrt_dispatch
from ._functional import sign as sign_dispatch, signbit as signbit_dispatch, square as square_dispatch
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
from ._functional import eq as eq_dispatch, ne as ne_dispatch, lt as lt_dispatch, le as le_dispatch, gt as gt_dispatch, ge as ge_dispatch
from ._functional import cumsum as cumsum_dispatch, cumprod as cumprod_dispatch, cummax as cummax_dispatch
from ._functional import argsort as argsort_dispatch, sort as sort_dispatch, topk as topk_dispatch
from ._functional import tril as tril_dispatch, triu as triu_dispatch, diag as diag_dispatch
from ._functional import reshape as reshape_dispatch
from ._functional import transpose as transpose_dispatch, view as view_dispatch, to as to_dispatch
from ._functional import nonzero as nonzero_dispatch, masked_select as masked_select_dispatch
from ._functional import gather as gather_dispatch, scatter as scatter_dispatch
from ._functional import index_select as index_select_dispatch, take as take_dispatch
from ._functional import narrow as narrow_dispatch, select as select_dispatch
from ._functional import expand as expand_dispatch
from ._functional import masked_fill_ as masked_fill__dispatch, masked_fill as masked_fill_dispatch
from ._functional import index_put_ as index_put__dispatch, index_put as index_put_dispatch
from ._functional import index_copy_ as index_copy__dispatch
from ._functional import index_fill_ as index_fill__dispatch
from ._functional import index_add_ as index_add__dispatch
from ._functional import scatter_ as scatter__dispatch, scatter_add_ as scatter_add__dispatch
from ._functional import masked_scatter_ as masked_scatter__dispatch
from ._functional import unfold as unfold_dispatch
from ._functional import squeeze as squeeze_dispatch, unsqueeze as unsqueeze_dispatch, permute as permute_dispatch
from ._functional import var as var_dispatch, norm as norm_dispatch, prod as prod_dispatch
from ._functional import mm as mm_dispatch, bmm as bmm_dispatch
from ._functional import floor_divide as floor_divide_dispatch
from ._functional import tile as tile_dispatch, flip as flip_dispatch, roll as roll_dispatch, rot90 as rot90_dispatch
from ._functional import reciprocal as reciprocal_dispatch, addmm as addmm_dispatch
from ._functional import log1p as log1p_dispatch, expm1 as expm1_dispatch
from ._autograd.engine import backward as _backward
from ._autograd.version_counter import VersionCounter
from ._printing import format_tensor


class _StrideTuple(tuple):
    """A tuple subclass that is also callable, matching PyTorch's stride() API.

    Supports both attribute access (t.stride) and method call (t.stride()),
    as well as per-dimension access (t.stride(dim)).
    """
    def __call__(self, dim=None):
        if dim is None:
            return tuple(self)
        if dim < 0:
            dim += len(self)
        return self[dim]


def _compute_strides(shape):
    stride = []
    acc = 1
    for d in reversed(shape):
        stride.append(acc)
        acc *= d
    return _StrideTuple(reversed(stride))


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
        self.stride = _StrideTuple(stride)
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

    @property
    def data(self):
        """Returns the underlying data tensor (detached from autograd graph)."""
        return self.detach()

    @data.setter
    def data(self, new_data):
        """Replace the tensor's data with new_data (in-place)."""
        if not isinstance(new_data, Tensor):
            raise TypeError(f"data must be a Tensor, got {type(new_data).__name__}")
        if new_data.shape != self.shape:
            raise RuntimeError(f"shape mismatch: expected {self.shape}, got {new_data.shape}")
        if new_data.dtype != self.dtype:
            raise RuntimeError(f"dtype mismatch: expected {self.dtype}, got {new_data.dtype}")
        # Replace storage
        self._storage = new_data._storage
        self.stride = new_data.stride
        self.offset = new_data.offset
        self._bump_version()

    @property
    def is_cuda(self):
        """Returns True if the tensor is stored on a CUDA GPU."""
        return self.device.type == "cuda"

    @property
    def is_cpu(self):
        """Returns True if the tensor is stored on CPU."""
        return self.device.type == "cpu"

    @property
    def is_npu(self):
        """Returns True if the tensor is stored on NPU."""
        return self.device.type == "npu"

    @property
    def is_meta(self):
        """Returns True if the tensor is a meta tensor."""
        return self.device.type == "meta"

    @property
    def is_leaf(self):
        """Returns True if this tensor is a leaf in the autograd graph."""
        return self.grad_fn is None

    @property
    def is_sparse(self):
        """Returns True if the tensor is sparse."""
        return False

    @property
    def is_quantized(self):
        """Returns True if the tensor is quantized."""
        return False

    def storage(self):
        return self._storage

    def storage_offset(self):
        """Returns the offset into the storage."""
        return self.offset

    def get_device(self):
        """Returns device index for GPU/NPU tensors, or -1 for CPU."""
        if self.device.type == "cpu":
            return -1
        return self.device.index if self.device.index is not None else 0

    def untyped_storage(self):
        """Return the underlying untyped storage.

        This is needed for compatibility with safetensors which calls
        tensor.untyped_storage().nbytes() to determine storage size.
        """
        return self._storage.untyped_storage()

    def dim(self):
        return len(self.shape)

    def ndimension(self):
        """Return the number of dimensions of the tensor (alias for dim())."""
        return len(self.shape)

    @property
    def ndim(self):
        return len(self.shape)

    def size(self, dim=None):
        if dim is None:
            return self.shape
        if dim < 0:
            dim += len(self.shape)
        if dim < 0 or dim >= len(self.shape):
            raise IndexError("Dimension out of range")
        return self.shape[dim]

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
        if self.device.type != "cpu":
            # Convert to CPU to get numpy view
            return self.to("cpu")._numpy_view()
        base = self._storage.data.ravel()
        itemsize = base.itemsize
        strides = tuple(s * itemsize for s in self.stride)
        return np.lib.stride_tricks.as_strided(
            base[self.offset:], shape=self.shape, strides=strides
        )

    def reshape(self, *shape):
        if not shape:
            raise TypeError("reshape() missing shape arguments")
        if len(shape) == 1 and isinstance(shape[0], (tuple, list)):
            shape = tuple(shape[0])
        return reshape_dispatch(self, shape)

    def view(self, *shape):
        if not shape:
            raise TypeError(
                "view() received an invalid combination of arguments - got (), but expected one of:\n"
                " * (torch.dtype dtype)\n"
                " * (tuple of ints size)\n"
            )
        if len(shape) == 1 and isinstance(shape[0], (tuple, list)):
            shape = shape[0]
        return view_dispatch(self, shape)

    def flatten(self, start_dim=0, end_dim=-1):
        ndim = len(self.shape)
        if ndim == 0:
            return self.reshape((1,))
        if start_dim < 0:
            start_dim += ndim
        if end_dim < 0:
            end_dim += ndim
        if start_dim < 0 or start_dim >= ndim:
            raise IndexError("Dimension out of range")
        if end_dim < 0 or end_dim >= ndim:
            raise IndexError("Dimension out of range")
        if start_dim > end_dim:
            raise RuntimeError("flatten() has invalid args: start_dim cannot come after end_dim")

        flattened = 1
        for d in self.shape[start_dim:end_dim + 1]:
            flattened *= d
        new_shape = self.shape[:start_dim] + (flattened,) + self.shape[end_dim + 1:]
        return self.reshape(new_shape)

    def transpose(self, dim0, dim1):
        return transpose_dispatch(self, dim0, dim1)

    def t(self):
        """Transpose for 2D tensors. Expects input to be <= 2-D tensor and transposes dimensions 0 and 1."""
        if len(self.shape) > 2:
            raise RuntimeError(f"t() expects a tensor with <= 2 dimensions, but self is {len(self.shape)}D")
        if len(self.shape) < 2:
            return self
        return self.transpose(0, 1)

    def t_(self):
        """In-place transpose for 2D tensors."""
        if len(self.shape) > 2:
            raise RuntimeError(f"t_() expects a tensor with <= 2 dimensions, but self is {len(self.shape)}D")
        if len(self.shape) < 2:
            return self
        # Swap shape and stride dimensions in-place
        self.shape = (self.shape[1], self.shape[0])
        self.stride = _StrideTuple((self.stride[1], self.stride[0]))
        return self

    @property
    def T(self):
        return self.t()

    def view_as(self, other):
        """Reshape this tensor to the same shape as other."""
        return self.view(other.shape)

    def new_empty(self, size, *, dtype=None, device=None, requires_grad=False):
        """Create a new empty tensor with the same dtype and device as self."""
        from ._creation import empty
        dt = dtype if dtype is not None else self.dtype
        dev = device if device is not None else self.device
        return empty(size, dtype=dt, device=dev)

    def new_tensor(self, data, *, dtype=None, device=None, requires_grad=False):
        """Create a new tensor with the given data using this tensor's dtype and device by default."""
        from ._creation import tensor
        dt = dtype if dtype is not None else self.dtype
        dev = device if device is not None else self.device
        return tensor(data, dtype=dt, device=dev)

    def new_empty_strided(self, size, stride, *, dtype=None, device=None, requires_grad=False):
        """Create a new empty tensor with the given size and stride."""
        dt = dtype if dtype is not None else self.dtype
        dev = device if device is not None else self.device
        if dev.type == "cpu":
            numel = 1
            for s in size:
                numel *= s
            arr = np.empty(numel, dtype=to_numpy_dtype(dt))
            storage = typed_storage_from_numpy(arr, dt, device=dev)
            return Tensor(storage, tuple(size), tuple(stride))
        else:
            from ._creation import empty
            t = empty(size, dtype=dt, device=dev)
            return t

    def as_strided(self, size, stride, storage_offset=None):
        """Create a view of the tensor with given size, stride, and storage_offset."""
        offset = storage_offset if storage_offset is not None else self.offset
        return Tensor(self._storage, tuple(size), tuple(stride), offset=offset,
                      requires_grad=self.requires_grad)

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
        if self.numel() == 0:
            raise RuntimeError("Boolean value of Tensor with no values is ambiguous")
        if self.numel() > 1:
            raise RuntimeError("Boolean value of Tensor with more than one value is ambiguous")
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

    def add_(self, other, *, alpha=1):
        from ._dispatch.dispatcher import dispatch

        self._check_inplace()
        if alpha != 1:
            other = mul(other, alpha)
        out = dispatch("add_", self.device.type, self, other)
        return out

    def add(self, other, *, alpha=1):
        return add(self, other, alpha=alpha)

    def sub(self, other, *, alpha=1):
        return sub_dispatch(self, other, alpha=alpha)

    def mul(self, other):
        return mul(self, other)

    def div(self, other, *, rounding_mode=None):
        return div_dispatch(self, other)

    def mul_(self, other):
        from ._dispatch.dispatcher import dispatch

        self._check_inplace()
        out = dispatch("mul_", self.device.type, self, other)
        return out

    def relu_(self):
        from ._dispatch.dispatcher import dispatch

        self._check_inplace()
        out = dispatch("relu_", self.device.type, self)
        return out

    def zero_(self):
        from ._dispatch.dispatcher import dispatch

        self._check_inplace()
        out = dispatch("zero_", self.device.type, self)
        return out

    def uniform_(self, low=0.0, high=1.0, *, generator=None):
        from ._dispatch.dispatcher import dispatch

        self._check_inplace()
        out = dispatch("uniform_", self.device.type, self, low, high, generator=generator)
        return out

    def normal_(self, mean=0.0, std=1.0, *, generator=None):
        from ._dispatch.dispatcher import dispatch

        self._check_inplace()
        out = dispatch("normal_", self.device.type, self, mean, std, generator=generator)
        return out

    def random_(self, from_=0, to=None, *, generator=None):
        from ._dispatch.dispatcher import dispatch

        self._check_inplace()
        out = dispatch("random_", self.device.type, self, from_, to, generator=generator)
        return out

    def randint_(self, low, high=None, *, generator=None):
        from ._dispatch.dispatcher import dispatch

        self._check_inplace()
        out = dispatch("randint_", self.device.type, self, low, high, generator=generator)
        return out

    def bernoulli_(self, p=0.5, *, generator=None):
        from ._dispatch.dispatcher import dispatch

        self._check_inplace()
        out = dispatch("bernoulli_", self.device.type, self, p, generator=generator)
        return out

    def exponential_(self, lambd=1.0, *, generator=None):
        from ._dispatch.dispatcher import dispatch

        self._check_inplace()
        out = dispatch("exponential_", self.device.type, self, lambd, generator=generator)
        return out

    def log_normal_(self, mean=1.0, std=2.0, *, generator=None):
        from ._dispatch.dispatcher import dispatch

        self._check_inplace()
        out = dispatch("log_normal_", self.device.type, self, mean, std, generator=generator)
        return out

    def cauchy_(self, median=0.0, sigma=1.0, *, generator=None):
        from ._dispatch.dispatcher import dispatch

        self._check_inplace()
        out = dispatch("cauchy_", self.device.type, self, median, sigma, generator=generator)
        return out

    def geometric_(self, p, *, generator=None):
        from ._dispatch.dispatcher import dispatch

        self._check_inplace()
        out = dispatch("geometric_", self.device.type, self, p, generator=generator)
        return out

    def fill_(self, value):
        from ._dispatch.dispatcher import dispatch

        self._check_inplace()
        out = dispatch("fill_", self.device.type, self, value)
        return out

    def clamp_(self, min=None, max=None):
        from ._dispatch.dispatcher import dispatch

        self._check_inplace()
        out = dispatch("clamp_", self.device.type, self, min, max)
        return out

    def copy_(self, src):
        from ._dispatch.dispatcher import dispatch

        self._check_inplace()
        out = dispatch("copy_", self.device.type, self, src)
        return out

    def erfinv_(self):
        from ._dispatch.dispatcher import dispatch

        self._check_inplace()
        out = dispatch("erfinv_", self.device.type, self)
        return out

    def sub_(self, other, *, alpha=1):
        from ._dispatch.dispatcher import dispatch

        self._check_inplace()
        if alpha != 1:
            other = mul(other, alpha)
        out = dispatch("sub_", self.device.type, self, other)
        return out

    def abs_(self):
        """In-place absolute value."""
        self._check_inplace()
        out = abs_dispatch(self)
        self._storage = out._storage
        self._bump_version()
        return self

    def neg_(self):
        """In-place negation."""
        self._check_inplace()
        out = neg_dispatch(self)
        self._storage = out._storage
        self._bump_version()
        return self

    def exp_(self):
        """In-place exponential."""
        self._check_inplace()
        out = exp_dispatch(self)
        self._storage = out._storage
        self._bump_version()
        return self

    def log_(self):
        """In-place natural logarithm."""
        self._check_inplace()
        out = log_dispatch(self)
        self._storage = out._storage
        self._bump_version()
        return self

    def log2_(self):
        """In-place base-2 logarithm."""
        self._check_inplace()
        out = log2_dispatch(self)
        self._storage = out._storage
        self._bump_version()
        return self

    def log10_(self):
        """In-place base-10 logarithm."""
        self._check_inplace()
        out = log10_dispatch(self)
        self._storage = out._storage
        self._bump_version()
        return self

    def sqrt_(self):
        """In-place square root."""
        self._check_inplace()
        out = sqrt_dispatch(self)
        self._storage = out._storage
        self._bump_version()
        return self

    def sin_(self):
        """In-place sine."""
        self._check_inplace()
        out = sin_dispatch(self)
        self._storage = out._storage
        self._bump_version()
        return self

    def cos_(self):
        """In-place cosine."""
        self._check_inplace()
        out = cos_dispatch(self)
        self._storage = out._storage
        self._bump_version()
        return self

    def tan_(self):
        """In-place tangent."""
        self._check_inplace()
        out = tan_dispatch(self)
        self._storage = out._storage
        self._bump_version()
        return self

    def tanh_(self):
        """In-place hyperbolic tangent."""
        self._check_inplace()
        out = tanh_dispatch(self)
        self._storage = out._storage
        self._bump_version()
        return self

    def sigmoid_(self):
        """In-place sigmoid."""
        self._check_inplace()
        out = sigmoid_dispatch(self)
        self._storage = out._storage
        self._bump_version()
        return self

    def floor_(self):
        """In-place floor."""
        self._check_inplace()
        out = floor_dispatch(self)
        self._storage = out._storage
        self._bump_version()
        return self

    def ceil_(self):
        """In-place ceiling."""
        self._check_inplace()
        out = ceil_dispatch(self)
        self._storage = out._storage
        self._bump_version()
        return self

    def round_(self):
        """In-place rounding."""
        self._check_inplace()
        out = round_dispatch(self)
        self._storage = out._storage
        self._bump_version()
        return self

    def trunc_(self):
        """In-place truncation."""
        self._check_inplace()
        out = trunc_dispatch(self)
        self._storage = out._storage
        self._bump_version()
        return self

    def pow_(self, exponent):
        """In-place power."""
        self._check_inplace()
        out = pow_dispatch(self, exponent)
        self._storage = out._storage
        self._bump_version()
        return self

    def reciprocal_(self):
        """In-place reciprocal."""
        self._check_inplace()
        out = reciprocal_dispatch(self)
        self._storage = out._storage
        self._bump_version()
        return self

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
        copy = kwargs.get("copy", False)
        memory_format = kwargs.get("memory_format", None)
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
            result = to_dispatch(
                result,
                device,
                dtype=dtype,
                non_blocking=non_blocking,
                copy=copy,
                memory_format=memory_format,
            )
        if result is self and dtype is None and device is None:
            return self
        return result

    def cpu(self, memory_format=None):
        if memory_format is None:
            return self.to("cpu")
        return self.to("cpu", memory_format=memory_format)

    def npu(self, device=None, non_blocking=False, memory_format=None):
        if device is None:
            device = "npu"
        return self.to(device, non_blocking=non_blocking, memory_format=memory_format)

    def cuda(self, device=None, non_blocking=False, memory_format=None):
        if device is None:
            target = "cuda"
        elif isinstance(device, str):
            target = device
        else:
            target = f"cuda:{int(device)}"
        return self.to(target, non_blocking=non_blocking, memory_format=memory_format)

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
        elif self.device.type == "npu":
            from ._backends.npu import ops as npu_ops
            from ._backends.npu import runtime as npu_runtime
            from ._backends.npu import state as npu_state

            runtime = npu_runtime.get_runtime((self.device.index or 0))
            stream = npu_state.current_stream((self.device.index or 0))

            # Allocate output buffer
            from ._backends.npu.ops import _numel, _dtype_itemsize, _unwrap_storage, _wrap_tensor
            out_size = _numel(self.shape) * _dtype_itemsize(dtype)
            out_ptr = npu_runtime._alloc_device(out_size, runtime=runtime)

            # Call aclnnCast
            from ._backends.npu import aclnn
            self_storage = _unwrap_storage(self)
            aclnn.cast(
                self_storage.data_ptr(),
                out_ptr,
                self.shape,
                self.stride,
                self.dtype,
                dtype,
                runtime,
                stream=stream.stream,
            )

            # Wrap result
            storage = npu_typed_storage_from_ptr(out_ptr, _numel(self.shape), dtype, device=self.device)
            return _wrap_tensor(storage, self.shape, self.stride)
        elif self.device.type == "mps":
            from ._storage import mps_typed_storage_from_numpy
            arr = self._numpy_view()
            src_dtype = self.dtype
            target_np = to_numpy_dtype(dtype)
            if src_dtype == bfloat16:
                arr = _bf16_to_f32(arr)
            if dtype == bfloat16:
                arr = arr.astype(np.float32)
                arr = _f32_to_bf16(arr)
            else:
                arr = arr.astype(target_np)
            storage = mps_typed_storage_from_numpy(
                np.ascontiguousarray(arr), dtype, device=self.device
            )
            stride = tuple(np.array(arr.strides) // arr.itemsize) if arr.ndim > 0 else ()
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

    def new_ones(self, size, dtype=None, device=None):
        from ._creation import ones

        if dtype is None:
            dtype = self.dtype
        if device is None:
            device = self.device
        return ones(size, dtype=dtype, device=device)

    def new_zeros(self, size, dtype=None, device=None):
        from ._creation import zeros

        if dtype is None:
            dtype = self.dtype
        if device is None:
            device = self.device
        return zeros(size, dtype=dtype, device=device)

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

    def __sub__(self, other):
        if isinstance(other, Tensor):
            return add(self, neg_dispatch(other))
        return add(self, -other)

    def __rsub__(self, other):
        return add(neg_dispatch(self), other)

    def __mul__(self, other):
        return mul(self, other)

    def __rmul__(self, other):
        return mul(self, other)

    def __truediv__(self, other):
        return true_divide_dispatch(self, other)

    def __rtruediv__(self, other):
        return true_divide_dispatch(other, self)

    def __pow__(self, exponent):
        return pow_dispatch(self, exponent)

    def __rpow__(self, base):
        from ._dispatch.dispatcher import dispatch
        return dispatch("pow", self.device.type, base, self)

    def __floordiv__(self, other):
        return floor_divide_dispatch(self, other)

    def __rfloordiv__(self, other):
        from ._dispatch.dispatcher import dispatch
        return dispatch("floor_divide", self.device.type, other, self)

    def __mod__(self, other):
        return remainder_dispatch(self, other)

    def __rmod__(self, other):
        from ._dispatch.dispatcher import dispatch
        return dispatch("remainder", self.device.type, other, self)

    def __iadd__(self, other):
        self._check_inplace()
        self.add_(other)
        return self

    def __isub__(self, other):
        self._check_inplace()
        self.sub_(other)
        return self

    def __imul__(self, other):
        self._check_inplace()
        self.mul_(other)
        return self

    def __itruediv__(self, other):
        self._check_inplace()
        self.div_(other)
        return self

    def __neg__(self):
        return neg_dispatch(self)

    def clone(self):
        from ._functional import to as to_dispatch

        return to_dispatch(self, self.device, copy=True)

    def matmul(self, other):
        return matmul(self, other)

    def __matmul__(self, other):
        return matmul(self, other)

    def __rmatmul__(self, other):
        return matmul(other, self)

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

    def tril(self, diagonal=0):
        return tril_dispatch(self, diagonal)

    def triu(self, diagonal=0):
        return triu_dispatch(self, diagonal)

    def diag(self, diagonal=0):
        return diag_dispatch(self, diagonal)

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

    def square(self):
        return square_dispatch(self)

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

    def clamp_(self, min=None, max=None):
        self._check_inplace()
        out = clamp_dispatch(self, min, max)
        return self.copy_(out)

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

    def squeeze(self, dim=None):
        return squeeze_dispatch(self, dim)

    def unsqueeze(self, dim):
        return unsqueeze_dispatch(self, dim)

    def permute(self, *dims):
        if len(dims) == 1 and isinstance(dims[0], (tuple, list)):
            dims = tuple(dims[0])
        return permute_dispatch(self, dims)

    def var(self, dim=None, keepdim=False, unbiased=True):
        return var_dispatch(self, dim=dim, keepdim=keepdim, unbiased=unbiased)

    def norm(self, p=2, dim=None, keepdim=False):
        return norm_dispatch(self, p=p, dim=dim, keepdim=keepdim)

    def prod(self, dim=None, keepdim=False):
        return prod_dispatch(self, dim=dim, keepdim=keepdim)

    def mm(self, mat2):
        return mm_dispatch(self, mat2)

    def bmm(self, batch2):
        return bmm_dispatch(self, batch2)
    def sum(self, dim=None, keepdim=False, *, dtype=None):
        return sum(self, dim=dim, keepdim=keepdim, dtype=dtype)

    def mean(self, dim=None, keepdim=False, *, dtype=None, axis=None):
        if axis is not None:
            dim = axis
        return mean_dispatch(self, dim=dim, keepdim=keepdim, dtype=dtype)

    def std(self, dim=None, keepdim=False, unbiased=True, axis=None):
        if axis is not None:
            dim = axis
        return std_dispatch(self, dim=dim, keepdim=keepdim, unbiased=unbiased)

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

    def argsort(self, dim=-1, descending=False, stable=False):
        return argsort_dispatch(self, dim=dim, descending=descending, stable=stable)

    def sort(self, dim=-1, descending=False, stable=False):
        return sort_dispatch(self, dim=dim, descending=descending, stable=stable)

    def topk(self, k, dim=-1, largest=True, sorted=True):
        return topk_dispatch(self, k, dim=dim, largest=largest, sorted=sorted)

    def split(self, split_size_or_sections, dim=0):
        return split_dispatch(self, split_size_or_sections, dim=dim)

    def chunk(self, chunks, dim=0):
        return chunk_dispatch(self, chunks, dim=dim)

    def repeat(self, *repeats):
        if len(repeats) == 1 and isinstance(repeats[0], (tuple, list)):
            repeats = tuple(repeats[0])
        return repeat_dispatch(self, repeats)

    def tile(self, *dims):
        if len(dims) == 1 and isinstance(dims[0], (tuple, list)):
            dims = tuple(dims[0])
        return tile_dispatch(self, dims)

    def flip(self, dims):
        if isinstance(dims, int):
            dims = [dims]
        return flip_dispatch(self, dims)

    def roll(self, shifts, dims=None):
        return roll_dispatch(self, shifts, dims)

    def rot90(self, k=1, dims=(0, 1)):
        return rot90_dispatch(self, k, dims)

    def reciprocal(self):
        return reciprocal_dispatch(self)

    def log1p(self):
        """Returns a new tensor with the natural logarithm of (1 + input)."""
        return log1p_dispatch(self)

    def expm1(self):
        """Returns a new tensor with the exponential of the elements minus 1."""
        return expm1_dispatch(self)

    def logsumexp(self, dim, keepdim=False):
        """Returns the log of summed exponentials of each row of the input tensor in the given dimension dim."""
        from ._dispatch.dispatcher import dispatch
        return dispatch("logsumexp", self.device.type, self, dim, keepdim)

    def trace(self):
        """Returns the sum of the elements of the diagonal of the input 2-D matrix."""
        from ._dispatch.dispatcher import dispatch
        return dispatch("trace", self.device.type, self)

    def det(self):
        """Returns the determinant of a square matrix."""
        from ._dispatch.dispatcher import dispatch
        return dispatch("det", self.device.type, self)

    def matrix_power(self, n):
        """Returns the matrix raised to the power n for square matrices."""
        from ._dispatch.dispatcher import dispatch
        return dispatch("matrix_power", self.device.type, self, n)

    def dist(self, other, p=2):
        """Returns the p-norm of (self - other)."""
        from ._dispatch.dispatcher import dispatch
        return dispatch("dist", self.device.type, self, other, p)

    def renorm(self, p, dim, maxnorm):
        """Returns a tensor where each sub-tensor along dimension dim is normalized."""
        from ._dispatch.dispatcher import dispatch
        return dispatch("renorm", self.device.type, self, p, dim, maxnorm)

    def nansum(self, dim=None, keepdim=False):
        """Returns the sum of all elements, treating NaNs as zero."""
        from ._dispatch.dispatcher import dispatch
        return dispatch("nansum", self.device.type, self, dim, keepdim)

    def nanmean(self, dim=None, keepdim=False):
        """Returns the mean of all elements, treating NaNs as zero."""
        from ._dispatch.dispatcher import dispatch
        return dispatch("nanmean", self.device.type, self, dim, keepdim)

    def argwhere(self):
        """Returns a tensor containing the indices of all non-zero elements."""
        from ._dispatch.dispatcher import dispatch
        return dispatch("argwhere", self.device.type, self)

    def addmm(self, mat1, mat2, *, beta=1, alpha=1):
        return addmm_dispatch(self, mat1, mat2, beta=beta, alpha=alpha)

    def baddbmm(self, batch1, batch2, *, beta=1, alpha=1):
        """Performs a batch matrix-matrix product with added input.

        out = beta * self + alpha * (batch1 @ batch2)

        Args:
            batch1: First batch of matrices (B x N x M)
            batch2: Second batch of matrices (B x M x P)
            beta: Multiplier for self (default: 1)
            alpha: Multiplier for batch1 @ batch2 (default: 1)

        Returns:
            Tensor of shape (B x N x P)
        """
        from ._dispatch.dispatcher import dispatch
        return dispatch("baddbmm", self.device.type, self, batch1, batch2, beta=beta, alpha=alpha)

    def type_as(self, other):
        return self.to(other.dtype)

    def reshape_as(self, other):
        return self.reshape(other.shape)

    def new_full(self, size, fill_value, *, dtype=None, device=None, requires_grad=False):
        from ._creation import full
        dt = dtype if dtype is not None else self.dtype
        dev = device if device is not None else self.device
        return full(size, fill_value, dtype=dt, device=dev)

    def div_(self, other):
        from ._dispatch.dispatcher import dispatch
        self._check_inplace()
        out = dispatch("div_", self.device.type, self, other)
        return out

    def unflatten(self, dim, sizes):
        ndim = len(self.shape)
        if dim < 0:
            dim += ndim
        new_shape = self.shape[:dim] + tuple(sizes) + self.shape[dim + 1:]
        return self.view(new_shape)

    # -----------------------------------------------------------------------
    # Indexing / selection methods
    # -----------------------------------------------------------------------

    def narrow(self, dim, start, length):
        return narrow_dispatch(self, dim, start, length)

    def select(self, dim, index):
        return select_dispatch(self, dim, index)

    def expand(self, *sizes):
        return expand_dispatch(self, *sizes)

    def expand_as(self, other):
        return expand_dispatch(self, *other.shape)

    def nonzero(self, as_tuple=False):
        return nonzero_dispatch(self, as_tuple=as_tuple)

    def masked_select(self, mask):
        return masked_select_dispatch(self, mask)

    def gather(self, dim, index):
        return gather_dispatch(self, dim, index)

    def scatter(self, dim, index, src):
        return scatter_dispatch(self, dim, index, src)

    def scatter_(self, dim, index, src):
        self._check_inplace()
        return scatter__dispatch(self, dim, index, src)

    def scatter_add_(self, dim, index, src):
        self._check_inplace()
        return scatter_add__dispatch(self, dim, index, src)

    def index_select(self, dim, index):
        return index_select_dispatch(self, dim, index)

    def take(self, index):
        return take_dispatch(self, index)

    def masked_fill(self, mask, value):
        return masked_fill_dispatch(self, mask, value)

    def masked_fill_(self, mask, value):
        self._check_inplace()
        return masked_fill__dispatch(self, mask, value)

    def masked_scatter_(self, mask, source):
        self._check_inplace()
        return masked_scatter__dispatch(self, mask, source)

    def index_put_(self, indices, values, accumulate=False):
        self._check_inplace()
        return index_put__dispatch(self, indices, values, accumulate)

    def index_put(self, indices, values, accumulate=False):
        return index_put_dispatch(self, indices, values, accumulate)

    def index_copy_(self, dim, index, source):
        self._check_inplace()
        return index_copy__dispatch(self, dim, index, source)

    def index_fill_(self, dim, index, value):
        self._check_inplace()
        return index_fill__dispatch(self, dim, index, value)

    def index_add_(self, dim, index, source, alpha=1.0):
        self._check_inplace()
        return index_add__dispatch(self, dim, index, source, alpha)

    def unfold(self, dimension, size, step):
        return unfold_dispatch(self, dimension, size, step)

    def allclose(self, other, rtol=1e-05, atol=1e-08, equal_nan=False):
        return allclose_dispatch(self, other, rtol=rtol, atol=atol, equal_nan=equal_nan)

    def isclose(self, other, rtol=1e-05, atol=1e-08, equal_nan=False):
        return isclose_dispatch(self, other, rtol=rtol, atol=atol, equal_nan=equal_nan)

    def equal(self, other):
        return equal_dispatch(self, other)

    def eq(self, other):
        return self.__eq__(other)

    def ne(self, other):
        return self.__ne__(other)

    def lt(self, other):
        """Element-wise less-than comparison."""
        return lt_dispatch(self, other)

    def le(self, other):
        """Element-wise less-than-or-equal comparison."""
        return le_dispatch(self, other)

    def gt(self, other):
        """Element-wise greater-than comparison."""
        return gt_dispatch(self, other)

    def ge(self, other):
        """Element-wise greater-than-or-equal comparison."""
        return ge_dispatch(self, other)

    def logical_and(self, other):
        """Element-wise logical AND."""
        from ._dispatch.dispatcher import dispatch
        return dispatch("logical_and", self.device.type, self, other)

    def logical_or(self, other):
        """Element-wise logical OR."""
        from ._dispatch.dispatcher import dispatch
        return dispatch("logical_or", self.device.type, self, other)

    def logical_xor(self, other):
        """Element-wise logical XOR."""
        from ._dispatch.dispatcher import dispatch
        return dispatch("logical_xor", self.device.type, self, other)

    def logical_not(self):
        """Element-wise logical NOT."""
        from ._dispatch.dispatcher import dispatch
        return dispatch("logical_not", self.device.type, self)

    def bitwise_and(self, other):
        """Element-wise bitwise AND."""
        from ._dispatch.dispatcher import dispatch
        return dispatch("bitwise_and", self.device.type, self, other)

    def bitwise_or(self, other):
        """Element-wise bitwise OR."""
        from ._dispatch.dispatcher import dispatch
        return dispatch("bitwise_or", self.device.type, self, other)

    def bitwise_xor(self, other):
        """Element-wise bitwise XOR."""
        from ._dispatch.dispatcher import dispatch
        return dispatch("bitwise_xor", self.device.type, self, other)

    def bitwise_not(self):
        """Element-wise bitwise NOT."""
        from ._dispatch.dispatcher import dispatch
        return dispatch("bitwise_not", self.device.type, self)

    def bitwise_and_(self, other):
        """In-place element-wise bitwise AND."""
        self._check_inplace()
        from ._dispatch.dispatcher import dispatch
        out = dispatch("bitwise_and", self.device.type, self, other)
        self._storage = out._storage
        self.shape = out.shape
        self.stride = out.stride
        self._bump_version()
        return self

    def bitwise_or_(self, other):
        """In-place element-wise bitwise OR."""
        self._check_inplace()
        from ._dispatch.dispatcher import dispatch
        out = dispatch("bitwise_or", self.device.type, self, other)
        self._storage = out._storage
        self.shape = out.shape
        self.stride = out.stride
        self._bump_version()
        return self

    def bitwise_xor_(self, other):
        """In-place element-wise bitwise XOR."""
        self._check_inplace()
        from ._dispatch.dispatcher import dispatch
        out = dispatch("bitwise_xor", self.device.type, self, other)
        self._storage = out._storage
        self.shape = out.shape
        self.stride = out.stride
        self._bump_version()
        return self

    def movedim(self, source, destination):
        """Move dimensions to new positions."""
        from ._dispatch.dispatcher import dispatch
        return dispatch("movedim", self.device.type, self, source, destination)

    def moveaxis(self, source, destination):
        """Alias for movedim."""
        return self.movedim(source, destination)

    def swapdims(self, dim0, dim1):
        """Swap two dimensions (alias for transpose with positional args)."""
        return self.transpose(dim0, dim1)

    def swapaxes(self, axis0, axis1):
        """Alias for swapdims."""
        return self.swapdims(axis0, axis1)

    def diagonal(self, offset=0, dim1=0, dim2=1):
        """Returns partial view of input with the diagonal elements of input."""
        from ._dispatch.dispatcher import dispatch
        return dispatch("diagonal", self.device.type, self, offset, dim1, dim2)

    def unbind(self, dim=0):
        """Remove a tensor dimension, returning a tuple of all slices along dim."""
        from ._dispatch.dispatcher import dispatch
        return dispatch("unbind", self.device.type, self, dim)

    def vsplit(self, split_size_or_sections):
        """Split a tensor into multiple sub-tensors vertically (row-wise)."""
        from ._dispatch.dispatcher import dispatch
        return dispatch("vsplit", self.device.type, self, split_size_or_sections)

    def hsplit(self, split_size_or_sections):
        """Split a tensor into multiple sub-tensors horizontally (column-wise)."""
        from ._dispatch.dispatcher import dispatch
        return dispatch("hsplit", self.device.type, self, split_size_or_sections)

    def dsplit(self, split_size_or_sections):
        """Split a tensor into multiple sub-tensors along the third axis."""
        from ._dispatch.dispatcher import dispatch
        return dispatch("dsplit", self.device.type, self, split_size_or_sections)

    def take_along_dim(self, indices, dim):
        """Take values along an axis at the given indices."""
        from ._dispatch.dispatcher import dispatch
        return dispatch("take_along_dim", self.device.type, self, indices, dim)

    def scatter_add(self, dim, index, src):
        """Non-inplace scatter_add: adds all values from src into self at index positions."""
        out = self.clone()
        out.scatter_add_(dim, index, src)
        return out

    def index_fill(self, dim, index, value):
        """Non-inplace version of index_fill_: fills self tensor with value along dim at index."""
        out = self.clone()
        out.index_fill_(dim, index, value)
        return out

    def index_copy(self, dim, index, source):
        """Non-inplace version of index_copy_: copies values from source into self along dim."""
        out = self.clone()
        out.index_copy_(dim, index, source)
        return out

    def index_add(self, dim, index, source, alpha=1):
        """Non-inplace version of index_add_: adds values from source (scaled by alpha) into self along dim."""
        out = self.clone()
        out.index_add_(dim, index, source, alpha)
        return out

    def put_(self, indices, values, accumulate=False):
        """Copies elements from values into self tensor at positions specified by indices.

        Treats self as a flat (1-D) tensor and uses flat indices.
        """
        self._check_inplace()
        # Work on a contiguous version for flat indexing
        if not self.is_contiguous():
            cont = self.contiguous()
            self._storage = cont._storage
            self.stride = cont.stride
        numel_idx = indices.numel()
        shape = self.shape
        for i in range(numel_idx):
            idx = int(indices.reshape((numel_idx,))[i].item())
            val = values.reshape((numel_idx,))[i]
            # Calculate multi-dim index from flat index
            multi_idx = []
            tmp = idx
            for d in reversed(shape):
                multi_idx.append(tmp % d)
                tmp //= d
            multi_idx = tuple(reversed(multi_idx))
            if accumulate:
                self[multi_idx] = self[multi_idx] + val
            else:
                self[multi_idx] = val
        self._bump_version()
        return self

    def cummin(self, dim):
        """Returns a namedtuple (values, indices) of cumulative minimum of elements along dim."""
        from ._dispatch.dispatcher import dispatch
        return dispatch("cummin", self.device.type, self, dim)

    def __getitem__(self, key):
        from ._dispatch.dispatcher import dispatch

        return dispatch("getitem", self.device.type, self, key)

    def __setitem__(self, key, value):
        from ._dispatch.dispatcher import dispatch

        self._check_inplace()
        dispatch("setitem", self.device.type, self, key, value)

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

    @staticmethod
    def _is_scalar_comparable(other):
        return isinstance(other, (int, float, bool))

    def __gt__(self, other):
        if isinstance(other, Tensor) or self._is_scalar_comparable(other):
            return gt_dispatch(self, other)
        return NotImplemented

    def __lt__(self, other):
        if isinstance(other, Tensor) or self._is_scalar_comparable(other):
            return lt_dispatch(self, other)
        return NotImplemented

    def __ge__(self, other):
        if isinstance(other, Tensor) or self._is_scalar_comparable(other):
            return ge_dispatch(self, other)
        return NotImplemented

    def __le__(self, other):
        if isinstance(other, Tensor) or self._is_scalar_comparable(other):
            return le_dispatch(self, other)
        return NotImplemented

    def __eq__(self, other):
        if isinstance(other, Tensor) or self._is_scalar_comparable(other):
            return eq_dispatch(self, other)
        return False

    def __ne__(self, other):
        if isinstance(other, Tensor) or self._is_scalar_comparable(other):
            return ne_dispatch(self, other)
        return True

    def __and__(self, other):
        return mul(self.bool(), other.bool() if isinstance(other, Tensor) else bool(other))

    def __or__(self, other):
        return add(self.bool(), other.bool() if isinstance(other, Tensor) else bool(other))

    def __xor__(self, other):
        return ne_dispatch(self.bool(), other.bool() if isinstance(other, Tensor) else bool(other))

    def __hash__(self):
        return id(self)
