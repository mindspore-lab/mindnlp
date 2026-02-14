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
from ._dtype import float32, to_numpy_dtype
from ._functional import add, mul, matmul, relu, sum, reshape as reshape_dispatch
from ._functional import transpose as transpose_dispatch, view as view_dispatch, to as to_dispatch
from ._autograd.engine import backward as _backward
from ._autograd.version_counter import VersionCounter
from ._printing import format_tensor


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
        """Check if tensor is of a floating point dtype."""
        floating_point_dtypes = {'float16', 'float32', 'float64', 'bfloat16',
                                 'half', 'float', 'double'}
        return str(self.dtype).split('.')[-1] in floating_point_dtypes

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
        result = dispatch("contiguous", self)

        # Track autograd if needed
        if self._requires_grad:
            from ._autograd import is_grad_enabled
            from ._autograd.node import AccumulateGrad
            from ._autograd.functions import ContiguousBackward

            if is_grad_enabled():
                grad_fn = ContiguousBackward()

                if self._grad_fn is not None:
                    grad_fn._next_functions = ((self._grad_fn, 0),)
                else:
                    acc_grad = AccumulateGrad(self)
                    grad_fn._next_functions = ((acc_grad, 0),)

                result._grad_fn = grad_fn
                result._requires_grad = True  # Propagate requires_grad

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

    def backward(self, gradient=None, retain_graph=False, create_graph=False):
        _backward(self, gradient, retain_graph=retain_graph, create_graph=create_graph)

    def pin_memory(self):
        if self.device.type != "cpu":
            raise RuntimeError("pin_memory only supports CPU tensors")
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
            raise RuntimeError("a leaf Variable that requires grad is being used in an in-place operation")
        if self._is_view() and self._base is not None and self._base.grad_fn is None and self._base.requires_grad:
            raise RuntimeError("a view of a leaf Variable that requires grad is being used in an in-place operation")

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

    def to(self, dev, non_blocking=False):
        if self._pending:
            from ._dispatch.pipeline import current_pipeline

            pipe = current_pipeline()
            if pipe is not None:
                pipe.flush()
        return to_dispatch(self, dev, non_blocking=non_blocking)

    def __add__(self, other):
        return add(self, other)

    def __mul__(self, other):
        return mul(self, other)

    def matmul(self, other):
        return matmul(self, other)

    def relu(self):
        return relu(self)

    def sum(self, dim=None, keepdim=False):
        return sum(self, dim=dim, keepdim=keepdim)

    def __repr__(self):
        return format_tensor(self)

    def __str__(self):
        return format_tensor(self)
