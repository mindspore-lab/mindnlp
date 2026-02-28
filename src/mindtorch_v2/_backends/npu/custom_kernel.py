"""AscendC custom kernel integration for mindtorch_v2.

Provides:
- KernelLauncher — loads compiled AscendC .so and launches kernels
- tensor_ptr() — extracts device memory pointer from a Tensor
- alloc_like() — allocates an output tensor with the same shape/dtype/device
- copy_h2d() — copies host bytes to a new device buffer, returns device pointer
- @ascendc_op — decorator that combines @custom_op with NPU-specific defaults
"""

import ctypes


class KernelLauncher:
    """Load a compiled AscendC shared library and launch kernels.

    Args:
        library_path: Path to the compiled .so file containing AscendC kernels.

    Example::

        launcher = KernelLauncher("/path/to/my_kernels.so")
        launcher.launch("add_custom", block_dim=8,
                        args=[tensor_ptr(x), tensor_ptr(y), tensor_ptr(out), x.numel()])
    """

    def __init__(self, library_path):
        # Ensure the ACL runtime is loaded first so symbols like
        # aclrtCtxGetCurrentDefaultStream / RegisterAscendBinary are
        # available.  RTLD_LAZY defers resolution; RTLD_GLOBAL exposes
        # runtime symbols to our kernel library.
        from . import runtime as _npu_runtime
        _npu_runtime.get_runtime(0)  # triggers ACL init

        import os
        _RTLD_LAZY = 0x00001  # POSIX RTLD_LAZY
        self._lib = ctypes.CDLL(library_path, mode=_RTLD_LAZY | os.RTLD_GLOBAL)
        self._cache = {}

    def _get_launch_fn(self, kernel_name):
        fn = self._cache.get(kernel_name)
        if fn is not None:
            return fn
        symbol = f"aclrtlaunch_{kernel_name}"
        try:
            fn = getattr(self._lib, symbol)
        except AttributeError:
            raise RuntimeError(
                f"Symbol {symbol!r} not found in {self._lib._name}. "
                f"Make sure the kernel is compiled with the correct entry point name."
            )
        self._cache[kernel_name] = fn
        return fn

    def launch(self, kernel_name, block_dim, args, stream=None):
        """Launch an AscendC kernel.

        Args:
            kernel_name: Entry point name (e.g. "add_custom").
            block_dim: Number of AI cores to use.
            args: List of kernel arguments. Tensors should be passed as
                  ``tensor_ptr(t)`` (int pointers). Scalars are converted
                  automatically.
            stream: ACL stream handle (int/ctypes pointer).  If None, uses
                    the current stream for the default device.
        """
        if stream is None:
            from . import state as npu_state
            stream = npu_state.current_stream().stream

        fn = self._get_launch_fn(kernel_name)

        # Convert Python args to ctypes
        c_args = []
        for a in args:
            if isinstance(a, int):
                c_args.append(ctypes.c_uint64(a))
            elif isinstance(a, float):
                c_args.append(ctypes.c_double(a))
            elif isinstance(a, ctypes._SimpleCData):
                c_args.append(a)
            else:
                c_args.append(ctypes.c_uint64(int(a)))

        # AscendC launch signature: fn(block_dim, stream, *args)
        ret = fn(ctypes.c_uint32(block_dim), stream, *c_args)
        if ret != 0:
            raise RuntimeError(
                f"AscendC kernel {kernel_name!r} launch failed with error code {ret}"
            )


def tensor_ptr(t):
    """Extract device memory pointer from a Tensor.

    Args:
        t: A Tensor (must be on NPU device).

    Returns:
        Integer device pointer.
    """
    return t.storage().data_ptr()


def alloc_like(t):
    """Allocate an output tensor with the same shape, dtype, and device.

    The tensor memory is allocated but not initialized.

    Args:
        t: A reference Tensor on NPU.

    Returns:
        A new Tensor with the same metadata but uninitialized storage.
    """
    from . import runtime as npu_runtime
    from ..._storage import npu_typed_storage_from_ptr
    from ..._tensor import Tensor

    device_id = (t.device.index if hasattr(t.device, "index") else None) or 0
    runtime = npu_runtime.get_runtime(device_id)
    nbytes = t.numel() * t.element_size()
    ptr = npu_runtime._alloc_device(nbytes, runtime=runtime)
    storage = npu_typed_storage_from_ptr(ptr, t.numel(), t.dtype, device=t.device)
    stride = _contiguous_stride(t.shape)
    return Tensor(storage, t.shape, stride)


def copy_h2d(data, device_id=0):
    """Copy host bytes to a newly allocated device buffer.

    Useful for uploading small host-side data (e.g. tiling parameters)
    to the NPU before a kernel launch.

    Args:
        data: A bytes-like object to copy to device memory.
        device_id: NPU device index (default 0).

    Returns:
        Integer device pointer to the newly allocated buffer.
    """
    import numpy as np
    from . import runtime as npu_runtime

    buf = bytes(data)
    arr = np.frombuffer(buf, dtype=np.uint8)
    runtime = npu_runtime.get_runtime(device_id)
    dev_ptr, _ = npu_runtime._copy_cpu_to_npu(arr, runtime=runtime)
    return dev_ptr


def _contiguous_stride(shape):
    stride = []
    acc = 1
    for d in reversed(shape):
        stride.append(acc)
        acc *= d
    return tuple(reversed(stride))


def ascendc_op(qualname, *, mutates_args=()):
    """Decorator to define an AscendC custom operator.

    Sugar over ``@custom_op(..., device_types="npu")`` with one addition:
    auto-generates a ``register_fake`` based on the return type hint
    (``empty_like`` for single Tensor return).  The user can override
    with an explicit ``@op.register_fake``.

    Usage::

        @ascendc_op("mylib::vector_add")
        def vector_add(x: Tensor, y: Tensor) -> Tensor:
            out = alloc_like(x)
            launcher.launch("add_custom", block_dim=8,
                            args=[tensor_ptr(x), tensor_ptr(y),
                                  tensor_ptr(out), x.numel()])
            return out

    Args:
        qualname: Qualified operator name, e.g. "mylib::vector_add".
        mutates_args: Tuple of argument names mutated in-place.

    Returns:
        CustomOpHandle wrapping the function.
    """
    from ...library import custom_op as _custom_op
    from ...library import _infer_schema, _return_annotation_to_schema
    from ..._dispatch.registry import registry
    from ..._dispatch.keys import DispatchKey
    import inspect

    def decorator(fn):
        handle = _custom_op(
            qualname, mutates_args=mutates_args, device_types="npu",
        )(fn)

        # Auto-register fake (Meta) kernel if return type is single Tensor
        sig = inspect.signature(fn)
        ret_schema = _return_annotation_to_schema(sig.return_annotation)
        entry = registry.get(qualname)
        if DispatchKey.Meta not in entry.kernels and ret_schema == "Tensor":
            # Default fake: empty_like the first Tensor argument
            def _auto_fake(*args, **kwargs):
                from ..._functional import empty_like
                for a in args:
                    if hasattr(a, "device") and hasattr(a, "dtype"):
                        return empty_like(a)
                raise RuntimeError("ascendc_op auto-fake requires at least one Tensor argument")
            registry.register_kernel(qualname, DispatchKey.Meta, _auto_fake)

        return handle

    return decorator
