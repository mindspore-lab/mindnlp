"""AscendC custom kernel integration for mindtorch_v2.

Provides:
- KernelLauncher — loads compiled AscendC .so and launches kernels
- AclnnCustomLauncher — loads custom OPP .so and runs ACLNN two-phase ops
- tensor_ptr() — extracts device memory pointer from a Tensor
- alloc_like() — allocates an output tensor with the same shape/dtype/device
- alloc_npu_tensor() — allocates an uninitialized NPU tensor with explicit shape/dtype
- tensor_to_acl() — creates ACL tensor descriptor from a mindtorch Tensor
- destroy_acl_tensor() — destroys an ACL tensor descriptor
- copy_h2d() — copies host bytes to a new device buffer, returns device pointer
- @ascendc_op — decorator that combines @custom_op with NPU-specific defaults
"""

import ctypes
import os


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


class AclnnCustomLauncher:
    """Load a custom OPP shared library and run ACLNN two-phase ops.

    Custom OPP packages (built via opdev) export pairs of symbols:
    - ``aclnn{OpName}GetWorkspaceSize`` — planning phase
    - ``aclnn{OpName}`` — execution phase

    This launcher handles the full two-phase call sequence: workspace
    sizing, workspace allocation, execution, and deferred cleanup of
    the executor and workspace memory.

    Args:
        library_path: Path to the compiled OPP .so (e.g. libcust_opapi.so).

    Example::

        launcher = AclnnCustomLauncher("/path/to/libcust_opapi.so")
        acl_x, kx = tensor_to_acl(x)
        acl_out, ko = tensor_to_acl(out)
        try:
            launcher.run("MyOp", [acl_x, acl_out])
        finally:
            destroy_acl_tensor(acl_x)
            destroy_acl_tensor(acl_out)
    """

    def __init__(self, library_path):
        from . import runtime as _npu_runtime
        _npu_runtime.get_runtime(0)
        from .aclnn import get_bindings as _get_bindings
        _get_bindings()

        _RTLD_LAZY = 0x00001
        self._lib = ctypes.CDLL(library_path, mode=_RTLD_LAZY | os.RTLD_GLOBAL)
        self._cache = {}

    def _get_symbols(self, op_name):
        """Bind and cache the GetWorkspaceSize + Execute pair for *op_name*.

        Returns:
            (get_ws_fn, exec_fn) — raw ctypes callables.
            The caller is responsible for passing correctly typed args
            to ``get_ws_fn``; ``exec_fn`` always has a fixed 4-arg
            signature (workspace, ws_size, executor, stream).
        """
        pair = self._cache.get(op_name)
        if pair is not None:
            return pair

        ws_name = f"aclnn{op_name}GetWorkspaceSize"
        exec_name = f"aclnn{op_name}"

        try:
            get_ws_fn = getattr(self._lib, ws_name)
        except AttributeError:
            raise RuntimeError(
                f"Symbol {ws_name!r} not found in {self._lib._name}. "
                f"Ensure the OPP package is compiled correctly."
            )
        try:
            exec_fn = getattr(self._lib, exec_name)
        except AttributeError:
            raise RuntimeError(
                f"Symbol {exec_name!r} not found in {self._lib._name}. "
                f"Ensure the OPP package is compiled correctly."
            )

        exec_fn.restype = ctypes.c_int32
        exec_fn.argtypes = [
            ctypes.c_void_p,   # workspace ptr
            ctypes.c_uint64,   # workspace size
            ctypes.c_void_p,   # executor
            ctypes.c_void_p,   # stream
        ]

        self._cache[op_name] = (get_ws_fn, exec_fn)
        return get_ws_fn, exec_fn

    def run(self, op_name, get_workspace_args, stream=None):
        """Execute a two-phase ACLNN custom op.

        Args:
            op_name: ACLNN op name without the ``aclnn`` prefix,
                e.g. ``"RopeEx"`` → calls ``aclnnRopeExGetWorkspaceSize``
                and ``aclnnRopeEx``.
            get_workspace_args: List of ctypes-compatible arguments for
                the GetWorkspaceSize call.  ``workspace_size`` and
                ``executor`` references are appended automatically.
            stream: ACL stream handle.  If None, uses the current
                stream for the default device.
        """
        from . import runtime as _npu_runtime
        from . import state as npu_state
        from .aclnn import _defer_executor, _maybe_sync

        runtime = _npu_runtime.get_runtime(0)
        if stream is None:
            stream = npu_state.current_stream().stream

        get_ws_fn, exec_fn = self._get_symbols(op_name)

        workspace_size = ctypes.c_uint64(0)
        executor = ctypes.c_void_p()
        workspace = None

        try:
            # Phase 1: GetWorkspaceSize
            full_args = list(get_workspace_args) + [
                ctypes.byref(workspace_size),
                ctypes.byref(executor),
            ]
            ret = get_ws_fn(*full_args)
            if ret != 0:
                raise RuntimeError(
                    f"aclnn{op_name}GetWorkspaceSize failed with error code {ret}"
                )

            # Allocate workspace if needed
            if workspace_size.value:
                from .acl_loader import ensure_acl
                acl = ensure_acl()
                workspace_ptr, ret = acl.rt.malloc(int(workspace_size.value), 0)
                if ret != 0:
                    raise RuntimeError(f"acl.rt.malloc failed: {ret}")
                workspace = workspace_ptr

            # Phase 2: Execute
            ret = exec_fn(
                ctypes.c_void_p(0 if workspace is None else int(workspace)),
                ctypes.c_uint64(workspace_size.value),
                executor,
                ctypes.c_void_p(int(stream)),
            )
            if ret != 0:
                raise RuntimeError(
                    f"aclnn{op_name} failed with error code {ret}"
                )
            _maybe_sync(runtime)
        finally:
            _defer_executor(executor)
            if workspace is not None:
                runtime.defer_raw_free(workspace)


def tensor_to_acl(t):
    """Create an ACL tensor descriptor from a mindtorch Tensor.

    Args:
        t: A mindtorch Tensor (must be on NPU device with valid storage).

    Returns:
        (acl_tensor_handle, keepalive_tuple) — the handle is a ctypes
        void pointer; *keepalive_tuple* must be kept alive while the
        handle is in use.
    """
    from .aclnn import get_bindings, _create_tensor

    bindings = get_bindings()
    return _create_tensor(
        bindings,
        t.shape,
        t.stride(),
        t.dtype,
        t.storage().data_ptr(),
    )


def destroy_acl_tensor(acl_tensor):
    """Destroy an ACL tensor descriptor previously created by ``tensor_to_acl``.

    Safe to call with None (no-op).
    """
    if acl_tensor is None:
        return
    from .aclnn import get_bindings

    bindings = get_bindings()
    bindings.acl_destroy_tensor(acl_tensor)


def alloc_npu_tensor(shape, dtype, device=None):
    """Allocate an uninitialized NPU tensor with explicit shape and dtype.

    Args:
        shape: Tuple of ints describing the tensor dimensions.
        dtype: Tensor dtype object (e.g. ``torch.float16``).
        device: Device string or Device object.  Defaults to ``"npu:0"``.

    Returns:
        A new Tensor with uninitialized storage.
    """
    from . import runtime as npu_runtime
    from ..._storage import npu_typed_storage_from_ptr
    from ..._tensor import Tensor
    from ..._device import device as Device

    if device is None:
        device = Device("npu", 0)
    elif isinstance(device, str):
        device = Device(device)

    device_id = device.index if hasattr(device, "index") and device.index is not None else 0
    runtime = npu_runtime.get_runtime(device_id)
    numel = 1
    for d in shape:
        numel *= d
    import numpy as np
    from ..._dtype import to_numpy_dtype
    nbytes = numel * np.dtype(to_numpy_dtype(dtype)).itemsize
    ptr = npu_runtime._alloc_device(nbytes, runtime=runtime)
    storage = npu_typed_storage_from_ptr(ptr, numel, dtype, device=device)
    stride = _contiguous_stride(shape)
    return Tensor(storage, tuple(shape), stride)


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
