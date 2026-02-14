import numpy as np

from ..._device import device as Device
from ..._storage import (
    empty_cpu_typed_storage,
    meta_typed_storage_from_shape,
    npu_typed_storage_from_ptr,
    typed_storage_from_numpy,
)
from ..._tensor import Tensor


def to_device(a, dev, non_blocking=False):
    if isinstance(dev, str):
        dev = Device(dev)
    if a.device.type == dev.type and a.device.index == dev.index:
        return a
    if a.device.type == "meta":
        if dev.type == "meta":
            return a
        if dev.type == "cpu":
            storage = empty_cpu_typed_storage(a.shape, a.dtype, device=dev)
            return Tensor(storage, a.shape, a.stride, a.offset, a.requires_grad)
        if dev.type == "npu":
            from ..npu import runtime as npu_runtime

            runtime = npu_runtime.get_runtime(dev.index or 0)
            size = int(np.prod(a.shape)) * np.dtype(npu_runtime._dtype_to_numpy(a.dtype)).itemsize
            ptr = npu_runtime._alloc_device(size, runtime=runtime)
            storage = npu_typed_storage_from_ptr(ptr, int(np.prod(a.shape)), a.dtype, device=dev)
            return Tensor(storage, a.shape, a.stride, a.offset, a.requires_grad)
        raise NotImplementedError(f"Unsupported device: {dev}")
    if a.device.type == "cpu" and dev.type == "meta":
        storage = meta_typed_storage_from_shape(a.shape, a.dtype, device=dev)
        return Tensor(storage, a.shape, a.stride, a.offset, a.requires_grad)
    if a.device.type == "cpu" and dev.type == "npu":
        from ..npu import runtime as npu_runtime

        runtime = npu_runtime.get_runtime(dev.index or 0)
        arr = a.storage().data
        do_non_blocking = bool(non_blocking)
        if do_non_blocking:
            from ... import npu as npu_api

            do_non_blocking = npu_api.is_pinned(a)
        stream = None
        if do_non_blocking:
            from ..npu import state as npu_state

            stream = npu_state.current_stream(dev.index or 0).stream
        ptr, _ = npu_runtime._copy_cpu_to_npu(arr, runtime=runtime, non_blocking=do_non_blocking, stream=stream)
        storage = npu_typed_storage_from_ptr(ptr, arr.size, a.dtype, device=dev)
        return Tensor(storage, a.shape, a.stride, a.offset, a.requires_grad)

    if a.device.type == "npu" and dev.type == "npu":
        from ..npu import runtime as npu_runtime

        src_runtime = npu_runtime.get_runtime(a.device.index or 0)
        src_runtime.synchronize()
        arr = npu_runtime._copy_npu_to_cpu(
            a.storage().data_ptr(), a.storage().nbytes(), a.shape, a.dtype, runtime=src_runtime
        )
        dst_runtime = npu_runtime.get_runtime(dev.index or 0)
        ptr, _ = npu_runtime._copy_cpu_to_npu(arr, runtime=dst_runtime)
        storage = npu_typed_storage_from_ptr(ptr, arr.size, a.dtype, device=dev)
        return Tensor(storage, a.shape, a.stride, a.offset, a.requires_grad)

    if a.device.type == "npu" and dev.type == "cpu":
        from ..npu import runtime as npu_runtime

        runtime = npu_runtime.get_runtime(a.device.index or 0)
        do_non_blocking = bool(non_blocking)
        stream = None
        if do_non_blocking:
            from ..npu import state as npu_state

            stream = npu_state.current_stream(a.device.index or 0).stream
        arr = npu_runtime._copy_npu_to_cpu(
            a.storage().data_ptr(),
            a.storage().nbytes(),
            a.shape,
            a.dtype,
            runtime=runtime,
            non_blocking=do_non_blocking,
            stream=stream,
        )
        storage = typed_storage_from_numpy(arr, a.dtype, device=dev)
        out = Tensor(storage, a.shape, a.stride, a.offset, a.requires_grad)
        if do_non_blocking:
            from ... import npu as npu_api

            npu_api.pin_memory(out)
        return out
    if a.device.type == "npu" and dev.type == "meta":
        storage = meta_typed_storage_from_shape(a.shape, a.dtype, device=dev)
        return Tensor(storage, a.shape, a.stride, a.offset, a.requires_grad)
    raise NotImplementedError(f"Unsupported device: {a.device} -> {dev}")
