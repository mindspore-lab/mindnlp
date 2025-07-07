import sys
import traceback


from mindnlp import core

element_size_map = {
    core.float16: 2,
    core.float32: 4,
    core.float64: 8,
    core.bfloat16: 2,
    core.int64: 8,
    core.int32: 4,
    core.int16: 2,
    core.int8: 1,
    core.uint8: 1,
    core.bool: 1
}

def _element_size(dtype):
    return element_size_map[dtype]

def _flatten_dense_tensors(tensors):
    """Flatten dense tensors into a contiguous 1D buffer. Assume tensors are of
    same dense type.

    Since inputs are dense, the resulting tensor will be a concatenated 1D
    buffer. Element-wise operation on this buffer will be equivalent to
    operating individually.

    Args:
        tensors (Iterable[Tensor]): dense tensors to flatten.

    Returns:
        A contiguous 1D buffer containing input tensors.
    """
    tensors = [tensor.view(-1) for tensor in tensors]
    return core.cat(tensors)


def _unflatten_dense_tensors(flat, tensors):
    """View a flat buffer using the sizes of tensors. Assume that tensors are of
    same dense type, and that flat is given by _flatten_dense_tensors.

    Args:
        flat (Tensor): flattened dense tensors to unflatten.
        tensors (Iterable[Tensor]): dense tensors whose sizes will be used to
          unflatten flat.

    Returns:
        Unflattened dense tensors with sizes same as tensors and values from
        flat.
    """
    outputs = []
    offset = 0
    for tensor in tensors:
        numel = tensor.numel()
        if numel == 0:
            outputs.append(core.empty(0, flat.dtype))
        else:
            outputs.append(core.narrow(flat, 0, offset, numel).view(tensor.shape))
            offset += numel
    return outputs

def _rebuild_tensor_v2(
    storage,
    storage_offset,
    size,
    stride,
    requires_grad,
    backward_hooks,
    metadata=None,
):
    return core.Tensor(storage)

class KeyErrorMessage(str):
    r"""str subclass that returns itself in repr"""

    def __repr__(self):
        return self

class ExceptionWrapper:
    r"""Wraps an exception plus traceback to communicate across threads"""

    def __init__(self, exc_info=None, where="in background"):
        # It is important that we don't store exc_info, see
        # NOTE [ Python Traceback Reference Cycle Problem ]
        if exc_info is None:
            exc_info = sys.exc_info()
        self.exc_type = exc_info[0]
        self.exc_msg = "".join(traceback.format_exception(*exc_info))
        self.where = where

    def reraise(self):
        r"""Reraises the wrapped exception in the current thread"""
        # Format a message such as: "Caught ValueError in DataLoader worker
        # process 2. Original Traceback:", followed by the traceback.
        msg = f"Caught {self.exc_type.__name__} {self.where}.\nOriginal {self.exc_msg}"
        if self.exc_type == KeyError:
            # KeyError calls repr() on its argument (usually a dict key). This
            # makes stack traces unreadable. It will not be changed in Python
            # (https://bugs.python.org/issue2651), so we work around it.
            msg = KeyErrorMessage(msg)
        elif getattr(self.exc_type, "message", None):
            # Some exceptions have first argument as non-str but explicitly
            # have message field
            raise self.exc_type(message=msg)
        try:
            exception = self.exc_type(msg)
        except Exception:
            # If the exception takes multiple arguments or otherwise can't
            # be constructed, don't try to instantiate since we don't know how to
            raise RuntimeError(msg) from None
        raise exception

def set_device_address(tensor):
    core._prims.cpu.tensor_shape_cpu(tensor)


def _type(self, dtype=None, non_blocking=False, **kwargs):
    """Returns the type if `dtype` is not provided, else casts this object to
    the specified type.

    If this is already of the correct type, no copy is performed and the
    original object is returned.

    Args:
        dtype (type or string): The desired type
        non_blocking (bool): If ``True``, and the source is in pinned memory
            and destination is on the GPU or vice versa, the copy is performed
            asynchronously with respect to the host. Otherwise, the argument
            has no effect.
        **kwargs: For compatibility, may contain the key ``async`` in place of
            the ``non_blocking`` argument. The ``async`` arg is deprecated.
    """
    non_blocking = _get_async_or_non_blocking("type", non_blocking, kwargs)
    if dtype is None:
        return self.__module__ + "." + self.__class__.__name__

    if isinstance(dtype, str):
        dtype = _import_dotted_name(dtype)
    if dtype == type(self):
        return self
    if self.is_sparse:
        if not dtype.is_sparse:
            raise RuntimeError("Cannot cast sparse tensor to dense tensor")
        new_module_name = dtype.__module__.replace(".sparse", "")
        new_values_type_name = new_module_name + "." + dtype.__name__
        new_values = core.Tensor._values(self).type(new_values_type_name, non_blocking)
        new_indices_type_name = new_module_name + ".LongTensor"
        new_indices = core.Tensor._indices(self).type(
            new_indices_type_name, non_blocking
        )
        return dtype(new_indices, new_values, self.size())
    if dtype.is_sparse:
        raise RuntimeError("Cannot cast dense tensor to sparse tensor")
    return dtype(self.size()).copy_(self, non_blocking)

def _to(self, device, non_blocking=False):
    """Returns a copy of this object in device memory.

    If this object is already on the correct device, then no copy is performed
    and the original object is returned.

    Args:
        device (int): The destination device.
        non_blocking (bool): If ``True`` and the source is in pinned memory,
            the copy will be asynchronous with respect to the host. Otherwise,
            the argument has no effect.
    """
    if self.device == device:
        return self

    if device.type == "cpu":
        pin_memory = non_blocking and self.device.type in (
            "cuda",
            core._C._get_privateuse1_backend_name(),
        )
        untyped_storage = core.empty(
            self.nbytes(), dtype=core.uint8, device=device, pin_memory=pin_memory
        ).untyped_storage()
        untyped_storage.copy_(self, non_blocking)
        return untyped_storage

    device_module = getattr(mindtorch, device.type, None)
    assert device_module is not None, (
        f"{device.type.upper()} device module is not loaded"
    )
    with device_module.device(device):
        if self.is_sparse and hasattr(device_module, "sparse"):
            new_type = getattr(device_module.sparse, self.__class__.__name__)
            indices = getattr(core.Tensor._indices(self), device.type)(
                device, non_blocking
            )
            values = getattr(core.Tensor._values(self), device.type)(
                device, non_blocking
            )
            return new_type(indices, values, self.size())
        else:
            assert not self.is_sparse, (
                f"sparse storage is not supported for {device.type.upper()} tensors"
            )
            untyped_storage = core.UntypedStorage(self.size(), device=device)
            untyped_storage.copy_(self, non_blocking)
            return untyped_storage