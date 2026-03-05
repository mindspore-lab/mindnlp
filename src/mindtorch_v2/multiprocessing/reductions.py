from multiprocessing import reduction

from .. import multiprocessing as _mt_mp
from .._storage import _CPUUntypedStorage, TypedStorage
from .._tensor import Tensor


_NON_LEAF_ERR_MSG = (
    "Cowardly refusing to serialize non-leaf tensor which requires_grad, "
    "since autograd does not support crossing process boundaries.  "
    "If you just want to transfer the data, call detach() on the tensor "
    "before serializing (e.g., putting it on the queue)."
)


def _rebuild_cpu_storage_from_meta(meta):
    mechanism = meta["mechanism"]
    if mechanism == "file_system":
        return _CPUUntypedStorage.from_shared_memory(meta["filename"], meta["nbytes"])
    if mechanism == "file_descriptor":
        fd = meta.get("fd")
        if fd is None:
            dup_fd = meta.get("dup_fd")
            if dup_fd is None:
                raise RuntimeError("file_descriptor storage metadata missing fd/dup_fd")
            fd = dup_fd.detach()
        return _CPUUntypedStorage.from_shared_fd(fd, meta["nbytes"], filename=meta.get("filename"))
    raise RuntimeError(f"Unsupported CPU shared storage mechanism: {mechanism}")


def _reduce_cpu_storage(storage):
    strategy = _mt_mp.get_sharing_strategy()
    storage = storage.share_memory_(strategy=strategy)
    meta = storage.shared_memory_meta()
    if meta["mechanism"] == "file_descriptor":
        from multiprocessing.reduction import DupFd

        meta = dict(meta)
        meta["dup_fd"] = DupFd(meta["fd"])
        del meta["fd"]
    return (_rebuild_cpu_storage_from_meta, (meta,))


def _rebuild_typed_storage(untyped, dtype, size):
    data = untyped.typed_view(dtype, size)
    return TypedStorage(untyped, dtype=dtype, size=size, data=data)


def _reduce_typed_storage(storage):
    untyped = storage.untyped_storage()
    if isinstance(untyped, _CPUUntypedStorage):
        reduced = _reduce_cpu_storage(untyped)
        rebuild_untyped, args = reduced
        return (
            _rebuild_typed_storage_from_reduced_untyped,
            (rebuild_untyped, args, storage.dtype, storage.size()),
        )
    raise RuntimeError(f"Cannot serialize storage on device {storage.device}")


def _rebuild_typed_storage_from_reduced_untyped(rebuild_untyped, args, dtype, size):
    untyped = rebuild_untyped(*args)
    return _rebuild_typed_storage(untyped, dtype, size)


def _rebuild_tensor(storage, shape, stride, offset, requires_grad):
    t = Tensor(storage, shape, stride, offset=offset, requires_grad=requires_grad)
    if requires_grad:
        t.grad_fn = None
    return t


def _reduce_tensor(tensor):
    if tensor.requires_grad and tensor.grad_fn is not None:
        raise RuntimeError(_NON_LEAF_ERR_MSG)

    storage = tensor.storage()
    meta = (
        tuple(tensor.shape),
        tuple(tensor.stride),
        int(tensor.offset),
        bool(tensor.requires_grad),
    )
    reduced_storage = _reduce_typed_storage(storage)
    rebuild_storage, storage_args = reduced_storage
    return (_rebuild_tensor_from_reduced_storage, (rebuild_storage, storage_args, meta))


def _rebuild_tensor_from_reduced_storage(rebuild_storage, storage_args, meta):
    storage = rebuild_storage(*storage_args)
    shape, stride, offset, requires_grad = meta
    return _rebuild_tensor(storage, shape, stride, offset, requires_grad)


def init_reductions():
    reduction.register(_CPUUntypedStorage, _reduce_cpu_storage)
    reduction.register(TypedStorage, _reduce_typed_storage)
    reduction.register(Tensor, _reduce_tensor)


def non_leaf_requires_grad_error_message():
    return _NON_LEAF_ERR_MSG
