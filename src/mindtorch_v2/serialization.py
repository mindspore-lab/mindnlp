"""Torch-compatible serialization without runtime torch dependency.

This module implements a subset of the PyTorch zip checkpoint format that is
sufficient for common checkpoint paths (state_dict, optimizer state, nested
containers) while keeping mindtorch_v2 save/load independent from torch
runtime imports.
"""

import io
import os
import pickle
import sys
import zipfile
from collections import OrderedDict

import numpy as np

from ._dtype import (
    bool as mt_bool,
    bfloat16,
    complex64,
    complex128,
    float16,
    float32,
    float64,
    int8,
    int16,
    int32,
    int64,
    uint8,
    to_numpy_dtype,
)
from ._storage import typed_storage_from_numpy
from ._tensor import Tensor as MindTensor


def _check_filelike_for_read(f):
    if isinstance(f, (str, os.PathLike)):
        return
    if not hasattr(f, "read"):
        raise AttributeError(
            "expected 'f' to be string, path, or a file-like object with a 'read' attribute"
        )


def _check_filelike_for_write(f):
    if isinstance(f, (str, os.PathLike)):
        return
    if not hasattr(f, "write"):
        raise AttributeError(
            "expected 'f' to be string, path, or a file-like object with a 'write' attribute"
        )


def _is_pathlike(f):
    return isinstance(f, (str, os.PathLike))


def _maybe_decode_ascii(value):
    if isinstance(value, bytes):
        return value.decode("ascii")
    return value


# Storage type proxies that get rewritten to torch globals in data.pkl.
class FloatStorage:  # pragma: no cover - marker type used by pickle
    pass


class DoubleStorage:  # pragma: no cover - marker type used by pickle
    pass


class HalfStorage:  # pragma: no cover - marker type used by pickle
    pass


class BFloat16Storage:  # pragma: no cover - marker type used by pickle
    pass


class LongStorage:  # pragma: no cover - marker type used by pickle
    pass


class IntStorage:  # pragma: no cover - marker type used by pickle
    pass


class ShortStorage:  # pragma: no cover - marker type used by pickle
    pass


class CharStorage:  # pragma: no cover - marker type used by pickle
    pass


class ByteStorage:  # pragma: no cover - marker type used by pickle
    pass


class BoolStorage:  # pragma: no cover - marker type used by pickle
    pass


class ComplexFloatStorage:  # pragma: no cover - marker type used by pickle
    pass


class ComplexDoubleStorage:  # pragma: no cover - marker type used by pickle
    pass


# Force global names emitted in pickle to be stable and patchable.
for _cls in (
    FloatStorage,
    DoubleStorage,
    HalfStorage,
    BFloat16Storage,
    LongStorage,
    IntStorage,
    ShortStorage,
    CharStorage,
    ByteStorage,
    BoolStorage,
    ComplexFloatStorage,
    ComplexDoubleStorage,
):
    _cls.__module__ = __name__


_DTYPE_NAME_TO_STORAGE = {
    "float32": FloatStorage,
    "float64": DoubleStorage,
    "float16": HalfStorage,
    "bfloat16": BFloat16Storage,
    "int64": LongStorage,
    "int32": IntStorage,
    "int16": ShortStorage,
    "int8": CharStorage,
    "uint8": ByteStorage,
    "bool": BoolStorage,
    "complex64": ComplexFloatStorage,
    "complex128": ComplexDoubleStorage,
}

_STORAGE_NAME_TO_DTYPE = {
    "FloatStorage": float32,
    "DoubleStorage": float64,
    "HalfStorage": float16,
    "BFloat16Storage": bfloat16,
    "LongStorage": int64,
    "IntStorage": int32,
    "ShortStorage": int16,
    "CharStorage": int8,
    "ByteStorage": uint8,
    "BoolStorage": mt_bool,
    "ComplexFloatStorage": complex64,
    "ComplexDoubleStorage": complex128,
}


class _StorageRef:
    __slots__ = ("storage_type", "key", "location", "numel", "raw_bytes")

    def __init__(self, storage_type, key, location, numel, raw_bytes):
        self.storage_type = storage_type
        self.key = str(key)
        self.location = location
        self.numel = int(numel)
        self.raw_bytes = raw_bytes


class _LegacyStorageView:
    __slots__ = ("storage", "base_offset")

    def __init__(self, storage, base_offset):
        self.storage = storage
        self.base_offset = int(base_offset)


class _TensorReduceProxy:
    __slots__ = ("storage_ref", "storage_offset", "size", "stride", "requires_grad")

    def __init__(self, storage_ref, storage_offset, size, stride, requires_grad):
        self.storage_ref = storage_ref
        self.storage_offset = int(storage_offset)
        self.size = tuple(size)
        self.stride = tuple(stride)
        self.requires_grad = bool(requires_grad)

    def __reduce_ex__(self, _protocol):
        # Match torch tensor pickle path: _rebuild_tensor_v2(storage, offset, size, stride, ...)
        return (
            _rebuild_tensor_v2,
            (
                self.storage_ref,
                self.storage_offset,
                self.size,
                self.stride,
                self.requires_grad,
                OrderedDict(),
            ),
        )


def _rebuild_tensor_v2(storage, storage_offset, size, stride, requires_grad, _backward_hooks, _metadata=None):
    offset = int(storage_offset)
    if isinstance(storage, _LegacyStorageView):
        offset += storage.base_offset
        storage = storage.storage

    tensor = MindTensor(
        storage,
        tuple(size),
        tuple(stride),
        offset=offset,
        requires_grad=bool(requires_grad),
    )
    if not requires_grad:
        tensor.grad_fn = None
    return tensor


def _tensor_to_proxy(tensor, storage_refs_by_id):
    cpu_tensor = tensor.detach().to("cpu") if tensor.device.type != "cpu" else tensor.detach()
    storage = cpu_tensor.storage()
    untyped = storage.untyped_storage()
    storage_id = id(untyped)
    storage_ref = storage_refs_by_id.get(storage_id)
    if storage_ref is None:
        storage_type = _DTYPE_NAME_TO_STORAGE.get(cpu_tensor.dtype.name)
        if storage_type is None:
            raise TypeError(f"unsupported dtype for serialization: {cpu_tensor.dtype}")
        raw = np.ascontiguousarray(storage.data).tobytes()
        storage_ref = _StorageRef(
            storage_type=storage_type,
            key=str(len(storage_refs_by_id)),
            location="cpu",
            numel=int(storage.size()),
            raw_bytes=raw,
        )
        storage_refs_by_id[storage_id] = storage_ref
    return _TensorReduceProxy(
        storage_ref=storage_ref,
        storage_offset=int(cpu_tensor.offset),
        size=tuple(int(s) for s in cpu_tensor.shape),
        stride=tuple(int(s) for s in cpu_tensor.stride),
        requires_grad=bool(tensor.requires_grad),
    )


def _prepare_for_pickle(obj, storage_refs_by_id):
    if isinstance(obj, MindTensor):
        return _tensor_to_proxy(obj, storage_refs_by_id)
    if isinstance(obj, OrderedDict):
        return OrderedDict((k, _prepare_for_pickle(v, storage_refs_by_id)) for k, v in obj.items())
    if isinstance(obj, dict):
        return {k: _prepare_for_pickle(v, storage_refs_by_id) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_prepare_for_pickle(v, storage_refs_by_id) for v in obj]
    if isinstance(obj, tuple):
        return tuple(_prepare_for_pickle(v, storage_refs_by_id) for v in obj)
    if isinstance(obj, set):
        return {_prepare_for_pickle(v, storage_refs_by_id) for v in obj}
    return obj


def _persistent_id(obj):
    if isinstance(obj, _StorageRef):
        return ("storage", obj.storage_type, obj.key, obj.location, obj.numel)
    return None


def _patch_pickle_globals_for_torch(data_bytes):
    patched = data_bytes
    local_mod = __name__

    # Rebuild op must resolve to torch._utils on torch.load.
    patched = patched.replace(
        f"{local_mod}\n_rebuild_tensor_v2\n".encode("utf-8"),
        b"torch._utils\n_rebuild_tensor_v2\n",
    )

    # Storage classes must resolve to torch.*Storage.
    for storage_name in _STORAGE_NAME_TO_DTYPE.keys():
        patched = patched.replace(
            f"{local_mod}\n{storage_name}\n".encode("utf-8"),
            f"torch\n{storage_name}\n".encode("utf-8"),
        )

    return patched


def _write_zip_checkpoint(obj, f, pickle_module, pickle_protocol):
    storage_refs_by_id = {}
    prepared = _prepare_for_pickle(obj, storage_refs_by_id)

    data_buf = io.BytesIO()
    pickler = pickle_module.Pickler(data_buf, protocol=pickle_protocol)
    pickler.persistent_id = _persistent_id
    pickler.dump(prepared)
    data_pkl = _patch_pickle_globals_for_torch(data_buf.getvalue())

    prefix = "archive"
    with zipfile.ZipFile(f, mode="w", compression=zipfile.ZIP_STORED) as zf:
        zf.writestr(f"{prefix}/data.pkl", data_pkl)
        zf.writestr(f"{prefix}/version", b"3")
        zf.writestr(f"{prefix}/byteorder", sys.byteorder.encode("ascii"))

        refs = sorted(storage_refs_by_id.values(), key=lambda r: int(r.key))
        for ref in refs:
            zf.writestr(f"{prefix}/data/{ref.key}", ref.raw_bytes)


def _detect_prefix(zf):
    for name in zf.namelist():
        if name.endswith("/data.pkl"):
            return name[: -len("/data.pkl")]
    if "data.pkl" in zf.namelist():
        return ""
    raise RuntimeError("checkpoint missing data.pkl record")


def _record_name(prefix, name):
    if not prefix:
        return name
    return f"{prefix}/{name}"


def _storage_dtype_from_type(storage_type):
    name = getattr(storage_type, "__name__", None)
    if name is None:
        name = str(storage_type)
    dtype = _STORAGE_NAME_TO_DTYPE.get(name)
    if dtype is None:
        raise TypeError(f"unsupported storage type in checkpoint: {storage_type}")
    return dtype


class _TorchCompatUnpickler(pickle.Unpickler):
    def find_class(self, mod_name, name):
        if mod_name == "torch._utils" and name in {"_rebuild_tensor_v2", "_rebuild_tensor"}:
            return _rebuild_tensor_v2
        if mod_name == "torch" and name in _STORAGE_NAME_TO_DTYPE:
            return globals()[name]
        if mod_name == "collections" and name == "OrderedDict":
            return OrderedDict
        return super().find_class(mod_name, name)




def _apply_map_location(storage, location, map_location):
    if map_location is None or map_location == "cpu":
        return storage

    if isinstance(map_location, dict):
        mapped = map_location.get(location, location)
        if mapped not in ("cpu", None):
            raise NotImplementedError(
                f"unsupported remapped location: {mapped}; only cpu is supported"
            )
        return storage

    if callable(map_location):
        remapped = map_location(storage, location)
        if remapped is None:
            return storage
        return remapped

    raise NotImplementedError(
        "mindtorch_v2.load supports map_location=None, 'cpu', dict, or callable for torch zip checkpoints"
    )


def _validate_map_location(map_location):
    if map_location in (None, "cpu"):
        return
    if isinstance(map_location, dict):
        for _, mapped in map_location.items():
            if mapped not in ("cpu", None):
                raise NotImplementedError(
                    f"unsupported remapped location: {mapped}; only cpu is supported"
                )
        return
    if callable(map_location):
        return
    raise NotImplementedError(
        "mindtorch_v2.load supports map_location=None, 'cpu', dict, or callable for torch zip checkpoints"
    )


def _load_zip_checkpoint(file_obj, map_location=None, **pickle_load_args):
    _validate_map_location(map_location)

    loaded_storages = {}
    with zipfile.ZipFile(file_obj, mode="r") as zf:
        prefix = _detect_prefix(zf)
        data_pkl = zf.read(_record_name(prefix, "data.pkl"))

        def persistent_load(saved_id):
            assert isinstance(saved_id, tuple)
            typename = _maybe_decode_ascii(saved_id[0])
            if typename != "storage":
                raise RuntimeError(
                    f"Unknown typename for persistent_load, expected 'storage' but got '{typename}'"
                )

            storage_type, key, location, numel = saved_id[1:]
            key = _maybe_decode_ascii(key)
            location = _maybe_decode_ascii(location)
            if location not in ("cpu", None):
                raise NotImplementedError(
                    f"unsupported checkpoint storage location: {location}; only cpu is supported"
                )

            if key in loaded_storages:
                return loaded_storages[key]

            dtype = _storage_dtype_from_type(storage_type)
            payload = zf.read(_record_name(prefix, f"data/{key}"))
            np_dtype = to_numpy_dtype(dtype)
            arr = np.frombuffer(payload, dtype=np_dtype, count=int(numel)).copy()
            storage = typed_storage_from_numpy(arr, dtype=dtype, device="cpu")
            storage = _apply_map_location(storage, location, map_location)
            loaded_storages[key] = storage
            return storage

        unpickler = _TorchCompatUnpickler(io.BytesIO(data_pkl), **pickle_load_args)
        unpickler.persistent_load = persistent_load
        return unpickler.load()




def _legacy_element_size(dtype):
    return int(np.dtype(to_numpy_dtype(dtype)).itemsize)


def _load_legacy_checkpoint(file_obj, map_location=None, **pickle_load_args):
    if map_location not in (None, "cpu"):
        raise NotImplementedError(
            "mindtorch_v2.load currently supports map_location=None or 'cpu' for legacy checkpoints"
        )

    deserialized_objects = {}

    class _LegacyUnpickler(pickle.Unpickler):
        def find_class(self, mod_name, name):
            if mod_name == "torch._utils" and name in {"_rebuild_tensor_v2", "_rebuild_tensor"}:
                return _rebuild_tensor_v2
            if mod_name == "torch" and name in _STORAGE_NAME_TO_DTYPE:
                return globals()[name]
            if mod_name == "collections" and name == "OrderedDict":
                return OrderedDict
            return super().find_class(mod_name, name)

    def persistent_load(saved_id):
        assert isinstance(saved_id, tuple)
        typename = _maybe_decode_ascii(saved_id[0])
        data = saved_id[1:]

        if typename == "module":
            return data[0]

        if typename == "storage":
            storage_type, root_key, location, numel, view_metadata = data
            root_key = _maybe_decode_ascii(root_key)
            location = _maybe_decode_ascii(location)
            if location not in ("cpu", None):
                raise NotImplementedError(
                    f"unsupported checkpoint storage location: {location}; only cpu is supported"
                )

            dtype = _storage_dtype_from_type(storage_type)
            if root_key not in deserialized_objects:
                arr = np.empty(int(numel), dtype=to_numpy_dtype(dtype))
                deserialized_objects[root_key] = typed_storage_from_numpy(arr, dtype=dtype, device="cpu")
            root_storage = deserialized_objects[root_key]

            if view_metadata is not None:
                view_key, offset, _view_size = view_metadata
                view_key = _maybe_decode_ascii(view_key)
                if view_key not in deserialized_objects:
                    deserialized_objects[view_key] = _LegacyStorageView(root_storage, int(offset))
                return deserialized_objects[view_key]
            return root_storage

        raise RuntimeError(f"Unknown saved id type: {saved_id[0]}")

    magic_number = pickle.load(file_obj, **pickle_load_args)
    if magic_number != 0x1950A86A20F9469CFC6C:
        raise RuntimeError("Invalid magic number; corrupt file?")

    protocol_version = pickle.load(file_obj, **pickle_load_args)
    if protocol_version != 1001:
        raise RuntimeError(f"Invalid protocol version: {protocol_version}")

    _ = pickle.load(file_obj, **pickle_load_args)

    unpickler = _LegacyUnpickler(file_obj, **pickle_load_args)
    unpickler.persistent_load = persistent_load
    result = unpickler.load()

    deserialized_storage_keys = pickle.load(file_obj, **pickle_load_args)

    for key in deserialized_storage_keys:
        key = _maybe_decode_ascii(key)
        storage = deserialized_objects[key]
        if isinstance(storage, _LegacyStorageView):
            storage = storage.storage

        # Legacy stream stores an 8-byte record header before each raw storage payload.
        header = file_obj.read(8)
        if len(header) != 8:
            raise RuntimeError("corrupt legacy checkpoint: missing storage record header")

        nbytes = storage.nbytes()
        payload = file_obj.read(nbytes)
        if len(payload) != nbytes:
            raise RuntimeError("corrupt legacy checkpoint: truncated storage payload")
        arr = np.frombuffer(payload, dtype=storage.data.dtype, count=storage.size()).copy()
        storage.data[:] = arr

    return result


def _is_zip_checkpoint(file_obj):
    try:
        cur = file_obj.tell()
    except Exception:
        cur = None

    try:
        return zipfile.is_zipfile(file_obj)
    finally:
        if cur is not None:
            try:
                file_obj.seek(cur)
            except Exception:
                pass


def save(obj, f, pickle_module=pickle, pickle_protocol=2, **kwargs):
    """Save object in torch-compatible zip checkpoint format without torch import."""
    _check_filelike_for_write(f)
    _ = kwargs

    if _is_pathlike(f):
        with open(f, "wb") as fh:
            _write_zip_checkpoint(obj, fh, pickle_module, pickle_protocol)
        return

    _write_zip_checkpoint(obj, f, pickle_module, pickle_protocol)


def load(f, map_location=None, pickle_module=pickle, *, weights_only=False, **kwargs):
    """Load checkpoint without torch runtime dependency.

    Supports torch zip checkpoints and the zip checkpoints produced by
    :func:`save`. Falls back to plain pickle for older mindtorch_v2 baseline
    files if needed.
    """
    _check_filelike_for_read(f)
    _ = pickle_module, weights_only, kwargs

    if _is_pathlike(f):
        with open(f, "rb") as fh:
            if _is_zip_checkpoint(fh):
                return _load_zip_checkpoint(fh, map_location=map_location, encoding="utf-8")
            return _load_legacy_checkpoint(fh, map_location=map_location, encoding="utf-8")

    if _is_zip_checkpoint(f):
        return _load_zip_checkpoint(f, map_location=map_location, encoding="utf-8")
    return _load_legacy_checkpoint(f, map_location=map_location, encoding="utf-8")
