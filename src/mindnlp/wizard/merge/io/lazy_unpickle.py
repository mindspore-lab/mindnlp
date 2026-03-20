# Copyright (C) 2025 Arcee AI
# SPDX-License-Identifier: LGPL-3.0-only
# Modified for MindSpore/Ascend NPU by MindNLP Wizard contributors.
#

"""
Lazy unpickler for PyTorch ``.bin`` checkpoint files.

Reads the pickle metadata without materialising tensors.  When a tensor
is actually needed, raw bytes are read from the ZIP archive and
reconstructed as a ``mindspore.Tensor`` via NumPy.
"""

import codecs
import collections
import contextlib
import logging
import operator
import os
import pickle
import zipfile
from functools import reduce
from typing import Any, Dict, Optional, Tuple, Union

import mindspore
import numpy

from ..dtype_policy import numpy_to_mindspore
from ._device import move_tensor_to_device

LOG = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# dtype helpers — map PyTorch storage class names to numpy dtypes
# ---------------------------------------------------------------------------

_STORAGE_DTYPE_MAP = {
    "DoubleStorage": numpy.float64,
    "FloatStorage": numpy.float32,
    "HalfStorage": numpy.float16,
    "LongStorage": numpy.int64,
    "IntStorage": numpy.int32,
    "ShortStorage": numpy.int16,
    "CharStorage": numpy.int8,
    "ByteStorage": numpy.uint8,
    "BoolStorage": numpy.bool_,
    "BFloat16Storage": "bfloat16",
}

_NUMPY_DTYPE_BYTES = {
    numpy.float64: 8,
    numpy.float32: 4,
    numpy.float16: 2,
    numpy.int64: 8,
    numpy.int32: 4,
    numpy.int16: 2,
    numpy.int8: 1,
    numpy.uint8: 1,
    numpy.bool_: 1,
    "bfloat16": 2,
}


def _resolve_numpy_dtype(name: str):
    """Return the numpy dtype (or ``'bfloat16'`` sentinel) for a storage name."""
    return _STORAGE_DTYPE_MAP.get(name, numpy.float32)


def _element_size(np_dtype) -> int:
    return _NUMPY_DTYPE_BYTES.get(np_dtype, 4)


# ---------------------------------------------------------------------------
# Placeholder storage types used during unpickling
# ---------------------------------------------------------------------------

class _StoragePlaceholder:
    """Stand-in for ``torch.*Storage`` during pickle loading."""
    def __init__(self, np_dtype):
        self.np_dtype = np_dtype
        self.dtype = np_dtype


def _make_storage_placeholder(name: str):
    np_dt = _resolve_numpy_dtype(name)
    return _StoragePlaceholder(np_dt)


_STORAGE_PLACEHOLDERS = {
    name: _make_storage_placeholder(name)
    for name in _STORAGE_DTYPE_MAP
}


# ---------------------------------------------------------------------------
# DeferredLoad
# ---------------------------------------------------------------------------

class DeferredLoad:
    """Describes a not-yet-materialised tensor inside a ``.bin`` archive."""

    def __init__(self, name: str, location: str, np_dtype):
        self.name = name
        self.location = location
        self.np_dtype = np_dtype
        self.file_offset: Optional[int] = None
        self.shape: Optional[Tuple[int, ...]] = None
        self.stride: Optional[Tuple[int, ...]] = None
        self.requires_grad = False

    @staticmethod
    def rebuild(
        load: "DeferredLoad",
        offset: int,
        shape: Union[Tuple[int, ...], Any],
        stride: Tuple[int, ...],
    ) -> "DeferredLoad":
        load.shape = tuple(shape)
        load.stride = tuple(stride)
        load.file_offset = offset * _element_size(load.np_dtype)
        return load

    def execute(
        self,
        reader: "TorchArchiveReader",
        map_location: Any = None,
    ) -> mindspore.Tensor:
        if self.shape is None or self.stride is None or self.file_offset is None:
            raise RuntimeError(
                f"DeferredLoad for '{self.name}' was not fully initialised"
            )

        total_params = reduce(operator.mul, self.shape, 1)
        elem_sz = _element_size(self.np_dtype)
        total_bytes = total_params * elem_sz

        f = reader.open_file(file_name=self.name, offset=self.file_offset)
        raw = f.read(total_bytes)

        if self.np_dtype == "bfloat16":
            import ml_dtypes  # noqa: F811
            arr = numpy.frombuffer(raw, dtype=ml_dtypes.bfloat16).copy()
        else:
            arr = numpy.frombuffer(raw, dtype=self.np_dtype).copy()

        arr = arr.reshape(self.shape)
        tensor = numpy_to_mindspore(arr)
        return move_tensor_to_device(tensor, map_location, caller="LazyPickleLoader")


# ---------------------------------------------------------------------------
# Custom unpickler
# ---------------------------------------------------------------------------

def _rebuild_tensor_v2_placeholder(storage, offset, shape, stride, *extra):
    # PyTorch may pass additional trailing args such as requires_grad,
    # backward_hooks, metadata, etc. We only need the storage/offset/shape/stride
    # information to reconstruct a DeferredLoad placeholder.
    load = DeferredLoad.rebuild(storage, offset, shape, stride)
    if extra:
        try:
            load.requires_grad = bool(extra[0])
        except Exception as exc:
            LOG.debug(
                "Failed to parse requires_grad flag from pickle metadata (%s: %s)",
                type(exc).__name__,
                exc,
            )
    return load


ACCEPTABLE_TYPES = {
    ("torch._utils", "_rebuild_tensor_v2"): _rebuild_tensor_v2_placeholder,
    ("collections", "OrderedDict"): collections.OrderedDict,
    ("numpy.core.multiarray", "scalar"): numpy.core.multiarray.scalar,
    ("numpy", "dtype"): numpy.core.multiarray.scalar,
    ("_codecs", "encode"): codecs.encode,
}
for _sname, _placeholder in _STORAGE_PLACEHOLDERS.items():
    ACCEPTABLE_TYPES[("torch", _sname)] = _placeholder


class LazyTorchUnpickler(pickle.Unpickler):
    def find_class(self, module: str, name: str) -> Any:
        if (module, name) in ACCEPTABLE_TYPES:
            return ACCEPTABLE_TYPES[(module, name)]
        raise pickle.UnpicklingError(f"Unsupported type {module}.{name}")

    def persistent_load(self, pid: Any) -> Any:
        if not isinstance(pid, tuple) or pid[0] != "storage":
            raise RuntimeError(
                f"Unpickling object with unexpected PID: {repr(pid)}"
            )
        storage_type, key, location, _ = pid[1:]
        if isinstance(storage_type, _StoragePlaceholder):
            np_dtype = storage_type.np_dtype
        else:
            np_dtype = numpy.float32
        return DeferredLoad(name=key, location=location, np_dtype=np_dtype)


class LazyUnpickleModule:
    """Drop-in ``pickle_module`` for ``torch.load`` replacement."""
    Unpickler = LazyTorchUnpickler

    @staticmethod
    def load(*args, **kwargs):
        return LazyTorchUnpickler(*args, **kwargs).load()


# ---------------------------------------------------------------------------
# Archive reader
# ---------------------------------------------------------------------------

class TorchArchiveReader:
    """Lazily reads files from a PyTorch ZIP archive."""

    def __init__(self, path: str):
        self.archive = zipfile.ZipFile(path, mode="r")
        self.archive_name = os.path.basename(
            os.path.normpath(path)
        ).split(".")[0]
        self.file_name: Optional[str] = None
        self.file = None

    def open_file(
        self, file_name: str, offset: int = 0
    ) -> zipfile.ZipExtFile:
        if self.file_name != file_name or (
            self.file is not None and self.file.tell() > offset
        ):
            if self.file is not None:
                self.file.close()
            try:
                fd = self.archive.open(
                    f"archive/data/{file_name}", mode="r"
                )
            except Exception:
                fd = self.archive.open(
                    f"{self.archive_name}/data/{file_name}", mode="r"
                )
            self.file = fd
            self.file_name = file_name

        skip_bytes = offset - self.file.tell()
        if skip_bytes < 0:
            raise RuntimeError(
                f"Cannot seek backwards in zip stream (need {skip_bytes})"
            )
        self.file.seek(skip_bytes, os.SEEK_CUR)
        return self.file

    def close(self):
        if self.file is not None:
            self.file.close()
            self.file = None
        self.archive.close()


# ---------------------------------------------------------------------------
# Context manager for lazy loading (replaces torch_lazy_load)
# ---------------------------------------------------------------------------

@contextlib.contextmanager
def lazy_load_context():
    """Context manager that monkey-patches ``pickle`` so that
    ``pickle.load`` on a ``.bin`` file returns ``DeferredLoad`` placeholders
    instead of real tensors.
    """
    old_unpickler = pickle.Unpickler
    old_load = pickle.load
    try:
        def load_mp(*args, **kwargs):
            return LazyTorchUnpickler(*args, **kwargs).load()

        pickle.Unpickler = LazyTorchUnpickler
        pickle.load = load_mp
        yield
    finally:
        pickle.Unpickler = old_unpickler
        pickle.load = old_load


def load_bin_lazy(path: str) -> Dict[str, DeferredLoad]:
    """Load a ``.bin`` file lazily, returning ``{name: DeferredLoad}``."""
    with lazy_load_context():
        if zipfile.is_zipfile(path):
            archive = zipfile.ZipFile(path, mode="r")
            archive_name = os.path.basename(os.path.normpath(path)).split(".")[0]
            try:
                try:
                    with archive.open("archive/data.pkl", mode="r") as f:
                        return LazyTorchUnpickler(f).load()
                except KeyError:
                    with archive.open(f"{archive_name}/data.pkl", mode="r") as f:
                        return LazyTorchUnpickler(f).load()
            finally:
                archive.close()

        with open(path, "rb") as f:
            return LazyTorchUnpickler(f).load()
