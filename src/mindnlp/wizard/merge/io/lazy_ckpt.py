# Copyright (c) MindNLP Wizard contributors.
# Licensed under the Apache License, Version 2.0.

"""
B-level lazy loader for MindSpore ``.ckpt`` checkpoint files.

Scans the protobuf wire format to build an index of tensor locations
(file offsets and byte lengths), then reads only the requested tensor
bytes from disk on demand — no full file materialisation.

Protobuf schema (from ``mindspore/ccsrc/utils/checkpoint.proto``)::

    message Checkpoint {
      message Value {
        required string tag = 1;
        oneof value {
          TensorProto tensor = 2;
          MapTensorProto maptensor = 3;
        }
      }
      repeated Value value = 1;
    }
    message TensorProto {
      repeated int64 dims = 1;
      required string tensor_type = 2;
      required bytes tensor_content = 3;
    }

Large tensors may be sliced into multiple ``Value`` entries sharing
the same ``tag``; their ``tensor_content`` chunks are concatenated on read.
"""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy

LOG = logging.getLogger(__name__)


class CkptFormatNotSupported(Exception):
    """The .ckpt file uses features not handled by the lazy loader
    (encryption, MapTensor, unknown wire types, …).

    Callers should fall back to ``DumbCkptLoader`` which delegates
    to ``mindspore.load_checkpoint``.
    """


# ── MindSpore dtype string → numpy dtype ──────────────────────────────────

_TENSOR_TYPE_TO_NUMPY = {
    # str(mindspore.dtype) format — produced by native MindSpore Tensor
    "Float32": numpy.float32,
    "Float16": numpy.float16,
    "Float64": numpy.float64,
    "Int8": numpy.int8,
    "Int16": numpy.int16,
    "Int32": numpy.int32,
    "Int64": numpy.int64,
    "UInt8": numpy.uint8,
    "UInt16": numpy.uint16,
    "UInt32": numpy.uint32,
    "UInt64": numpy.uint64,
    "Bool": numpy.bool_,
    "BFloat16": "bfloat16",
    # repr(mindspore.dtype) format — produced when mindtorch wraps tensors
    "mindspore.float32": numpy.float32,
    "mindspore.float16": numpy.float16,
    "mindspore.float64": numpy.float64,
    "mindspore.int8": numpy.int8,
    "mindspore.int16": numpy.int16,
    "mindspore.int32": numpy.int32,
    "mindspore.int64": numpy.int64,
    "mindspore.uint8": numpy.uint8,
    "mindspore.uint16": numpy.uint16,
    "mindspore.uint32": numpy.uint32,
    "mindspore.uint64": numpy.uint64,
    "mindspore.bool": numpy.bool_,
    "mindspore.bool_": numpy.bool_,
    "mindspore.bfloat16": "bfloat16",
}


# ── Protobuf wire-format primitives ───────────────────────────────────────

def _read_varint(f) -> Optional[int]:
    """Read a base-128 varint. Return *None* at EOF."""
    result = 0
    shift = 0
    while True:
        b = f.read(1)
        if not b:
            return None
        byte = b[0]
        result |= (byte & 0x7F) << shift
        if (byte & 0x80) == 0:
            return result
        shift += 7
        if shift > 63:
            raise CkptFormatNotSupported(
                "Varint exceeds 64 bits — file may be corrupted or encrypted"
            )


def _skip_wire_field(f, wire_type: int) -> None:
    """Advance past one protobuf field value."""
    if wire_type == 0:
        _read_varint(f)
    elif wire_type == 1:
        f.seek(8, os.SEEK_CUR)
    elif wire_type == 2:
        length = _read_varint(f)
        f.seek(length, os.SEEK_CUR)
    elif wire_type == 5:
        f.seek(4, os.SEEK_CUR)
    else:
        raise CkptFormatNotSupported(
            f"Unknown protobuf wire type {wire_type}"
        )


# ── Index data structures ─────────────────────────────────────────────────

@dataclass
class CkptTensorSlice:
    """One contiguous byte range inside a ``.ckpt`` file."""
    file_offset: int
    byte_length: int


@dataclass
class CkptTensorEntry:
    """All slices of a single parameter, plus its metadata."""
    slices: List[CkptTensorSlice]
    dims: Tuple[int, ...]
    tensor_type_str: str


# ── CkptIndex: scanner + on-demand reader ─────────────────────────────────

class CkptIndex:
    """Index of every tensor in a ``.ckpt`` file.

    Build with :meth:`from_file`, then call :meth:`read_tensor` to
    materialise individual parameters as numpy arrays.
    """

    def __init__(self, file_path: str, entries: Dict[str, CkptTensorEntry]):
        self.file_path = file_path
        self.entries = entries

    @classmethod
    def from_file(cls, file_path: str) -> "CkptIndex":
        entries = _scan_ckpt_file(file_path)
        return cls(file_path, entries)

    def read_tensor(self, key: str) -> numpy.ndarray:
        if key not in self.entries:
            raise KeyError(f"Tensor '{key}' not found in {self.file_path}")

        entry = self.entries[key]
        np_dtype = _TENSOR_TYPE_TO_NUMPY.get(entry.tensor_type_str)

        if np_dtype is None:
            raise CkptFormatNotSupported(
                f"Unsupported tensor_type '{entry.tensor_type_str}' "
                f"for key '{key}' in {self.file_path}"
            )

        chunks: List[bytes] = []
        with open(self.file_path, "rb") as f:
            for slc in entry.slices:
                f.seek(slc.file_offset)
                chunks.append(f.read(slc.byte_length))

        raw = b"".join(chunks)

        if np_dtype == "bfloat16":
            import ml_dtypes
            arr = numpy.frombuffer(raw, dtype=ml_dtypes.bfloat16).copy()
        else:
            arr = numpy.frombuffer(raw, dtype=np_dtype).copy()

        if entry.dims:
            arr = arr.reshape(entry.dims)
        return arr


# ── File scanner ──────────────────────────────────────────────────────────

def _scan_ckpt_file(path: str) -> Dict[str, CkptTensorEntry]:
    """Walk the protobuf wire format to index every tensor without
    loading ``tensor_content`` into memory."""
    file_size = os.path.getsize(path)
    entries: Dict[str, CkptTensorEntry] = {}

    try:
        with open(path, "rb") as f:
            while f.tell() < file_size:
                key = _read_varint(f)
                if key is None:
                    break
                field_num = key >> 3
                wire_type = key & 0x7

                if field_num == 1 and wire_type == 2:
                    value_len = _read_varint(f)
                    if value_len is None:
                        break
                    value_end = f.tell() + value_len
                    _parse_value(f, value_end, entries)
                    f.seek(value_end)
                else:
                    _skip_wire_field(f, wire_type)
    except (OSError, ValueError, UnicodeDecodeError) as exc:
        raise CkptFormatNotSupported(
            f"Failed to parse {path}: {type(exc).__name__}: {exc}"
        ) from exc

    return entries


def _parse_value(
    f,
    value_end: int,
    entries: Dict[str, CkptTensorEntry],
) -> None:
    """Parse one ``Checkpoint.Value`` submessage."""
    tag: Optional[str] = None

    dims: List[int] = []
    tensor_type_str: Optional[str] = None
    content_offset: Optional[int] = None
    content_length: int = 0

    while f.tell() < value_end:
        sub_key = _read_varint(f)
        if sub_key is None:
            break
        sub_field = sub_key >> 3
        sub_wire = sub_key & 0x7

        if sub_field == 1 and sub_wire == 2:
            # Value.tag  (string)
            str_len = _read_varint(f)
            tag = f.read(str_len).decode("utf-8")

        elif sub_field == 2 and sub_wire == 2:
            # Value.tensor  (TensorProto embedded message)
            tp_len = _read_varint(f)
            tp_end = f.tell() + tp_len
            dims, tensor_type_str, content_offset, content_length = (
                _parse_tensor_proto(f, tp_end)
            )
            f.seek(tp_end)

        elif sub_field == 3 and sub_wire == 2:
            raise CkptFormatNotSupported(
                f"MapTensor entries are not supported by the lazy loader "
                f"(tag='{tag}')"
            )

        else:
            _skip_wire_field(f, sub_wire)

    if tag is None:
        return
    if content_offset is None or tensor_type_str is None:
        LOG.debug("Skipping non-tensor entry '%s'", tag)
        return

    slc = CkptTensorSlice(file_offset=content_offset, byte_length=content_length)
    if tag in entries:
        entries[tag].slices.append(slc)
    else:
        entries[tag] = CkptTensorEntry(
            slices=[slc],
            dims=tuple(dims),
            tensor_type_str=tensor_type_str,
        )


def _parse_tensor_proto(
    f,
    tp_end: int,
) -> Tuple[List[int], Optional[str], Optional[int], int]:
    """Parse a ``TensorProto`` submessage, returning
    ``(dims, tensor_type_str, content_offset, content_length)``.
    """
    dims: List[int] = []
    tensor_type_str: Optional[str] = None
    content_offset: Optional[int] = None
    content_length: int = 0

    while f.tell() < tp_end:
        tp_key = _read_varint(f)
        if tp_key is None:
            break
        tp_field = tp_key >> 3
        tp_wire = tp_key & 0x7

        if tp_field == 1 and tp_wire == 2:
            # dims  (packed repeated int64)
            packed_len = _read_varint(f)
            packed_end = f.tell() + packed_len
            while f.tell() < packed_end:
                dims.append(_read_varint(f))
        elif tp_field == 1 and tp_wire == 0:
            # dims  (single int64 varint)
            dims.append(_read_varint(f))
        elif tp_field == 2 and tp_wire == 2:
            # tensor_type  (string, e.g. "Float32")
            str_len = _read_varint(f)
            tensor_type_str = f.read(str_len).decode("utf-8")
        elif tp_field == 3 and tp_wire == 2:
            # tensor_content  (bytes — record offset, skip data)
            content_length = _read_varint(f)
            content_offset = f.tell()
            f.seek(content_length, os.SEEK_CUR)
        else:
            _skip_wire_field(f, tp_wire)

    return dims, tensor_type_str, content_offset, content_length
