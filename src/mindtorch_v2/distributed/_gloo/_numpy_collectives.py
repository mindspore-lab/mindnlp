"""Numpy-based reduce operations and tensor serialization for Gloo backend."""

import struct
import numpy as np

from .._reduce_op import RedOpType


# ReduceOp int value -> numpy ufunc
_REDUCE_OPS = {
    int(RedOpType.SUM): np.add,
    int(RedOpType.PRODUCT): np.multiply,
    int(RedOpType.MAX): np.maximum,
    int(RedOpType.MIN): np.minimum,
    int(RedOpType.BAND): np.bitwise_and,
    int(RedOpType.BOR): np.bitwise_or,
    int(RedOpType.BXOR): np.bitwise_xor,
    # AVG is handled specially by the caller (sum then divide)
    int(RedOpType.AVG): np.add,
}


def apply_reduce_op(op, a, b):
    """Apply a reduce operation element-wise: result = op(a, b)."""
    op_int = int(op)
    fn = _REDUCE_OPS.get(op_int)
    if fn is None:
        raise ValueError(f"Unsupported reduce op: {op}")
    return fn(a, b)


def serialize_array(arr):
    """Serialize a numpy array to bytes.

    Wire format:
      1 byte  - length of dtype name
      N bytes - dtype name (ascii)
      4 bytes - ndim (uint32 big-endian)
      ndim*8 bytes - shape dims (uint64 big-endian each)
      rest    - raw data bytes
    """
    arr = np.ascontiguousarray(arr)
    dtype_name = arr.dtype.str.encode("ascii")
    header = struct.pack("!B", len(dtype_name)) + dtype_name
    header += struct.pack("!I", arr.ndim)
    for s in arr.shape:
        header += struct.pack("!Q", s)
    return header + arr.tobytes()


def deserialize_array(data):
    """Deserialize bytes back to a numpy array."""
    offset = 0
    dtype_len = data[offset]
    offset += 1
    dtype_name = data[offset:offset + dtype_len].decode("ascii")
    offset += dtype_len
    ndim = struct.unpack("!I", data[offset:offset + 4])[0]
    offset += 4
    shape = []
    for _ in range(ndim):
        shape.append(struct.unpack("!Q", data[offset:offset + 8])[0])
        offset += 8
    shape = tuple(shape)
    body = data[offset:]
    return np.frombuffer(body, dtype=np.dtype(dtype_name)).reshape(shape)
