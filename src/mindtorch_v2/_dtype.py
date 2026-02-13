import numpy as np


class DType:
    def __init__(self, name):
        self.name = name

    def __repr__(self):
        return f"torch.{self.name}"


float32 = DType("float32")
float16 = DType("float16")
int64 = DType("int64")


_NUMPY_DTYPE_MAP = {
    float32: np.float32,
    float16: np.float16,
    int64: np.int64,
}


def to_numpy_dtype(dtype):
    return _NUMPY_DTYPE_MAP.get(dtype, np.float32)
