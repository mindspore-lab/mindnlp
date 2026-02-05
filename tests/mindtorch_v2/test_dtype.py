# tests/mindtorch_v2/test_dtype.py
import mindtorch_v2 as torch


def test_dtype_exists():
    """Core dtypes are accessible as torch.float32, etc."""
    assert torch.float32 is not None
    assert torch.float64 is not None
    assert torch.float16 is not None
    assert torch.bfloat16 is not None
    assert torch.int8 is not None
    assert torch.int16 is not None
    assert torch.int32 is not None
    assert torch.int64 is not None
    assert torch.uint8 is not None
    assert torch.bool is not None
    assert torch.complex64 is not None
    assert torch.complex128 is not None


def test_dtype_aliases():
    """PyTorch aliases like torch.float, torch.long, torch.half."""
    assert torch.float is torch.float32
    assert torch.double is torch.float64
    assert torch.half is torch.float16
    assert torch.long is torch.int64
    assert torch.int is torch.int32


def test_dtype_properties():
    """dtype has is_floating_point and is_complex properties."""
    assert torch.float32.is_floating_point
    assert torch.float64.is_floating_point
    assert not torch.int32.is_floating_point
    assert not torch.bool.is_floating_point
    assert torch.complex64.is_complex
    assert torch.complex128.is_complex
    assert not torch.float32.is_complex


def test_dtype_itemsize():
    """dtype.itemsize returns element size in bytes."""
    assert torch.float32.itemsize == 4
    assert torch.float64.itemsize == 8
    assert torch.float16.itemsize == 2
    assert torch.int64.itemsize == 8
    assert torch.int32.itemsize == 4
    assert torch.int8.itemsize == 1
    assert torch.bool.itemsize == 1


def test_dtype_to_numpy():
    """dtype can be converted to numpy dtype."""
    import numpy as np
    assert torch.dtype_to_numpy(torch.float32) == np.float32
    assert torch.dtype_to_numpy(torch.int64) == np.int64
    assert torch.dtype_to_numpy(torch.bool) == np.bool_


def test_numpy_to_dtype():
    """numpy dtype can be converted to torch dtype."""
    import numpy as np
    assert torch.numpy_to_dtype(np.float32) is torch.float32
    assert torch.numpy_to_dtype(np.int64) is torch.int64
    assert torch.numpy_to_dtype(np.bool_) is torch.bool


def test_from_mindspore_dtype():
    """MindSpore dtypes convert to torch dtypes."""
    import mindspore
    assert torch.from_mindspore_dtype(mindspore.float32) is torch.float32
    assert torch.from_mindspore_dtype(mindspore.int64) is torch.int64
    assert torch.from_mindspore_dtype(mindspore.bool_) is torch.bool


def test_dtype_to_numpy_bfloat16():
    """bfloat16 has no numpy equivalent, should return None."""
    assert torch.dtype_to_numpy(torch.bfloat16) is None


def test_numpy_to_dtype_complex():
    """Complex dtypes should convert correctly."""
    import numpy as np
    assert torch.numpy_to_dtype(np.complex64) is torch.complex64
    assert torch.numpy_to_dtype(np.complex128) is torch.complex128
