"""Tests for P0 dtype utilities & query functions:
finfo, iinfo, is_tensor, is_floating_point, is_complex, numel, square.
"""
import math
import pytest
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "src"))
import mindtorch_v2 as torch


# ---- finfo ----

class TestFinfo:
    @pytest.mark.parametrize("dt", [torch.float16, torch.float32, torch.float64])
    def test_finfo_standard_floats(self, dt):
        info = torch.finfo(dt)
        assert info.dtype is dt
        assert info.bits > 0
        assert info.eps > 0
        assert info.max > 0
        assert info.min < 0
        assert info.smallest_normal > 0
        assert info.tiny > 0
        assert info.resolution > 0

    def test_finfo_float32_values(self):
        info = torch.finfo(torch.float32)
        assert info.bits == 32
        assert abs(info.eps - 1.1920928955078125e-07) < 1e-15

    def test_finfo_float64_values(self):
        info = torch.finfo(torch.float64)
        assert info.bits == 64
        assert abs(info.eps - 2.220446049250313e-16) < 1e-25

    def test_finfo_float16_values(self):
        info = torch.finfo(torch.float16)
        assert info.bits == 16

    def test_finfo_bfloat16(self):
        info = torch.finfo(torch.bfloat16)
        assert info.bits == 16
        assert abs(info.eps - 0.0078125) < 1e-10
        assert info.max > 3e38
        assert info.min < -3e38
        assert info.smallest_normal > 0
        assert info.dtype is torch.bfloat16

    def test_finfo_accepts_tensor(self):
        t = torch.tensor([1.0, 2.0], dtype=torch.float32)
        info = torch.finfo(t)
        assert info.dtype is torch.float32
        assert info.bits == 32

    def test_finfo_repr(self):
        info = torch.finfo(torch.float32)
        r = repr(info)
        assert "finfo" in r
        assert "float32" in r
        assert "eps=" in r

    def test_finfo_rejects_int(self):
        with pytest.raises(TypeError):
            torch.finfo(torch.int32)


# ---- iinfo ----

class TestIinfo:
    @pytest.mark.parametrize("dt", [torch.int8, torch.int16, torch.int32, torch.int64, torch.uint8])
    def test_iinfo_integer_dtypes(self, dt):
        info = torch.iinfo(dt)
        assert info.dtype is dt
        assert info.bits > 0
        assert isinstance(info.max, int)
        assert isinstance(info.min, int)
        assert info.max > info.min

    def test_iinfo_int8_values(self):
        info = torch.iinfo(torch.int8)
        assert info.bits == 8
        assert info.min == -128
        assert info.max == 127

    def test_iinfo_uint8_values(self):
        info = torch.iinfo(torch.uint8)
        assert info.bits == 8
        assert info.min == 0
        assert info.max == 255

    def test_iinfo_int32_values(self):
        info = torch.iinfo(torch.int32)
        assert info.bits == 32
        assert info.min == -(2**31)
        assert info.max == 2**31 - 1

    def test_iinfo_int64_values(self):
        info = torch.iinfo(torch.int64)
        assert info.bits == 64
        assert info.min == -(2**63)
        assert info.max == 2**63 - 1

    def test_iinfo_accepts_tensor(self):
        t = torch.tensor([1, 2, 3], dtype=torch.int64)
        info = torch.iinfo(t)
        assert info.dtype is torch.int64
        assert info.bits == 64

    def test_iinfo_repr(self):
        info = torch.iinfo(torch.int32)
        r = repr(info)
        assert "iinfo" in r
        assert "int32" in r


# ---- is_tensor ----

class TestIsTensor:
    def test_tensor_returns_true(self):
        t = torch.tensor([1.0])
        assert torch.is_tensor(t) is True

    def test_non_tensor_returns_false(self):
        assert torch.is_tensor(42) is False
        assert torch.is_tensor("hello") is False
        assert torch.is_tensor([1, 2, 3]) is False
        assert torch.is_tensor(None) is False


# ---- is_floating_point ----

class TestIsFloatingPoint:
    def test_float_tensor(self):
        t = torch.tensor([1.0], dtype=torch.float32)
        assert torch.is_floating_point(t) is True

    def test_int_tensor(self):
        t = torch.tensor([1], dtype=torch.int64)
        assert torch.is_floating_point(t) is False


# ---- is_complex ----

class TestIsComplex:
    def test_complex_tensor(self):
        t = torch.tensor([1.0 + 2.0j], dtype=torch.complex64)
        assert torch.is_complex(t) is True

    def test_real_tensor(self):
        t = torch.tensor([1.0], dtype=torch.float32)
        assert torch.is_complex(t) is False


# ---- numel ----

class TestNumel:
    def test_1d(self):
        t = torch.tensor([1.0, 2.0, 3.0])
        assert torch.numel(t) == 3

    def test_2d(self):
        t = torch.zeros(3, 4)
        assert torch.numel(t) == 12

    def test_scalar(self):
        t = torch.tensor(42.0)
        assert torch.numel(t) == 1


# ---- square ----

class TestSquare:
    def test_basic(self):
        t = torch.tensor([1.0, 2.0, 3.0, -4.0])
        result = torch.square(t)
        expected = [1.0, 4.0, 9.0, 16.0]
        for i in range(4):
            assert abs(result[i].item() - expected[i]) < 1e-6

    def test_zeros(self):
        t = torch.zeros(5)
        result = torch.square(t)
        for i in range(5):
            assert result[i].item() == 0.0

    def test_integer(self):
        t = torch.tensor([2, 3, 4], dtype=torch.int32)
        result = torch.square(t)
        assert result[0].item() == 4
        assert result[1].item() == 9
        assert result[2].item() == 16

    def test_preserves_shape(self):
        t = torch.randn(3, 4)
        result = torch.square(t)
        assert result.shape == (3, 4)
