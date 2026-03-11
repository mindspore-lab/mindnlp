"""Tests for torch.testing module."""

import sys
import os
import math
import warnings
import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "src"))

import mindtorch_v2 as torch


class TestAssertClose:
    """Tests for torch.testing.assert_close."""

    def test_equal_tensors(self):
        a = torch.tensor([1.0, 2.0, 3.0])
        b = torch.tensor([1.0, 2.0, 3.0])
        torch.testing.assert_close(a, b)

    def test_close_tensors(self):
        a = torch.tensor([1.0, 2.0, 3.0])
        b = torch.tensor([1.0 + 1e-7, 2.0 - 1e-7, 3.0 + 1e-7])
        torch.testing.assert_close(a, b)

    def test_not_close_raises(self):
        a = torch.tensor([1.0, 2.0, 3.0])
        b = torch.tensor([1.0, 2.0, 4.0])
        with pytest.raises(AssertionError, match="Tensors are not close"):
            torch.testing.assert_close(a, b)

    def test_custom_tolerances(self):
        a = torch.tensor([1.0, 2.0, 3.0])
        b = torch.tensor([1.1, 2.1, 3.1])
        # Fails with default tolerance
        with pytest.raises(AssertionError):
            torch.testing.assert_close(a, b)
        # Passes with relaxed tolerance
        torch.testing.assert_close(a, b, atol=0.2, rtol=0.0)

    def test_shape_mismatch(self):
        a = torch.tensor([1.0, 2.0])
        b = torch.tensor([1.0, 2.0, 3.0])
        with pytest.raises(AssertionError, match="shape mismatch"):
            torch.testing.assert_close(a, b)

    def test_dtype_mismatch(self):
        a = torch.tensor([1.0, 2.0], dtype=torch.float32)
        b = torch.tensor([1.0, 2.0], dtype=torch.float64)
        with pytest.raises(AssertionError, match="dtype mismatch"):
            torch.testing.assert_close(a, b)

    def test_dtype_check_disabled(self):
        a = torch.tensor([1.0, 2.0], dtype=torch.float32)
        b = torch.tensor([1.0, 2.0], dtype=torch.float64)
        torch.testing.assert_close(a, b, check_dtype=False)

    def test_equal_nan_false(self):
        a = torch.tensor([1.0, float('nan')])
        b = torch.tensor([1.0, float('nan')])
        with pytest.raises(AssertionError, match="NaN"):
            torch.testing.assert_close(a, b, equal_nan=False)

    def test_equal_nan_true(self):
        a = torch.tensor([1.0, float('nan')])
        b = torch.tensor([1.0, float('nan')])
        torch.testing.assert_close(a, b, equal_nan=True)

    def test_inf_equal(self):
        a = torch.tensor([float('inf'), float('-inf'), 1.0])
        b = torch.tensor([float('inf'), float('-inf'), 1.0])
        torch.testing.assert_close(a, b)

    def test_inf_sign_mismatch(self):
        a = torch.tensor([float('inf')])
        b = torch.tensor([float('-inf')])
        with pytest.raises(AssertionError):
            torch.testing.assert_close(a, b)

    def test_scalar_values(self):
        torch.testing.assert_close(torch.tensor(1.0), torch.tensor(1.0))

    def test_python_scalar_conversion(self):
        torch.testing.assert_close(torch.tensor(5.0), 5.0)

    def test_2d_tensors(self):
        a = torch.tensor([[1.0, 2.0], [3.0, 4.0]])
        b = torch.tensor([[1.0, 2.0], [3.0, 4.0]])
        torch.testing.assert_close(a, b)

    def test_error_message_format(self):
        a = torch.tensor([1.0, 2.0, 3.0])
        b = torch.tensor([1.0, 2.0, 5.0])
        with pytest.raises(AssertionError, match="Mismatched elements"):
            torch.testing.assert_close(a, b)

    def test_custom_msg(self):
        a = torch.tensor([1.0])
        b = torch.tensor([2.0])
        with pytest.raises(AssertionError, match="my custom msg"):
            torch.testing.assert_close(a, b, msg="my custom msg")

    def test_callable_msg(self):
        a = torch.tensor([1.0])
        b = torch.tensor([2.0])
        with pytest.raises(AssertionError, match="callable msg"):
            torch.testing.assert_close(a, b, msg=lambda: "callable msg")

    def test_integer_tensors(self):
        a = torch.tensor([1, 2, 3], dtype=torch.int32)
        b = torch.tensor([1, 2, 3], dtype=torch.int32)
        torch.testing.assert_close(a, b)

    def test_integer_mismatch(self):
        a = torch.tensor([1, 2, 3], dtype=torch.int32)
        b = torch.tensor([1, 2, 4], dtype=torch.int32)
        with pytest.raises(AssertionError):
            torch.testing.assert_close(a, b)

    def test_bool_tensors(self):
        a = torch.tensor([True, False, True])
        b = torch.tensor([True, False, True])
        torch.testing.assert_close(a, b)

    def test_relative_tolerance(self):
        a = torch.tensor([100.0])
        b = torch.tensor([100.01])
        # atol=0 means only rtol matters
        torch.testing.assert_close(a, b, atol=0.0, rtol=0.001)

    def test_absolute_tolerance(self):
        a = torch.tensor([0.0])
        b = torch.tensor([0.001])
        # rtol=0 means only atol matters
        torch.testing.assert_close(a, b, atol=0.01, rtol=0.0)


class TestAssertAllclose:
    """Tests for torch.testing.assert_allclose (deprecated)."""

    def test_basic_usage(self):
        a = torch.tensor([1.0, 2.0, 3.0])
        b = torch.tensor([1.0, 2.0, 3.0])
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", FutureWarning)
            torch.testing.assert_allclose(a, b)

    def test_deprecation_warning(self):
        a = torch.tensor([1.0])
        b = torch.tensor([1.0])
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            torch.testing.assert_allclose(a, b)
            assert len(w) == 1
            assert issubclass(w[0].category, FutureWarning)
            assert "deprecated" in str(w[0].message).lower()

    def test_no_dtype_check(self):
        """assert_allclose should not check dtype by default."""
        a = torch.tensor([1.0], dtype=torch.float32)
        b = torch.tensor([1.0], dtype=torch.float64)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", FutureWarning)
            torch.testing.assert_allclose(a, b)

    def test_nan_equal_by_default(self):
        """assert_allclose has equal_nan=True by default."""
        a = torch.tensor([float('nan')])
        b = torch.tensor([float('nan')])
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", FutureWarning)
            torch.testing.assert_allclose(a, b)


class TestMakeTensor:
    """Tests for torch.testing.make_tensor."""

    def test_basic_float(self):
        t = torch.testing.make_tensor((3, 4), dtype=torch.float32)
        assert t.shape == (3, 4)
        assert t.dtype == torch.float32

    def test_basic_int(self):
        t = torch.testing.make_tensor((5,), dtype=torch.int64)
        assert t.shape == (5,)
        assert t.dtype == torch.int64

    def test_bool_dtype(self):
        t = torch.testing.make_tensor((10,), dtype=torch.bool)
        assert t.shape == (10,)
        assert t.dtype == torch.bool
        # Only 0s and 1s
        vals = t.numpy()
        assert np.all((vals == 0) | (vals == 1))

    def test_custom_range(self):
        t = torch.testing.make_tensor((100,), dtype=torch.float32, low=0.0, high=1.0)
        vals = t.numpy()
        assert np.all(vals >= 0.0)
        assert np.all(vals < 1.0)

    def test_custom_int_range(self):
        t = torch.testing.make_tensor((100,), dtype=torch.int32, low=5, high=10)
        vals = t.numpy()
        assert np.all(vals >= 5)
        assert np.all(vals < 10)

    def test_exclude_zero(self):
        t = torch.testing.make_tensor((100,), dtype=torch.float32, exclude_zero=True)
        vals = t.numpy()
        assert not np.any(vals == 0.0)

    def test_exclude_zero_int(self):
        t = torch.testing.make_tensor((100,), dtype=torch.int32, low=0, high=3, exclude_zero=True)
        vals = t.numpy()
        assert not np.any(vals == 0)

    def test_requires_grad(self):
        t = torch.testing.make_tensor((3,), dtype=torch.float32, requires_grad=True)
        assert t.requires_grad is True

    def test_shape_as_varargs(self):
        t = torch.testing.make_tensor(3, 4, dtype=torch.float32)
        assert t.shape == (3, 4)

    def test_shape_as_tuple(self):
        t = torch.testing.make_tensor((3, 4), dtype=torch.float32)
        assert t.shape == (3, 4)

    def test_scalar_shape(self):
        t = torch.testing.make_tensor((), dtype=torch.float32)
        assert t.shape == ()

    def test_float64(self):
        t = torch.testing.make_tensor((5,), dtype=torch.float64)
        assert t.dtype == torch.float64

    def test_int8(self):
        t = torch.testing.make_tensor((5,), dtype=torch.int8)
        assert t.dtype == torch.int8

    def test_noncontiguous(self):
        t = torch.testing.make_tensor((4, 3), dtype=torch.float32, noncontiguous=True)
        # Shape should be preserved
        assert t.shape == (4, 3)
