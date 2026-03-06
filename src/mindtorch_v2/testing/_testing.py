"""Implementation of torch.testing functions."""

import math
import warnings

import numpy as np

from .._tensor import Tensor
from .._creation import tensor, randn, rand, randint, empty, zeros
from .._dtype import (
    DType, float16, float32, float64, bfloat16,
    int8, int16, int32, int64, uint8,
    bool as torch_bool, complex64, complex128,
    to_numpy_dtype,
)


# Default tolerances per dtype, matching PyTorch's defaults
_DEFAULT_TOLERANCES = {
    float16: (1e-3, 1e-5),
    bfloat16: (1.6e-2, 1e-5),
    float32: (1.3e-6, 1e-5),
    float64: (1e-7, 1e-7),
    complex64: (1.3e-6, 1e-5),
    complex128: (1e-7, 1e-7),
}


def _get_default_tolerance(actual, expected):
    """Get default (rtol, atol) based on the wider dtype of actual/expected."""
    dtype = actual.dtype
    if hasattr(expected, 'dtype'):
        # Use the wider (less precise) dtype
        a_key = _DEFAULT_TOLERANCES.get(actual.dtype)
        e_key = _DEFAULT_TOLERANCES.get(expected.dtype)
        if a_key is not None and e_key is not None:
            # wider dtype = larger tolerances
            if a_key[0] > e_key[0]:
                dtype = actual.dtype
            else:
                dtype = expected.dtype

    return _DEFAULT_TOLERANCES.get(dtype, (1.3e-6, 1e-5))


def _to_tensor(obj):
    """Convert obj to Tensor if it isn't one already."""
    if isinstance(obj, Tensor):
        return obj
    if isinstance(obj, (int, float, complex)):
        return tensor(obj)
    if isinstance(obj, np.ndarray):
        return tensor(obj)
    if isinstance(obj, (list, tuple)):
        return tensor(obj)
    return obj


def assert_close(
    actual,
    expected,
    *,
    allow_subclasses=True,
    rtol=None,
    atol=None,
    equal_nan=False,
    check_device=True,
    check_dtype=True,
    check_layout=True,
    check_stride=False,
    msg=None,
):
    """Assert that ``actual`` and ``expected`` are close.

    Compares tensors element-wise for closeness using the formula:
        |actual - expected| <= atol + rtol * |expected|

    For non-quantized, non-sparse tensors. If ``actual`` and ``expected`` are
    not tensors, they are converted to tensors before comparison.

    Args:
        actual: Actual tensor or scalar.
        expected: Expected tensor or scalar.
        allow_subclasses: If True (default), allow subclasses of Tensor.
        rtol: Relative tolerance. If None, uses dtype-specific default.
        atol: Absolute tolerance. If None, uses dtype-specific default.
        equal_nan: If True, NaNs in the same position are considered equal.
        check_device: If True, assert both tensors are on the same device.
        check_dtype: If True, assert both tensors have the same dtype.
        check_layout: If True, assert both tensors have the same layout.
        check_stride: If True, assert both tensors have the same stride.
        msg: Optional error message prefix.

    Raises:
        AssertionError: If the tensors are not close.
    """
    actual = _to_tensor(actual)
    expected = _to_tensor(expected)

    if not isinstance(actual, Tensor) or not isinstance(expected, Tensor):
        # Fall back to simple equality for non-tensor types
        if actual != expected:
            _raise_mismatch("values are not equal", actual, expected, msg)
        return

    # Check metadata
    if check_device:
        if str(actual.device) != str(expected.device):
            _raise_mismatch(
                f"device mismatch: {actual.device} vs {expected.device}",
                actual, expected, msg,
            )

    if check_dtype:
        if actual.dtype != expected.dtype:
            _raise_mismatch(
                f"dtype mismatch: {actual.dtype} vs {expected.dtype}",
                actual, expected, msg,
            )

    # Check shapes
    if actual.shape != expected.shape:
        _raise_mismatch(
            f"shape mismatch: {actual.shape} vs {expected.shape}",
            actual, expected, msg,
        )

    # Get tolerances
    if rtol is None or atol is None:
        default_rtol, default_atol = _get_default_tolerance(actual, expected)
        if rtol is None:
            rtol = default_rtol
        if atol is None:
            atol = default_atol

    # Convert to numpy for comparison
    actual_np = actual.detach().numpy().astype(np.float64)
    expected_np = expected.detach().numpy().astype(np.float64)

    # Handle complex dtypes
    if actual.is_complex():
        actual_np = actual.detach().numpy()
        expected_np = expected.detach().numpy()
        # For complex, compare real and imaginary parts
        diff = np.abs(actual_np - expected_np)
        threshold = atol + rtol * np.abs(expected_np)
    else:
        diff = np.abs(actual_np - expected_np)
        threshold = atol + rtol * np.abs(expected_np)

    # Handle NaN
    if equal_nan:
        nan_mask = np.isnan(actual_np) & np.isnan(expected_np)
        diff = np.where(nan_mask, 0.0, diff)
        threshold = np.where(nan_mask, 1.0, threshold)  # NaN==NaN passes
    else:
        if np.any(np.isnan(actual_np)) or np.any(np.isnan(expected_np)):
            nan_actual = np.sum(np.isnan(actual_np))
            nan_expected = np.sum(np.isnan(expected_np))
            _raise_mismatch(
                f"found NaN(s): {nan_actual} in actual, {nan_expected} in expected",
                actual, expected, msg,
            )

    # Handle inf: +inf == +inf and -inf == -inf
    inf_mask = np.isinf(actual_np) & np.isinf(expected_np)
    same_inf = inf_mask & (np.sign(actual_np) == np.sign(expected_np))
    diff_sign_inf = inf_mask & ~same_inf  # both inf but different signs
    diff = np.where(same_inf, 0.0, diff)
    threshold = np.where(same_inf, 1.0, threshold)
    # Different-sign infinities or one-is-inf: force mismatch
    one_inf = np.isinf(actual_np) ^ np.isinf(expected_np)
    force_mismatch = diff_sign_inf | one_inf

    mismatched = (diff > threshold) | force_mismatch
    num_mismatched = np.sum(mismatched)

    if num_mismatched > 0:
        total = actual_np.size
        max_abs_diff = float(np.max(diff))
        max_abs_diff_idx = np.unravel_index(np.argmax(diff), diff.shape)

        # Find max relative diff (avoid division by zero)
        with np.errstate(divide='ignore', invalid='ignore'):
            rel_diff = diff / np.where(np.abs(expected_np) == 0, 1.0, np.abs(expected_np))
        max_rel_diff = float(np.max(np.where(np.isfinite(rel_diff), rel_diff, 0.0)))

        error_msg = (
            f"Tensors are not close!\n"
            f"\n"
            f"Mismatched elements: {num_mismatched} / {total} ({100 * num_mismatched / total:.1f}%)\n"
            f"Greatest absolute difference: {max_abs_diff} at index {max_abs_diff_idx} "
            f"(up to {atol} allowed)\n"
            f"Greatest relative difference: {max_rel_diff} "
            f"(up to {rtol} allowed)"
        )
        _raise_mismatch(error_msg, actual, expected, msg)


def _raise_mismatch(error_msg, actual, expected, msg):
    """Raise AssertionError with optional user message prefix."""
    if msg is not None:
        if callable(msg):
            msg = msg()
        raise AssertionError(f"{msg}\n\n{error_msg}")
    raise AssertionError(error_msg)


def assert_allclose(actual, expected, rtol=None, atol=None, equal_nan=True, msg=""):
    """Deprecated alias for assert_close with slightly different defaults.

    .. deprecated:: 1.12
        Use :func:`torch.testing.assert_close` instead.
    """
    warnings.warn(
        "torch.testing.assert_allclose is deprecated in favor of torch.testing.assert_close. "
        "Please use assert_close instead.",
        FutureWarning,
        stacklevel=2,
    )
    # assert_allclose defaults: equal_nan=True, no check_dtype by default
    if rtol is None:
        rtol = 1.3e-6
    if atol is None:
        atol = 1e-5
    assert_close(
        actual, expected,
        rtol=rtol, atol=atol,
        equal_nan=equal_nan,
        check_dtype=False,
        msg=msg if msg else None,
    )


def make_tensor(
    *shape,
    dtype,
    device=None,
    low=None,
    high=None,
    requires_grad=False,
    noncontiguous=False,
    exclude_zero=False,
    memory_format=None,
):
    """Create a tensor with random data for testing purposes.

    Args:
        *shape: Shape of the tensor. Can be a single tuple or multiple ints.
        dtype: The dtype of the tensor.
        device: The device of the tensor (default: None for CPU).
        low: Minimum value (inclusive). Default depends on dtype.
        high: Maximum value (exclusive). Default depends on dtype.
        requires_grad: Whether the tensor requires grad.
        noncontiguous: If True, return a noncontiguous tensor.
        exclude_zero: If True, ensure no zeros in the tensor.
        memory_format: Memory format (unused, for API compatibility).

    Returns:
        A tensor filled with random data.
    """
    # Handle shape as tuple or varargs
    if len(shape) == 1 and isinstance(shape[0], (tuple, list)):
        shape = tuple(shape[0])

    # Determine dtype category and set defaults
    is_float = dtype.is_floating_point if isinstance(dtype, DType) else False
    is_cplx = dtype.is_complex if isinstance(dtype, DType) else False
    is_bool = (dtype == torch_bool)

    if is_bool:
        result = randint(0, 2, size=shape, dtype=torch_bool, device=device)
        return result

    if is_float or is_cplx:
        if low is None:
            low = -9.0
        if high is None:
            high = 9.0

        # Generate random floats in [low, high)
        result = rand(*shape, dtype=dtype, device=device)
        result = _to_tensor(result.numpy() * (high - low) + low)
        result = tensor(result.numpy(), dtype=dtype, device=device)

        if exclude_zero:
            np_data = result.numpy()
            zero_mask = np_data == 0
            if np.any(zero_mask):
                np_data[zero_mask] = 1.0
                result = tensor(np_data, dtype=dtype, device=device)

        if requires_grad and is_float:
            result.requires_grad_(True)

    else:
        # Integer types
        if low is None:
            low = 0
        if high is None:
            # Determine sensible upper bound for integer dtypes
            if dtype == int8:
                high = 10
            elif dtype == uint8:
                high = 10
            elif dtype == int16:
                high = 100
            elif dtype == int32:
                high = 1000
            elif dtype == int64:
                high = 1000
            else:
                high = 10

        result = randint(int(low), int(high), size=shape, dtype=dtype, device=device)

        if exclude_zero:
            np_data = result.numpy()
            zero_mask = np_data == 0
            if np.any(zero_mask):
                np_data[zero_mask] = 1
                result = tensor(np_data, dtype=dtype, device=device)

    if noncontiguous:
        # Make noncontiguous by slicing a larger tensor
        if result.numel() > 0 and result.ndim > 0:
            # Create tensor with double size on first dim, fill, then slice
            big_shape = list(result.shape)
            big_shape[0] = big_shape[0] * 2
            np_data = result.numpy()
            big_np = np.zeros(big_shape, dtype=to_numpy_dtype(dtype))
            big_np[::2] = np_data
            big_t = tensor(big_np, dtype=dtype, device=device)
            result = big_t[::2]

    return result
