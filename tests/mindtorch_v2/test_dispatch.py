# tests/mindtorch_v2/test_dispatch.py
"""Tests for dispatch system - device agnostic."""
from mindtorch_v2._dispatch import DispatchKey, register_op, get_op_impl, dispatch
from mindtorch_v2.configs import DEVICE_TARGET
import mindtorch_v2 as torch


def _get_current_backend_key():
    """Get the dispatch key for the current device."""
    if DEVICE_TARGET == 'Ascend':
        return DispatchKey.Backend_Ascend
    elif DEVICE_TARGET == 'GPU':
        return DispatchKey.Backend_CUDA
    else:
        return DispatchKey.Backend_CPU


def test_dispatch_key_enum():
    """DispatchKey enum has required keys."""
    assert hasattr(DispatchKey, 'Autograd')
    assert hasattr(DispatchKey, 'Backend_CPU')
    assert hasattr(DispatchKey, 'Backend_CUDA')
    assert hasattr(DispatchKey, 'Backend_Ascend')
    assert hasattr(DispatchKey, 'CompositeExplicit')


def test_dispatch_key_ordering():
    """Autograd comes before Backend keys."""
    assert DispatchKey.Autograd.value < DispatchKey.Backend_CPU.value


def test_register_and_get_op():
    """Can register and retrieve op implementations."""
    backend_key = _get_current_backend_key()

    @register_op("test_add_device", backend_key)
    def test_add_impl(a, b):
        return a + b

    impl = get_op_impl("test_add_device", backend_key)
    assert impl is not None
    assert impl(2, 3) == 5


def test_get_unregistered_op():
    """Getting unregistered op returns None."""
    impl = get_op_impl("nonexistent_op_xyz123", DispatchKey.Backend_CPU)
    assert impl is None


def test_register_multiple_backends():
    """Can register same op for multiple backends."""
    @register_op("test_mul_multi", DispatchKey.Backend_CPU)
    def test_mul_cpu(a, b):
        return a * b

    @register_op("test_mul_multi", DispatchKey.Backend_CUDA)
    def test_mul_cuda(a, b):
        return a * b * 1  # Different impl

    cpu_impl = get_op_impl("test_mul_multi", DispatchKey.Backend_CPU)
    cuda_impl = get_op_impl("test_mul_multi", DispatchKey.Backend_CUDA)

    assert cpu_impl is not None
    assert cuda_impl is not None
    assert cpu_impl is not cuda_impl


def test_dispatch_basic():
    """Dispatch routes to correct backend."""
    import numpy as np
    backend_key = _get_current_backend_key()

    # Register a test op for the current backend
    @register_op("dispatch_test", backend_key)
    def dispatch_test_impl(x):
        # Work with numpy, then wrap result
        arr = x.numpy() * 2
        return torch.tensor(arr)

    # Create tensor and dispatch
    t = torch.tensor([1.0, 2.0, 3.0])
    result = dispatch("dispatch_test", t)

    # Should use current backend impl - result is 2x the input
    np.testing.assert_array_almost_equal(result.numpy(), [2.0, 4.0, 6.0])


def test_dispatch_determines_backend_from_tensor():
    """Dispatch determines backend from tensor device."""
    backend_key = _get_current_backend_key()
    expected_result = "current_backend"

    @register_op("backend_test", backend_key)
    def backend_test_impl(x):
        return expected_result

    t = torch.tensor([1.0])
    result = dispatch("backend_test", t)
    assert result == expected_result
