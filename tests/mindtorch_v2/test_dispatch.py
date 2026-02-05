# tests/mindtorch_v2/test_dispatch.py
from mindtorch_v2._dispatch import DispatchKey, register_op, get_op_impl, dispatch
import mindtorch_v2 as torch


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
    @register_op("test_add", DispatchKey.Backend_CPU)
    def test_add_cpu(a, b):
        return a + b

    impl = get_op_impl("test_add", DispatchKey.Backend_CPU)
    assert impl is not None
    assert impl(2, 3) == 5


def test_get_unregistered_op():
    """Getting unregistered op returns None."""
    impl = get_op_impl("nonexistent_op", DispatchKey.Backend_CPU)
    assert impl is None


def test_register_multiple_backends():
    """Can register same op for multiple backends."""
    @register_op("test_mul", DispatchKey.Backend_CPU)
    def test_mul_cpu(a, b):
        return a * b

    @register_op("test_mul", DispatchKey.Backend_CUDA)
    def test_mul_cuda(a, b):
        return a * b * 1  # Different impl

    cpu_impl = get_op_impl("test_mul", DispatchKey.Backend_CPU)
    cuda_impl = get_op_impl("test_mul", DispatchKey.Backend_CUDA)

    assert cpu_impl is not None
    assert cuda_impl is not None
    assert cpu_impl is not cuda_impl


def test_dispatch_basic():
    """Dispatch routes to correct backend."""
    import numpy as np

    # Register a test op that works with tensors
    @register_op("dispatch_test", DispatchKey.Backend_CPU)
    def dispatch_test_cpu(x):
        # Work with numpy, then wrap result
        arr = x.numpy() * 2
        return torch.tensor(arr)

    # Create tensor and dispatch
    t = torch.tensor([1.0, 2.0, 3.0])
    result = dispatch("dispatch_test", t)

    # Should use CPU impl - result is 2x the input
    np.testing.assert_array_almost_equal(result.numpy(), [2.0, 4.0, 6.0])


def test_dispatch_determines_backend_from_tensor():
    """Dispatch determines backend from tensor device."""
    @register_op("backend_test", DispatchKey.Backend_CPU)
    def backend_test_cpu(x):
        return "cpu"

    t = torch.tensor([1.0])  # CPU tensor
    result = dispatch("backend_test", t)
    assert result == "cpu"
