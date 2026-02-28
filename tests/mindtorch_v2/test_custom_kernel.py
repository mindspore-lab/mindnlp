import pytest
import ctypes
import os
import tempfile
from unittest import mock

import mindtorch_v2 as torch
from mindtorch_v2._backends.npu.custom_kernel import (
    KernelLauncher,
    tensor_ptr,
    alloc_like,
    ascendc_op,
    _contiguous_stride,
)
from mindtorch_v2._dispatch.registry import registry
from mindtorch_v2._dispatch.keys import DispatchKey
from mindtorch_v2._tensor import Tensor


@pytest.fixture(autouse=True)
def _restore_registry():
    snap = registry.snapshot()
    yield
    registry.restore(snap)


# ---------------------------------------------------------------------------
# _contiguous_stride
# ---------------------------------------------------------------------------

class TestContiguousStride:
    def test_1d(self):
        assert _contiguous_stride((5,)) == (1,)

    def test_2d(self):
        assert _contiguous_stride((3, 4)) == (4, 1)

    def test_3d(self):
        assert _contiguous_stride((2, 3, 4)) == (12, 4, 1)

    def test_scalar(self):
        assert _contiguous_stride(()) == ()


# ---------------------------------------------------------------------------
# tensor_ptr
# ---------------------------------------------------------------------------

class TestTensorPtr:
    def test_cpu_tensor(self):
        t = torch.tensor([1.0, 2.0, 3.0])
        ptr = tensor_ptr(t)
        assert isinstance(ptr, int)
        assert ptr != 0


# ---------------------------------------------------------------------------
# KernelLauncher — test Python-level logic with mock .so
# ---------------------------------------------------------------------------

class TestKernelLauncher:
    def test_symbol_lookup_caches(self):
        """Test that _get_launch_fn caches the function pointer."""
        mock_lib = mock.MagicMock()
        mock_fn = mock.MagicMock(return_value=0)
        mock_lib.aclrtlaunch_my_kernel = mock_fn

        launcher = KernelLauncher.__new__(KernelLauncher)
        launcher._lib = mock_lib
        launcher._cache = {}

        fn1 = launcher._get_launch_fn("my_kernel")
        fn2 = launcher._get_launch_fn("my_kernel")
        assert fn1 is fn2
        assert fn1 is mock_fn

    def test_missing_symbol_raises(self):
        """Test that a missing symbol raises RuntimeError."""
        mock_lib = mock.MagicMock()
        mock_lib.aclrtlaunch_missing = mock.PropertyMock(
            side_effect=AttributeError("not found")
        )
        # Configure getattr to raise
        del mock_lib.aclrtlaunch_missing

        launcher = KernelLauncher.__new__(KernelLauncher)
        launcher._lib = mock_lib
        launcher._cache = {}

        with pytest.raises(RuntimeError, match="Symbol.*not found"):
            launcher._get_launch_fn("missing")

    def test_launch_converts_args(self):
        """Test that launch converts args to ctypes and calls the function."""
        mock_fn = mock.MagicMock(return_value=0)
        mock_lib = mock.MagicMock()
        mock_lib.aclrtlaunch_test_kernel = mock_fn

        launcher = KernelLauncher.__new__(KernelLauncher)
        launcher._lib = mock_lib
        launcher._cache = {}

        fake_stream = 42

        launcher.launch("test_kernel", block_dim=8,
                        args=[100, 3.14, 200], stream=fake_stream)

        mock_fn.assert_called_once()
        call_args = mock_fn.call_args[0]
        # First arg: block_dim as c_uint32
        assert call_args[0].value == 8
        # Second arg: stream
        assert call_args[1] == 42
        # Remaining: converted args
        assert call_args[2].value == 100  # int -> c_uint64
        assert abs(call_args[3].value - 3.14) < 1e-10  # float -> c_double
        assert call_args[4].value == 200  # int -> c_uint64

    def test_launch_nonzero_return_raises(self):
        """Test that a non-zero return from the kernel raises RuntimeError."""
        mock_fn = mock.MagicMock(return_value=1)
        mock_lib = mock.MagicMock()
        mock_lib.aclrtlaunch_bad_kernel = mock_fn

        launcher = KernelLauncher.__new__(KernelLauncher)
        launcher._lib = mock_lib
        launcher._cache = {}

        with pytest.raises(RuntimeError, match="launch failed"):
            launcher.launch("bad_kernel", block_dim=1, args=[], stream=0)


# ---------------------------------------------------------------------------
# ascendc_op — test Python-level decorator logic
# ---------------------------------------------------------------------------

class TestAscendcOp:
    def test_registers_npu_kernel(self):
        @ascendc_op("test_ascendc::my_op")
        def my_op(x: Tensor) -> Tensor:
            return x

        entry = registry.get("test_ascendc::my_op")
        assert DispatchKey.NPU in entry.kernels

    def test_auto_registers_fake(self):
        """ascendc_op should auto-register a Meta kernel for single Tensor return."""
        @ascendc_op("test_ascendc::auto_fake")
        def auto_fake(x: Tensor) -> Tensor:
            return x

        entry = registry.get("test_ascendc::auto_fake")
        assert DispatchKey.Meta in entry.kernels

    def test_returns_handle(self):
        from mindtorch_v2.library import CustomOpHandle

        @ascendc_op("test_ascendc::handle_op")
        def handle_op(x: Tensor) -> Tensor:
            return x

        assert isinstance(handle_op, CustomOpHandle)

    def test_no_auto_fake_for_non_tensor_return(self):
        """If return type is not a single Tensor, no auto-fake is registered."""
        @ascendc_op("test_ascendc::int_ret")
        def int_ret(x: Tensor) -> int:
            return 0

        entry = registry.get("test_ascendc::int_ret")
        # No auto-fake because return is "int" not "Tensor"
        assert DispatchKey.Meta not in entry.kernels

    def test_explicit_fake_overrides_auto(self):
        """User-registered fake should be used even if auto-fake would apply."""
        @ascendc_op("test_ascendc::explicit_fake")
        def explicit_fake(x: Tensor) -> Tensor:
            return x

        sentinel = object()

        @explicit_fake.register_fake
        def fake_fn(x):
            return sentinel

        entry = registry.get("test_ascendc::explicit_fake")
        assert entry.kernels[DispatchKey.Meta] is fake_fn
