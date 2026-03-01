import pytest
import mindtorch_v2 as torch
from mindtorch_v2.library import Library, impl, register_fake
from mindtorch_v2._dispatch.registry import registry, dispatch_key_from_string
from mindtorch_v2._dispatch.keys import DispatchKey


@pytest.fixture(autouse=True)
def _restore_registry():
    """Save/restore registry state so tests don't pollute each other."""
    snap = registry.snapshot()
    yield
    registry.restore(snap)


# ---------------------------------------------------------------------------
# dispatch_key_from_string
# ---------------------------------------------------------------------------

class TestDispatchKeyFromString:
    def test_cpu(self):
        assert dispatch_key_from_string("CPU") is DispatchKey.CPU

    def test_npu(self):
        assert dispatch_key_from_string("NPU") is DispatchKey.NPU

    def test_cuda_maps_to_privateuse1(self):
        assert dispatch_key_from_string("CUDA") is DispatchKey.PrivateUse1

    def test_privateuse1_maps_to_privateuse1(self):
        assert dispatch_key_from_string("PrivateUse1") is DispatchKey.PrivateUse1

    def test_meta(self):
        assert dispatch_key_from_string("Meta") is DispatchKey.Meta

    def test_autograd(self):
        assert dispatch_key_from_string("Autograd") is DispatchKey.Autograd

    def test_autograd_cpu(self):
        assert dispatch_key_from_string("AutogradCPU") is DispatchKey.AutogradCPU

    def test_autograd_npu(self):
        assert dispatch_key_from_string("AutogradNPU") is DispatchKey.AutogradNPU

    def test_autograd_cuda_maps_to_autogradxpu(self):
        assert dispatch_key_from_string("AutogradCUDA") is DispatchKey.AutogradXPU

    def test_autograd_privateuse1_maps_to_autogradxpu(self):
        assert dispatch_key_from_string("AutogradPrivateUse1") is DispatchKey.AutogradXPU

    def test_composite_implicit(self):
        assert dispatch_key_from_string("CompositeImplicitAutograd") is DispatchKey.CompositeImplicitAutograd

    def test_unknown_raises(self):
        with pytest.raises(ValueError, match="Unknown dispatch key string"):
            dispatch_key_from_string("NoSuchKey")


# ---------------------------------------------------------------------------
# Library.define + Library.impl
# ---------------------------------------------------------------------------

class TestLibrary:
    def test_define_registers_schema(self):
        lib = Library("testops", "DEF")
        lib.define("my_add(Tensor x, Tensor y) -> Tensor")
        entry = registry.get("testops::my_add")
        assert entry.schema_obj is not None

    def test_impl_direct_call(self):
        lib = Library("testops", "DEF")
        lib.define("my_add(Tensor x, Tensor y) -> Tensor")

        def my_add_cpu(x, y):
            return torch.add(x, y)

        lib.impl("my_add", my_add_cpu, "CPU")
        entry = registry.get("testops::my_add")
        assert DispatchKey.CPU in entry.kernels
        assert entry.kernels[DispatchKey.CPU] is my_add_cpu

    def test_impl_decorator_form(self):
        lib = Library("testops", "DEF")
        lib.define("my_mul(Tensor x, Tensor y) -> Tensor")

        @lib.impl("my_mul", dispatch_key="CPU")
        def my_mul_cpu(x, y):
            return torch.mul(x, y)

        entry = registry.get("testops::my_mul")
        assert DispatchKey.CPU in entry.kernels
        assert entry.kernels[DispatchKey.CPU] is my_mul_cpu

    def test_impl_before_define_raises(self):
        lib = Library("testops", "DEF")
        with pytest.raises(RuntimeError, match="schema must be registered"):
            lib.impl("nonexistent", lambda x: x, "CPU")

    def test_multiple_dispatch_keys(self):
        lib = Library("testops", "DEF")
        lib.define("my_op(Tensor x) -> Tensor")
        lib.impl("my_op", lambda x: x, "CPU")
        lib.impl("my_op", lambda x: x, "Meta")
        entry = registry.get("testops::my_op")
        assert DispatchKey.CPU in entry.kernels
        assert DispatchKey.Meta in entry.kernels


# ---------------------------------------------------------------------------
# Standalone impl() decorator
# ---------------------------------------------------------------------------

class TestStandaloneImpl:
    def test_impl_decorator(self):
        lib = Library("testops", "DEF")
        lib.define("standalone_add(Tensor x, Tensor y) -> Tensor")

        @impl("testops::standalone_add", "CPU")
        def standalone_add_cpu(x, y):
            return torch.add(x, y)

        entry = registry.get("testops::standalone_add")
        assert DispatchKey.CPU in entry.kernels
        assert entry.kernels[DispatchKey.CPU] is standalone_add_cpu


# ---------------------------------------------------------------------------
# register_fake
# ---------------------------------------------------------------------------

class TestRegisterFake:
    def test_register_fake(self):
        lib = Library("testops", "DEF")
        lib.define("fake_op(Tensor x) -> Tensor")

        @register_fake("testops::fake_op")
        def fake_op_meta(x):
            return x

        entry = registry.get("testops::fake_op")
        assert DispatchKey.Meta in entry.kernels
        assert entry.kernels[DispatchKey.Meta] is fake_op_meta
