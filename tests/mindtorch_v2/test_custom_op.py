import pytest
import inspect
import numpy as np
import mindtorch_v2 as torch
from mindtorch_v2.library import custom_op, _infer_schema, CustomOpHandle
from mindtorch_v2._dispatch.registry import registry
from mindtorch_v2._dispatch.keys import DispatchKey
from mindtorch_v2._autograd.engine import backward, grad
from mindtorch_v2._tensor import Tensor


@pytest.fixture(autouse=True)
def _restore_registry():
    snap = registry.snapshot()
    yield
    registry.restore(snap)


def _allclose(t, expected, atol=1e-6):
    arr = t._numpy_view().flatten()
    return np.allclose(arr, expected, atol=atol)


# ---------------------------------------------------------------------------
# Schema inference
# ---------------------------------------------------------------------------

class TestSchemaInference:
    def test_basic_tensor_args(self):
        def my_add(x: Tensor, y: Tensor) -> Tensor:
            pass
        schema = _infer_schema(my_add)
        assert schema == "my_add(Tensor x, Tensor y) -> Tensor"

    def test_scalar_args_with_defaults(self):
        def scaled_add(x: Tensor, y: Tensor, scale: float = 1.0) -> Tensor:
            pass
        schema = _infer_schema(scaled_add)
        assert schema == "scaled_add(Tensor x, Tensor y, float scale=1.0) -> Tensor"

    def test_bool_arg(self):
        def my_op(x: Tensor, flag: bool = False) -> Tensor:
            pass
        schema = _infer_schema(my_op)
        assert schema == "my_op(Tensor x, bool flag=False) -> Tensor"

    def test_int_arg(self):
        def my_op(x: Tensor, dim: int = 0) -> Tensor:
            pass
        schema = _infer_schema(my_op)
        assert schema == "my_op(Tensor x, int dim=0) -> Tensor"

    def test_optional_tensor(self):
        from typing import Optional
        def my_op(x: Tensor, bias: Optional[Tensor] = None) -> Tensor:
            pass
        schema = _infer_schema(my_op)
        assert schema == "my_op(Tensor x, Tensor? bias=None) -> Tensor"

    def test_list_int(self):
        from typing import List
        def my_op(x: Tensor, dims: List[int]) -> Tensor:
            pass
        schema = _infer_schema(my_op)
        assert schema == "my_op(Tensor x, int[] dims) -> Tensor"

    def test_mutates_args(self):
        def my_op(x: Tensor, y: Tensor) -> Tensor:
            pass
        schema = _infer_schema(my_op, mutates_args=("x",))
        assert schema == "my_op(Tensor! x, Tensor y) -> Tensor"

    def test_no_annotations_defaults_to_tensor(self):
        def my_op(x, y):
            pass
        schema = _infer_schema(my_op)
        assert schema == "my_op(Tensor x, Tensor y) -> Tensor"


# ---------------------------------------------------------------------------
# @custom_op basic
# ---------------------------------------------------------------------------

class TestCustomOp:
    def test_custom_op_registers_schema(self):
        @custom_op("testns::my_add", mutates_args=())
        def my_add(x: Tensor, y: Tensor) -> Tensor:
            return torch.add(x, y)

        assert registry.has("testns::my_add")
        entry = registry.get("testns::my_add")
        assert entry.schema_obj is not None

    def test_custom_op_returns_handle(self):
        @custom_op("testns::my_add2", mutates_args=())
        def my_add2(x: Tensor, y: Tensor) -> Tensor:
            return torch.add(x, y)

        assert isinstance(my_add2, CustomOpHandle)
        assert my_add2.__name__ == "my_add2"

    def test_custom_op_callable(self):
        @custom_op("testns::my_add3", mutates_args=())
        def my_add3(x: Tensor, y: Tensor) -> Tensor:
            return torch.add(x, y)

        x = torch.tensor([1.0, 2.0])
        y = torch.tensor([3.0, 4.0])
        result = my_add3(x, y)
        assert _allclose(result, [4.0, 6.0])

    def test_custom_op_default_registration(self):
        @custom_op("testns::my_scale", mutates_args=())
        def my_scale(x: Tensor, s: float = 2.0) -> Tensor:
            return torch.mul(x, torch.tensor([s]))

        entry = registry.get("testns::my_scale")
        assert DispatchKey.CPU in entry.kernels
        assert DispatchKey.NPU in entry.kernels

    def test_custom_op_device_types(self):
        @custom_op("testns::cpu_only", mutates_args=(), device_types="cpu")
        def cpu_only(x: Tensor) -> Tensor:
            return x

        entry = registry.get("testns::cpu_only")
        assert DispatchKey.CPU in entry.kernels
        assert DispatchKey.CompositeImplicitAutograd not in entry.kernels

    def test_custom_op_explicit_schema(self):
        @custom_op("testns::explicit", mutates_args=(), schema="explicit(Tensor a, Tensor b) -> Tensor")
        def explicit(a, b):
            return torch.add(a, b)

        entry = registry.get("testns::explicit")
        assert entry.schema_obj is not None


# ---------------------------------------------------------------------------
# register_fake
# ---------------------------------------------------------------------------

class TestRegisterFakeOnHandle:
    def test_register_fake(self):
        @custom_op("testns::with_fake", mutates_args=())
        def with_fake(x: Tensor) -> Tensor:
            return torch.mul(x, torch.tensor([2.0]))

        @with_fake.register_fake
        def fake_fn(x):
            return x  # just return same shape

        entry = registry.get("testns::with_fake")
        assert DispatchKey.Meta in entry.kernels


# ---------------------------------------------------------------------------
# register_kernel
# ---------------------------------------------------------------------------

class TestRegisterKernel:
    def test_register_kernel_cpu(self):
        @custom_op("testns::with_kernel", mutates_args=())
        def with_kernel(x: Tensor) -> Tensor:
            return x

        @with_kernel.register_kernel("cpu")
        def cpu_impl(x):
            return torch.mul(x, torch.tensor([3.0]))

        entry = registry.get("testns::with_kernel")
        assert DispatchKey.CPU in entry.kernels


# ---------------------------------------------------------------------------
# register_autograd
# ---------------------------------------------------------------------------

class TestRegisterAutograd:
    def test_autograd_backward(self):
        @custom_op("testns::triple", mutates_args=(), device_types="cpu")
        def triple(x: Tensor) -> Tensor:
            return torch.mul(x, torch.tensor([3.0]))

        def setup(ctx, inputs, output):
            pass

        def bwd(ctx, grad_output):
            return (torch.mul(grad_output, torch.tensor([3.0])),)

        triple.register_autograd(bwd, setup_context=setup)

        x = torch.tensor([2.0], requires_grad=True)
        y = triple(x)
        assert y.grad_fn is not None
        backward(y.sum())
        assert _allclose(x.grad, [3.0])

    def test_autograd_with_save_for_backward(self):
        @custom_op("testns::square", mutates_args=(), device_types="cpu")
        def square(x: Tensor) -> Tensor:
            return torch.mul(x, x)

        def setup(ctx, inputs, output):
            x, = inputs
            ctx.save_for_backward(x)

        def bwd(ctx, grad_output):
            x, = ctx.saved_tensors
            return (torch.mul(torch.mul(grad_output, x), torch.tensor([2.0])),)

        square.register_autograd(bwd, setup_context=setup)

        x = torch.tensor([3.0], requires_grad=True)
        y = square(x)
        backward(y.sum())
        # d(x^2)/dx = 2x = 6.0
        assert _allclose(x.grad, [6.0])

    def test_autograd_grad(self):
        @custom_op("testns::double_ag", mutates_args=(), device_types="cpu")
        def double_ag(x: Tensor) -> Tensor:
            return torch.mul(x, torch.tensor([2.0]))

        def bwd(ctx, grad_output):
            return (torch.mul(grad_output, torch.tensor([2.0])),)

        double_ag.register_autograd(bwd)

        x = torch.tensor([5.0], requires_grad=True)
        y = double_ag(x)
        (dx,) = grad(y.sum(), (x,))
        assert _allclose(dx, [2.0])

    def test_no_grad_skips_autograd(self):
        @custom_op("testns::no_grad_op", mutates_args=(), device_types="cpu")
        def no_grad_op(x: Tensor) -> Tensor:
            return torch.mul(x, torch.tensor([2.0]))

        def bwd(ctx, grad_output):
            return (grad_output,)

        no_grad_op.register_autograd(bwd)

        x = torch.tensor([3.0])  # no requires_grad
        y = no_grad_op(x)
        assert y.grad_fn is None
