"""Tests for Round 2 backward ops (31 ops across 3 tiers)."""
import sys
import math
import numpy as np
import pytest

sys.path.insert(0, "src")
import mindtorch_v2 as torch


def _check_grad(func, inputs, eps=1e-4, atol=1e-3, rtol=1e-3):
    """Numerical gradient check via finite differences."""
    for i, inp in enumerate(inputs):
        if not inp.requires_grad:
            continue
        flat = inp.detach().numpy().flatten()
        analytical_grads = []
        numerical_grads = []
        for j in range(flat.size):
            # Forward at x+eps
            flat_p = flat.copy()
            flat_p[j] += eps
            inp_p = torch.tensor(flat_p.reshape(inp.shape), dtype=inp.dtype, requires_grad=False)
            args_p = list(inputs)
            args_p[i] = inp_p
            out_p = func(*args_p)
            if isinstance(out_p, tuple):
                out_p = out_p[0]
            val_p = out_p.sum().item()

            # Forward at x-eps
            flat_m = flat.copy()
            flat_m[j] -= eps
            inp_m = torch.tensor(flat_m.reshape(inp.shape), dtype=inp.dtype, requires_grad=False)
            args_m = list(inputs)
            args_m[i] = inp_m
            out_m = func(*args_m)
            if isinstance(out_m, tuple):
                out_m = out_m[0]
            val_m = out_m.sum().item()

            numerical_grads.append((val_p - val_m) / (2 * eps))

        # Get analytical grad
        out = func(*inputs)
        if isinstance(out, tuple):
            out = out[0]
        out.sum().backward()
        analytical = inp.grad.numpy().flatten()

        numerical_arr = np.array(numerical_grads)
        np.testing.assert_allclose(analytical, numerical_arr, atol=atol, rtol=rtol,
                                   err_msg=f"Gradient mismatch for input {i}")
        # Reset grads
        for inp2 in inputs:
            if hasattr(inp2, 'grad') and inp2.grad is not None:
                inp2.grad = None


# ============================================================================
# Tier 1: Training-Critical
# ============================================================================

class TestClampMinBackward:
    def test_basic(self):
        x = torch.tensor([-2.0, -0.5, 0.0, 0.5, 2.0], requires_grad=True)
        y = torch.clamp_min(x, 0.0)
        y.sum().backward()
        expected = np.array([0.0, 0.0, 1.0, 1.0, 1.0])
        np.testing.assert_allclose(x.grad.numpy(), expected)

    def test_numerical(self):
        x = torch.tensor([[-1.0, 0.5], [1.5, -0.3]], requires_grad=True)
        _check_grad(lambda a: torch.clamp_min(a, -0.5), [x])


class TestClampMaxBackward:
    def test_basic(self):
        x = torch.tensor([-2.0, -0.5, 0.0, 0.5, 2.0], requires_grad=True)
        y = torch.clamp_max(x, 0.5)
        y.sum().backward()
        expected = np.array([1.0, 1.0, 1.0, 1.0, 0.0])
        np.testing.assert_allclose(x.grad.numpy(), expected)

    def test_numerical(self):
        x = torch.tensor([[0.1, 0.8], [1.5, -0.3]], requires_grad=True)
        _check_grad(lambda a: torch.clamp_max(a, 1.0), [x])


class TestMinBackward:
    def test_basic(self):
        a = torch.tensor([1.0, 3.0, 2.0], requires_grad=True)
        b = torch.tensor([2.0, 1.0, 2.0], requires_grad=True)
        y = torch.min(a, b)
        y.sum().backward()
        np.testing.assert_allclose(a.grad.numpy(), [1.0, 0.0, 1.0])
        np.testing.assert_allclose(b.grad.numpy(), [0.0, 1.0, 0.0])

    def test_numerical(self):
        a = torch.tensor([1.0, 3.0, 0.5], requires_grad=True)
        b = torch.tensor([2.0, 1.0, 2.0], requires_grad=True)
        _check_grad(lambda x, y: torch.min(x, y), [a, b])


class TestMaxBackward:
    def test_basic(self):
        a = torch.tensor([1.0, 3.0, 2.0], requires_grad=True)
        b = torch.tensor([2.0, 1.0, 2.0], requires_grad=True)
        y = torch.max(a, b)
        y.sum().backward()
        np.testing.assert_allclose(a.grad.numpy(), [0.0, 1.0, 1.0])
        np.testing.assert_allclose(b.grad.numpy(), [1.0, 0.0, 0.0])

    def test_numerical(self):
        a = torch.tensor([1.0, 3.0, 0.5], requires_grad=True)
        b = torch.tensor([2.0, 1.0, 2.0], requires_grad=True)
        _check_grad(lambda x, y: torch.max(x, y), [a, b])


class TestCumprodBackward:
    def test_basic(self):
        x = torch.tensor([1.0, 2.0, 3.0], requires_grad=True)
        y = torch.cumprod(x, 0)
        # y = [1, 2, 6]; sum = 9
        y.sum().backward()
        # d(sum(cumprod))/dx[0] = 1 + 2 + 6 = 9 (x[0] appears in all products)
        # d(sum(cumprod))/dx[1] = 0 + 1 + 3 = 4 (x[1] appears in products 1,2)
        # d(sum(cumprod))/dx[2] = 0 + 0 + 2 = 2 (x[2] appears in product 2 only)
        expected = np.array([9.0, 4.0, 2.0])
        np.testing.assert_allclose(x.grad.numpy(), expected, rtol=1e-5)

    def test_numerical(self):
        x = torch.tensor([[1.0, 2.0], [3.0, 0.5]], requires_grad=True)
        _check_grad(lambda a: torch.cumprod(a, 0), [x])


class TestRepeatInterleaveBackward:
    def test_scalar_repeats(self):
        x = torch.tensor([1.0, 2.0, 3.0], requires_grad=True)
        y = torch.repeat_interleave(x, 2, dim=0)
        y.sum().backward()
        # Each element repeated 2 times → grad is 2
        np.testing.assert_allclose(x.grad.numpy(), [2.0, 2.0, 2.0])

    def test_no_dim(self):
        x = torch.tensor([[1.0, 2.0], [3.0, 4.0]], requires_grad=True)
        y = torch.repeat_interleave(x, 3)
        y.sum().backward()
        np.testing.assert_allclose(x.grad.numpy(), [[3.0, 3.0], [3.0, 3.0]])


class TestScatterBackward:
    def test_basic(self):
        src = torch.tensor([[1.0, 2.0], [3.0, 4.0]], requires_grad=True)
        index = torch.tensor([[0, 1], [1, 0]], dtype=torch.long)
        a = torch.tensor(np.zeros((2, 3), dtype=np.float32), requires_grad=True)
        y = torch.scatter(a, 1, index, src)
        y.sum().backward()
        # src grad: gathered from output grad at index positions
        np.testing.assert_allclose(src.grad.numpy(), [[1.0, 1.0], [1.0, 1.0]])


class TestFloorDivideBackward:
    def test_basic(self):
        a = torch.tensor([7.0, 5.0], requires_grad=True)
        b = torch.tensor([2.0, 3.0], requires_grad=True)
        y = torch.floor_divide(a, b)
        y.sum().backward()
        # floor_divide is not differentiable; grad should be 0
        np.testing.assert_allclose(a.grad.numpy(), [0.0, 0.0])
        np.testing.assert_allclose(b.grad.numpy(), [0.0, 0.0])


# ============================================================================
# Tier 2: Trig / Math
# ============================================================================

class TestTanBackward:
    def test_numerical(self):
        x = torch.tensor([0.1, 0.5, -0.3, 1.0], requires_grad=True)
        _check_grad(torch.tan, [x])


class TestAsinBackward:
    def test_numerical(self):
        x = torch.tensor([0.1, 0.5, -0.3, 0.8], requires_grad=True)
        _check_grad(torch.asin, [x])


class TestAcosBackward:
    def test_numerical(self):
        x = torch.tensor([0.1, 0.5, -0.3, 0.8], requires_grad=True)
        _check_grad(torch.acos, [x])


class TestAtanBackward:
    def test_numerical(self):
        x = torch.tensor([0.1, 1.0, -0.5, 2.0], requires_grad=True)
        _check_grad(torch.atan, [x])


class TestAtan2Backward:
    def test_numerical(self):
        a = torch.tensor([1.0, -1.0, 2.0], requires_grad=True)
        b = torch.tensor([2.0, 1.0, -1.0], requires_grad=True)
        _check_grad(torch.atan2, [a, b])


class TestSinhBackward:
    def test_numerical(self):
        x = torch.tensor([0.1, 0.5, -0.3, 1.0], requires_grad=True)
        _check_grad(torch.sinh, [x])


class TestCoshBackward:
    def test_numerical(self):
        x = torch.tensor([0.1, 0.5, -0.3, 1.0], dtype=torch.float64, requires_grad=True)
        _check_grad(torch.cosh, [x])


class TestAsinhBackward:
    def test_numerical(self):
        x = torch.tensor([0.1, 0.5, -0.3, 2.0], requires_grad=True)
        _check_grad(torch.asinh, [x])


class TestAcoshBackward:
    def test_numerical(self):
        x = torch.tensor([1.1, 1.5, 2.0, 3.0], requires_grad=True)
        _check_grad(torch.acosh, [x])


class TestAtanhBackward:
    def test_numerical(self):
        x = torch.tensor([0.1, 0.5, -0.3, 0.8], requires_grad=True)
        _check_grad(torch.atanh, [x])


class TestLog10Backward:
    def test_numerical(self):
        x = torch.tensor([0.5, 1.0, 2.0, 10.0], requires_grad=True)
        _check_grad(torch.log10, [x])


class TestErfcBackward:
    def test_numerical(self):
        x = torch.tensor([0.1, 0.5, -0.3, 1.0], requires_grad=True)
        _check_grad(torch.erfc, [x])


class TestLogaddexpBackward:
    def test_numerical(self):
        a = torch.tensor([1.0, -1.0, 2.0], requires_grad=True)
        b = torch.tensor([2.0, 1.0, -1.0], requires_grad=True)
        _check_grad(torch.logaddexp, [a, b])


class TestLogaddexp2Backward:
    def test_numerical(self):
        a = torch.tensor([1.0, -1.0, 2.0], requires_grad=True)
        b = torch.tensor([2.0, 1.0, -1.0], requires_grad=True)
        _check_grad(torch.logaddexp2, [a, b])


# ============================================================================
# Tier 3: Less Common
# ============================================================================

class TestFminBackward:
    def test_numerical(self):
        a = torch.tensor([1.0, 3.0, 0.5], requires_grad=True)
        b = torch.tensor([2.0, 1.0, 2.0], requires_grad=True)
        _check_grad(torch.fmin, [a, b])


class TestFmaxBackward:
    def test_numerical(self):
        a = torch.tensor([1.0, 3.0, 0.5], requires_grad=True)
        b = torch.tensor([2.0, 1.0, 2.0], requires_grad=True)
        _check_grad(torch.fmax, [a, b])


class TestFmodBackward:
    def test_numerical(self):
        a = torch.tensor([7.0, 5.5, -3.0], requires_grad=True)
        b = torch.tensor([2.0, 3.0, 2.0], requires_grad=True)
        _check_grad(torch.fmod, [a, b])


class TestHypotBackward:
    def test_numerical(self):
        a = torch.tensor([3.0, 1.0, 2.0], dtype=torch.float64, requires_grad=True)
        b = torch.tensor([4.0, 2.0, 1.0], dtype=torch.float64, requires_grad=True)
        _check_grad(torch.hypot, [a, b])


class TestRemainderBackward:
    def test_numerical(self):
        a = torch.tensor([7.0, 5.5, 9.0], requires_grad=True)
        b = torch.tensor([2.0, 3.0, 4.0], requires_grad=True)
        _check_grad(torch.remainder, [a, b])


class TestRot90Backward:
    def test_basic(self):
        x = torch.tensor([[1.0, 2.0], [3.0, 4.0]], requires_grad=True)
        y = torch.rot90(x, 1, (0, 1))
        y.sum().backward()
        # rot90 is a permutation — grad should be all 1s
        np.testing.assert_allclose(x.grad.numpy(), [[1.0, 1.0], [1.0, 1.0]])

    def test_numerical(self):
        x = torch.tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], dtype=torch.float64, requires_grad=True)
        _check_grad(lambda a: torch.rot90(a, 1, (0, 1)), [x])


class TestTakeBackward:
    def test_basic(self):
        x = torch.tensor([1.0, 2.0, 3.0, 4.0], requires_grad=True)
        idx = torch.tensor([0, 2, 2], dtype=torch.long)
        y = torch.take(x, idx)
        y.sum().backward()
        # Index 0 taken once, index 2 taken twice
        np.testing.assert_allclose(x.grad.numpy(), [1.0, 0.0, 2.0, 0.0])


class TestTakeAlongDimBackward:
    def test_basic(self):
        x = torch.tensor([[1.0, 2.0], [3.0, 4.0]], requires_grad=True)
        idx = torch.tensor([[0, 0], [1, 0]], dtype=torch.long)
        y = torch.take_along_dim(x, idx, dim=1)
        y.sum().backward()
        # Grad scattered back along dim 1
        # idx[0,0]=0→x[0,0], idx[0,1]=0→x[0,0]: x[0,0] gets 2 grads
        # idx[1,0]=1→x[1,1], idx[1,1]=0→x[1,0]: x[1,0] gets 1, x[1,1] gets 1
        expected = np.array([[2.0, 0.0], [1.0, 1.0]])
        np.testing.assert_allclose(x.grad.numpy(), expected)


class TestCummaxBackward:
    def test_basic(self):
        x = torch.tensor([1.0, 3.0, 2.0, 5.0], requires_grad=True)
        values, indices = torch.cummax(x, 0)
        # cummax: [1, 3, 3, 5], indices: [0, 1, 1, 3]
        values.sum().backward()
        # Grad flows to positions indicated by indices
        # index 0 appears once, index 1 appears twice, index 3 appears once
        expected = np.array([1.0, 2.0, 0.0, 1.0])
        np.testing.assert_allclose(x.grad.numpy(), expected)


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
