"""PyTorch-parity gradient tests for mindtorch_v2 autograd backward ops."""
import math

import numpy as np
import pytest

import mindtorch_v2 as torch
import mindtorch_v2.nn.functional as F


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _check_grad(x, expected, *, atol=1e-5, rtol=1e-5):
    """Assert that x.grad matches expected (numpy array or list)."""
    assert x.grad is not None, "x.grad is None"
    actual = x.grad.numpy()
    np.testing.assert_allclose(actual, np.array(expected, dtype=actual.dtype),
                               atol=atol, rtol=rtol)


def _tensor(vals, requires_grad=True):
    return torch.tensor(vals, dtype=torch.float32).requires_grad_(requires_grad)


# ===================================================================
# Phase 1: Basic unary/binary math ops
# ===================================================================

class TestExpBackward:
    def test_basic(self):
        x = _tensor([1.0, 2.0, -1.0])
        y = torch.exp(x)
        y.sum().backward()
        _check_grad(x, [math.exp(v) for v in [1.0, 2.0, -1.0]])

    def test_zeros(self):
        x = _tensor([0.0])
        torch.exp(x).sum().backward()
        _check_grad(x, [1.0])


class TestLogBackward:
    def test_basic(self):
        x = _tensor([1.0, 2.0, 0.5])
        torch.log(x).sum().backward()
        _check_grad(x, [1.0 / v for v in [1.0, 2.0, 0.5]])


class TestLog2Backward:
    def test_basic(self):
        x = _tensor([1.0, 2.0, 4.0])
        torch.log2(x).sum().backward()
        _check_grad(x, [1.0 / (v * math.log(2.0)) for v in [1.0, 2.0, 4.0]])


class TestExp2Backward:
    def test_basic(self):
        x = _tensor([0.0, 1.0, 2.0])
        torch.exp2(x).sum().backward()
        _check_grad(x, [2.0 ** v * math.log(2.0) for v in [0.0, 1.0, 2.0]])


class TestSqrtBackward:
    def test_basic(self):
        x = _tensor([1.0, 4.0, 9.0])
        torch.sqrt(x).sum().backward()
        _check_grad(x, [1.0 / (2.0 * math.sqrt(v)) for v in [1.0, 4.0, 9.0]])


class TestRsqrtBackward:
    def test_basic(self):
        x = _tensor([1.0, 4.0, 9.0])
        torch.rsqrt(x).sum().backward()
        expected = [-0.5 * v ** (-1.5) for v in [1.0, 4.0, 9.0]]
        _check_grad(x, expected)


class TestSigmoidBackward:
    def test_basic(self):
        x = _tensor([0.0, 1.0, -1.0])
        torch.sigmoid(x).sum().backward()
        def sig(v):
            s = 1.0 / (1.0 + math.exp(-v))
            return s * (1.0 - s)
        _check_grad(x, [sig(v) for v in [0.0, 1.0, -1.0]])


class TestTanhBackward:
    def test_basic(self):
        x = _tensor([0.0, 1.0, -1.0])
        torch.tanh(x).sum().backward()
        _check_grad(x, [1.0 - math.tanh(v) ** 2 for v in [0.0, 1.0, -1.0]])


class TestSinBackward:
    def test_basic(self):
        x = _tensor([0.0, math.pi / 4, math.pi / 2])
        torch.sin(x).sum().backward()
        _check_grad(x, [math.cos(v) for v in [0.0, math.pi / 4, math.pi / 2]])


class TestCosBackward:
    def test_basic(self):
        x = _tensor([0.0, math.pi / 4, math.pi / 2])
        torch.cos(x).sum().backward()
        _check_grad(x, [-math.sin(v) for v in [0.0, math.pi / 4, math.pi / 2]])


class TestErfBackward:
    def test_basic(self):
        x = _tensor([0.0, 0.5, -0.5])
        torch.erf(x).sum().backward()
        def erf_grad(v):
            return (2.0 / math.sqrt(math.pi)) * math.exp(-v * v)
        _check_grad(x, [erf_grad(v) for v in [0.0, 0.5, -0.5]])


class TestSoftplusBackward:
    def test_basic(self):
        x = _tensor([0.0, 1.0, -1.0])
        x.softplus().sum().backward()
        def sig(v):
            return 1.0 / (1.0 + math.exp(-v))
        _check_grad(x, [sig(v) for v in [0.0, 1.0, -1.0]])


class TestPowBackward:
    def test_base_grad(self):
        a = _tensor([2.0, 3.0])
        b = torch.tensor([3.0, 2.0])
        (a ** b).sum().backward()
        # d/da (a^b) = b * a^(b-1)
        _check_grad(a, [3.0 * 2.0 ** 2.0, 2.0 * 3.0 ** 1.0])

    def test_exp_grad(self):
        a = torch.tensor([2.0, 3.0])
        b = _tensor([3.0, 2.0])
        (a ** b).sum().backward()
        # d/db (a^b) = a^b * log(a)
        _check_grad(b, [2.0 ** 3.0 * math.log(2.0), 3.0 ** 2.0 * math.log(3.0)])


# ===================================================================
# Phase 2: Fixed passthrough ops
# ===================================================================

class TestSoftmaxBackward:
    def test_basic(self):
        x = _tensor([[1.0, 2.0, 3.0]])
        s = F.softmax(x, dim=-1)
        # Pick one output element: d softmax_i / d x_j = s_i (delta_ij - s_j)
        loss = s[0, 1]  # second element
        loss.backward()
        s_np = s.detach().numpy()[0]
        # Jacobian row for element 1
        expected = s_np[1] * (-s_np)
        expected[1] += s_np[1]
        _check_grad(x, [expected])

    def test_dim0(self):
        x = _tensor([[1.0], [2.0], [3.0]])
        s = F.softmax(x, dim=0)
        s.sum().backward()
        # sum of softmax is 1 (constant), gradient should be 0
        _check_grad(x, np.zeros_like(x.detach().numpy()), atol=1e-5)


class TestGeluBackward:
    def test_basic(self):
        x = _tensor([0.0, 1.0, -1.0])
        F.gelu(x).sum().backward()
        def gelu_grad(v):
            cdf = 0.5 * (1.0 + math.erf(v / math.sqrt(2.0)))
            pdf = math.exp(-v * v / 2.0) / math.sqrt(2.0 * math.pi)
            return cdf + v * pdf
        _check_grad(x, [gelu_grad(v) for v in [0.0, 1.0, -1.0]])


class TestLayerNormBackward:
    def test_basic(self):
        x = _tensor([[1.0, 2.0, 3.0, 4.0]])
        y = F.layer_norm(x, [4])
        y.sum().backward()
        assert x.grad is not None
        assert x.grad.shape == x.shape
        # sum of layer_norm output is constant-ish; gradient should be near-zero
        np.testing.assert_allclose(x.grad.numpy(), np.zeros_like(x.detach().numpy()), atol=1e-5)

    def test_with_weight_bias(self):
        x = _tensor([[1.0, 2.0, 3.0]])
        w = _tensor([1.0, 1.0, 1.0])
        b = _tensor([0.0, 0.0, 0.0])
        y = F.layer_norm(x, [3], weight=w, bias=b)
        (y * y).sum().backward()
        assert x.grad is not None
        assert x.grad.shape == x.shape


# ===================================================================
# Phase 3: View/shape ops
# ===================================================================

class TestSqueezeBackward:
    def test_basic(self):
        x = _tensor([[1.0, 2.0, 3.0]])  # shape (1, 3)
        y = x.squeeze(0)  # shape (3,)
        y.sum().backward()
        _check_grad(x, np.ones((1, 3)))


class TestUnsqueezeBackward:
    def test_basic(self):
        x = _tensor([1.0, 2.0, 3.0])  # shape (3,)
        y = x.unsqueeze(0)  # shape (1, 3)
        y.sum().backward()
        _check_grad(x, np.ones(3))


class TestExpandBackward:
    def test_basic(self):
        x = _tensor([[1.0], [2.0]])  # shape (2, 1)
        y = x.expand(2, 3)  # shape (2, 3)
        y.sum().backward()
        # gradient for expand: sum over expanded dim
        _check_grad(x, [[3.0], [3.0]])


class TestPermuteBackward:
    def test_basic(self):
        x = _tensor([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])  # (3, 2)
        y = x.permute(1, 0)  # (2, 3)
        y.sum().backward()
        _check_grad(x, np.ones((3, 2)))


class TestNarrowBackward:
    def test_basic(self):
        x = _tensor([1.0, 2.0, 3.0, 4.0, 5.0])
        y = x.narrow(0, 1, 3)  # elements [2, 3, 4]
        y.sum().backward()
        _check_grad(x, [0.0, 1.0, 1.0, 1.0, 0.0])


class TestSelectBackward:
    def test_basic(self):
        x = _tensor([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])  # (3, 2)
        y = x.select(0, 1)  # [3.0, 4.0]
        y.sum().backward()
        _check_grad(x, [[0.0, 0.0], [1.0, 1.0], [0.0, 0.0]])


class TestCatBackward:
    def test_basic(self):
        a = _tensor([1.0, 2.0])
        b = _tensor([3.0, 4.0, 5.0])
        y = torch.cat([a, b], dim=0)
        y.sum().backward()
        _check_grad(a, [1.0, 1.0])
        _check_grad(b, [1.0, 1.0, 1.0])

    def test_dim1(self):
        a = _tensor([[1.0, 2.0]])
        b = _tensor([[3.0]])
        y = torch.cat([a, b], dim=1)
        y.sum().backward()
        _check_grad(a, [[1.0, 1.0]])
        _check_grad(b, [[1.0]])


class TestStackBackward:
    def test_basic(self):
        a = _tensor([1.0, 2.0])
        b = _tensor([3.0, 4.0])
        y = torch.stack([a, b], dim=0)
        y.sum().backward()
        _check_grad(a, [1.0, 1.0])
        _check_grad(b, [1.0, 1.0])


class TestSplitBackward:
    def test_basic(self):
        x = _tensor([1.0, 2.0, 3.0, 4.0])
        a, b = x.split(2)
        (a.sum() + b.sum()).backward()
        _check_grad(x, [1.0, 1.0, 1.0, 1.0])

    def test_sections(self):
        x = _tensor([1.0, 2.0, 3.0, 4.0, 5.0])
        a, b = x.split([2, 3])
        (a.sum() + b.sum()).backward()
        _check_grad(x, [1.0, 1.0, 1.0, 1.0, 1.0])

    def test_partial_use(self):
        x = _tensor([1.0, 2.0, 3.0, 4.0])
        a, b = x.split(2)
        a.sum().backward()
        _check_grad(x, [1.0, 1.0, 0.0, 0.0])


class TestUnbindBackward:
    def test_basic(self):
        x = _tensor([[1.0, 2.0], [3.0, 4.0]])
        a, b = torch.unbind(x, dim=0)
        (a.sum() + b.sum()).backward()
        _check_grad(x, [[1.0, 1.0], [1.0, 1.0]])

    def test_partial_use(self):
        x = _tensor([[1.0, 2.0], [3.0, 4.0]])
        a, b = torch.unbind(x, dim=0)
        a.sum().backward()
        _check_grad(x, [[1.0, 1.0], [0.0, 0.0]])


# ===================================================================
# Phase 4: Advanced ops
# ===================================================================

class TestClampBackward:
    def test_basic(self):
        x = _tensor([-2.0, -0.5, 0.5, 2.0])
        y = torch.clamp(x, -1.0, 1.0)
        y.sum().backward()
        # grad is 1 where -1 <= x <= 1, else 0
        _check_grad(x, [0.0, 1.0, 1.0, 0.0])

    def test_min_only(self):
        x = _tensor([-2.0, 0.0, 2.0])
        torch.clamp(x, 0.0).sum().backward()
        _check_grad(x, [0.0, 1.0, 1.0])


class TestHardtanhBackward:
    def test_basic(self):
        x = _tensor([-2.0, -0.5, 0.5, 2.0])
        x.hardtanh(-1.0, 1.0).sum().backward()
        _check_grad(x, [0.0, 1.0, 1.0, 0.0])


class TestRelu6Backward:
    def test_basic(self):
        x = _tensor([-1.0, 0.0, 3.0, 6.0, 7.0])
        x.relu6().sum().backward()
        # grad is 1 where 0 <= x <= 6
        _check_grad(x, [0.0, 1.0, 1.0, 1.0, 0.0])


class TestLogSoftmaxBackward:
    def test_basic(self):
        x = _tensor([[1.0, 2.0, 3.0]])
        ls = F.log_softmax(x, dim=-1)
        ls.sum().backward()
        # sum of log_softmax is log(1) + ... hmm, not constant
        # sum(log_softmax(x)) grad = 1 - n * softmax(x)
        s = np.exp(ls.detach().numpy())
        n = x.shape[-1]
        expected = 1.0 - n * s
        _check_grad(x, expected, atol=1e-4)


class TestBatchNormBackward:
    def test_training(self):
        x = _tensor([[1.0, 2.0], [3.0, 4.0]].copy())
        x_4d = x.unsqueeze(-1).unsqueeze(-1)  # (2, 2, 1, 1) for batch_norm
        y = F.batch_norm(x_4d, None, None, training=True)
        y.sum().backward()
        assert x.grad is not None
        assert x.grad.shape == x.shape


class TestEmbeddingBackward:
    def test_basic(self):
        w = _tensor([[0.1, 0.2], [0.3, 0.4], [0.5, 0.6]])
        idx = torch.tensor([0, 2, 1, 0], dtype=torch.int64)
        y = F.embedding(idx, w)
        y.sum().backward()
        # Row 0 used twice, row 1 once, row 2 once
        expected = np.array([[2.0, 2.0], [1.0, 1.0], [1.0, 1.0]], dtype=np.float32)
        _check_grad(w, expected)


class TestWhereBackward:
    def test_basic(self):
        cond = torch.tensor([True, False, True])
        x = _tensor([1.0, 2.0, 3.0])
        y = _tensor([4.0, 5.0, 6.0])
        z = torch.where(cond, x, y)
        z.sum().backward()
        _check_grad(x, [1.0, 0.0, 1.0])
        _check_grad(y, [0.0, 1.0, 0.0])


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
