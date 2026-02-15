import numpy as np
import mindtorch_v2 as torch


def test_add_mul_matmul_relu_sum():
    a = torch.tensor([[1.0, 2.0], [3.0, 4.0]])
    b = torch.tensor([[2.0, 0.5], [1.0, -1.0]])
    c = torch.add(a, b)
    d = torch.mul(a, b)
    e = torch.matmul(a, b)
    f = torch.relu(torch.tensor([-1.0, 2.0]))
    s = torch.sum(a, dim=1, keepdim=True)
    assert c.shape == (2, 2)
    assert d.shape == (2, 2)
    assert e.shape == (2, 2)
    assert f.shape == (2,)
    assert s.shape == (2, 1)


def test_abs_cpu():
    x = torch.tensor([-1.0, 0.5, 2.0])
    expected = np.abs(x.numpy())
    np.testing.assert_allclose(torch.abs(x).numpy(), expected)
    np.testing.assert_allclose(x.abs().numpy(), expected)


def test_neg_cpu():
    x = torch.tensor([-1.0, 0.5, 2.0])
    expected = -x.numpy()
    np.testing.assert_allclose(torch.neg(x).numpy(), expected)
    np.testing.assert_allclose((-x).numpy(), expected)


def test_exp_cpu():
    x = torch.tensor([0.5, 1.0, 2.0])
    expected = np.exp(x.numpy())
    np.testing.assert_allclose(torch.exp(x).numpy(), expected)
    np.testing.assert_allclose(x.exp().numpy(), expected)


def test_log_cpu():
    x = torch.tensor([0.5, 1.0, 2.0])
    expected = np.log(x.numpy())
    np.testing.assert_allclose(torch.log(x).numpy(), expected)
    np.testing.assert_allclose(x.log().numpy(), expected)


def test_sqrt_cpu():
    x = torch.tensor([0.5, 1.0, 4.0])
    expected = np.sqrt(x.numpy())
    np.testing.assert_allclose(torch.sqrt(x).numpy(), expected)
    np.testing.assert_allclose(x.sqrt().numpy(), expected)
