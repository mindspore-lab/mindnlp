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


def test_sin_cpu():
    x = torch.tensor([0.0, 0.5, 1.0])
    expected = np.sin(x.numpy())
    np.testing.assert_allclose(torch.sin(x).numpy(), expected)
    np.testing.assert_allclose(x.sin().numpy(), expected)


def test_cos_cpu():
    x = torch.tensor([0.0, 0.5, 1.0])
    expected = np.cos(x.numpy())
    np.testing.assert_allclose(torch.cos(x).numpy(), expected)
    np.testing.assert_allclose(x.cos().numpy(), expected)


def test_tan_cpu():
    x = torch.tensor([0.0, 0.5, 1.0])
    expected = np.tan(x.numpy())
    np.testing.assert_allclose(torch.tan(x).numpy(), expected)
    np.testing.assert_allclose(x.tan().numpy(), expected)


def test_tanh_cpu():
    x = torch.tensor([0.0, 0.5, 1.0])
    expected = np.tanh(x.numpy())
    np.testing.assert_allclose(torch.tanh(x).numpy(), expected)
    np.testing.assert_allclose(x.tanh().numpy(), expected)


def test_sigmoid_cpu():
    x = torch.tensor([-1.0, 0.0, 1.0])
    expected = 1.0 / (1.0 + np.exp(-x.numpy()))
    np.testing.assert_allclose(torch.sigmoid(x).numpy(), expected)
    np.testing.assert_allclose(x.sigmoid().numpy(), expected)


def test_floor_cpu():
    x = torch.tensor([-1.2, 0.5, 1.7])
    expected = np.floor(x.numpy())
    np.testing.assert_allclose(torch.floor(x).numpy(), expected)
    np.testing.assert_allclose(x.floor().numpy(), expected)


def test_ceil_cpu():
    x = torch.tensor([-1.2, 0.5, 1.7])
    expected = np.ceil(x.numpy())
    np.testing.assert_allclose(torch.ceil(x).numpy(), expected)
    np.testing.assert_allclose(x.ceil().numpy(), expected)


def test_round_cpu():
    x = torch.tensor([-1.2, 0.5, 1.7])
    expected = np.round(x.numpy())
    np.testing.assert_allclose(torch.round(x).numpy(), expected)
    np.testing.assert_allclose(x.round().numpy(), expected)


def test_trunc_cpu():
    x = torch.tensor([-1.2, 0.5, 1.7])
    expected = np.trunc(x.numpy())
    np.testing.assert_allclose(torch.trunc(x).numpy(), expected)
    np.testing.assert_allclose(x.trunc().numpy(), expected)


def test_frac_cpu():
    x = torch.tensor([-1.2, 0.5, 1.7])
    expected = x.numpy() - np.trunc(x.numpy())
    np.testing.assert_allclose(torch.frac(x).numpy(), expected)
    np.testing.assert_allclose(x.frac().numpy(), expected)


def test_pow_cpu():
    x = torch.tensor([1.0, 2.0, 3.0])
    expected = np.power(x.numpy(), 2.0)
    np.testing.assert_allclose(torch.pow(x, 2.0).numpy(), expected)


def test_log2_cpu():
    x = torch.tensor([0.5, 1.0, 2.0])
    expected = np.log2(x.numpy())
    np.testing.assert_allclose(torch.log2(x).numpy(), expected)
    np.testing.assert_allclose(x.log2().numpy(), expected)


def test_log10_cpu():
    x = torch.tensor([0.5, 1.0, 10.0])
    expected = np.log10(x.numpy())
    np.testing.assert_allclose(torch.log10(x).numpy(), expected)
    np.testing.assert_allclose(x.log10().numpy(), expected)


def test_exp2_cpu():
    x = torch.tensor([0.0, 1.0, 2.0])
    expected = np.exp2(x.numpy())
    np.testing.assert_allclose(torch.exp2(x).numpy(), expected)
    np.testing.assert_allclose(x.exp2().numpy(), expected)


def test_rsqrt_cpu():
    x = torch.tensor([0.5, 1.0, 4.0])
    expected = 1.0 / np.sqrt(x.numpy())
    np.testing.assert_allclose(torch.rsqrt(x).numpy(), expected)
    np.testing.assert_allclose(x.rsqrt().numpy(), expected)
