import math
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


def test_sign_cpu():
    x = torch.tensor([-2.0, 0.0, 3.0])
    expected = np.sign(x.numpy())
    np.testing.assert_allclose(torch.sign(x).numpy(), expected)
    np.testing.assert_allclose(x.sign().numpy(), expected)


def test_signbit_cpu():
    x = torch.tensor([-2.0, 0.0, 3.0])
    expected = np.signbit(x.numpy())
    np.testing.assert_array_equal(torch.signbit(x).numpy(), expected)
    np.testing.assert_array_equal(x.signbit().numpy(), expected)


def test_isnan_cpu():
    x = torch.tensor([0.0, float('nan'), 1.0])
    expected = np.isnan(x.numpy())
    np.testing.assert_array_equal(torch.isnan(x).numpy(), expected)
    np.testing.assert_array_equal(x.isnan().numpy(), expected)


def test_isinf_cpu():
    x = torch.tensor([0.0, float('inf'), -float('inf')])
    expected = np.isinf(x.numpy())
    np.testing.assert_array_equal(torch.isinf(x).numpy(), expected)
    np.testing.assert_array_equal(x.isinf().numpy(), expected)


def test_isfinite_cpu():
    x = torch.tensor([0.0, float('nan'), float('inf')])
    expected = np.isfinite(x.numpy())
    np.testing.assert_array_equal(torch.isfinite(x).numpy(), expected)
    np.testing.assert_array_equal(x.isfinite().numpy(), expected)


def test_sinh_cpu():
    x = torch.tensor([0.0, 0.5, 1.0])
    expected = np.sinh(x.numpy())
    np.testing.assert_allclose(torch.sinh(x).numpy(), expected)
    np.testing.assert_allclose(x.sinh().numpy(), expected)


def test_cosh_cpu():
    x = torch.tensor([0.0, 0.5, 1.0])
    expected = np.cosh(x.numpy())
    np.testing.assert_allclose(torch.cosh(x).numpy(), expected)
    np.testing.assert_allclose(x.cosh().numpy(), expected)


def test_erf_cpu():
    x = torch.tensor([-1.0, 0.0, 1.0])
    expected = np.vectorize(math.erf)(x.numpy())
    np.testing.assert_allclose(torch.erf(x).numpy(), expected)
    np.testing.assert_allclose(x.erf().numpy(), expected)


def test_erfc_cpu():
    x = torch.tensor([-1.0, 0.0, 1.0])
    expected = np.vectorize(math.erfc)(x.numpy())
    np.testing.assert_allclose(torch.erfc(x).numpy(), expected)
    np.testing.assert_allclose(x.erfc().numpy(), expected)


def test_softplus_cpu():
    x = torch.tensor([-1.0, 0.0, 1.0])
    expected = np.log1p(np.exp(x.numpy()))
    np.testing.assert_allclose(torch.softplus(x).numpy(), expected)
    np.testing.assert_allclose(x.softplus().numpy(), expected)


def test_clamp_cpu():
    x = torch.tensor([-1.0, 0.5, 2.0])
    expected = np.clip(x.numpy(), 0.0, 1.0)
    np.testing.assert_allclose(torch.clamp(x, 0.0, 1.0).numpy(), expected)
    np.testing.assert_allclose(x.clamp(0.0, 1.0).numpy(), expected)


def test_clamp_min_cpu():
    x = torch.tensor([-1.0, 0.5, 2.0])
    expected = np.maximum(x.numpy(), 0.5)
    np.testing.assert_allclose(torch.clamp_min(x, 0.5).numpy(), expected)
    np.testing.assert_allclose(x.clamp_min(0.5).numpy(), expected)


def test_clamp_max_cpu():
    x = torch.tensor([-1.0, 0.5, 2.0])
    expected = np.minimum(x.numpy(), 0.5)
    np.testing.assert_allclose(torch.clamp_max(x, 0.5).numpy(), expected)
    np.testing.assert_allclose(x.clamp_max(0.5).numpy(), expected)


def test_relu6_cpu():
    x = torch.tensor([-1.0, 0.5, 7.0])
    expected = np.minimum(np.maximum(x.numpy(), 0.0), 6.0)
    np.testing.assert_allclose(torch.relu6(x).numpy(), expected)
    np.testing.assert_allclose(x.relu6().numpy(), expected)


def test_hardtanh_cpu():
    x = torch.tensor([-2.0, 0.0, 2.0])
    expected = np.clip(x.numpy(), -1.0, 1.0)
    np.testing.assert_allclose(torch.hardtanh(x, -1.0, 1.0).numpy(), expected)
    np.testing.assert_allclose(x.hardtanh(-1.0, 1.0).numpy(), expected)


def test_min_cpu():
    x = torch.tensor([1.0, 2.0, 3.0])
    y = torch.tensor([3.0, 1.0, 2.0])
    expected = np.minimum(x.numpy(), y.numpy())
    np.testing.assert_allclose(torch.min(x, y).numpy(), expected)
    np.testing.assert_allclose(x.min(y).numpy(), expected)


def test_max_cpu():
    x = torch.tensor([1.0, 2.0, 3.0])
    y = torch.tensor([3.0, 1.0, 2.0])
    expected = np.maximum(x.numpy(), y.numpy())
    np.testing.assert_allclose(torch.max(x, y).numpy(), expected)
    np.testing.assert_allclose(x.max(y).numpy(), expected)


def test_amin_cpu():
    x = torch.tensor([[1.0, 2.0], [3.0, 0.5]])
    expected = np.amin(x.numpy(), axis=1)
    np.testing.assert_allclose(torch.amin(x, dim=1).numpy(), expected)


def test_amax_cpu():
    x = torch.tensor([[1.0, 2.0], [3.0, 0.5]])
    expected = np.amax(x.numpy(), axis=1)
    np.testing.assert_allclose(torch.amax(x, dim=1).numpy(), expected)


def test_fmin_cpu():
    x = torch.tensor([1.0, float('nan'), 2.0])
    y = torch.tensor([0.5, 1.0, float('nan')])
    expected = np.fmin(x.numpy(), y.numpy())
    np.testing.assert_allclose(torch.fmin(x, y).numpy(), expected)


def test_fmax_cpu():
    x = torch.tensor([1.0, float('nan'), 2.0])
    y = torch.tensor([0.5, 1.0, float('nan')])
    expected = np.fmax(x.numpy(), y.numpy())
    np.testing.assert_allclose(torch.fmax(x, y).numpy(), expected)


def test_where_scalar_cpu():
    cond = torch.tensor([True, False, True])
    x = torch.tensor([1.0, 2.0, 3.0])
    expected = np.where(cond.numpy(), x.numpy(), 0.5)
    np.testing.assert_allclose(torch.where(cond, x, 0.5).numpy(), expected)


def test_where_tensor_cpu():
    cond = torch.tensor([True, False, True])
    x = torch.tensor([1.0, 2.0, 3.0])
    y = torch.tensor([3.0, 2.0, 1.0])
    expected = np.where(cond.numpy(), x.numpy(), y.numpy())
    np.testing.assert_allclose(torch.where(cond, x, y).numpy(), expected)


def test_tensor_where_cpu():
    cond = torch.tensor([True, False, True])
    x = torch.tensor([1.0, 2.0, 3.0])
    y = torch.tensor([3.0, 2.0, 1.0])
    expected = np.where(cond.numpy(), x.numpy(), y.numpy())
    np.testing.assert_allclose(x.where(cond, y).numpy(), expected)


def test_tensor_where_scalar_cpu():
    cond = torch.tensor([True, False, True])
    x = torch.tensor([1.0, 2.0, 3.0])
    expected = np.where(cond.numpy(), x.numpy(), 0.5)
    np.testing.assert_allclose(x.where(cond, 0.5).numpy(), expected)


def test_lerp_scalar_cpu():
    x = torch.tensor([0.0, 1.0, 2.0])
    y = torch.tensor([2.0, 3.0, 4.0])
    weight = 0.25
    expected = x.numpy() + weight * (y.numpy() - x.numpy())
    np.testing.assert_allclose(torch.lerp(x, y, weight).numpy(), expected)
    np.testing.assert_allclose(x.lerp(y, weight).numpy(), expected)


def test_lerp_tensor_cpu():
    x = torch.tensor([0.0, 1.0, 2.0])
    y = torch.tensor([2.0, 3.0, 4.0])
    weight = torch.tensor([0.0, 0.5, 1.0])
    expected = x.numpy() + weight.numpy() * (y.numpy() - x.numpy())
    np.testing.assert_allclose(torch.lerp(x, y, weight).numpy(), expected)
    np.testing.assert_allclose(x.lerp(y, weight).numpy(), expected)


def test_atan_cpu():
    x = torch.tensor([-1.0, 0.0, 1.0])
    expected = np.arctan(x.numpy())
    np.testing.assert_allclose(torch.atan(x).numpy(), expected)
    np.testing.assert_allclose(x.atan().numpy(), expected)


def test_atan2_cpu():
    x = torch.tensor([1.0, -1.0, 2.0])
    y = torch.tensor([2.0, 2.0, -2.0])
    expected = np.arctan2(x.numpy(), y.numpy())
    np.testing.assert_allclose(torch.atan2(x, y).numpy(), expected)
    np.testing.assert_allclose(x.atan2(y).numpy(), expected)


def test_asin_cpu():
    x = torch.tensor([-0.5, 0.0, 0.5])
    expected = np.arcsin(x.numpy())
    np.testing.assert_allclose(torch.asin(x).numpy(), expected)
    np.testing.assert_allclose(x.asin().numpy(), expected)


def test_acos_cpu():
    x = torch.tensor([-0.5, 0.0, 0.5])
    expected = np.arccos(x.numpy())
    np.testing.assert_allclose(torch.acos(x).numpy(), expected)
    np.testing.assert_allclose(x.acos().numpy(), expected)
