import math
import numpy as np
import pytest
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


def test_asinh_cpu():
    x = torch.tensor([-1.0, 0.0, 1.0])
    expected = np.arcsinh(x.numpy())
    np.testing.assert_allclose(torch.asinh(x).numpy(), expected)
    np.testing.assert_allclose(x.asinh().numpy(), expected)


def test_acosh_cpu():
    x = torch.tensor([1.0, 2.0, 3.0])
    expected = np.arccosh(x.numpy())
    np.testing.assert_allclose(torch.acosh(x).numpy(), expected)
    np.testing.assert_allclose(x.acosh().numpy(), expected)


def test_atanh_cpu():
    x = torch.tensor([-0.5, 0.0, 0.5])
    expected = np.arctanh(x.numpy())
    np.testing.assert_allclose(torch.atanh(x).numpy(), expected)
    np.testing.assert_allclose(x.atanh().numpy(), expected)


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


def test_all_cpu():
    x = torch.tensor([[True, False], [True, True]], dtype=torch.bool)
    expected = np.all(x.numpy(), axis=1)
    np.testing.assert_array_equal(torch.all(x, dim=1).numpy(), expected)
    expected_keep = np.all(x.numpy(), axis=1, keepdims=True)
    np.testing.assert_array_equal(torch.all(x, dim=1, keepdim=True).numpy(), expected_keep)


def test_any_cpu():
    x = torch.tensor([[False, False], [True, False]], dtype=torch.bool)
    expected = np.any(x.numpy(), axis=1)
    np.testing.assert_array_equal(torch.any(x, dim=1).numpy(), expected)
    expected_keep = np.any(x.numpy(), axis=1, keepdims=True)
    np.testing.assert_array_equal(torch.any(x, dim=1, keepdim=True).numpy(), expected_keep)


def test_argmax_cpu():
    x = torch.tensor([[1.0, 3.0, 2.0], [4.0, 0.0, 5.0]])
    expected = np.argmax(x.numpy(), axis=1)
    np.testing.assert_array_equal(torch.argmax(x, dim=1).numpy(), expected)
    expected_keep = np.argmax(x.numpy(), axis=1)
    np.testing.assert_array_equal(torch.argmax(x, dim=1, keepdim=True).numpy(), expected_keep.reshape(2, 1))


def test_allclose_cpu():
    x = torch.tensor([1.0, 1.0, 1.0001])
    y = torch.tensor([1.0, 1.0, 1.0])
    assert torch.allclose(x, y, rtol=1e-3, atol=1e-4)
    assert not torch.allclose(x, y, rtol=1e-6, atol=1e-8)


def test_isclose_cpu():
    x = torch.tensor([1.0, 1.0, 1.0001])
    y = torch.tensor([1.0, 1.0, 1.0])
    expected = np.isclose(x.numpy(), y.numpy(), rtol=1e-3, atol=1e-4)
    np.testing.assert_array_equal(torch.isclose(x, y, rtol=1e-3, atol=1e-4).numpy(), expected)


def test_equal_cpu():
    x = torch.tensor([1.0, 2.0])
    y = torch.tensor([1.0, 2.0])
    z = torch.tensor([1.0, 3.0])
    assert torch.equal(x, y)
    assert not torch.equal(x, z)


def test_argmin_cpu():
    x = torch.tensor([[1.0, 3.0, 2.0], [4.0, 0.0, 5.0]])
    expected = np.argmin(x.numpy(), axis=1)
    np.testing.assert_array_equal(torch.argmin(x, dim=1).numpy(), expected)
    expected_keep = np.argmin(x.numpy(), axis=1)
    np.testing.assert_array_equal(torch.argmin(x, dim=1, keepdim=True).numpy(), expected_keep.reshape(2, 1))


def test_count_nonzero_cpu():
    x = torch.tensor([[0.0, 1.0, 2.0], [0.0, 0.0, 3.0]])
    expected = np.count_nonzero(x.numpy(), axis=1)
    np.testing.assert_array_equal(torch.count_nonzero(x, dim=1).numpy(), expected)
    expected_keep = np.count_nonzero(x.numpy(), axis=1, keepdims=True)
    np.testing.assert_array_equal(
        torch.count_nonzero(x, dim=1, keepdim=True).numpy(),
        expected_keep,
    )


def test_cumsum_cpu():
    x = torch.tensor([[1.0, 2.0], [3.0, 4.0]])
    expected = np.cumsum(x.numpy(), axis=1)
    np.testing.assert_allclose(torch.cumsum(x, dim=1).numpy(), expected)


def test_cumprod_cpu():
    x = torch.tensor([[1.0, 2.0], [3.0, 4.0]])
    expected = np.cumprod(x.numpy(), axis=1)
    np.testing.assert_allclose(torch.cumprod(x, dim=1).numpy(), expected)


def test_cummax_cpu():
    x = torch.tensor([[1.0, 3.0, 2.0], [4.0, 0.0, 5.0]])
    values, indices = torch.cummax(x, dim=1)
    expected_vals = np.maximum.accumulate(x.numpy(), axis=1)
    np.testing.assert_allclose(values.numpy(), expected_vals)
    expected_idx = np.array([[0, 1, 1], [0, 0, 2]], dtype=np.int64)
    np.testing.assert_array_equal(indices.numpy(), expected_idx)


def test_argsort_cpu():
    x = torch.tensor([[3.0, 1.0, 2.0], [0.0, -1.0, 5.0]])
    expected = np.argsort(x.numpy(), axis=1)
    np.testing.assert_array_equal(torch.argsort(x, dim=1).numpy(), expected)
    expected_desc = np.argsort(-x.numpy(), axis=1)
    np.testing.assert_array_equal(torch.argsort(x, dim=1, descending=True).numpy(), expected_desc)


def test_sort_cpu():
    x = torch.tensor([[3.0, 1.0, 2.0], [0.0, -1.0, 5.0]])
    values, indices = torch.sort(x, dim=1)
    expected_indices = np.argsort(x.numpy(), axis=1)
    expected_values = np.take_along_axis(x.numpy(), expected_indices, axis=1)
    np.testing.assert_allclose(values.numpy(), expected_values)
    np.testing.assert_array_equal(indices.numpy(), expected_indices)


def test_topk_cpu():
    x = torch.tensor([[3.0, 1.0, 2.0], [0.0, -1.0, 5.0]])
    values, indices = torch.topk(x, k=2, dim=1, largest=True, sorted=True)
    expected_indices = np.argsort(-x.numpy(), axis=1)[:, :2]
    expected_values = np.take_along_axis(x.numpy(), expected_indices, axis=1)
    np.testing.assert_allclose(values.numpy(), expected_values)
    np.testing.assert_array_equal(indices.numpy(), expected_indices)


def test_sort_out_cpu():
    x = torch.tensor([[3.0, 1.0, 2.0]])
    out_values = torch.empty((1, 3))
    out_indices = torch.empty((1, 3), dtype=torch.int64)
    values, indices = torch.sort(x, dim=1, out=(out_values, out_indices))
    assert values is out_values
    assert indices is out_indices
    expected_indices = np.argsort(x.numpy(), axis=1)
    expected_values = np.take_along_axis(x.numpy(), expected_indices, axis=1)
    np.testing.assert_allclose(out_values.numpy(), expected_values)
    np.testing.assert_array_equal(out_indices.numpy(), expected_indices)


def test_topk_out_cpu():
    x = torch.tensor([[3.0, 1.0, 2.0]])
    out_values = torch.empty((1, 2))
    out_indices = torch.empty((1, 2), dtype=torch.int64)
    values, indices = torch.topk(x, k=2, dim=1, out=(out_values, out_indices))
    assert values is out_values
    assert indices is out_indices
    expected_indices = np.argsort(-x.numpy(), axis=1)[:, :2]
    expected_values = np.take_along_axis(x.numpy(), expected_indices, axis=1)
    np.testing.assert_allclose(out_values.numpy(), expected_values)
    np.testing.assert_array_equal(out_indices.numpy(), expected_indices)


def test_stack_cpu():
    a = torch.tensor([1.0, 2.0])
    b = torch.tensor([3.0, 4.0])
    expected = np.stack([a.numpy(), b.numpy()], axis=0)
    np.testing.assert_allclose(torch.stack([a, b], dim=0).numpy(), expected)


def test_cat_cpu():
    a = torch.tensor([[1.0, 2.0]])
    b = torch.tensor([[3.0, 4.0]])
    expected = np.concatenate([a.numpy(), b.numpy()], axis=0)
    np.testing.assert_allclose(torch.cat([a, b], dim=0).numpy(), expected)


def test_concat_cpu():
    a = torch.tensor([[1.0, 2.0]])
    b = torch.tensor([[3.0, 4.0]])
    expected = np.concatenate([a.numpy(), b.numpy()], axis=1)
    np.testing.assert_allclose(torch.concat([a, b], dim=1).numpy(), expected)


def test_concatenate_cpu():
    a = torch.tensor([[1.0, 2.0]])
    b = torch.tensor([[3.0, 4.0]])
    expected = np.concatenate([a.numpy(), b.numpy()], axis=0)
    np.testing.assert_allclose(torch.concatenate([a, b], dim=0).numpy(), expected)
    expected_axis = np.concatenate([a.numpy(), b.numpy()], axis=1)
    np.testing.assert_allclose(torch.concatenate([a, b], axis=1).numpy(), expected_axis)


def test_stack_out_cpu():
    a = torch.tensor([1.0, 2.0])
    b = torch.tensor([3.0, 4.0])
    out = torch.empty((2, 2))
    result = torch.stack([a, b], dim=0, out=out)
    assert result is out
    expected = np.stack([a.numpy(), b.numpy()], axis=0)
    np.testing.assert_allclose(out.numpy(), expected)


def test_cat_out_cpu():
    a = torch.tensor([[1.0, 2.0]])
    b = torch.tensor([[3.0, 4.0]])
    out = torch.empty((2, 2))
    result = torch.cat([a, b], dim=0, out=out)
    assert result is out
    expected = np.concatenate([a.numpy(), b.numpy()], axis=0)
    np.testing.assert_allclose(out.numpy(), expected)


def test_hstack_cpu():
    a = torch.tensor([1.0, 2.0])
    b = torch.tensor([3.0, 4.0])
    expected = np.hstack([a.numpy(), b.numpy()])
    np.testing.assert_allclose(torch.hstack([a, b]).numpy(), expected)


def test_vstack_cpu():
    a = torch.tensor([1.0, 2.0])
    b = torch.tensor([3.0, 4.0])
    expected = np.vstack([a.numpy(), b.numpy()])
    np.testing.assert_allclose(torch.vstack([a, b]).numpy(), expected)


def test_row_stack_cpu():
    a = torch.tensor([1.0, 2.0])
    b = torch.tensor([3.0, 4.0])
    expected = np.vstack([a.numpy(), b.numpy()])
    np.testing.assert_allclose(torch.row_stack([a, b]).numpy(), expected)
    c = torch.tensor([[1.0, 2.0]])
    d = torch.tensor([[3.0, 4.0]])
    expected_2d = np.vstack([c.numpy(), d.numpy()])
    np.testing.assert_allclose(torch.row_stack([c, d]).numpy(), expected_2d)


def test_dstack_cpu():
    a = torch.tensor([1.0, 2.0])
    b = torch.tensor([3.0, 4.0])
    expected = np.dstack([a.numpy(), b.numpy()])
    np.testing.assert_allclose(torch.dstack([a, b]).numpy(), expected)
    c = torch.tensor([[1.0, 2.0], [3.0, 4.0]])
    d = torch.tensor([[5.0, 6.0], [7.0, 8.0]])
    expected_2d = np.dstack([c.numpy(), d.numpy()])
    np.testing.assert_allclose(torch.dstack([c, d]).numpy(), expected_2d)


def test_column_stack_cpu():
    a = torch.tensor([1.0, 2.0])
    b = torch.tensor([3.0, 4.0])
    expected = np.column_stack([a.numpy(), b.numpy()])
    np.testing.assert_allclose(torch.column_stack([a, b]).numpy(), expected)


def test_hstack_out_cpu():
    a = torch.tensor([1.0, 2.0])
    b = torch.tensor([3.0, 4.0])
    out = torch.empty((4,))
    result = torch.hstack([a, b], out=out)
    assert result is out
    expected = np.hstack([a.numpy(), b.numpy()])
    np.testing.assert_allclose(out.numpy(), expected)


def test_pad_sequence_cpu_right():
    a = torch.tensor([1.0, 2.0])
    b = torch.tensor([3.0])
    out = torch.pad_sequence(
        [a, b],
        batch_first=True,
        padding_value=0.0,
        padding_side="right",
    )
    expected = np.array([[1.0, 2.0], [3.0, 0.0]])
    np.testing.assert_allclose(out.numpy(), expected)


def test_pad_sequence_cpu_left():
    a = torch.tensor([1.0, 2.0])
    b = torch.tensor([3.0])
    out = torch.pad_sequence(
        [a, b],
        batch_first=True,
        padding_value=-1.0,
        padding_side="left",
    )
    expected = np.array([[1.0, 2.0], [-1.0, 3.0]])
    np.testing.assert_allclose(out.numpy(), expected)


def test_block_diag_cpu():
    a = torch.tensor([[1.0, 2.0], [3.0, 4.0]])
    b = torch.tensor([[5.0]])
    out = torch.block_diag(a, b)
    expected = np.array([
        [1.0, 2.0, 0.0],
        [3.0, 4.0, 0.0],
        [0.0, 0.0, 5.0],
    ])
    np.testing.assert_allclose(out.numpy(), expected)


def test_cartesian_prod_cpu():
    a = torch.tensor([1.0, 2.0])
    b = torch.tensor([3.0, 4.0])
    out = torch.cartesian_prod(a, b)
    expected = np.array([[1.0, 3.0], [1.0, 4.0], [2.0, 3.0], [2.0, 4.0]])
    np.testing.assert_allclose(out.numpy(), expected)


def test_chunk_cpu():
    x = torch.tensor([1.0, 2.0, 3.0, 4.0, 5.0])
    out = torch.chunk(x, 2)
    assert len(out) == 2
    np.testing.assert_allclose(out[0].numpy(), np.array([1.0, 2.0, 3.0]))
    np.testing.assert_allclose(out[1].numpy(), np.array([4.0, 5.0]))


def test_split_int_cpu():
    x = torch.tensor([1.0, 2.0, 3.0, 4.0, 5.0])
    out = torch.split(x, 2)
    assert len(out) == 3
    np.testing.assert_allclose(out[0].numpy(), np.array([1.0, 2.0]))
    np.testing.assert_allclose(out[1].numpy(), np.array([3.0, 4.0]))
    np.testing.assert_allclose(out[2].numpy(), np.array([5.0]))


def test_split_sections_cpu():
    x = torch.tensor([1.0, 2.0, 3.0, 4.0, 5.0])
    out = torch.split(x, [2, 1, 2])
    assert len(out) == 3
    np.testing.assert_allclose(out[0].numpy(), np.array([1.0, 2.0]))
    np.testing.assert_allclose(out[1].numpy(), np.array([3.0]))
    np.testing.assert_allclose(out[2].numpy(), np.array([4.0, 5.0]))


def test_unbind_cpu():
    x = torch.tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
    out = torch.unbind(x, dim=1)
    assert len(out) == 3
    np.testing.assert_allclose(out[0].numpy(), np.array([1.0, 4.0]))
    np.testing.assert_allclose(out[1].numpy(), np.array([2.0, 5.0]))
    np.testing.assert_allclose(out[2].numpy(), np.array([3.0, 6.0]))


def test_tril_cpu():
    x = torch.tensor([[1.0, 2.0], [3.0, 4.0]])
    expected = np.tril(x.numpy(), k=0)
    np.testing.assert_allclose(torch.tril(x).numpy(), expected)
    expected_offset = np.tril(x.numpy(), k=-1)
    np.testing.assert_allclose(torch.tril(x, diagonal=-1).numpy(), expected_offset)


def test_triu_cpu():
    x = torch.tensor([[1.0, 2.0], [3.0, 4.0]])
    expected = np.triu(x.numpy(), k=0)
    np.testing.assert_allclose(torch.triu(x).numpy(), expected)
    expected_offset = np.triu(x.numpy(), k=1)
    np.testing.assert_allclose(torch.triu(x, diagonal=1).numpy(), expected_offset)


def test_diag_cpu():
    x = torch.tensor([1.0, 2.0, 3.0])
    expected = np.diag(x.numpy(), k=0)
    np.testing.assert_allclose(torch.diag(x).numpy(), expected)
    expected_offset = np.diag(x.numpy(), k=1)
    np.testing.assert_allclose(torch.diag(x, diagonal=1).numpy(), expected_offset)
    y = torch.tensor([[1.0, 2.0], [3.0, 4.0]])
    expected_y = np.diag(y.numpy(), k=0)
    np.testing.assert_allclose(torch.diag(y).numpy(), expected_y)


def test_tril_indices_cpu():
    row, col, offset = 3, 4, 1
    expected = np.vstack(np.tril_indices(row, k=offset, m=col))
    out = torch.tril_indices(row, col, offset=offset)
    np.testing.assert_allclose(out.numpy(), expected)


def test_triu_indices_cpu():
    row, col, offset = 3, 4, -1
    expected = np.vstack(np.triu_indices(row, k=offset, m=col))
    out = torch.triu_indices(row, col, offset=offset)
    np.testing.assert_allclose(out.numpy(), expected)


def test_vsplit_cpu():
    x = torch.tensor([1.0, 2.0, 3.0, 4.0])
    out = torch.vsplit(x, 2)
    assert len(out) == 2
    np.testing.assert_allclose(out[0].numpy(), np.array([1.0, 2.0]))
    np.testing.assert_allclose(out[1].numpy(), np.array([3.0, 4.0]))
    y = torch.tensor([[1.0, 2.0], [3.0, 4.0]])
    out = torch.vsplit(y, 2)
    assert len(out) == 2
    np.testing.assert_allclose(out[0].numpy(), np.array([[1.0, 2.0]]))
    np.testing.assert_allclose(out[1].numpy(), np.array([[3.0, 4.0]]))


def test_hsplit_cpu():
    x = torch.tensor([1.0, 2.0, 3.0, 4.0])
    out = torch.hsplit(x, 2)
    assert len(out) == 2
    np.testing.assert_allclose(out[0].numpy(), np.array([1.0, 2.0]))
    np.testing.assert_allclose(out[1].numpy(), np.array([3.0, 4.0]))
    y = torch.tensor([[1.0, 2.0], [3.0, 4.0]])
    out = torch.hsplit(y, 2)
    assert len(out) == 2
    np.testing.assert_allclose(out[0].numpy(), np.array([[1.0], [3.0]]))
    np.testing.assert_allclose(out[1].numpy(), np.array([[2.0], [4.0]]))


def test_dsplit_cpu():
    x = torch.tensor([[[1.0, 2.0], [3.0, 4.0]]])
    out = torch.dsplit(x, 2)
    assert len(out) == 2
    np.testing.assert_allclose(out[0].numpy(), np.array([[[1.0], [3.0]]]))
    np.testing.assert_allclose(out[1].numpy(), np.array([[[2.0], [4.0]]]))
    y = torch.tensor([[1.0, 2.0], [3.0, 4.0]])
    with pytest.raises(ValueError):
        torch.dsplit(y, 2)


def test_take_cpu():
    x = torch.tensor([[1.0, 2.0], [3.0, 4.0]])
    index = torch.tensor([0, 3, 1], dtype=torch.int64)
    expected = np.take(x.numpy().reshape(-1), index.numpy().astype(np.int64))
    np.testing.assert_allclose(torch.take(x, index).numpy(), expected)
    neg_index = torch.tensor([-1, 0], dtype=torch.int64)
    expected_neg = np.take(x.numpy().reshape(-1), neg_index.numpy().astype(np.int64))
    np.testing.assert_allclose(torch.take(x, neg_index).numpy(), expected_neg)
    out_of_range = torch.tensor([4], dtype=torch.int64)
    with pytest.raises(IndexError):
        torch.take(x, out_of_range)


def test_take_along_dim_cpu():
    x = torch.tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
    indices = torch.tensor([[0, 2, 1], [2, 0, 1]], dtype=torch.int64)
    expected = np.take_along_axis(x.numpy(), indices.numpy().astype(np.int64), axis=1)
    np.testing.assert_allclose(torch.take_along_dim(x, indices, dim=1).numpy(), expected)
    neg_indices = torch.tensor([[-1, 0, 1], [1, -2, 0]], dtype=torch.int64)
    expected_neg = np.take_along_axis(x.numpy(), neg_indices.numpy().astype(np.int64), axis=1)
    np.testing.assert_allclose(torch.take_along_dim(x, neg_indices, dim=1).numpy(), expected_neg)
    out_of_range = torch.tensor([[3, 0, 1], [1, 2, 0]], dtype=torch.int64)
    with pytest.raises(IndexError):
        torch.take_along_dim(x, out_of_range, dim=1)


def test_index_select_cpu():
    x = torch.tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
    index = torch.tensor([2, 0], dtype=torch.int64)
    expected = np.take(x.numpy(), index.numpy().astype(np.int64), axis=1)
    np.testing.assert_allclose(torch.index_select(x, dim=1, index=index).numpy(), expected)
    neg_index = torch.tensor([-1, 0], dtype=torch.int64)
    expected_neg = np.take(x.numpy(), neg_index.numpy().astype(np.int64), axis=1)
    np.testing.assert_allclose(torch.index_select(x, dim=1, index=neg_index).numpy(), expected_neg)
    out_of_range = torch.tensor([3], dtype=torch.int64)
    with pytest.raises(IndexError):
        torch.index_select(x, dim=1, index=out_of_range)
    bad_index = torch.tensor([[0, 1]], dtype=torch.int64)
    with pytest.raises(ValueError):
        torch.index_select(x, dim=1, index=bad_index)


def test_gather_cpu():
    x = torch.tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
    index = torch.tensor([[0, 2], [1, 0]], dtype=torch.int64)
    expected = np.take_along_axis(x.numpy(), index.numpy(), axis=1)
    np.testing.assert_allclose(torch.gather(x, dim=1, index=index).numpy(), expected)
    neg_index = torch.tensor([[0, -1], [1, 0]], dtype=torch.int64)
    with pytest.raises(IndexError):
        torch.gather(x, dim=1, index=neg_index)
    out_of_range = torch.tensor([[3, 0], [1, 0]], dtype=torch.int64)
    with pytest.raises(IndexError):
        torch.gather(x, dim=1, index=out_of_range)


def test_scatter_cpu():
    x = torch.tensor([[0.0, 0.0, 0.0], [1.0, 1.0, 1.0]])
    index = torch.tensor([[0, 2], [1, 0]], dtype=torch.int64)
    src = torch.tensor([[5.0, 6.0], [7.0, 8.0]])
    expected = x.numpy().copy()
    np.put_along_axis(expected, index.numpy(), src.numpy(), axis=1)
    np.testing.assert_allclose(torch.scatter(x, dim=1, index=index, src=src).numpy(), expected)
    expected_scalar = x.numpy().copy()
    np.put_along_axis(expected_scalar, index.numpy(), 3.0, axis=1)
    np.testing.assert_allclose(torch.scatter(x, dim=1, index=index, src=3.0).numpy(), expected_scalar)
    out_of_range = torch.tensor([[3, 0], [1, 0]], dtype=torch.int64)
    with pytest.raises(IndexError):
        torch.scatter(x, dim=1, index=out_of_range, src=1.0)


def test_logspace_cpu():
    x = torch.logspace(0.0, 2.0, 3)
    expected = np.logspace(0.0, 2.0, 3)
    np.testing.assert_allclose(x.numpy(), expected)


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


def test_masked_select_cpu():
    x = torch.tensor([[1, 2], [3, 4]])
    mask = torch.tensor([[True, False], [False, True]])
    out = torch.masked_select(x, mask)
    np.testing.assert_array_equal(out.numpy(), np.array([1, 4]))


def test_nonzero_cpu():
    x = torch.tensor([[0, 1], [2, 0]])
    out = torch.nonzero(x)
    np.testing.assert_array_equal(out.numpy(), np.array([[0, 1], [1, 0]]))


def test_nonzero_as_tuple_cpu():
    x = torch.tensor([[0, 1], [2, 0]])
    out = torch.nonzero(x, as_tuple=True)
    assert isinstance(out, tuple)
    np.testing.assert_array_equal(out[0].numpy(), np.array([0, 1]))
    np.testing.assert_array_equal(out[1].numpy(), np.array([1, 0]))


def test_where_condition_cpu():
    cond = torch.tensor([[True, False], [False, True]])
    out = torch.where(cond)
    assert isinstance(out, tuple)
    np.testing.assert_array_equal(out[0].numpy(), np.array([0, 1]))
    np.testing.assert_array_equal(out[1].numpy(), np.array([0, 1]))


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


def test_addcmul_cpu():
    x = torch.tensor([1.0, 2.0, 3.0])
    y = torch.tensor([2.0, 2.0, 2.0])
    z = torch.tensor([3.0, 4.0, 5.0])
    value = 0.5
    expected = x.numpy() + value * (y.numpy() * z.numpy())
    np.testing.assert_allclose(torch.addcmul(x, y, z, value=value).numpy(), expected)
    np.testing.assert_allclose(x.addcmul(y, z, value=value).numpy(), expected)


def test_addcdiv_cpu():
    x = torch.tensor([1.0, 2.0, 3.0])
    y = torch.tensor([2.0, 4.0, 6.0])
    z = torch.tensor([1.0, 2.0, 3.0])
    value = 0.25
    expected = x.numpy() + value * (y.numpy() / z.numpy())
    np.testing.assert_allclose(torch.addcdiv(x, y, z, value=value).numpy(), expected)
    np.testing.assert_allclose(x.addcdiv(y, z, value=value).numpy(), expected)


def test_logaddexp_cpu():
    x = torch.tensor([0.0, 1.0, 2.0])
    y = torch.tensor([2.0, 1.0, 0.0])
    expected = np.logaddexp(x.numpy(), y.numpy())
    np.testing.assert_allclose(torch.logaddexp(x, y).numpy(), expected)
    np.testing.assert_allclose(x.logaddexp(y).numpy(), expected)


def test_logaddexp2_cpu():
    x = torch.tensor([0.0, 1.0, 2.0])
    y = torch.tensor([2.0, 1.0, 0.0])
    expected = np.logaddexp2(x.numpy(), y.numpy())
    np.testing.assert_allclose(torch.logaddexp2(x, y).numpy(), expected)
    np.testing.assert_allclose(x.logaddexp2(y).numpy(), expected)


def test_hypot_cpu():
    x = torch.tensor([3.0, 5.0, 8.0])
    y = torch.tensor([4.0, 12.0, 15.0])
    expected = np.hypot(x.numpy(), y.numpy())
    np.testing.assert_allclose(torch.hypot(x, y).numpy(), expected)
    np.testing.assert_allclose(x.hypot(y).numpy(), expected)


def test_remainder_cpu():
    x = torch.tensor([-3.0, 3.0, -3.0])
    y = torch.tensor([2.0, 2.0, -2.0])
    expected = np.remainder(x.numpy(), y.numpy())
    np.testing.assert_allclose(torch.remainder(x, y).numpy(), expected)
    np.testing.assert_allclose(x.remainder(y).numpy(), expected)


def test_fmod_cpu():
    x = torch.tensor([-3.0, 3.0, -3.0])
    y = torch.tensor([2.0, 2.0, -2.0])
    expected = np.fmod(x.numpy(), y.numpy())
    np.testing.assert_allclose(torch.fmod(x, y).numpy(), expected)
    np.testing.assert_allclose(x.fmod(y).numpy(), expected)


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
