import pytest
import mindtorch_v2 as torch


def test_meta_tensor_blocks_numpy():
    t = torch.tensor([1.0, 2.0], device="meta")
    with pytest.raises(RuntimeError, match="meta tensor has no data"):
        _ = t.numpy()


def test_meta_to_cpu_materializes():
    t = torch.tensor([1.0, 2.0], device="meta")
    out = t.to("cpu")
    assert out.device.type == "cpu"
    assert out.shape == (2,)


def test_meta_ops_shape_propagation():
    a = torch.tensor([[1.0, 2.0]], device="meta")
    b = torch.tensor([[3.0, 4.0]], device="meta")
    c = a + b
    d = c.relu()
    e = d.sum()
    assert c.device.type == "meta"
    assert d.device.type == "meta"
    assert e.device.type == "meta"
    assert c.shape == a.shape
    assert e.shape == ()


def test_meta_unary_elementwise_ops_shape():
    x = torch.tensor([1.0, 2.0], device="meta")
    for op in (
        torch.abs,
        torch.neg,
        torch.exp,
        torch.log,
        torch.sqrt,
        torch.sin,
        torch.cos,
        torch.tan,
        torch.tanh,
        torch.sigmoid,
        torch.floor,
        torch.ceil,
        torch.round,
        torch.trunc,
        torch.frac,
        torch.log2,
        torch.log10,
        torch.exp2,
        torch.rsqrt,
        torch.sign,
        torch.signbit,
        torch.isnan,
        torch.isinf,
        torch.isfinite,
        torch.sinh,
        torch.cosh,
        torch.asinh,
        torch.acosh,
        torch.atanh,
        torch.erf,
        torch.erfc,
        torch.softplus,
        torch.relu6,
    ):
        out = op(x)
        assert out.device.type == "meta"
        assert out.shape == x.shape

    for op in (
        lambda t: torch.clamp(t, 0.0, 1.0),
        lambda t: torch.clamp_min(t, 0.0),
        lambda t: torch.clamp_max(t, 1.0),
        lambda t: torch.hardtanh(t, -1.0, 1.0),
        lambda t: torch.min(t, t),
        lambda t: torch.max(t, t),
        lambda t: torch.fmin(t, t),
        lambda t: torch.fmax(t, t),
        lambda t: torch.where(t, t, t),
        lambda t: torch.atan(t),
        lambda t: torch.atan2(t, t),
        lambda t: torch.asin(t),
        lambda t: torch.acos(t),
        lambda t: torch.lerp(t, t, 0.5),
        lambda t: torch.addcmul(t, t, t, value=0.5),
        lambda t: torch.addcdiv(t, t, t, value=0.5),
        lambda t: torch.logaddexp(t, t),
        lambda t: torch.logaddexp2(t, t),
        lambda t: torch.hypot(t, t),
        lambda t: torch.remainder(t, t),
        lambda t: torch.fmod(t, t),
    ):
        out = op(x)
        assert out.device.type == "meta"
        assert out.shape == x.shape

    for op in (
        lambda t: torch.amin(t, dim=0),
        lambda t: torch.amax(t, dim=0),
        lambda t: torch.all(t, dim=0),
        lambda t: torch.any(t, dim=0),
        lambda t: torch.argmax(t, dim=0),
        lambda t: torch.argmin(t, dim=0),
        lambda t: torch.count_nonzero(t, dim=0),
    ):
        out = op(x)
        assert out.device.type == "meta"
        assert out.shape == ()

    for op in (
        lambda t: torch.allclose(t, t),
        lambda t: torch.isclose(t, t),
        lambda t: torch.equal(t, t),
    ):
        out = op(x)
        assert out.device.type == "meta"
        assert out.shape in ((), x.shape)


def test_meta_logspace_shape():
    x = torch.logspace(0.0, 2.0, 3, device="meta")
    assert x.device.type == "meta"
    assert x.shape == (3,)


def test_meta_eye_shape():
    x = torch.eye(3, 2, device="meta")
    assert x.device.type == "meta"
    assert x.shape == (3, 2)


def test_meta_range_shape():
    x = torch.range(0.0, 2.0, 0.5, device="meta")
    assert x.device.type == "meta"
    assert x.shape == (5,)


def test_meta_cum_ops_shape():
    x = torch.tensor([[1.0, 2.0]], device="meta")
    out = torch.cumsum(x, dim=1)
    assert out.device.type == "meta"
    assert out.shape == x.shape
    out = torch.cumprod(x, dim=1)
    assert out.device.type == "meta"
    assert out.shape == x.shape


def test_meta_cummax_shape():
    x = torch.tensor([[1.0, 2.0]], device="meta")
    values, indices = torch.cummax(x, dim=1)
    assert values.device.type == "meta"
    assert indices.device.type == "meta"
    assert values.shape == x.shape
    assert indices.shape == x.shape


def test_meta_sort_shape():
    x = torch.tensor([[1.0, 2.0]], device="meta")
    values, indices = torch.sort(x, dim=1)
    assert values.device.type == "meta"
    assert indices.device.type == "meta"
    assert values.shape == x.shape
    assert indices.shape == x.shape


def test_meta_argsort_shape():
    x = torch.tensor([[1.0, 2.0]], device="meta")
    out = torch.argsort(x, dim=1)
    assert out.device.type == "meta"
    assert out.shape == x.shape


def test_meta_topk_shape():
    x = torch.tensor([[1.0, 2.0, 3.0]], device="meta")
    values, indices = torch.topk(x, k=2, dim=1)
    assert values.device.type == "meta"
    assert indices.device.type == "meta"
    assert values.shape == (1, 2)
    assert indices.shape == (1, 2)


def test_meta_stack_shape():
    a = torch.tensor([1.0, 2.0], device="meta")
    b = torch.tensor([3.0, 4.0], device="meta")
    out = torch.stack([a, b], dim=0)
    assert out.device.type == "meta"
    assert out.shape == (2, 2)


def test_meta_cat_shape():
    a = torch.tensor([[1.0, 2.0]], device="meta")
    b = torch.tensor([[3.0, 4.0]], device="meta")
    out = torch.cat([a, b], dim=0)
    assert out.device.type == "meta"
    assert out.shape == (2, 2)


def test_meta_concat_shape():
    a = torch.tensor([[1.0, 2.0]], device="meta")
    b = torch.tensor([[3.0, 4.0]], device="meta")
    out = torch.concat([a, b], dim=1)
    assert out.device.type == "meta"
    assert out.shape == (1, 4)


def test_meta_hstack_shape():
    a = torch.tensor([1.0, 2.0], device="meta")
    b = torch.tensor([3.0, 4.0], device="meta")
    out = torch.hstack([a, b])
    assert out.device.type == "meta"
    assert out.shape == (4,)


def test_meta_vstack_shape():
    a = torch.tensor([1.0, 2.0], device="meta")
    b = torch.tensor([3.0, 4.0], device="meta")
    out = torch.vstack([a, b])
    assert out.device.type == "meta"
    assert out.shape == (2, 2)


def test_meta_column_stack_shape():
    a = torch.tensor([1.0, 2.0], device="meta")
    b = torch.tensor([3.0, 4.0], device="meta")
    out = torch.column_stack([a, b])
    assert out.device.type == "meta"
    assert out.shape == (2, 2)


def test_meta_pad_sequence_shape():
    a = torch.tensor([1.0, 2.0], device="meta")
    b = torch.tensor([3.0], device="meta")
    out = torch.pad_sequence([a, b], batch_first=True)
    assert out.device.type == "meta"
    assert out.shape == (2, 2)


def test_meta_block_diag_shape():
    a = torch.tensor([[1.0, 2.0]], device="meta")
    b = torch.tensor([[3.0]], device="meta")
    out = torch.block_diag(a, b)
    assert out.device.type == "meta"
    assert out.shape == (2, 3)


def test_meta_cartesian_prod_shape():
    a = torch.tensor([1.0, 2.0], device="meta")
    b = torch.tensor([3.0], device="meta")
    out = torch.cartesian_prod(a, b)
    assert out.device.type == "meta"
    assert out.shape == (2, 2)


def test_meta_chunk_shape():
    x = torch.tensor([1.0, 2.0, 3.0, 4.0, 5.0], device="meta")
    out = torch.chunk(x, 2)
    assert len(out) == 2
    assert out[0].shape == (3,)
    assert out[1].shape == (2,)


def test_meta_split_shape():
    x = torch.tensor([1.0, 2.0, 3.0, 4.0, 5.0], device="meta")
    out = torch.split(x, 2)
    assert len(out) == 3
    assert out[0].shape == (2,)
    assert out[1].shape == (2,)
    assert out[2].shape == (1,)


def test_meta_unbind_shape():
    x = torch.tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], device="meta")
    out = torch.unbind(x, dim=1)
    assert len(out) == 3
    assert out[0].shape == (2,)
    assert out[1].shape == (2,)
    assert out[2].shape == (2,)


def test_meta_concatenate_shape():
    a = torch.tensor([[1.0, 2.0]], device="meta")
    b = torch.tensor([[3.0, 4.0]], device="meta")
    out = torch.concatenate([a, b], dim=0)
    assert out.device.type == "meta"
    assert out.shape == (2, 2)


def test_meta_row_stack_shape():
    a = torch.tensor([1.0, 2.0], device="meta")
    b = torch.tensor([3.0, 4.0], device="meta")
    out = torch.row_stack([a, b])
    assert out.device.type == "meta"
    assert out.shape == (2, 2)


def test_meta_dstack_shape():
    a = torch.tensor([1.0, 2.0], device="meta")
    b = torch.tensor([3.0, 4.0], device="meta")
    out = torch.dstack([a, b])
    assert out.device.type == "meta"
    assert out.shape == (1, 2, 2)


def test_meta_tril_triu_indices_shape():
    out = torch.tril_indices(3, 4, offset=1, device="meta")
    assert out.device.type == "meta"
    assert out.shape == (2, 9)
    out = torch.triu_indices(3, 4, offset=-1, device="meta")
    assert out.device.type == "meta"
    assert out.shape == (2, 11)


def test_meta_vsplit_shape():
    x = torch.tensor([1.0, 2.0, 3.0, 4.0], device="meta")
    out = torch.vsplit(x, 2)
    assert len(out) == 2
    assert out[0].shape == (2,)
    assert out[1].shape == (2,)


def test_meta_hsplit_shape():
    x = torch.tensor([1.0, 2.0, 3.0, 4.0], device="meta")
    out = torch.hsplit(x, 2)
    assert len(out) == 2
    assert out[0].shape == (2,)
    assert out[1].shape == (2,)
    y = torch.tensor([[1.0, 2.0], [3.0, 4.0]], device="meta")
    out = torch.hsplit(y, 2)
    assert len(out) == 2
    assert out[0].shape == (2, 1)
    assert out[1].shape == (2, 1)


def test_meta_dsplit_shape():
    x = torch.tensor([[[1.0, 2.0], [3.0, 4.0]]], device="meta")
    out = torch.dsplit(x, 2)
    assert len(out) == 2
    assert out[0].shape == (1, 2, 1)
    assert out[1].shape == (1, 2, 1)
    y = torch.tensor([[1.0, 2.0], [3.0, 4.0]], device="meta")
    with pytest.raises(ValueError):
        torch.dsplit(y, 2)


def test_meta_take_shape():
    x = torch.tensor([[1.0, 2.0], [3.0, 4.0]], device="meta")
    index = torch.tensor([0, 1, 3], device="meta", dtype=torch.int64)
    out = torch.take(x, index)
    assert out.device.type == "meta"
    assert out.shape == index.shape


def test_meta_take_along_dim_shape():
    x = torch.tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], device="meta")
    indices = torch.tensor([[0, 2, 1], [2, 0, 1]], device="meta", dtype=torch.int64)
    out = torch.take_along_dim(x, indices, dim=1)
    assert out.device.type == "meta"
    assert out.shape == indices.shape
    bad_indices = torch.tensor([[0, 1, 2]], device="meta", dtype=torch.int64)
    with pytest.raises(ValueError):
        torch.take_along_dim(x, bad_indices, dim=1)


def test_meta_index_select_shape():
    x = torch.tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], device="meta")
    index = torch.tensor([0, 2], device="meta", dtype=torch.int64)
    out = torch.index_select(x, dim=1, index=index)
    assert out.device.type == "meta"
    assert out.shape == (2, 2)
    bad_index = torch.tensor([[0, 1]], device="meta", dtype=torch.int64)
    with pytest.raises(ValueError):
        torch.index_select(x, dim=1, index=bad_index)


def test_meta_gather_shape():
    x = torch.tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], device="meta")
    index = torch.tensor([[0, 2], [1, 0]], device="meta", dtype=torch.int64)
    out = torch.gather(x, dim=1, index=index)
    assert out.device.type == "meta"
    assert out.shape == index.shape
    bad_index = torch.tensor([[0, 1, 2]], device="meta", dtype=torch.int64)
    with pytest.raises(ValueError):
        torch.gather(x, dim=1, index=bad_index)


def test_meta_scatter_shape():
    x = torch.tensor([[0.0, 0.0, 0.0], [1.0, 1.0, 1.0]], device="meta")
    index = torch.tensor([[0, 2], [1, 0]], device="meta", dtype=torch.int64)
    src = torch.tensor([[5.0, 6.0], [7.0, 8.0]], device="meta")
    out = torch.scatter(x, dim=1, index=index, src=src)
    assert out.device.type == "meta"
    assert out.shape == x.shape
    bad_index = torch.tensor([[0, 1, 2]], device="meta", dtype=torch.int64)
    with pytest.raises(ValueError):
        torch.scatter(x, dim=1, index=bad_index, src=src)


def test_meta_tril_triu_shape():
    x = torch.tensor([[1.0, 2.0]], device="meta")
    out = torch.tril(x, diagonal=-1)
    assert out.device.type == "meta"
    assert out.shape == x.shape
    out = torch.triu(x, diagonal=1)
    assert out.device.type == "meta"
    assert out.shape == x.shape


def test_meta_diag_shape():
    x = torch.tensor([1.0, 2.0], device="meta")
    out = torch.diag(x, diagonal=1)
    assert out.device.type == "meta"
    assert out.shape == (3, 3)
    y = torch.tensor([[1.0, 2.0], [3.0, 4.0]], device="meta")
    out = torch.diag(y, diagonal=0)
    assert out.device.type == "meta"
    assert out.shape == (2,)


def test_meta_pow_shape():
    x = torch.tensor([1.0, 2.0, 3.0], device="meta")
    out = torch.pow(x, 2.0)
    assert out.device.type == "meta"
    assert out.shape == x.shape


def test_meta_masked_select_shape():
    x = torch.empty((2, 3), device="meta")
    mask = torch.empty((2, 3), device="meta", dtype=torch.bool)
    out = torch.masked_select(x, mask)
    assert out.shape == (0,)


def test_meta_flip_shape():
    x = torch.empty((2, 3, 4), device="meta")
    out = torch.flip(x, dims=(0, 2))
    assert out.shape == x.shape


def test_meta_roll_shape():
    x = torch.empty((2, 3, 4), device="meta")
    out = torch.roll(x, shifts=1, dims=2)
    assert out.shape == x.shape


def test_meta_rot90_shape():
    x = torch.empty((2, 3, 4), device="meta")
    out = torch.rot90(x, k=1, dims=(0, 2))
    assert out.shape == (4, 3, 2)


def test_meta_nonzero_shape():
    x = torch.empty((2, 3), device="meta")
    out = torch.nonzero(x)
    assert out.shape == (0, 2)


def test_meta_nonzero_as_tuple_shape():
    x = torch.empty((2, 3), device="meta")
    out = torch.nonzero(x, as_tuple=True)
    assert isinstance(out, tuple)
    assert len(out) == 2
    assert out[0].shape == (0,)
    assert out[1].shape == (0,)


def test_meta_where_condition_shape():
    cond = torch.empty((2, 3), device="meta", dtype=torch.bool)
    out = torch.where(cond)
    assert isinstance(out, tuple)
    assert len(out) == 2
    assert out[0].shape == (0,)
    assert out[1].shape == (0,)
