"""Tests for top-level torch.* function gap-fill (Categories A, B, C1, C2)."""
import math
import pytest

import mindtorch_v2 as torch


# ===========================================================================
# Category A: Export-only (already implemented, now exported)
# ===========================================================================

class TestCategoryA:
    """Functions that were in _functional.py but not exported."""

    def test_eq(self):
        a = torch.tensor([1, 2, 3])
        b = torch.tensor([1, 0, 3])
        out = torch.eq(a, b)
        assert out.tolist() == [True, False, True]

    def test_ne(self):
        a = torch.tensor([1, 2, 3])
        b = torch.tensor([1, 0, 3])
        out = torch.ne(a, b)
        assert out.tolist() == [False, True, False]

    def test_lt(self):
        a = torch.tensor([1, 2, 3])
        b = torch.tensor([2, 2, 1])
        out = torch.lt(a, b)
        assert out.tolist() == [True, False, False]

    def test_le(self):
        a = torch.tensor([1, 2, 3])
        b = torch.tensor([2, 2, 1])
        out = torch.le(a, b)
        assert out.tolist() == [True, True, False]

    def test_gt(self):
        a = torch.tensor([1, 2, 3])
        b = torch.tensor([2, 2, 1])
        out = torch.gt(a, b)
        assert out.tolist() == [False, False, True]

    def test_ge(self):
        a = torch.tensor([1, 2, 3])
        b = torch.tensor([2, 2, 1])
        out = torch.ge(a, b)
        assert out.tolist() == [False, True, True]

    def test_select(self):
        a = torch.tensor([[1, 2], [3, 4], [5, 6]])
        out = torch.select(a, 0, 1)
        assert out.tolist() == [3, 4]

    def test_expand(self):
        a = torch.tensor([[1], [2], [3]])
        out = torch.expand(a, 3, 4)
        assert out.shape == (3, 4)
        assert out[0].tolist() == [1, 1, 1, 1]

    def test_masked_fill(self):
        a = torch.tensor([1.0, 2.0, 3.0])
        mask = torch.tensor([True, False, True])
        out = torch.masked_fill(a, mask, 0.0)
        assert out.tolist() == [0.0, 2.0, 0.0]

    def test_unfold(self):
        a = torch.arange(1, 8, dtype=torch.float32)
        out = torch.unfold(a, 0, 3, 2)
        assert out.shape[0] == 3
        assert out.shape[1] == 3

    def test_scatter_(self):
        src = torch.tensor([[1, 2], [3, 4]], dtype=torch.float32)
        index = torch.tensor([[0, 1], [1, 0]], dtype=torch.int64)
        out = torch.zeros(2, 3, dtype=torch.float32)
        torch.scatter_(out, 1, index, src)

    def test_scatter_add_(self):
        src = torch.ones(2, 3, dtype=torch.float32)
        index = torch.tensor([[0, 1, 2], [0, 1, 2]], dtype=torch.int64)
        out = torch.zeros(2, 3, dtype=torch.float32)
        torch.scatter_add_(out, 1, index, src)

    def test_index_put(self):
        a = torch.zeros(3, dtype=torch.float32)
        indices = (torch.tensor([0, 2], dtype=torch.int64),)
        values = torch.tensor([10.0, 20.0])
        out = torch.index_put(a, indices, values)
        assert out[0].item() == 10.0
        assert out[2].item() == 20.0

    def test_index_put_(self):
        a = torch.zeros(3, dtype=torch.float32)
        indices = (torch.tensor([0, 2], dtype=torch.int64),)
        values = torch.tensor([10.0, 20.0])
        torch.index_put_(a, indices, values)
        assert a[0].item() == 10.0

    def test_index_add_(self):
        a = torch.zeros(3, dtype=torch.float32)
        index = torch.tensor([0, 2])
        source = torch.tensor([1.0, 2.0])
        torch.index_add_(a, 0, index, source)
        assert a[0].item() == 1.0
        assert a[2].item() == 2.0

    def test_index_copy_(self):
        a = torch.zeros(3, dtype=torch.float32)
        index = torch.tensor([0, 2])
        source = torch.tensor([10.0, 20.0])
        torch.index_copy_(a, 0, index, source)
        assert a[0].item() == 10.0
        assert a[2].item() == 20.0

    def test_index_fill_(self):
        a = torch.zeros(3, dtype=torch.float32)
        index = torch.tensor([0, 2])
        torch.index_fill_(a, 0, index, 5.0)
        assert a[0].item() == 5.0
        assert a[1].item() == 0.0
        assert a[2].item() == 5.0

    def test_masked_fill_(self):
        a = torch.tensor([1.0, 2.0, 3.0])
        mask = torch.tensor([True, False, True])
        torch.masked_fill_(a, mask, 0.0)
        assert a[0].item() == 0.0
        assert a[1].item() == 2.0

    def test_masked_scatter_(self):
        a = torch.zeros(4, dtype=torch.float32)
        mask = torch.tensor([True, False, True, False])
        source = torch.tensor([10.0, 20.0])
        torch.masked_scatter_(a, mask, source)
        assert a[0].item() == 10.0
        assert a[2].item() == 20.0


# ===========================================================================
# Category B: Wrapper + Export
# ===========================================================================

class TestCategoryB:
    """Functions with existing schema+kernel that needed wrappers."""

    def test_nansum_all(self):
        a = torch.tensor([1.0, float('nan'), 3.0])
        out = torch.nansum(a)
        assert abs(out.item() - 4.0) < 1e-6

    def test_nansum_dim(self):
        a = torch.tensor([[1.0, float('nan')], [3.0, 4.0]])
        out = torch.nansum(a, dim=1)
        assert abs(out[0].item() - 1.0) < 1e-6
        assert abs(out[1].item() - 7.0) < 1e-6

    def test_nanmean_all(self):
        a = torch.tensor([1.0, float('nan'), 3.0])
        out = torch.nanmean(a)
        assert abs(out.item() - 2.0) < 1e-6

    def test_nanmean_dim(self):
        a = torch.tensor([[1.0, float('nan')], [3.0, 4.0]])
        out = torch.nanmean(a, dim=1)
        assert abs(out[0].item() - 1.0) < 1e-6
        assert abs(out[1].item() - 3.5) < 1e-6

    def test_det(self):
        a = torch.tensor([[1.0, 2.0], [3.0, 4.0]])
        out = torch.det(a)
        assert abs(out.item() - (-2.0)) < 1e-5

    def test_dist(self):
        a = torch.tensor([1.0, 2.0, 3.0])
        b = torch.tensor([4.0, 5.0, 6.0])
        out = torch.dist(a, b)
        expected = math.sqrt(27.0)
        assert abs(out.item() - expected) < 1e-5

    def test_matrix_power(self):
        a = torch.tensor([[1.0, 2.0], [0.0, 1.0]])
        out = torch.matrix_power(a, 3)
        # [[1,2],[0,1]]^3 = [[1,6],[0,1]]
        assert abs(out[0][1].item() - 6.0) < 1e-5

    def test_argwhere(self):
        a = torch.tensor([0, 1, 0, 1, 1])
        out = torch.argwhere(a)
        assert out.shape == (3, 1)
        assert out[0][0].item() == 1
        assert out[1][0].item() == 3
        assert out[2][0].item() == 4


# ===========================================================================
# Category C1: Pure-Python functions
# ===========================================================================

class TestCategoryC1:
    """Pure-Python implementations without dispatch."""

    def test_meshgrid_ij(self):
        x = torch.tensor([1, 2, 3])
        y = torch.tensor([4, 5])
        gx, gy = torch.meshgrid(x, y, indexing='ij')
        assert gx.shape == (3, 2)
        assert gy.shape == (3, 2)
        assert gx[0].tolist() == [1, 1]
        assert gy[0].tolist() == [4, 5]

    def test_meshgrid_xy(self):
        x = torch.tensor([1, 2, 3])
        y = torch.tensor([4, 5])
        gx, gy = torch.meshgrid(x, y, indexing='xy')
        assert gx.shape == (2, 3)
        assert gy.shape == (2, 3)

    def test_atleast_1d_scalar(self):
        a = torch.tensor(5.0)
        out = torch.atleast_1d(a)
        assert out.ndim == 1
        assert out.shape == (1,)

    def test_atleast_1d_already(self):
        a = torch.tensor([1.0, 2.0])
        out = torch.atleast_1d(a)
        assert out.ndim == 1

    def test_atleast_1d_multi(self):
        a = torch.tensor(1.0)
        b = torch.tensor([2.0, 3.0])
        results = torch.atleast_1d(a, b)
        assert len(results) == 2
        assert results[0].ndim >= 1
        assert results[1].ndim >= 1

    def test_atleast_2d_scalar(self):
        a = torch.tensor(5.0)
        out = torch.atleast_2d(a)
        assert out.ndim == 2
        assert out.shape == (1, 1)

    def test_atleast_2d_1d(self):
        a = torch.tensor([1.0, 2.0])
        out = torch.atleast_2d(a)
        assert out.ndim == 2
        assert out.shape == (1, 2)

    def test_atleast_3d_scalar(self):
        a = torch.tensor(5.0)
        out = torch.atleast_3d(a)
        assert out.ndim == 3
        assert out.shape == (1, 1, 1)

    def test_atleast_3d_1d(self):
        a = torch.tensor([1.0, 2.0])
        out = torch.atleast_3d(a)
        assert out.ndim == 3
        assert out.shape == (1, 2, 1)

    def test_atleast_3d_2d(self):
        a = torch.tensor([[1.0, 2.0]])
        out = torch.atleast_3d(a)
        assert out.ndim == 3
        assert out.shape == (1, 2, 1)

    def test_broadcast_shapes(self):
        assert torch.broadcast_shapes((2, 1), (1, 3)) == (2, 3)
        assert torch.broadcast_shapes((5,), (1,)) == (5,)
        assert torch.broadcast_shapes((3, 1), (3, 4)) == (3, 4)

    def test_broadcast_shapes_error(self):
        with pytest.raises(RuntimeError):
            torch.broadcast_shapes((2,), (3,))

    def test_broadcast_tensors(self):
        a = torch.tensor([[1], [2], [3]])
        b = torch.tensor([4, 5])
        ra, rb = torch.broadcast_tensors(a, b)
        assert ra.shape == (3, 2)
        assert rb.shape == (3, 2)

    def test_complex(self):
        real = torch.tensor([1.0, 2.0])
        imag = torch.tensor([3.0, 4.0])
        out = torch.complex(real, imag)
        assert out.shape == (2,)

    def test_polar(self):
        abs_t = torch.tensor([1.0, 1.0])
        angle = torch.tensor([0.0, math.pi / 2])
        out = torch.polar(abs_t, angle)
        assert out.shape == (2,)


# ===========================================================================
# Category C2: Dispatch-based functions
# ===========================================================================

class TestCategoryC2:
    """Functions requiring schema + kernel + wrapper."""

    def test_diff_basic(self):
        a = torch.tensor([1.0, 2.0, 4.0, 7.0])
        out = torch.diff(a)
        assert out.tolist() == [1.0, 2.0, 3.0]

    def test_diff_n2(self):
        a = torch.tensor([1.0, 2.0, 4.0, 7.0])
        out = torch.diff(a, n=2)
        assert out.tolist() == [1.0, 1.0]

    def test_diff_dim(self):
        a = torch.tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
        out = torch.diff(a, dim=1)
        assert out.shape == (2, 2)
        assert out[0].tolist() == [1.0, 1.0]

    def test_diff_prepend(self):
        a = torch.tensor([1.0, 2.0, 4.0])
        prepend = torch.tensor([0.0])
        out = torch.diff(a, prepend=prepend)
        assert out.tolist() == [1.0, 1.0, 2.0]

    def test_bincount(self):
        a = torch.tensor([0, 1, 1, 3, 2, 1], dtype=torch.int64)
        out = torch.bincount(a)
        assert out[0].item() == 1
        assert out[1].item() == 3
        assert out[2].item() == 1
        assert out[3].item() == 1

    def test_bincount_minlength(self):
        a = torch.tensor([0, 1], dtype=torch.int64)
        out = torch.bincount(a, minlength=5)
        assert out.numel() >= 5

    def test_cdist_2d(self):
        x1 = torch.tensor([[0.0, 0.0], [1.0, 1.0]])
        x2 = torch.tensor([[0.0, 0.0], [1.0, 0.0]])
        out = torch.cdist(x1, x2)
        assert out.shape == (2, 2)
        assert abs(out[0][0].item()) < 1e-5
        assert abs(out[0][1].item() - 1.0) < 1e-5

    def test_cdist_3d_batch(self):
        x1 = torch.randn(2, 3, 4)
        x2 = torch.randn(2, 5, 4)
        out = torch.cdist(x1, x2)
        assert out.shape == (2, 3, 5)

    def test_aminmax_all(self):
        a = torch.tensor([3.0, 1.0, 4.0, 1.0, 5.0])
        mn, mx = torch.aminmax(a)
        assert mn.item() == 1.0
        assert mx.item() == 5.0

    def test_aminmax_dim(self):
        a = torch.tensor([[3.0, 1.0], [4.0, 2.0]])
        mn, mx = torch.aminmax(a, dim=1)
        assert mn.tolist() == [1.0, 2.0]
        assert mx.tolist() == [3.0, 4.0]

    def test_quantile(self):
        a = torch.tensor([0.0, 1.0, 2.0, 3.0], dtype=torch.float32)
        out = torch.quantile(a, 0.5)
        assert abs(out.item() - 1.5) < 1e-5

    def test_quantile_dim(self):
        a = torch.tensor([[0.0, 1.0, 2.0], [3.0, 4.0, 5.0]], dtype=torch.float32)
        out = torch.quantile(a, 0.5, dim=1)
        assert abs(out[0].item() - 1.0) < 1e-5
        assert abs(out[1].item() - 4.0) < 1e-5

    def test_nanquantile(self):
        a = torch.tensor([0.0, 1.0, float('nan'), 3.0], dtype=torch.float32)
        out = torch.nanquantile(a, 0.5)
        assert abs(out.item() - 1.0) < 1e-5

    def test_nanmedian_all(self):
        a = torch.tensor([1.0, float('nan'), 3.0, 2.0])
        out = torch.nanmedian(a)
        assert abs(out.item() - 2.0) < 1e-5

    def test_nanmedian_dim(self):
        a = torch.tensor([[1.0, float('nan'), 3.0], [4.0, 5.0, 6.0]])
        vals, idx = torch.nanmedian(a, dim=1)
        assert abs(vals[1].item() - 5.0) < 1e-5

    def test_histc(self):
        a = torch.tensor([1.0, 2.0, 1.0, 3.0, 2.0])
        out = torch.histc(a, bins=3, min=1, max=3)
        assert out.numel() == 3

    def test_histogram(self):
        a = torch.tensor([1.0, 2.0, 3.0, 4.0])
        hist, edges = torch.histogram(a, bins=2)
        assert hist.numel() == 2
        assert edges.numel() == 3

    def test_bucketize(self):
        boundaries = torch.tensor([1.0, 3.0, 5.0, 7.0])
        a = torch.tensor([2.0, 4.0, 6.0, 8.0])
        out = torch.bucketize(a, boundaries)
        assert out.tolist() == [1, 2, 3, 4]

    def test_bucketize_right(self):
        boundaries = torch.tensor([1.0, 3.0, 5.0])
        a = torch.tensor([1.0, 3.0, 5.0])
        out = torch.bucketize(a, boundaries, right=True)
        assert out[0].item() == 0
        assert out[1].item() == 1
        assert out[2].item() == 2

    def test_isneginf(self):
        a = torch.tensor([float('-inf'), 0.0, float('inf')])
        out = torch.isneginf(a)
        assert out.tolist() == [True, False, False]

    def test_isposinf(self):
        a = torch.tensor([float('-inf'), 0.0, float('inf')])
        out = torch.isposinf(a)
        assert out.tolist() == [False, False, True]

    def test_isreal(self):
        a = torch.tensor([1.0, 2.0, 3.0])
        out = torch.isreal(a)
        assert all(out.tolist())

    def test_isin(self):
        elements = torch.tensor([1, 2, 3, 4, 5])
        test_elements = torch.tensor([2, 4])
        out = torch.isin(elements, test_elements)
        assert out.tolist() == [False, True, False, True, False]

    def test_heaviside(self):
        a = torch.tensor([-1.0, 0.0, 1.0])
        values = torch.tensor([0.5, 0.5, 0.5])
        out = torch.heaviside(a, values)
        assert out[0].item() == 0.0
        assert out[1].item() == 0.5
        assert out[2].item() == 1.0
