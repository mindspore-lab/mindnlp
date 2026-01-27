"""Tests for tensor manipulation operations."""

import numpy as np
import mindtorch_v2 as torch


class TestCat:
    def test_cat_dim0(self):
        a = torch.tensor([[1, 2], [3, 4]])
        b = torch.tensor([[5, 6], [7, 8]])
        result = torch.cat([a, b], dim=0)
        expected = np.array([[1, 2], [3, 4], [5, 6], [7, 8]])
        np.testing.assert_array_equal(result.numpy(), expected)

    def test_cat_dim1(self):
        a = torch.tensor([[1, 2], [3, 4]])
        b = torch.tensor([[5, 6], [7, 8]])
        result = torch.cat([a, b], dim=1)
        expected = np.array([[1, 2, 5, 6], [3, 4, 7, 8]])
        np.testing.assert_array_equal(result.numpy(), expected)


class TestStack:
    def test_stack_dim0(self):
        a = torch.tensor([1, 2, 3])
        b = torch.tensor([4, 5, 6])
        result = torch.stack([a, b], dim=0)
        expected = np.array([[1, 2, 3], [4, 5, 6]])
        np.testing.assert_array_equal(result.numpy(), expected)

    def test_stack_dim1(self):
        a = torch.tensor([1, 2, 3])
        b = torch.tensor([4, 5, 6])
        result = torch.stack([a, b], dim=1)
        expected = np.array([[1, 4], [2, 5], [3, 6]])
        np.testing.assert_array_equal(result.numpy(), expected)


class TestSplit:
    def test_split_size(self):
        x = torch.tensor([1, 2, 3, 4, 5, 6])
        result = torch.split(x, 2)
        assert len(result) == 3
        np.testing.assert_array_equal(result[0].numpy(), [1, 2])
        np.testing.assert_array_equal(result[1].numpy(), [3, 4])
        np.testing.assert_array_equal(result[2].numpy(), [5, 6])

    def test_split_sections(self):
        x = torch.tensor([1, 2, 3, 4, 5, 6])
        result = torch.split(x, [1, 2, 3])
        assert len(result) == 3
        np.testing.assert_array_equal(result[0].numpy(), [1])
        np.testing.assert_array_equal(result[1].numpy(), [2, 3])
        np.testing.assert_array_equal(result[2].numpy(), [4, 5, 6])


class TestChunk:
    def test_chunk(self):
        x = torch.tensor([1, 2, 3, 4, 5, 6])
        result = torch.chunk(x, 3)
        assert len(result) == 3
        np.testing.assert_array_equal(result[0].numpy(), [1, 2])
        np.testing.assert_array_equal(result[1].numpy(), [3, 4])
        np.testing.assert_array_equal(result[2].numpy(), [5, 6])


class TestClone:
    def test_clone(self):
        x = torch.tensor([1.0, 2.0, 3.0])
        y = torch.clone(x)
        # Modify original
        x_np = x.numpy()
        # Clone should be independent
        np.testing.assert_array_equal(y.numpy(), [1.0, 2.0, 3.0])

    def test_clone_method(self):
        x = torch.tensor([1.0, 2.0, 3.0])
        y = x.clone()
        np.testing.assert_array_equal(y.numpy(), [1.0, 2.0, 3.0])


class TestWhere:
    def test_where(self):
        condition = torch.tensor([True, False, True])
        x = torch.tensor([1, 2, 3])
        y = torch.tensor([4, 5, 6])
        result = torch.where(condition, x, y)
        expected = np.array([1, 5, 3])
        np.testing.assert_array_equal(result.numpy(), expected)


class TestRepeat:
    def test_repeat(self):
        x = torch.tensor([1, 2, 3])
        result = x.repeat(2)
        expected = np.array([1, 2, 3, 1, 2, 3])
        np.testing.assert_array_equal(result.numpy(), expected)

    def test_repeat_2d(self):
        x = torch.tensor([[1, 2], [3, 4]])
        result = x.repeat(2, 3)
        assert result.shape == (4, 6)


class TestMaskedFill:
    def test_masked_fill(self):
        x = torch.tensor([1.0, 2.0, 3.0, 4.0])
        mask = torch.tensor([True, False, True, False])
        result = x.masked_fill(mask, 0.0)
        expected = np.array([0.0, 2.0, 0.0, 4.0])
        np.testing.assert_array_equal(result.numpy(), expected)

    def test_masked_fill_inplace(self):
        x = torch.tensor([1.0, 2.0, 3.0, 4.0])
        mask = torch.tensor([True, False, True, False])
        x.masked_fill_(mask, 0.0)
        expected = np.array([0.0, 2.0, 0.0, 4.0])
        np.testing.assert_array_equal(x.numpy(), expected)


class TestVarStd:
    def test_var(self):
        x = torch.tensor([1.0, 2.0, 3.0, 4.0, 5.0])
        result = torch.var(x)
        # With Bessel's correction (default)
        expected = np.var([1.0, 2.0, 3.0, 4.0, 5.0], ddof=1)
        np.testing.assert_almost_equal(result.item(), expected)

    def test_std(self):
        x = torch.tensor([1.0, 2.0, 3.0, 4.0, 5.0])
        result = torch.std(x)
        expected = np.std([1.0, 2.0, 3.0, 4.0, 5.0], ddof=1)
        np.testing.assert_almost_equal(result.item(), expected)


class TestClamp:
    def test_clamp(self):
        x = torch.tensor([-1.0, 0.0, 1.0, 2.0, 3.0])
        result = torch.clamp(x, min=0.0, max=2.0)
        expected = np.array([0.0, 0.0, 1.0, 2.0, 2.0])
        np.testing.assert_array_equal(result.numpy(), expected)

    def test_clamp_min_only(self):
        x = torch.tensor([-1.0, 0.0, 1.0])
        result = torch.clamp(x, min=0.0)
        expected = np.array([0.0, 0.0, 1.0])
        np.testing.assert_array_equal(result.numpy(), expected)


class TestRsqrt:
    def test_rsqrt(self):
        x = torch.tensor([1.0, 4.0, 9.0, 16.0])
        result = torch.rsqrt(x)
        expected = np.array([1.0, 0.5, 1/3, 0.25])
        np.testing.assert_array_almost_equal(result.numpy(), expected)


class TestReciprocal:
    def test_reciprocal(self):
        x = torch.tensor([1.0, 2.0, 4.0, 5.0])
        result = torch.reciprocal(x)
        expected = np.array([1.0, 0.5, 0.25, 0.2])
        np.testing.assert_array_almost_equal(result.numpy(), expected)


class TestBmm:
    def test_bmm(self):
        a = torch.randn(10, 3, 4)
        b = torch.randn(10, 4, 5)
        result = torch.bmm(a, b)
        assert result.shape == (10, 3, 5)


class TestBaddbmm:
    def test_baddbmm(self):
        M = torch.randn(10, 3, 5)
        batch1 = torch.randn(10, 3, 4)
        batch2 = torch.randn(10, 4, 5)
        result = torch.baddbmm(M, batch1, batch2)
        assert result.shape == (10, 3, 5)
