"""Tests for random distribution ops in mindtorch_v2."""

import pytest
import torch


class TestBernoulliInplace:
    def test_shape_and_range(self):
        """bernoulli_ output should be 0 or 1 only."""
        x = torch.empty(1000)
        x.bernoulli_(0.5)
        vals = set(x.numpy().flatten().tolist())
        assert vals <= {0.0, 1.0}

    def test_with_generator(self):
        """bernoulli_ with same generator seed should be reproducible."""
        g1 = torch.Generator()
        g1.manual_seed(42)
        a = torch.empty(100)
        a.bernoulli_(0.5, generator=g1)

        g2 = torch.Generator()
        g2.manual_seed(42)
        b = torch.empty(100)
        b.bernoulli_(0.5, generator=g2)

        assert torch.equal(a, b)

    def test_tensor_p(self):
        """bernoulli_ with a tensor of probabilities."""
        p = torch.tensor([0.0, 1.0, 0.0, 1.0])
        x = torch.empty(4)
        # p=0 -> always 0, p=1 -> always 1
        x.bernoulli_(p)
        assert x[0].item() == 0.0
        assert x[1].item() == 1.0
        assert x[2].item() == 0.0
        assert x[3].item() == 1.0


class TestExponentialInplace:
    def test_positive(self):
        """exponential_ output should be strictly positive."""
        x = torch.empty(1000)
        x.exponential_(1.0)
        assert (x.numpy() > 0).all()

    def test_with_generator(self):
        """exponential_ with same generator seed should be reproducible."""
        g1 = torch.Generator()
        g1.manual_seed(123)
        a = torch.empty(100)
        a.exponential_(2.0, generator=g1)

        g2 = torch.Generator()
        g2.manual_seed(123)
        b = torch.empty(100)
        b.exponential_(2.0, generator=g2)

        assert torch.equal(a, b)


class TestLogNormalInplace:
    def test_positive(self):
        """log_normal_ output should be strictly positive."""
        x = torch.empty(1000)
        x.log_normal_(1.0, 0.5)
        assert (x.numpy() > 0).all()

    def test_with_generator(self):
        """log_normal_ with same generator seed should be reproducible."""
        g1 = torch.Generator()
        g1.manual_seed(99)
        a = torch.empty(100)
        a.log_normal_(0.0, 1.0, generator=g1)

        g2 = torch.Generator()
        g2.manual_seed(99)
        b = torch.empty(100)
        b.log_normal_(0.0, 1.0, generator=g2)

        assert torch.equal(a, b)


class TestCauchyInplace:
    def test_with_generator(self):
        """cauchy_ with same generator seed should be reproducible."""
        g1 = torch.Generator()
        g1.manual_seed(77)
        a = torch.empty(100)
        a.cauchy_(0.0, 1.0, generator=g1)

        g2 = torch.Generator()
        g2.manual_seed(77)
        b = torch.empty(100)
        b.cauchy_(0.0, 1.0, generator=g2)

        assert torch.equal(a, b)


class TestGeometricInplace:
    def test_positive_int(self):
        """geometric_ output should be >= 1 (integer-valued)."""
        x = torch.empty(1000)
        x.geometric_(0.3)
        arr = x.numpy()
        assert (arr >= 1).all()
        # Values should be integer-like
        assert ((arr - arr.astype(int).astype(arr.dtype)) == 0).all()

    def test_with_generator(self):
        """geometric_ with same generator seed should be reproducible."""
        g1 = torch.Generator()
        g1.manual_seed(55)
        a = torch.empty(100)
        a.geometric_(0.5, generator=g1)

        g2 = torch.Generator()
        g2.manual_seed(55)
        b = torch.empty(100)
        b.geometric_(0.5, generator=g2)

        assert torch.equal(a, b)


class TestPoisson:
    def test_shape(self):
        """poisson output shape should match input."""
        lam = torch.ones(3, 4) * 5.0
        out = torch.poisson(lam)
        assert out.shape == lam.shape

    def test_with_generator(self):
        """poisson with same generator seed should be reproducible."""
        g1 = torch.Generator()
        g1.manual_seed(33)
        lam = torch.ones(100) * 3.0
        a = torch.poisson(lam, generator=g1)

        g2 = torch.Generator()
        g2.manual_seed(33)
        b = torch.poisson(lam, generator=g2)

        assert torch.equal(a, b)


class TestAlphaDropout:
    def test_preserves_shape(self):
        """AlphaDropout should preserve the input shape."""
        m = torch.nn.AlphaDropout(p=0.5)
        m.train()
        x = torch.randn(4, 8)
        out = m(x)
        assert out.shape == x.shape

    def test_no_effect_eval(self):
        """AlphaDropout should be a no-op in eval mode."""
        m = torch.nn.AlphaDropout(p=0.5)
        m.eval()
        x = torch.randn(4, 8)
        out = m(x)
        assert torch.equal(out, x)
