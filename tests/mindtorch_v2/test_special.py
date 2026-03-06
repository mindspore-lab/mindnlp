"""Tests for torch.special module."""

import sys
import os
import math
import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "src"))

import mindtorch_v2 as torch

try:
    from scipy import special as sp
    HAS_SCIPY = True
except ImportError:
    HAS_SCIPY = False

needs_scipy = pytest.mark.skipif(not HAS_SCIPY, reason="scipy not available")


class TestErrorFunctions:
    """Tests for error function variants."""

    def test_erf(self):
        x = torch.tensor([0.0, 0.5, 1.0, 2.0])
        result = torch.special.erf(x)
        expected = np.array([math.erf(v) for v in [0, 0.5, 1.0, 2.0]])
        np.testing.assert_allclose(result.numpy(), expected, atol=1e-5)

    def test_erfc(self):
        x = torch.tensor([0.0, 0.5, 1.0, 2.0])
        result = torch.special.erfc(x)
        expected = np.array([math.erfc(v) for v in [0, 0.5, 1.0, 2.0]])
        np.testing.assert_allclose(result.numpy(), expected, atol=1e-5)

    @needs_scipy
    def test_erfcx(self):
        x = torch.tensor([0.0, 0.5, 1.0, 2.0])
        result = torch.special.erfcx(x)
        expected = sp.erfcx([0, 0.5, 1.0, 2.0])
        np.testing.assert_allclose(result.numpy(), expected, atol=1e-5)

    @needs_scipy
    def test_erfinv(self):
        x = torch.tensor([0.0, 0.5, -0.5, 0.9])
        result = torch.special.erfinv(x)
        expected = sp.erfinv([0, 0.5, -0.5, 0.9])
        np.testing.assert_allclose(result.numpy(), expected, atol=1e-5)


class TestExponentialLog:
    """Tests for exponential and log functions."""

    def test_expit(self):
        x = torch.tensor([0.0, 1.0, -1.0, 10.0])
        result = torch.special.expit(x)
        expected = 1.0 / (1.0 + np.exp(-np.array([0, 1, -1, 10])))
        np.testing.assert_allclose(result.numpy(), expected, atol=1e-5)

    def test_exp2(self):
        x = torch.tensor([0.0, 1.0, 2.0, 3.0])
        result = torch.special.exp2(x)
        expected = np.array([1.0, 2.0, 4.0, 8.0])
        np.testing.assert_allclose(result.numpy(), expected, atol=1e-5)

    def test_expm1(self):
        x = torch.tensor([0.0, 1e-10, 1.0])
        result = torch.special.expm1(x)
        expected = np.expm1([0, 1e-10, 1.0])
        np.testing.assert_allclose(result.numpy(), expected, atol=1e-5)

    def test_log1p(self):
        x = torch.tensor([0.0, 1e-10, 1.0])
        result = torch.special.log1p(x)
        expected = np.log1p([0, 1e-10, 1.0])
        np.testing.assert_allclose(result.numpy(), expected, atol=1e-5)

    @needs_scipy
    def test_logit(self):
        x = torch.tensor([0.1, 0.5, 0.9])
        result = torch.special.logit(x)
        expected = sp.logit([0.1, 0.5, 0.9])
        np.testing.assert_allclose(result.numpy(), expected, atol=1e-5)

    @needs_scipy
    def test_logit_with_eps(self):
        x = torch.tensor([0.0, 0.5, 1.0])
        result = torch.special.logit(x, eps=0.01)
        expected = sp.logit(np.clip([0, 0.5, 1.0], 0.01, 0.99))
        np.testing.assert_allclose(result.numpy(), expected, atol=1e-5)


class TestGammaFunctions:
    """Tests for gamma-related functions."""

    @needs_scipy
    def test_gammaln(self):
        x = torch.tensor([1.0, 2.0, 3.0, 5.0])
        result = torch.special.gammaln(x)
        expected = sp.gammaln([1, 2, 3, 5])
        np.testing.assert_allclose(result.numpy(), expected, atol=1e-5)

    @needs_scipy
    def test_digamma(self):
        x = torch.tensor([1.0, 2.0, 3.0])
        result = torch.special.digamma(x)
        expected = sp.digamma([1, 2, 3])
        np.testing.assert_allclose(result.numpy(), expected, atol=1e-5)

    @needs_scipy
    def test_psi_alias(self):
        x = torch.tensor([1.0, 2.0, 3.0])
        result = torch.special.psi(x)
        expected = sp.digamma([1, 2, 3])
        np.testing.assert_allclose(result.numpy(), expected, atol=1e-5)

    @needs_scipy
    def test_polygamma(self):
        x = torch.tensor([1.0, 2.0, 3.0])
        result = torch.special.polygamma(1, x)
        expected = sp.polygamma(1, [1, 2, 3])
        np.testing.assert_allclose(result.numpy(), expected, atol=1e-5)

    @needs_scipy
    def test_multigammaln(self):
        x = torch.tensor([3.0, 4.0, 5.0])
        result = torch.special.multigammaln(x, 2)
        expected = sp.multigammaln([3, 4, 5], 2)
        np.testing.assert_allclose(result.numpy(), expected, atol=1e-5)

    @needs_scipy
    def test_gammainc(self):
        a = torch.tensor([1.0, 2.0, 3.0])
        x = torch.tensor([0.5, 1.0, 2.0])
        result = torch.special.gammainc(a, x)
        expected = sp.gammainc([1, 2, 3], [0.5, 1, 2])
        np.testing.assert_allclose(result.numpy(), expected, atol=1e-5)

    @needs_scipy
    def test_gammaincc(self):
        a = torch.tensor([1.0, 2.0, 3.0])
        x = torch.tensor([0.5, 1.0, 2.0])
        result = torch.special.gammaincc(a, x)
        expected = sp.gammaincc([1, 2, 3], [0.5, 1, 2])
        np.testing.assert_allclose(result.numpy(), expected, atol=1e-5)


class TestBesselFunctions:
    """Tests for Bessel function variants."""

    @needs_scipy
    def test_i0(self):
        x = torch.tensor([0.0, 1.0, 2.0, 3.0])
        result = torch.special.i0(x)
        expected = sp.i0([0, 1, 2, 3])
        np.testing.assert_allclose(result.numpy(), expected, atol=1e-5)

    @needs_scipy
    def test_i0e(self):
        x = torch.tensor([0.0, 1.0, 2.0, 3.0])
        result = torch.special.i0e(x)
        expected = sp.i0e([0, 1, 2, 3])
        np.testing.assert_allclose(result.numpy(), expected, atol=1e-5)

    @needs_scipy
    def test_i1(self):
        x = torch.tensor([0.0, 1.0, 2.0, 3.0])
        result = torch.special.i1(x)
        expected = sp.i1([0, 1, 2, 3])
        np.testing.assert_allclose(result.numpy(), expected, atol=1e-5)

    @needs_scipy
    def test_i1e(self):
        x = torch.tensor([0.0, 1.0, 2.0, 3.0])
        result = torch.special.i1e(x)
        expected = sp.i1e([0, 1, 2, 3])
        np.testing.assert_allclose(result.numpy(), expected, atol=1e-5)


class TestNormalDistribution:
    """Tests for normal distribution functions."""

    @needs_scipy
    def test_ndtr(self):
        x = torch.tensor([-2.0, -1.0, 0.0, 1.0, 2.0])
        result = torch.special.ndtr(x)
        expected = sp.ndtr([-2, -1, 0, 1, 2])
        np.testing.assert_allclose(result.numpy(), expected, atol=1e-5)

    @needs_scipy
    def test_ndtri(self):
        x = torch.tensor([0.1, 0.25, 0.5, 0.75, 0.9])
        result = torch.special.ndtri(x)
        expected = sp.ndtri([0.1, 0.25, 0.5, 0.75, 0.9])
        np.testing.assert_allclose(result.numpy(), expected, atol=1e-5)

    @needs_scipy
    def test_log_ndtr(self):
        x = torch.tensor([-2.0, 0.0, 2.0])
        result = torch.special.log_ndtr(x)
        expected = sp.log_ndtr([-2, 0, 2])
        np.testing.assert_allclose(result.numpy(), expected, atol=1e-5)


class TestMiscSpecial:
    """Tests for miscellaneous special functions."""

    def test_sinc(self):
        x = torch.tensor([0.0, 0.5, 1.0, 2.0])
        result = torch.special.sinc(x)
        expected = np.sinc([0, 0.5, 1.0, 2.0])
        np.testing.assert_allclose(result.numpy(), expected, atol=1e-5)

    def test_entr_positive(self):
        x = torch.tensor([0.5, 1.0, 2.0])
        result = torch.special.entr(x)
        expected = -np.array([0.5, 1.0, 2.0]) * np.log([0.5, 1.0, 2.0])
        np.testing.assert_allclose(result.numpy(), expected, atol=1e-5)

    def test_entr_zero(self):
        x = torch.tensor([0.0])
        result = torch.special.entr(x)
        assert result.item() == 0.0

    def test_entr_negative(self):
        x = torch.tensor([-1.0])
        result = torch.special.entr(x)
        assert result.item() == float('-inf')

    def test_round(self):
        x = torch.tensor([0.5, 1.5, 2.4, 3.6])
        result = torch.special.round(x)
        expected = np.round([0.5, 1.5, 2.4, 3.6])
        np.testing.assert_allclose(result.numpy(), expected, atol=1e-5)

    @needs_scipy
    def test_xlogy(self):
        x = torch.tensor([0.0, 1.0, 2.0])
        y = torch.tensor([1.0, 2.0, 3.0])
        result = torch.special.xlogy(x, y)
        expected = sp.xlogy([0, 1, 2], [1, 2, 3])
        np.testing.assert_allclose(result.numpy(), expected, atol=1e-5)

    @needs_scipy
    def test_xlogy_zero(self):
        x = torch.tensor([0.0])
        y = torch.tensor([0.0])
        result = torch.special.xlogy(x, y)
        assert result.item() == 0.0

    @needs_scipy
    def test_xlog1py(self):
        x = torch.tensor([0.0, 1.0, 2.0])
        y = torch.tensor([0.0, 1.0, 2.0])
        result = torch.special.xlog1py(x, y)
        expected = sp.xlog1py([0, 1, 2], [0, 1, 2])
        np.testing.assert_allclose(result.numpy(), expected, atol=1e-5)

    @needs_scipy
    def test_zeta(self):
        x = torch.tensor([2.0, 3.0, 4.0])
        q = torch.tensor([1.0, 1.0, 1.0])
        result = torch.special.zeta(x, q)
        expected = sp.zeta([2, 3, 4], [1, 1, 1])
        np.testing.assert_allclose(result.numpy(), expected, atol=1e-5)

    def test_logsumexp(self):
        x = torch.tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
        result = torch.special.logsumexp(x, dim=1)
        expected = np.log(np.sum(np.exp([[1, 2, 3], [4, 5, 6]]), axis=1))
        np.testing.assert_allclose(result.numpy(), expected, atol=1e-5)

    def test_softmax(self):
        x = torch.tensor([1.0, 2.0, 3.0])
        result = torch.special.softmax(x, dim=0)
        expected = np.exp([1, 2, 3]) / np.sum(np.exp([1, 2, 3]))
        np.testing.assert_allclose(result.numpy(), expected, atol=1e-5)

    def test_log_softmax(self):
        x = torch.tensor([1.0, 2.0, 3.0])
        result = torch.special.log_softmax(x, dim=0)
        arr = np.array([1, 2, 3], dtype=np.float64)
        expected = arr - np.log(np.sum(np.exp(arr)))
        np.testing.assert_allclose(result.numpy(), expected, atol=1e-5)
