"""Tests for torch.fft module."""

import sys
import os
import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "src"))

import mindtorch_v2 as torch


class TestFFT1D:
    """Tests for 1D FFT functions."""

    def test_fft(self):
        x = torch.tensor([1.0, 2.0, 3.0, 4.0])
        result = torch.fft.fft(x)
        expected = np.fft.fft([1, 2, 3, 4])
        np.testing.assert_allclose(result.numpy(), expected, atol=1e-10)

    def test_ifft(self):
        x = torch.tensor([1.0, 2.0, 3.0, 4.0])
        fft_result = torch.fft.fft(x)
        recovered = torch.fft.ifft(fft_result)
        np.testing.assert_allclose(np.real(recovered.numpy()), x.numpy(), atol=1e-10)

    def test_fft_roundtrip(self):
        x = torch.tensor([1.0, -1.0, 2.0, -2.0, 3.0, -3.0, 4.0, -4.0])
        recovered = torch.fft.ifft(torch.fft.fft(x))
        np.testing.assert_allclose(np.real(recovered.numpy()), x.numpy(), atol=1e-10)

    def test_fft_with_n(self):
        x = torch.tensor([1.0, 2.0, 3.0, 4.0])
        result = torch.fft.fft(x, n=8)
        expected = np.fft.fft([1, 2, 3, 4], n=8)
        np.testing.assert_allclose(result.numpy(), expected, atol=1e-10)

    def test_fft_normalized(self):
        x = torch.tensor([1.0, 2.0, 3.0, 4.0])
        result = torch.fft.fft(x, norm="ortho")
        expected = np.fft.fft([1, 2, 3, 4], norm="ortho")
        np.testing.assert_allclose(result.numpy(), expected, atol=1e-10)


class TestRFFT:
    """Tests for real-valued FFT functions."""

    def test_rfft(self):
        x = torch.tensor([1.0, 2.0, 3.0, 4.0])
        result = torch.fft.rfft(x)
        expected = np.fft.rfft([1, 2, 3, 4])
        np.testing.assert_allclose(result.numpy(), expected, atol=1e-10)

    def test_irfft(self):
        x = torch.tensor([1.0, 2.0, 3.0, 4.0])
        freq = torch.fft.rfft(x)
        recovered = torch.fft.irfft(freq)
        np.testing.assert_allclose(recovered.numpy(), x.numpy(), atol=1e-10)

    def test_rfft_output_size(self):
        x = torch.tensor([1.0, 2.0, 3.0, 4.0])
        result = torch.fft.rfft(x)
        # For n=4, rfft output is n//2+1 = 3
        assert result.shape == (3,)


class TestFFT2D:
    """Tests for 2D FFT functions."""

    def test_fft2(self):
        x = torch.tensor([[1.0, 2.0], [3.0, 4.0]])
        result = torch.fft.fft2(x)
        expected = np.fft.fft2([[1, 2], [3, 4]])
        np.testing.assert_allclose(result.numpy(), expected, atol=1e-10)

    def test_ifft2(self):
        x = torch.tensor([[1.0, 2.0], [3.0, 4.0]])
        fft_result = torch.fft.fft2(x)
        recovered = torch.fft.ifft2(fft_result)
        np.testing.assert_allclose(np.real(recovered.numpy()), x.numpy(), atol=1e-10)

    def test_rfft2(self):
        x = torch.tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
        result = torch.fft.rfft2(x)
        expected = np.fft.rfft2([[1, 2, 3], [4, 5, 6]])
        np.testing.assert_allclose(result.numpy(), expected, atol=1e-10)


class TestFFTND:
    """Tests for N-D FFT functions."""

    def test_fftn(self):
        x = torch.tensor([[[1.0, 2.0], [3.0, 4.0]], [[5.0, 6.0], [7.0, 8.0]]])
        result = torch.fft.fftn(x)
        expected = np.fft.fftn([[[1, 2], [3, 4]], [[5, 6], [7, 8]]])
        np.testing.assert_allclose(result.numpy(), expected, atol=1e-10)

    def test_ifftn(self):
        x = torch.tensor([[[1.0, 2.0], [3.0, 4.0]], [[5.0, 6.0], [7.0, 8.0]]])
        recovered = torch.fft.ifftn(torch.fft.fftn(x))
        np.testing.assert_allclose(np.real(recovered.numpy()), x.numpy(), atol=1e-10)

    def test_rfftn(self):
        x = torch.tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
        result = torch.fft.rfftn(x)
        expected = np.fft.rfftn([[1, 2, 3], [4, 5, 6]])
        np.testing.assert_allclose(result.numpy(), expected, atol=1e-10)

    def test_irfftn(self):
        x = torch.tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
        freq = torch.fft.rfftn(x)
        recovered = torch.fft.irfftn(freq, s=(2, 3))
        np.testing.assert_allclose(recovered.numpy(), x.numpy(), atol=1e-10)


class TestHFFT:
    """Tests for Hermitian FFT functions."""

    def test_hfft(self):
        x = torch.tensor([1.0, 2.0, 3.0, 4.0])
        result = torch.fft.hfft(x)
        expected = np.fft.hfft([1, 2, 3, 4])
        np.testing.assert_allclose(result.numpy(), expected, atol=1e-10)

    def test_ihfft(self):
        x = torch.tensor([1.0, 2.0, 3.0, 4.0])
        result = torch.fft.ihfft(x)
        expected = np.fft.ihfft([1, 2, 3, 4])
        np.testing.assert_allclose(result.numpy(), expected, atol=1e-10)


class TestFFTUtilities:
    """Tests for FFT utility functions."""

    def test_fftfreq(self):
        result = torch.fft.fftfreq(5, d=1.0)
        expected = np.fft.fftfreq(5, d=1.0)
        np.testing.assert_allclose(result.numpy(), expected, atol=1e-10)

    def test_rfftfreq(self):
        result = torch.fft.rfftfreq(5, d=1.0)
        expected = np.fft.rfftfreq(5, d=1.0)
        np.testing.assert_allclose(result.numpy(), expected, atol=1e-10)

    def test_fftshift(self):
        x = torch.tensor([0.0, 1.0, 2.0, 3.0, 4.0])
        result = torch.fft.fftshift(x)
        expected = np.fft.fftshift([0, 1, 2, 3, 4])
        np.testing.assert_allclose(result.numpy(), expected, atol=1e-10)

    def test_ifftshift(self):
        x = torch.tensor([0.0, 1.0, 2.0, 3.0, 4.0])
        shifted = torch.fft.fftshift(x)
        recovered = torch.fft.ifftshift(shifted)
        np.testing.assert_allclose(recovered.numpy(), x.numpy(), atol=1e-10)

    def test_fftshift_2d(self):
        x = torch.tensor([[0.0, 1.0, 2.0], [3.0, 4.0, 5.0], [6.0, 7.0, 8.0]])
        result = torch.fft.fftshift(x)
        expected = np.fft.fftshift([[0, 1, 2], [3, 4, 5], [6, 7, 8]])
        np.testing.assert_allclose(result.numpy(), expected, atol=1e-10)

    def test_fftfreq_custom_d(self):
        result = torch.fft.fftfreq(8, d=0.5)
        expected = np.fft.fftfreq(8, d=0.5)
        np.testing.assert_allclose(result.numpy(), expected, atol=1e-10)
