"""Implementation of torch.fft functions via dispatch."""

from .._dispatch.dispatcher import dispatch
from .._creation import tensor as _create_tensor
from .._dtype import float64 as _float64
import numpy as np


def fft(input, n=None, dim=-1, norm=None):
    """Computes the 1D discrete Fourier Transform."""
    return dispatch("fft_fft", input.device.type, input, n, dim, norm)


def ifft(input, n=None, dim=-1, norm=None):
    """Computes the 1D inverse discrete Fourier Transform."""
    return dispatch("fft_ifft", input.device.type, input, n, dim, norm)


def fft2(input, s=None, dim=(-2, -1), norm=None):
    """Computes the 2D discrete Fourier Transform."""
    return dispatch("fft_fft2", input.device.type, input, s, dim, norm)


def ifft2(input, s=None, dim=(-2, -1), norm=None):
    """Computes the 2D inverse discrete Fourier Transform."""
    return dispatch("fft_ifft2", input.device.type, input, s, dim, norm)


def fftn(input, s=None, dim=None, norm=None):
    """Computes the N-D discrete Fourier Transform."""
    return dispatch("fft_fftn", input.device.type, input, s, dim, norm)


def ifftn(input, s=None, dim=None, norm=None):
    """Computes the N-D inverse discrete Fourier Transform."""
    return dispatch("fft_ifftn", input.device.type, input, s, dim, norm)


def rfft(input, n=None, dim=-1, norm=None):
    """Computes the 1D FFT of real-valued input."""
    return dispatch("fft_rfft", input.device.type, input, n, dim, norm)


def irfft(input, n=None, dim=-1, norm=None):
    """Computes the inverse of rfft."""
    return dispatch("fft_irfft", input.device.type, input, n, dim, norm)


def rfft2(input, s=None, dim=(-2, -1), norm=None):
    """Computes the 2D FFT of real-valued input."""
    return dispatch("fft_rfft2", input.device.type, input, s, dim, norm)


def irfft2(input, s=None, dim=(-2, -1), norm=None):
    """Computes the inverse of rfft2."""
    return dispatch("fft_irfft2", input.device.type, input, s, dim, norm)


def rfftn(input, s=None, dim=None, norm=None):
    """Computes the N-D FFT of real-valued input."""
    return dispatch("fft_rfftn", input.device.type, input, s, dim, norm)


def irfftn(input, s=None, dim=None, norm=None):
    """Computes the inverse of rfftn."""
    return dispatch("fft_irfftn", input.device.type, input, s, dim, norm)


def hfft(input, n=None, dim=-1, norm=None):
    """Computes the 1D FFT of a Hermitian symmetric signal."""
    return dispatch("fft_hfft", input.device.type, input, n, dim, norm)


def ihfft(input, n=None, dim=-1, norm=None):
    """Computes the inverse of hfft."""
    return dispatch("fft_ihfft", input.device.type, input, n, dim, norm)


def fftfreq(n, d=1.0, *, dtype=None, device=None):
    """Computes the DFT sample frequencies."""
    freqs = np.fft.fftfreq(n, d=d)
    result = _create_tensor(freqs.tolist(), dtype=dtype or _float64)
    return result


def rfftfreq(n, d=1.0, *, dtype=None, device=None):
    """Computes the sample frequencies for rfft."""
    freqs = np.fft.rfftfreq(n, d=d)
    result = _create_tensor(freqs.tolist(), dtype=dtype or _float64)
    return result


def fftshift(input, dim=None):
    """Reorders an N-D FFT output by shifting the zero-frequency component to the center."""
    return dispatch("fft_fftshift", input.device.type, input, dim)


def ifftshift(input, dim=None):
    """Inverse of fftshift."""
    return dispatch("fft_ifftshift", input.device.type, input, dim)
