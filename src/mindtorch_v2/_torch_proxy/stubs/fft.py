"""Stub implementation of torch.fft module using numpy.fft."""
import numpy as np


def _ensure_tensor(arr):
    """Convert numpy array back to Tensor."""
    from mindtorch_v2._tensor import Tensor
    return Tensor(arr)


def fft(input, n=None, dim=-1, norm=None):
    """Compute 1D FFT."""
    arr = input.numpy()
    result = np.fft.fft(arr, n=n, axis=dim, norm=norm)
    return _ensure_tensor(result)


def ifft(input, n=None, dim=-1, norm=None):
    """Compute 1D inverse FFT."""
    arr = input.numpy()
    result = np.fft.ifft(arr, n=n, axis=dim, norm=norm)
    return _ensure_tensor(result)


def fft2(input, s=None, dim=(-2, -1), norm=None):
    """Compute 2D FFT."""
    arr = input.numpy()
    result = np.fft.fft2(arr, s=s, axes=dim, norm=norm)
    return _ensure_tensor(result)


def ifft2(input, s=None, dim=(-2, -1), norm=None):
    """Compute 2D inverse FFT."""
    arr = input.numpy()
    result = np.fft.ifft2(arr, s=s, axes=dim, norm=norm)
    return _ensure_tensor(result)


def fftn(input, s=None, dim=None, norm=None):
    """Compute N-dimensional FFT."""
    arr = input.numpy()
    result = np.fft.fftn(arr, s=s, axes=dim, norm=norm)
    return _ensure_tensor(result)


def ifftn(input, s=None, dim=None, norm=None):
    """Compute N-dimensional inverse FFT."""
    arr = input.numpy()
    result = np.fft.ifftn(arr, s=s, axes=dim, norm=norm)
    return _ensure_tensor(result)


def rfft(input, n=None, dim=-1, norm=None):
    """Compute 1D FFT of real input."""
    arr = input.numpy()
    result = np.fft.rfft(arr, n=n, axis=dim, norm=norm)
    return _ensure_tensor(result)


def irfft(input, n=None, dim=-1, norm=None):
    """Compute 1D inverse FFT returning real output."""
    arr = input.numpy()
    result = np.fft.irfft(arr, n=n, axis=dim, norm=norm)
    return _ensure_tensor(result)


def rfft2(input, s=None, dim=(-2, -1), norm=None):
    """Compute 2D FFT of real input."""
    arr = input.numpy()
    result = np.fft.rfft2(arr, s=s, axes=dim, norm=norm)
    return _ensure_tensor(result)


def irfft2(input, s=None, dim=(-2, -1), norm=None):
    """Compute 2D inverse FFT returning real output."""
    arr = input.numpy()
    result = np.fft.irfft2(arr, s=s, axes=dim, norm=norm)
    return _ensure_tensor(result)


def rfftn(input, s=None, dim=None, norm=None):
    """Compute N-dimensional FFT of real input."""
    arr = input.numpy()
    result = np.fft.rfftn(arr, s=s, axes=dim, norm=norm)
    return _ensure_tensor(result)


def irfftn(input, s=None, dim=None, norm=None):
    """Compute N-dimensional inverse FFT returning real output."""
    arr = input.numpy()
    result = np.fft.irfftn(arr, s=s, axes=dim, norm=norm)
    return _ensure_tensor(result)


def hfft(input, n=None, dim=-1, norm=None):
    """Compute 1D FFT of Hermitian-symmetric input."""
    arr = input.numpy()
    result = np.fft.hfft(arr, n=n, axis=dim, norm=norm)
    return _ensure_tensor(result)


def ihfft(input, n=None, dim=-1, norm=None):
    """Compute inverse FFT returning Hermitian-symmetric output."""
    arr = input.numpy()
    result = np.fft.ihfft(arr, n=n, axis=dim, norm=norm)
    return _ensure_tensor(result)


def fftfreq(n, d=1.0, *, dtype=None, layout=None, device=None, requires_grad=False):
    """Return discrete Fourier Transform sample frequencies."""
    result = np.fft.fftfreq(n, d=d)
    if dtype is not None:
        from mindtorch_v2._dtype import DTYPE_MAP
        np_dtype = DTYPE_MAP.get(dtype, np.float32)
        result = result.astype(np_dtype)
    return _ensure_tensor(result)


def rfftfreq(n, d=1.0, *, dtype=None, layout=None, device=None, requires_grad=False):
    """Return discrete Fourier Transform sample frequencies (real FFT)."""
    result = np.fft.rfftfreq(n, d=d)
    if dtype is not None:
        from mindtorch_v2._dtype import DTYPE_MAP
        np_dtype = DTYPE_MAP.get(dtype, np.float32)
        result = result.astype(np_dtype)
    return _ensure_tensor(result)


def fftshift(input, dim=None):
    """Shift zero-frequency component to center."""
    arr = input.numpy()
    result = np.fft.fftshift(arr, axes=dim)
    return _ensure_tensor(result)


def ifftshift(input, dim=None):
    """Inverse of fftshift."""
    arr = input.numpy()
    result = np.fft.ifftshift(arr, axes=dim)
    return _ensure_tensor(result)
