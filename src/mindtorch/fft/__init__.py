"""fft"""
import mindtorch
from ..executor import execute


def rfft(input, n=None, dim=-1, norm="backward"):
    return execute('rfft', input, n, dim, norm)

def irfft(input, n=None, dim=-1, norm="backward"):
    return execute('irfft', input, n, dim, norm)

def fftn(input, s=None, dim=None, norm=None):
    return execute('fftn', input, s, dim, norm)

def fft(input, s=None, dim=-1, norm=None):
    return execute('fft', input, s, dim, norm)

def fftshift(input, dim=None):
    return execute('fftshift', input, dim)

def ifftn(input, s=None, dim=None, norm=None, *, out=None):
    return execute('ifftn', input, s, dim, norm)

def ifftshift(input, dim=None):
    return execute('ifftshift', input, dim)

def fft2(input, s=None, dim=(-2, -1), norm=None):
    return execute('fft2', input, s, dim, norm)

def ifft2(input, s=None, dim=(-2, -1), norm=None):
    return execute('ifft2', input, s, dim, norm)

def ifft(input, s=None, dim=-1, norm=None):
    return execute('ifft', input, s, dim, norm)

__all__ = ['fft', 'fftn', 'irfft', 'rfft']
