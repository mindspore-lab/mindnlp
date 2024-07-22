"""fft"""
from mindspore import ops

def rfft(input, n=None, dim=-1, norm=None):
    return ops.rfft(input, n, dim, norm)

def irfft(input, n=None, dim=-1, norm=None):
    return ops.irfft(input, n, dim, norm)
