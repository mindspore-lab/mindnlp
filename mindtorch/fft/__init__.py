"""fft"""
import mindtorch
from ..executor import execute


def rfft(input, n=None, dim=-1, norm="backward"):
    return execute('rfft', input, n, dim, norm)
    # if use_pyboost():
    #     return ops.rfft(input, n, dim, norm)
    # if input.shape[dim] < n:
    #     pad_inf = (0, n - input.shape[dim])
    #     pad_dims = (0, 0) * (input.ndim - (dim + 1)) + pad_inf
    #     input = F.pad(input, pad_dims)
    # else:
    #     input = narrow(input, dim, 0, n)
    # _rfft = _get_cache_prim(ops.FFTWithSize)(input.ndim, False, True, norm)
    # return _rfft(input)

def irfft(input, n=None, dim=-1, norm="backward"):
    return execute('irfft', input, n, dim, norm)
    # if use_pyboost():
    #     return ops.irfft(input, n, dim, norm)
    # if input.shape[dim] < n:
    #     pad_inf = (0, n - input.shape[dim])
    #     pad_dims = (0, 0) * (input.ndim - (dim + 1)) + pad_inf
    #     input = pad(input, pad_dims)
    # else:
    #     input = narrow(input, dim, 0, n)
    # _irfft = _get_cache_prim(ops.FFTWithSize)(input.ndim, True, True, norm)
    # return _irfft(input)

def fftn(input, s=None, dim=None, norm=None):
    if input.device.type == 'npu':
        return execute('fftn', input, s, dim, norm)
    if dim is None:
        dim = tuple(range(input.dim()))
    if s is None:
        s = [input.size(d) for d in dim]
    
    # 确保s和dim是序列且长度相同
    if not isinstance(s, (list, tuple)):
        s = (s,)
    if not isinstance(dim, (list, tuple)):
        dim = (dim,)
    if len(s) != len(dim):
        raise ValueError("参数 's' 和 'dim' 必须具有相同的长度。")
    
    output = input.to(mindtorch.complex64) if input.is_floating_point() else input.clone()

    # 逐个维度进行FFT
    for d, n in zip(dim, s):
        output = fft(output, s=n, dim=d, norm=norm)
    return output

def fft(input, s=None, dim=-1, norm=None):
    return execute('fft', input, s, dim, norm)

def fftshift(x, dim=None):
    return ops.fftshift(x, dim)

def ifftn(input, s=None, dim=None, norm=None, *, out=None):
    return ops.ifftn(input, s, dim, norm)

def ifftshift(input, dim=None):
    return ops.ifftshift(input, dim)

def fft2(input, s=None, dim=(-2, -1), norm=None):
    return ops.fft2(input, s, dim, norm)

def ifft2(input, s=None, dim=(-2, -1), norm=None):
    return ops.ifft2(input, s, dim, norm)

def ifft(input, s=None, dim=-1, norm=None):
    return ops.ifft(input, s, dim, norm)

__all__ = ['fft', 'fftn', 'irfft', 'rfft']
