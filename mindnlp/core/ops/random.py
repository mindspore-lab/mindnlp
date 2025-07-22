"""random op"""
import numpy as np
import mindspore
from mindspore import ops
from mindspore.ops._primitive_cache import _get_cache_prim
from ..configs import use_pyboost, DEVICE_TARGET, ON_A1
from .other import cumsum, searchsorted
from .comparison import topk
from .pointwise import div, log
from .._bind import get_default_dtype
from ._inner import call_ms_func

# bernoulli
has_bernoulli = hasattr(mindspore.mint, 'bernoulli')
def bernoulli(input, *, generator=None, out=None):
    if use_pyboost() and has_bernoulli:
        return call_ms_func(mindspore.mint.bernoulli, input, generator=generator, out=out)
    random_numbers = rand(*input.shape, dtype=mindspore.float32)
    samples = random_numbers < 0.5
    samples = samples.int()
    if out is None:
        return samples
    else:
        return out.copy_(samples)

# multinomial
has_multinomial = hasattr(mindspore.mint, 'multinomial')
def multinomial(input, num_samples, replacement=False, *, generator=None):
    """custom multinomial"""
    if use_pyboost() and has_multinomial and not ON_A1:
        return mindspore.mint.multinomial(input, num_samples, replacement=replacement, generator=generator)
    if replacement:
        # with replacement
        cumulative_probs = cumsum(input, dim=-1)
        uniform_samples = rand(*input.shape[:-1] + (num_samples,))
        if cumulative_probs.dtype == mindspore.float16:
            cumulative_probs = cumulative_probs.astype(mindspore.float32)
        samples = searchsorted(cumulative_probs, uniform_samples, right=True)
    else:
        # without replacement
        n_dist = 1
        if input.ndim > 1:
            n_dist = input.shape[-2]
        random_uniform = rand(*(n_dist * input.shape[-1],))
        if n_dist != 1:
            random_uniform = random_uniform.reshape(n_dist, input.shape[-1])

        vals = div(log(random_uniform), input + 1e-10)
        _, samples = topk(vals, num_samples)

    return samples.astype(mindspore.int64)

# normal
has_normal = hasattr(mindspore.mint, 'normal')
def normal(mean=0.0, std=1.0, size=None, *, generator=None, out=None):
    if use_pyboost() and has_normal:
        return call_ms_func(mindspore.mint.normal, float(mean), float(std), size, generator, out=out)
    if size is None:
        if isinstance(mean, mindspore.Tensor):
            size = mean.shape
        else:
            size = ()
    return call_ms_func(ops.normal, size, mean, std, out=out)

# poisson


# rand
has_rand = hasattr(mindspore.mint, 'rand')
def rand(*size, generator=None, out=None, dtype=None, device=None, pin_memory=False, **kwargs):
    size = kwargs.pop('size', size)
    if size[0] == []:
        size = ()
    elif isinstance(size[0], (tuple, list)):
        size = size[0]
    if dtype is None:
        dtype = get_default_dtype()
    if use_pyboost() and has_rand:
        return call_ms_func(mindspore.mint.rand, *size, generator=generator, dtype=dtype, out=out)
    return call_ms_func(ops.rand, *size, dtype=dtype, out=out)

# rand_like
has_rand_like = hasattr(mindspore.mint, 'rand_like')
def rand_like(input, *, dtype=None):
    if use_pyboost() and has_rand_like:
        return mindspore.mint.rand_like(input, dtype=dtype)
    return ops.rand_like(input, dtype=dtype)

# randint
has_randint = hasattr(mindspore.mint, 'randint')
def randint(*args, **kwargs):
    device = kwargs.pop('device', None)
    high = kwargs.pop('high', None)
    size = kwargs.pop('size', None)
    if high is not None:
        args += (high,)
    
    if size is not None:
        args += (size,)

    if use_pyboost() and has_randint:
        return mindspore.mint.randint(*args, **kwargs)
    return ops.randint(*args, **kwargs)

# randint_like
def randint_like(*args, **kwargs):
    if use_pyboost() and has_randint:
        return mindspore.mint.randint_like(*args, **kwargs)
    return ops.randint_like(*args, **kwargs)

# randn
has_randn = hasattr(mindspore.mint, 'randn')
def randn(*size, generator=None, dtype=None, **kwargs):
    size = kwargs.pop('size', size)
    if dtype is None:
        dtype = get_default_dtype()
    if use_pyboost() and has_randn:
        return mindspore.mint.randn(*size, generator=generator, dtype=dtype)
    return ops.randn(*size, dtype=dtype)

# randn_like
has_randn_like = hasattr(mindspore.mint, 'randn_like')
def randn_like(input, *, dtype=None):
    if use_pyboost() and has_randn_like:
        return mindspore.mint.randn_like(input, dtype=dtype)
    return ops.randn_like(input, dtype=dtype)

# randperm
has_randperm = hasattr(mindspore.mint, 'randperm')
def randperm(n, *, generator=None, dtype=mindspore.int64):
    """randperm"""
    if use_pyboost() and has_randperm:
        return mindspore.mint.randperm(n, generator=generator, dtype=dtype)
    if DEVICE_TARGET == 'CPU':
        seed, offset = 0, 0
        randperm_v2_op = _get_cache_prim(ops.RandpermV2)(seed, offset, dtype)
        return randperm_v2_op(n)

    randperm_op = _get_cache_prim(ops.Randperm)(max_length=n, dtype=dtype)
    return randperm_op(mindspore.tensor([n]))

def gamma(shape, alpha, beta):
    if DEVICE_TARGET != 'Ascend':
        return mindspore.tensor(np.random.gamma(alpha, 1/beta, shape))
    return ops.gamma(shape, alpha, beta)

__all__ = ['bernoulli', 'gamma', 'multinomial', 'normal', 'rand', 'rand_like', 'randint', 'randn', 'randn_like', 'randperm', 'randint_like']
