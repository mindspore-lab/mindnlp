"""random op"""
import mindspore
from mindspore import ops
from mindspore.ops._primitive_cache import _get_cache_prim
from mindnlp.configs import USE_PYBOOST, DEVICE_TARGET, GENERATOR_SEED
from .other import cumsum, searchsorted
from .comparison import topk
from .pointwise import div, log
from ..utils import get_default_dtype

# bernoulli
def bernoulli(input, p=0.5):
    if DEVICE_TARGET == 'Ascend':
        random_numbers = rand(*input.shape, dtype=input.dtype)
        samples = random_numbers < p
        samples = samples.int()
        return samples
    return ops.bernoulli(input, p)

# multinomial
def multinomial(input, num_samples, replacement=False):
    """custom multinomial"""
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
def normal(mean=0.0, std=1.0, size=None):
    if USE_PYBOOST:
        return mindspore.mint.normal(mean, std, size)
    return ops.normal(size, mean, std)

# poisson


# rand
def rand(*size, dtype=None):
    if size[0] == []:
        size = ()
    elif isinstance(size[0], (tuple, list)):
        size = size[0]
    if dtype is None:
        dtype = get_default_dtype()
    if USE_PYBOOST:
        return mindspore.mint.rand(*size, dtype=dtype)
    return ops.rand(*size, dtype=dtype)

# rand_like
def rand_like(input, *, dtype=None):
    if USE_PYBOOST:
        return mindspore.mint.rand_like(input, dtype=dtype)
    return ops.rand_like(input, dtype=dtype)

# randint
def randint(low=0, high=None, size=None, *, dtype=None):
    return ops.randint(low, high, size, dtype=dtype)

# randint_like
def ranint_like(input, low, high, *, dtype=None):
    if dtype is None:
        dtype = input.dtype
    return randint(low, high, input.shape, dtype=dtype)

# randn
def randn(*size, dtype=None):
    if dtype is None:
        dtype = get_default_dtype()
    return ops.randn(*size, dtype=dtype)

# randn_like
def randn_like(input, *, dtype):
    return ops.randn_like(input, dtype=dtype)

# randperm
def randperm(n, seed=0, offset=0, dtype=mindspore.int64):
    """randperm"""
    if DEVICE_TARGET == 'CPU':
        if GENERATOR_SEED:
            randperm_v2_op = _get_cache_prim(ops.RandpermV2)(seed, offset, dtype)
            return randperm_v2_op(n)
        else:
            randperm_v2_op = _get_cache_prim(ops.RandpermV2)(dtype)
            return randperm_v2_op(mindspore.tensor([n]), seed, offset)

    randperm_op = _get_cache_prim(ops.Randperm)(max_length=n, dtype=dtype)
    return randperm_op(mindspore.tensor([n]))
