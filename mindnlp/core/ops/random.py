"""random op"""
import mindspore
from mindspore import ops
from mindnlp.configs import USE_PYBOOST, DEVICE_TARGET
from .other import cumsum, searchsorted
from .comparison import topk
from .pointwise import div, log

# bernoulli
def bernoulli(input):
    if DEVICE_TARGET == 'Ascend':
        random_numbers = rand(input.shape, dtype=input.dtype)
        samples = random_numbers < input
        samples = samples.int()
        return samples
    return ops.bernoulli(input)

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
    if USE_PYBOOST:
        return mindspore.mint.rand(*size, dtype=dtype)
    return ops.rand(*size, dtype=dtype)

# rand_like
def rand_like(input, *, dtype=None):
    if USE_PYBOOST:
        return mindspore.mint.rand_like(input, dtype=dtype)
    return ops.rand_like(input, dtype=dtype)

# randint
def randint(low, high, size, *, dtype=None):
    return ops.randint(low, high, size, dtype=dtype)

# randint_like
def ranint_like(input, low, high, *, dtype=None):
    if dtype is None:
        dtype = input.dtype
    return randint(low, high, input.shape, dtype=dtype)

# randn
def randn(*size, dtype=None):
    return ops.randn(*size, dtype=dtype)

# randn_like
def randn_like(input, *, dtype):
    return ops.randn_like(input, dtype=dtype)

# randperm
