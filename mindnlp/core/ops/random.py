"""random op"""
from mindnlp import core
from mindnlp.core._C import default_generator
from mindnlp.core.executor import execute
from .._bind import get_default_dtype, get_device_in_context
from ..configs import ON_A1

generator_step_ = 12


# bernoulli
def bernoulli(input, *, generator=None, out=None):
    if generator is None:
        generator = default_generator
    output = execute("bernoulli_ext", input, generator)
    if out is None:
        return output
    out.data = output
    return out


# multinomial
def multinomial(input, num_samples, replacement=False, *, generator=None, out=None):
    """custom multinomial"""
    if generator is None:
        generator = default_generator
    if not ON_A1:
        output = execute("multinomial_ext", input, num_samples, replacement, generator)

    else:
        if replacement:
            # with replacement
            cumulative_probs = core.cumsum(input, dim=-1)
            uniform_samples = rand(*input.shape[:-1] + (num_samples,), device=input.device)
            if cumulative_probs.dtype == core.float16:
                cumulative_probs = cumulative_probs.astype(core.float32)
            samples = core.searchsorted(cumulative_probs, uniform_samples, right=True)
        else:
            # without replacement
            n_dist = 1
            if input.ndim > 1:
                n_dist = input.shape[-2]
            random_uniform = rand(*(n_dist * input.shape[-1],), device=input.device)
            if n_dist != 1:
                random_uniform = random_uniform.reshape(n_dist, input.shape[-1])

            vals = core.div(core.log(random_uniform), input + 1e-10)
            _, samples = core.topk(vals, num_samples)
    
        output = samples.astype(core.int64)

    if out is None:
        return output
    out.data = output
    return out


# normal
def normal(mean=0.0, std=1.0, *, size=None, generator=None, out=None,
           dtype=None, layout=None, device=None, pin_memory=None, requires_grad=False):
    if generator is None:
        generator = default_generator
    seed, offset = generator._step(generator_step_)  # pylint: disable=protected-access
    if device is None:
        if out is None:
            device = get_device_in_context()
        else:
            device = out.device

    is_mean_tensor = isinstance(mean, core.Tensor)
    is_std_tensor = isinstance(std, core.Tensor)

    if device.type == 'cpu':
        if is_mean_tensor and is_std_tensor:
            size = (mean * std).shape
        if is_mean_tensor and not is_std_tensor:
            size = mean.shape
        if not is_mean_tensor and is_std_tensor:
            size = std.shape
        if out is not None:
            size = out.shape
        output = execute('normal', size)
        output = output * std - mean

    else:
        if is_mean_tensor and is_std_tensor:
            output = execute("normal_tensor_tensor", mean, std, seed, offset, device=device)
        if is_mean_tensor and not is_std_tensor:
            output = execute("normal_tensor_float", mean, std, seed, offset, device=device)
        if not is_mean_tensor and is_std_tensor:
            output = execute("normal_float_tensor", mean, std, seed, offset, device=device)
        if out is not None:
            size = out.shape
        output = execute("normal_float_float", float(mean), float(std), size, seed, offset, device=device)

    if out is None:
        return output
    out.data = output
    return out

# poisson


# rand
def rand(
    *size,
    generator=None,
    out=None,
    dtype=None,
    layout=None,
    device=None,
    requires_grad=False,
    pin_memory=False
):
    if device is None:
        device = get_device_in_context()
    if isinstance(device, str):
        device = core.device(device)
    if dtype is None:
        dtype = get_default_dtype()
    if not generator:
        generator = default_generator
    seed, offset = generator._step(generator_step_)  # pylint: disable=protected-access
    if size and isinstance(size[0], (tuple, list)):
        size = size[0]
    output = execute(
        "rand_ext",
        size,
        seed,
        offset,
        dtype,
        device=device,
        requires_grad=requires_grad,
        user_created=True,
    )
    if out is None:
        return output
    out.data = output
    return out


# rand_like
def rand_like(
    input,
    *,
    dtype=None,
    layout=None,
    device=None,
    requires_grad=False,
    memory_format=None
):
    if device is None:
        device = input.device
    if isinstance(device, str):
        device = core.device(device)

    if dtype is None:
        dtype = input.dtype
    seed, offset = default_generator._step(  # pylint: disable=protected-access
        generator_step_
    )
    return execute(
        "rand_like_ext",
        input,
        seed,
        offset,
        dtype,
        device=device,
        requires_grad=requires_grad,
    )


# randint
def randint(
    low=0, high=0, size=None, *,
    generator=None,
    out=None,
    dtype=None,
    layout=None,
    device=None,
    requires_grad=False,
    **kwargs
):
    if dtype is None:
        dtype = core.int64
    if device is None:
        device = get_device_in_context()
    if isinstance(device, str):
        device = core.device(device)

    if generator is None:
        generator = default_generator

    if size is None and isinstance(high, (tuple, list)):
        low, high, size = 0, low, high

    output = execute(
        "randint",
        low, high, size,
        dtype,
        generator,
        device=device,
    )
    if out is None:
        return output
    out.data = output
    return out


# randint_like
def randint_like(
    input,
    low,
    high=0,
    *,
    dtype=None,
    layout=None,
    device=None,
    requires_grad=False,
    memory_format=None
):
    if high == 0:
        low, high = 0, low
    if device is None:
        device = input.device
    if isinstance(device, str):
        device = core.device(device)

    if dtype is None:
        dtype = input.dtype
    seed, offset = default_generator._step(  # pylint: disable=protected-access
        generator_step_
    )
    return execute(
        "randint_like_ext",
        input,
        low,
        high,
        seed,
        offset,
        dtype,
        device=device,
        requires_grad=requires_grad,
    )


# randn
def randn(
    *size,
    generator=None,
    out=None,
    dtype=None,
    layout=None,
    device=None,
    requires_grad=False,
    pin_memory=False
):
    if device is None:
        device = get_device_in_context()
    if isinstance(device, str):
        device = core.device(device)

    if dtype is None:
        dtype = get_default_dtype()
    if not generator:
        generator = default_generator
    seed, offset = generator._step(generator_step_)  # pylint: disable=protected-access
    if size and isinstance(size[0], (tuple, list)):
        size = size[0]
    output = execute(
        "randn",
        size,
        seed,
        offset,
        dtype,
        device=device,
        requires_grad=requires_grad,
        user_created=True,
    )
    if out is None:
        return output
    out.data = output
    return out


# randn_like
def randn_like(
    input,
    *,
    dtype=None,
    layout=None,
    device=None,
    requires_grad=False,
    memory_format=None
):
    if device is None:
        device = input.device
    if isinstance(device, str):
        device = core.device(device)

    if dtype is None:
        dtype = input.dtype
    seed, offset = default_generator._step(  # pylint: disable=protected-access
        generator_step_
    )
    return execute(
        "rand_like_ext",
        input,
        seed,
        offset,
        dtype,
        device=device,
        requires_grad=requires_grad,
    )


# randperm
def randperm(
    n,
    *,
    generator=None,
    out=None,
    dtype=core.int64,
    layout=None,
    device=None,
    requires_grad=False,
    pin_memory=False
):
    if device is None:
        device = get_device_in_context()
    if isinstance(device, str):
        device = core.device(device)

    if not generator:
        generator = default_generator
    seed, offset = generator._step(generator_step_)  # pylint: disable=protected-access
    output = execute(
        "randperm_ext",
        n,
        seed,
        offset,
        dtype,
        device=device,
        requires_grad=requires_grad,
    )
    if out is None:
        return output
    out.data = output
    return out

def gamma(shape, alpha, beta):
    return execute('gamma', shape, alpha, beta)

__all__ = [
    "bernoulli",
    "multinomial",
    "normal",
    "rand",
    "rand_like",
    "randint",
    "randn",
    "randn_like",
    "randperm",
    "randint_like",
    "gamma"
]
