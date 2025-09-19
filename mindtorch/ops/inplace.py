import numbers
import mindtorch
from mindtorch._C import default_generator
from mindtorch.executor import execute

generator_step_ = 12

def inplace_copy(self, other):
    if self.device != other.device:
        other = other.to(self.device)
    execute('inplace_copy', self, other)
    return self

def inplace_zero(input):
    execute('inplace_zero', input)
    return input

def inplace_fill(input, value):
    if isinstance(value, (int, float, bool)):
        execute('inplace_fill_scalar', input, value)
    else:
        execute('inplace_fill_tensor', input, value)

    return input

def inplace_normal(input, mean=0, std=1, *, generator=None):
    if generator is None:
        generator = default_generator

    if isinstance(mean, mindtorch.Tensor):
        mean = mean.item()
    if isinstance(std, mindtorch.Tensor):
        std = std.item()

    execute('inplace_normal', input, mean, std, generator, device=input.device)
    return input

# uniform_
def inplace_uniform(input, *args, **kwargs):
    if len(args) == 1:
        from_ = args[0]
        to_ = None
    elif len(args) == 2:
        from_ = args[0]
        to_ = args[1]
    elif len(args) == 3:
        from_ = args[0]
        to_ = args[1]
    else:
        from_ = 0
        to_ = 1

    from_ = kwargs.get("from", 0) if from_ is None else from_
    # to_ = kwargs.get("to", 1)
    generator_ = kwargs.get("generator", None)
    if generator_ is None:
        generator_ = default_generator

    execute("inplace_uniform", input, from_, to_, generator_)

    return input

def inplace_add(input, other, alpha):
    if isinstance(other, numbers.Number):
        other = mindtorch.tensor(other, dtype=input.dtype, device=input.device)
    execute('inplace_add', input, other, alpha)
    return input


def inplace_random(self, from_=0, to=None, *, generator=None):
    if not generator:
        generator = default_generator
    execute('inplace_random', self, from_, to, generator, device=self.device)

    return self

def inplace_exponential(self, lambd, generator):
    if not generator:
        generator = default_generator
    execute('inplace_exponential', self, lambd, generator, device=self.device)
    return self

def inplace_fill_diagonal(input, value, wrap):
    execute("inplace_fill_diagonal", input, value, wrap)
    return input

__all__ = [
    'inplace_copy',
    'inplace_zero',
    'inplace_normal',
    'inplace_fill',
    'inplace_uniform',
    'inplace_add',
    'inplace_random',
    'inplace_exponential',
    'inplace_fill_diagonal'
]
