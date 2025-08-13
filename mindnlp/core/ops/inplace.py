from mindnlp import core
from mindnlp.core._C import default_generator
from mindnlp.core.executor import execute

generator_step_ = 12

def inplace_copy(self, other):
    if self.device != other.device:
        other = other.to(self.device)
    execute('inplace_copy', self, other)
    return self

def inplace_zero(input):
    if input.device.type == 'npu':
        execute('inplace_zero', input)
    elif input.device.type == 'meta':
        pass
    else:
        input.data = core.zeros_like(input)
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
    seed, offset = generator._step(generator_step_)

    if isinstance(mean, core.Tensor):
        mean = mean.item()
    if isinstance(std, core.Tensor):
        std = std.item()

    execute('inplace_normal', input, mean, std, seed, offset, device=input.device)

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
    execute('inplace_add_ext', input, other, alpha)
    return input

def inplace_random(self, from_=0, to=None, *, generator=None):
    if not generator:
        generator = default_generator
    seed, offset = generator._step(  # pylint: disable=protected-access
        generator_step_)
    execute('inplace_random', self, from_, to, seed, offset, device=self.device)
    return self


__all__ = [
    'inplace_copy',
    'inplace_zero',
    'inplace_normal',
    'inplace_fill',
    'inplace_uniform',
    'inplace_add',
    'inplace_random'
]
