
from mindspore.common.generator import default_generator


generator_step_ = 12

def inplace_copy(self, other):
    if self.device != other.device:
        other = other.to(self.device)
    if self.device.type == 'cpu':
        # execute('assign', self, other)
        # # self._data.assign_value_cpp(other._data)
        self.data = other
    else:
        execute('inplace_copy', self, other)
    return self

def inplace_zero(input):
    device = input.device
    if input.device == 'npu':
        execute('inplace_zero', input)
    elif input.device.type == 'cpu':
        out = execute('zeros', input.shape, input.dtype, device=device)
        input.data = out
    return input

def inplace_fill(input, value):
    device = input.device
    if input.device == 'npu':
        if isinstance(value, (int, float, bool)):
            execute('inplace_fill_scalar', input, value)
        execute('inplace_fill_tensor', input, value)
    elif input.device.type == 'cpu':
        out = execute('full', input.shape, value, device=device)
        input.data = out
    return input

def inplace_normal(input, mean=0, std=1, *, generator=None):
    if generator is None:
        generator = default_generator
    seed, offset = generator._step(generator_step_)
    if input.device.type == 'npu':
        execute('inplace_normal', input, mean, std, seed, offset)
    elif input.device.type == 'cpu':
        core.normal(mean, std, size=input.size, generator=generator, out=input)

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
    seed, offset = generator_._step(generator_step_)
    if input.device.type == 'npu':
        execute("inplace_uniform", input, from_, to_, seed, offset)
    elif input.device.type == 'cpu':
        input.data = core.rand(input.shape, generator=generator_, dtype=input.dtype) * (to_ - from_) + from_
    return input

def inplace_add(input, other, alpha):
    execute('inplace_add_ext', input, other, alpha)
    return input

def inplace_scatter(input, dim, index, src):
    if not isinstance(src, core.Tensor):
        return execute('inplace_scatter_value', input, dim, index, src)
    return execute('inplace_scatter', input, dim, index, src)


__all__ = [
    'inplace_copy',
    'inplace_zero',
    'inplace_normal',
    'inplace_fill',
    'inplace_uniform',
    'inplace_add',
    'inplace_scatter'
]
