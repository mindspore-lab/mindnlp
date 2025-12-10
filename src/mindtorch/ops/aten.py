from mindtorch.executor import execute
from .._ops import OpOverload

def new_empty(input, size, dtype, device):
    """
    Create a tensor with the same type as the given tensor.

    Args:
        *args (tuple): The arguments for creating a tensor.
        size (tuple): The size of the tensor.
        dtype (dtype): The data type of the tensor.
        device (device): The device of the tensor.
        requires_grad (bool): Whether the tensor requires gradients.

    Returns:
        Tensor: The created tensor.
    """
    execute('new_empty', input, size, dtype, device=device)

new_empty.default = OpOverload(new_empty)

def new_full():
    pass

new_full.default = OpOverload(new_full)

def new_ones(): 
    pass

new_ones.default = OpOverload(new_ones)

def new_zeros():
    pass

new_zeros.default = OpOverload(new_zeros)

def new_empty_strided():
    pass

new_empty_strided.default = OpOverload(new_empty_strided)

def expand():
    pass

expand.default = OpOverload(expand)

def reshape():
    pass

reshape.default = OpOverload(reshape)

def view():
    pass

view.default = OpOverload(view)

def _unsafe_view():
    pass

_unsafe_view.default = OpOverload(_unsafe_view)

def native_dropout():
    pass

native_dropout.default = OpOverload(native_dropout)

def normal_():
    pass

normal_.default = OpOverload(normal_)

def rand_like():
    pass

rand_like.default = OpOverload(rand_like)

def randn_like():
    pass

randn_like.default = OpOverload(randn_like)

def randint_like():
    pass

randint_like.default = OpOverload(randint_like)
randint_like.low_dtype = OpOverload(randint_like)
randint_like.low_dtype_out = OpOverload(randint_like)

def uniform_():
    pass

uniform_.default = OpOverload(uniform_)

def bernoulli_():
    pass

bernoulli_.float = OpOverload(bernoulli_)

def bernoulli():
    pass

bernoulli.default = OpOverload(bernoulli)

def linear():
    pass

linear.default = OpOverload(linear)

def matmul():
    pass

matmul.default = OpOverload(matmul)

def is_same_size():
    pass

is_same_size.default = OpOverload(is_same_size)

def convolution():
    pass

convolution.default = OpOverload(convolution)

def convolution_backward():
    pass

convolution_backward.default = OpOverload(convolution_backward)

def _amp_foreach_non_finite_check_and_unscale_():
    pass

_amp_foreach_non_finite_check_and_unscale_.default = OpOverload(_amp_foreach_non_finite_check_and_unscale_)

def embedding():
    pass

embedding.default = OpOverload(embedding)

