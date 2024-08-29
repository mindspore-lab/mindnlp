# mypy: allow-untyped-defs
"""This file contains utilities for initializing neural network parameters."""
import math
import numbers
import warnings
import numpy as np
import mindspore
from mindspore import Tensor
from mindspore.common.initializer import Initializer, _INITIALIZER_ALIAS, _init_random_uniform, _assignment, \
    Normal, Constant, TruncatedNormal, Dirac, Orthogonal, Sparse


def initializer(init, shape=None, dtype=mindspore.float32):
    """
    Create and initialize a tensor.

    Args:
        init (Union[Tensor, str, Initializer, numbers.Number]): Initialize value.

            - `str`: The `init` should be the alias of the class inheriting from `Initializer` and the corresponding
              class will be called in practice. The value of `init` can be ``"normal"``, ``"ones"`` or
              ``"zeros"``, etc.

            - `Initializer`: The `init` should be the class inheriting from `Initializer` to initialize tensor.

            - `numbers.Number`: The `Constant` will be called to initialize tensor.

            - `Tensor`: The tensor will be called to initialize tensor.

        shape (Union[tuple, list, int]): The shape of the initialized tensor. Default: ``None`` .
        dtype (:class:`mindspore.dtype`): The type of data in initialized tensor. Default: ``mstype.float32`` .

    Returns:
        Tensor, return is Tensor object.

    Raises:
        TypeError: The type of the argument 'init' is not correct.
        ValueError: The shape of the tensor which is passed through 'init' is not the same as that passed by 'shape'.


    Examples:
        >>> import numpy as np
        >>> import mindspore
        >>> from mindspore import Tensor
        >>> from mindspore.common.initializer import initializer, One
        >>> from mindspore import Parameter
        >>> data = Tensor(np.zeros([1, 2, 3]), mindspore.float32)
        >>> w1 = Parameter(initializer(data, [1, 2, 3], mindspore.float32))
        >>> w2 = Parameter(initializer('ones', [1, 2, 3], mindspore.float32))
        >>> w3 = Parameter(initializer(One(), [1, 2, 3], mindspore.float32))
        >>> w4 = Parameter(initializer(0, [1, 2, 3], mindspore.float32))
    """
    if not isinstance(init, (Tensor, numbers.Number, str, Initializer)):
        raise TypeError("For 'initializer', the type of the 'init' argument should be 'Tensor', 'number', 'string' "
                        "or 'initializer', but got {}.".format(type(init)))

    if isinstance(init, Tensor):
        init_shape = init.shape
        shape = shape if isinstance(shape, (tuple, list)) else [shape]
        if shape is not None and init_shape != tuple(shape):
            raise ValueError("For 'initializer', the shape of the 'init' argument should be same as "
                             "the argument 'shape', but got the "
                             "'init' shape {} and the 'shape' {}.".format(list(init.shape), shape))
        return init

    if isinstance(shape, list):
        shape = tuple(shape)
    elif isinstance(shape, numbers.Number):
        shape = (shape,)

    for value in shape if shape is not None else ():
        if not isinstance(value, int) or value <= 0:
            raise ValueError(f"For 'initializer', the argument 'shape' is invalid, the value of 'shape' "
                             f"must be positive integer, "
                             f"but got {shape}")

    if isinstance(init, str):
        class_name = _INITIALIZER_ALIAS.get(init.lower())
        if class_name is None:
            raise ValueError(f"For 'initializer', the class corresponding to '{init}' was not found.")
        init = class_name()
    elif isinstance(init, numbers.Number):
        init = Constant(init)
    shape = shape if shape is not None else init.shape
    data = np.ndarray(shape, dtype=mindspore.dtype_to_nptype(dtype))
    init(data)
    tensor = Tensor(data)
    return tensor


class Uniform(Initializer):
    r"""
    Generates an array with values sampled from Uniform distribution :math:`{U}(-\text{scale}, \text{scale})` in order
    to initialize a tensor.

    Args:
        scale (float): The bound of the Uniform distribution. Default: ``0.07`` .


    Examples:
        >>> import mindspore
        >>> from mindspore.common.initializer import initializer, Uniform
        >>> from mindspore import Parameter
        >>> w1 = Parameter(initializer(Uniform(), [1, 2, 3], mindspore.float32))
        >>> w2 = Parameter(initializer('uniform', [1, 2, 3], mindspore.float32))
    """

    def __init__(self, a=0.0, b=1.0):
        super(Uniform, self).__init__()
        self.a = a
        self.b = b

    def _initialize(self, arr):
        tmp = _init_random_uniform(self.a, self.b, arr.shape)
        _assignment(arr, tmp)



def calculate_gain(nonlinearity, param=None):
    r"""Return the recommended gain value for the given nonlinearity function.

    The values are as follows:

    ================= ====================================================
    nonlinearity      gain
    ================= ====================================================
    Linear / Identity :math:`1`
    Conv{1,2,3}D      :math:`1`
    Sigmoid           :math:`1`
    Tanh              :math:`\frac{5}{3}`
    ReLU              :math:`\sqrt{2}`
    Leaky Relu        :math:`\sqrt{\frac{2}{1 + \text{negative\_slope}^2}}`
    SELU              :math:`\frac{3}{4}`
    ================= ====================================================

    .. warning::
        In order to implement `Self-Normalizing Neural Networks`_ ,
        you should use ``nonlinearity='linear'`` instead of ``nonlinearity='selu'``.
        This gives the initial weights a variance of ``1 / N``,
        which is necessary to induce a stable fixed point in the forward pass.
        In contrast, the default gain for ``SELU`` sacrifices the normalization
        effect for more stable gradient flow in rectangular layers.

    Args:
        nonlinearity: the non-linear function (`nn.functional` name)
        param: optional parameter for the non-linear function

    Examples:
        >>> gain = nn.init.calculate_gain('leaky_relu', 0.2)  # leaky_relu with negative_slope=0.2

    .. _Self-Normalizing Neural Networks: https://papers.nips.cc/paper/2017/hash/5d44ee6f2c3f71b73125876103c8f6c4-Abstract.html
    """
    linear_fns = [
        "linear",
        "conv1d",
        "conv2d",
        "conv3d",
        "conv_transpose1d",
        "conv_transpose2d",
        "conv_transpose3d",
    ]
    if nonlinearity in linear_fns or nonlinearity == "sigmoid":
        return 1
    elif nonlinearity == "tanh":
        return 5.0 / 3
    elif nonlinearity == "relu":
        return math.sqrt(2.0)
    elif nonlinearity == "leaky_relu":
        if param is None:
            negative_slope = 0.01
        elif (
            not isinstance(param, bool)
            and isinstance(param, int)
            or isinstance(param, float)
        ):
            # True/False are instances of int, hence check above
            negative_slope = param
        else:
            raise ValueError(f"negative_slope {param} not a valid number")
        return math.sqrt(2.0 / (1 + negative_slope**2))
    elif nonlinearity == "selu":
        return (
            3.0 / 4
        )  # Value found empirically (https://github.com/pytorch/pytorch/pull/50664)
    else:
        raise ValueError(f"Unsupported nonlinearity {nonlinearity}")


def uniform_(
    tensor: Tensor,
    a: float = 0.0,
    b: float = 1.0,
) -> Tensor:
    r"""Fill the input Tensor with values drawn from the uniform distribution.

    :math:`\mathcal{U}(a, b)`.

    Args:
        tensor: an n-dimensional `mindspore.Tensor`
        a: the lower bound of the uniform distribution
        b: the upper bound of the uniform distribution

    Examples:
        >>> w = torch.empty(3, 5)
        >>> nn.init.uniform_(w)
    """
    tensor.data_sync(True)
    tensor.assign_value(initializer(Uniform(a, b), tensor.shape, tensor.dtype))
    return tensor

def normal_(
    tensor: Tensor,
    mean: float = 0.0,
    std: float = 1.0,
) -> Tensor:
    r"""Fill the input Tensor with values drawn from the normal distribution.

    :math:`\mathcal{N}(\text{mean}, \text{std}^2)`.

    Args:
        tensor: an n-dimensional `mindspore.Tensor`
        mean: the mean of the normal distribution
        std: the standard deviation of the normal distribution

    Examples:
        >>> w = torch.empty(3, 5)
        >>> nn.init.normal_(w)
    """
    tensor.data_sync(True)
    tensor.assign_value(initializer(Normal(std, mean), tensor.shape, tensor.dtype))
    return tensor


def trunc_normal_(
    tensor: Tensor,
    mean: float = 0.0,
    std: float = 1.0,
    a: float = -2.0,
    b: float = 2.0,
) -> Tensor:
    r"""Fill the input Tensor with values drawn from a truncated normal distribution.

    The values are effectively drawn from the
    normal distribution :math:`\mathcal{N}(\text{mean}, \text{std}^2)`
    with values outside :math:`[a, b]` redrawn until they are within
    the bounds. The method used for generating the random values works
    best when :math:`a \leq \text{mean} \leq b`.

    Args:
        tensor: an n-dimensional `mindspore.Tensor`
        mean: the mean of the normal distribution
        std: the standard deviation of the normal distribution
        a: the minimum cutoff value
        b: the maximum cutoff value

    Examples:
        >>> w = torch.empty(3, 5)
        >>> nn.init.trunc_normal_(w)
    """
    tensor.data_sync(True)
    tensor.assign_value(initializer(TruncatedNormal(std, mean, a, b), tensor.shape, tensor.dtype))
    return tensor


def constant_(tensor: Tensor, val: float) -> Tensor:
    r"""Fill the input Tensor with the value :math:`\text{val}`.

    Args:
        tensor: an n-dimensional `mindspore.Tensor`
        val: the value to fill the tensor with

    Examples:
        >>> w = torch.empty(3, 5)
        >>> nn.init.constant_(w, 0.3)
    """
    tensor.data_sync(True)
    tensor.assign_value(initializer(Constant(val), tensor.shape, tensor.dtype))
    return tensor


def ones_(tensor: Tensor) -> Tensor:
    r"""Fill the input Tensor with the scalar value `1`.

    Args:
        tensor: an n-dimensional `mindspore.Tensor`

    Examples:
        >>> w = torch.empty(3, 5)
        >>> nn.init.ones_(w)
    """
    tensor.data_sync(True)
    tensor.assign_value(initializer('ones', tensor.shape, tensor.dtype))
    return tensor


def zeros_(tensor: Tensor) -> Tensor:
    r"""Fill the input Tensor with the scalar value `0`.

    Args:
        tensor: an n-dimensional `mindspore.Tensor`

    Examples:
        >>> w = torch.empty(3, 5)
        >>> nn.init.zeros_(w)
    """
    tensor.data_sync(True)
    tensor.assign_value(initializer('zeros', tensor.shape, tensor.dtype))
    return tensor


def dirac_(tensor, groups=1):
    r"""Fill the {3, 4, 5}-dimensional input `Tensor` with the Dirac delta function.

    Preserves the identity of the inputs in `Convolutional`
    layers, where as many input channels are preserved as possible. In case
    of groups>1, each group of channels preserves identity

    Args:
        tensor: a {3, 4, 5}-dimensional `mindspore.Tensor`
        groups (int, optional): number of groups in the conv layer (default: 1)
    Examples:
        >>> w = torch.empty(3, 16, 5, 5)
        >>> nn.init.dirac_(w)
        >>> w = torch.empty(3, 24, 5, 5)
        >>> nn.init.dirac_(w, 3)
    """
    tensor.data_sync(True)
    tensor.assign_value(initializer(Dirac(groups), tensor.shape, tensor.dtype))
    return tensor


def _calculate_fan_in_and_fan_out(tensor):
    dimensions = tensor.ndim
    if dimensions < 2:
        raise ValueError(
            "Fan in and fan out can not be computed for tensor with fewer than 2 dimensions"
        )

    num_input_fmaps = tensor.shape[1]
    num_output_fmaps = tensor.shape[0]
    receptive_field_size = 1
    if tensor.ndim > 2:
        # math.prod is not always available, accumulate the product manually
        # we could use functools.reduce but that is not supported by TorchScript
        for s in tensor.shape[2:]:
            receptive_field_size *= s
    fan_in = num_input_fmaps * receptive_field_size
    fan_out = num_output_fmaps * receptive_field_size

    return fan_in, fan_out


def xavier_uniform_(
    tensor: Tensor,
    gain: float = 1.0,
) -> Tensor:
    r"""Fill the input `Tensor` with values using a Xavier uniform distribution.

    The method is described in `Understanding the difficulty of training
    deep feedforward neural networks` - Glorot, X. & Bengio, Y. (2010).
    The resulting tensor will have values sampled from
    :math:`\mathcal{U}(-a, a)` where

    .. math::
        a = \text{gain} \times \sqrt{\frac{6}{\text{fan\_in} + \text{fan\_out}}}

    Also known as Glorot initialization.

    Args:
        tensor: an n-dimensional `mindspore.Tensor`
        gain: an optional scaling factor
        generator: the torch Generator to sample from (default: None)

    Examples:
        >>> w = torch.empty(3, 5)
        >>> nn.init.xavier_uniform_(w, gain=nn.init.calculate_gain('relu'))

    Note:
        Be aware that ``fan_in`` and ``fan_out`` are calculated assuming
        that the weight matrix is used in a transposed manner,
        (i.e., ``x @ w.T`` in ``Linear`` layers, where ``w.shape = [fan_out, fan_in]``).
        This is important for correct initialization.
        If you plan to use ``x @ w``, where ``w.shape = [fan_in, fan_out]``,
        pass in a transposed weight matrix, i.e. ``nn.init.xavier_uniform_(w.T, ...)``.
    """
    fan_in, fan_out = _calculate_fan_in_and_fan_out(tensor)
    std = gain * math.sqrt(2.0 / float(fan_in + fan_out))
    a = math.sqrt(3.0) * std  # Calculate uniform bounds from standard deviation

    return uniform_(tensor, -a, a)


def xavier_normal_(
    tensor: Tensor,
    gain: float = 1.0,
) -> Tensor:
    r"""Fill the input `Tensor` with values using a Xavier normal distribution.

    The method is described in `Understanding the difficulty of training deep feedforward
    neural networks` - Glorot, X. & Bengio, Y. (2010). The resulting tensor
    will have values sampled from :math:`\mathcal{N}(0, \text{std}^2)` where

    .. math::
        \text{std} = \text{gain} \times \sqrt{\frac{2}{\text{fan\_in} + \text{fan\_out}}}

    Also known as Glorot initialization.

    Args:
        tensor: an n-dimensional `mindspore.Tensor`
        gain: an optional scaling factor
        generator: the torch Generator to sample from (default: None)

    Examples:
        >>> w = torch.empty(3, 5)
        >>> nn.init.xavier_normal_(w)

    Note:
        Be aware that ``fan_in`` and ``fan_out`` are calculated assuming
        that the weight matrix is used in a transposed manner,
        (i.e., ``x @ w.T`` in ``Linear`` layers, where ``w.shape = [fan_out, fan_in]``).
        This is important for correct initialization.
        If you plan to use ``x @ w``, where ``w.shape = [fan_in, fan_out]``,
        pass in a transposed weight matrix, i.e. ``nn.init.xavier_normal_(w.T, ...)``.
    """
    fan_in, fan_out = _calculate_fan_in_and_fan_out(tensor)
    std = gain * math.sqrt(2.0 / float(fan_in + fan_out))

    return normal_(tensor, 0.0, std)


def _calculate_correct_fan(tensor, mode):
    mode = mode.lower()
    valid_modes = ["fan_in", "fan_out"]
    if mode not in valid_modes:
        raise ValueError(f"Mode {mode} not supported, please use one of {valid_modes}")

    fan_in, fan_out = _calculate_fan_in_and_fan_out(tensor)
    return fan_in if mode == "fan_in" else fan_out


def kaiming_uniform_(
    tensor: Tensor,
    a: float = 0,
    mode: str = "fan_in",
    nonlinearity: str = "leaky_relu",
):
    r"""Fill the input `Tensor` with values using a Kaiming uniform distribution.

    The method is described in `Delving deep into rectifiers: Surpassing
    human-level performance on ImageNet classification` - He, K. et al. (2015).
    The resulting tensor will have values sampled from
    :math:`\mathcal{U}(-\text{bound}, \text{bound})` where

    .. math::
        \text{bound} = \text{gain} \times \sqrt{\frac{3}{\text{fan\_mode}}}

    Also known as He initialization.

    Args:
        tensor: an n-dimensional `mindspore.Tensor`
        a: the negative slope of the rectifier used after this layer (only
            used with ``'leaky_relu'``)
        mode: either ``'fan_in'`` (default) or ``'fan_out'``. Choosing ``'fan_in'``
            preserves the magnitude of the variance of the weights in the
            forward pass. Choosing ``'fan_out'`` preserves the magnitudes in the
            backwards pass.
        nonlinearity: the non-linear function (`nn.functional` name),
            recommended to use only with ``'relu'`` or ``'leaky_relu'`` (default).
        generator: the torch Generator to sample from (default: None)

    Examples:
        >>> w = torch.empty(3, 5)
        >>> nn.init.kaiming_uniform_(w, mode='fan_in', nonlinearity='relu')

    Note:
        Be aware that ``fan_in`` and ``fan_out`` are calculated assuming
        that the weight matrix is used in a transposed manner,
        (i.e., ``x @ w.T`` in ``Linear`` layers, where ``w.shape = [fan_out, fan_in]``).
        This is important for correct initialization.
        If you plan to use ``x @ w``, where ``w.shape = [fan_in, fan_out]``,
        pass in a transposed weight matrix, i.e. ``nn.init.kaiming_uniform_(w.T, ...)``.
    """

    if 0 in tensor.shape:
        warnings.warn("Initializing zero-element tensors is a no-op")
        return tensor
    fan = _calculate_correct_fan(tensor, mode)
    gain = calculate_gain(nonlinearity, a)
    std = gain / math.sqrt(fan)
    bound = math.sqrt(3.0) * std  # Calculate uniform bounds from standard deviation
    return uniform_(tensor, -bound, bound)


def kaiming_normal_(
    tensor: Tensor,
    a: float = 0,
    mode: str = "fan_in",
    nonlinearity: str = "leaky_relu",
):
    r"""Fill the input `Tensor` with values using a Kaiming normal distribution.

    The method is described in `Delving deep into rectifiers: Surpassing
    human-level performance on ImageNet classification` - He, K. et al. (2015).
    The resulting tensor will have values sampled from
    :math:`\mathcal{N}(0, \text{std}^2)` where

    .. math::
        \text{std} = \frac{\text{gain}}{\sqrt{\text{fan\_mode}}}

    Also known as He initialization.

    Args:
        tensor: an n-dimensional `mindspore.Tensor`
        a: the negative slope of the rectifier used after this layer (only
            used with ``'leaky_relu'``)
        mode: either ``'fan_in'`` (default) or ``'fan_out'``. Choosing ``'fan_in'``
            preserves the magnitude of the variance of the weights in the
            forward pass. Choosing ``'fan_out'`` preserves the magnitudes in the
            backwards pass.
        nonlinearity: the non-linear function (`nn.functional` name),
            recommended to use only with ``'relu'`` or ``'leaky_relu'`` (default).
        generator: the torch Generator to sample from (default: None)

    Examples:
        >>> w = torch.empty(3, 5)
        >>> nn.init.kaiming_normal_(w, mode='fan_out', nonlinearity='relu')

    Note:
        Be aware that ``fan_in`` and ``fan_out`` are calculated assuming
        that the weight matrix is used in a transposed manner,
        (i.e., ``x @ w.T`` in ``Linear`` layers, where ``w.shape = [fan_out, fan_in]``).
        This is important for correct initialization.
        If you plan to use ``x @ w``, where ``w.shape = [fan_in, fan_out]``,
        pass in a transposed weight matrix, i.e. ``nn.init.kaiming_normal_(w.T, ...)``.
    """
    if 0 in tensor.shape:
        warnings.warn("Initializing zero-element tensors is a no-op")
        return tensor
    fan = _calculate_correct_fan(tensor, mode)
    gain = calculate_gain(nonlinearity, a)
    std = gain / math.sqrt(fan)
    return normal_(tensor, 0, std)


def orthogonal_(
    tensor,
    gain=1,
):
    r"""Fill the input `Tensor` with a (semi) orthogonal matrix.

    Described in `Exact solutions to the nonlinear dynamics of learning in deep
    linear neural networks` - Saxe, A. et al. (2013). The input tensor must have
    at least 2 dimensions, and for tensors with more than 2 dimensions the
    trailing dimensions are flattened.

    Args:
        tensor: an n-dimensional `mindspore.Tensor`, where :math:`n \geq 2`
        gain: optional scaling factor

    Examples:
        >>> # xdoctest: +REQUIRES(env:TORCH_DOCTEST_LAPACK)
        >>> w = torch.empty(3, 5)
        >>> nn.init.orthogonal_(w)
    """
    tensor.data_sync(True)
    tensor.assign_value(initializer(Orthogonal(gain), tensor.shape, tensor.dtype))
    return tensor


def sparse_(
    tensor,
    sparsity,
    std=0.01,
):
    r"""Fill the 2D input `Tensor` as a sparse matrix.

    The non-zero elements will be drawn from the normal distribution
    :math:`\mathcal{N}(0, 0.01)`, as described in `Deep learning via
    Hessian-free optimization` - Martens, J. (2010).

    Args:
        tensor: an n-dimensional `mindspore.Tensor`
        sparsity: The fraction of elements in each column to be set to zero
        std: the standard deviation of the normal distribution used to generate
            the non-zero values
        generator: the torch Generator to sample from (default: None)

    Examples:
        >>> w = torch.empty(3, 5)
        >>> nn.init.sparse_(w, sparsity=0.1)
    """
    tensor.data_sync(True)
    tensor.assign_value(initializer(Sparse(sparsity, std), tensor.shape, tensor.dtype))
    return tensor


uniform = uniform_
normal = normal_
constant = constant_
dirac = dirac_
xavier_uniform = xavier_uniform_
xavier_normal = xavier_normal_
kaiming_uniform = kaiming_uniform_
kaiming_normal = kaiming_normal_
orthogonal = orthogonal_
sparse = sparse_
