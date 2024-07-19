"""nn init"""
# pylint: disable=unused-import
import numbers
import numpy as np
import mindspore
from mindspore.common.initializer import (
    Initializer,
    Normal,
    HeNormal,
    Uniform,
    HeUniform,
    XavierNormal,
    XavierUniform,
    Constant,
    One,
    Zero,
    _INITIALIZER_ALIAS,
    _calculate_fan_in_and_fan_out
)
from mindspore import Tensor

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
    if not isinstance(init, (numbers.Number, str, Initializer)):
        raise TypeError("For 'initializer', the type of the 'init' argument should be 'Tensor', 'number', 'string' "
                        "or 'initializer', but got {}.".format(type(init)))

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

    data = np.ndarray(shape, dtype=mindspore.dtype_to_nptype(dtype))
    init(data)
    tensor = Tensor(data)
    return tensor
