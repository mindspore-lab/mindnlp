from typing import Any, Callable, Literal, Optional, overload, Union
from collections.abc import Iterable, Sequence

from mindtorch import _C

device_types_t = Optional[Union[str, Sequence[str]]]

def register_fake(*args, **kwargs):
    def register(func):
        return func
    return register

def custom_op(
    name: str,
    fn: Optional[Callable] = None,
    /,
    *,
    mutates_args: Union[str, Iterable[str]],
    device_types: device_types_t = None,
    schema: Optional[str] = None,
    tags: Optional[Sequence[_C.Tag]] = None,
) -> Union[Callable[[Callable[..., object]], "CustomOpDef"], "CustomOpDef"]:
    """Wraps a function into custom operator.

    Reasons why you may want to create a custom op include:
    - Wrapping a third-party library or custom kernel to work with PyTorch
    subsystems like Autograd.
    - Preventing torch.compile/export/FX tracing from peeking inside your function.

    This API is used as a decorator around a function (please see examples).
    The provided function must have type hints; these are needed to interface
    with PyTorch's various subsystems.

    Args:
        name (str): A name for the custom op that looks like "{namespace}::{name}",
            e.g. "mylib::my_linear". The name is used as the op's stable identifier
            in PyTorch subsystems (e.g. torch.export, FX graphs).
            To avoid name collisions, please use your project name as the namespace;
            e.g. all custom ops in pytorch/fbgemm use "fbgemm" as the namespace.
        mutates_args (Iterable[str] or "unknown"): The names of args that the function mutates.
            This MUST be accurate, otherwise, the behavior is undefined. If "unknown",
            it pessimistically assumes that all inputs to the operator are being mutated.
        device_types (None | str | Sequence[str]): The device type(s) the function
            is valid for. If no device type is provided, then the function
            is used as the default implementation for all device types.
            Examples: "cpu", "cuda".
            When registering a device-specific implementation for an operator that accepts no Tensors,
            we require the operator to have a "device: torch.device argument".
        schema (None | str): A schema string for the operator. If None
            (recommended) we'll infer a schema for the operator from its type
            annotations. We recommend letting us infer a schema unless you
            have a specific reason not to.
            Example: "(Tensor x, int y) -> (Tensor, Tensor)".

    .. note::
        We recommend not passing in a ``schema`` arg and instead letting us infer
        it from the type annotations. It is error-prone to write your own schema.
        You may wish to provide your own schema if our interpretation of
        the type annotation is not what you want.
        For more info on how to write a schema string, see
        `here <https://github.com/pytorch/pytorch/blob/main/aten/src/ATen/native/README.md#func>`_

    Examples::
        >>> import torch
        >>> from torch import Tensor
        >>> from torch.library import custom_op
        >>> import numpy as np
        >>>
        >>> @custom_op("mylib::numpy_sin", mutates_args=())
        >>> def numpy_sin(x: Tensor) -> Tensor:
        >>>     x_np = x.cpu().numpy()
        >>>     y_np = np.sin(x_np)
        >>>     return torch.from_numpy(y_np).to(device=x.device)
        >>>
        >>> x = torch.randn(3)
        >>> y = numpy_sin(x)
        >>> assert torch.allclose(y, x.sin())
        >>>
        >>> # Example of a custom op that only works for one device type.
        >>> @custom_op("mylib::numpy_sin_cpu", mutates_args=(), device_types="cpu")
        >>> def numpy_sin_cpu(x: Tensor) -> Tensor:
        >>>     x_np = x.numpy()
        >>>     y_np = np.sin(x_np)
        >>>     return torch.from_numpy(y_np)
        >>>
        >>> x = torch.randn(3)
        >>> y = numpy_sin_cpu(x)
        >>> assert torch.allclose(y, x.sin())
        >>>
        >>> # Example of a custom op that mutates an input
        >>> @custom_op("mylib::numpy_sin_inplace", mutates_args={"x"}, device_types="cpu")
        >>> def numpy_sin_inplace(x: Tensor) -> None:
        >>>     x_np = x.numpy()
        >>>     np.sin(x_np, out=x_np)
        >>>
        >>> x = torch.randn(3)
        >>> expected = x.sin()
        >>> numpy_sin_inplace(x)
        >>> assert torch.allclose(x, expected)
        >>>
        >>> # Example of a factory function
        >>> @torch.library.custom_op("mylib::bar", mutates_args={}, device_types="cpu")
        >>> def bar(device: torch.device) -> Tensor:
        >>>     return torch.ones(3)
        >>>
        >>> bar("cpu")

    """

    def inner(fn: Callable[..., object]):
        import torch

        if schema is None:
            # schema_str = torch.library.infer_schema(fn, mutates_args=mutates_args)
            schema_str = None
        else:
            schema_str = schema

        namespace, opname = name.split("::")
        # result = CustomOpDef(namespace, opname, schema_str, fn, tags)
        # if schema is not None:
        #     # Check that schema's alias annotations match those of `mutates_args`.
        #     expected = set()
        #     for arg in result._opoverload._schema.arguments:
        #         if arg.alias_info is not None and arg.alias_info.is_write:
        #             expected.add(arg.name)
        #     if expected != set(mutates_args):
        #         raise ValueError(
        #             f"Attempted to create a custom op with `mutates_args={mutates_args}` "
        #             f"and `schema={schema}. The schema suggests that the op mutates {expected}"
        #             f"which is different from what was provided to us in `mutates_args`. "
        #             f"Please make these consistent."
        #         )
        # result.register_kernel(device_types)(fn)
        # return result
        return None

    if fn is None:
        return inner
    return inner(fn)
