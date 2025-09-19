import inspect
import functools

from typing import Any, Callable, Literal, NoReturn, TypeVar as _TypeVar
from typing_extensions import Concatenate as _Concatenate, ParamSpec as _ParamSpec

_T = _TypeVar("_T")
_U = _TypeVar("_U")
_P = _ParamSpec("_P")

_ValueDescriptor = Literal[
    "v",
    "i",
    "is",
    "f",
    "fs",
    "b",
    "s",
    "t",
    "none",
]

def parse_args(
    *arg_descriptors: _ValueDescriptor,
) -> Callable[[Callable[_Concatenate[_U, _P], _T]], Callable[_Concatenate[_U, _P], _T]]:
    """A decorator which converts args from torch._C.Value to built-in types.

    For example:

    ```
    @parse_args('v', 'i', 'fs')
    foo(g, a, b, c):
        assert isinstance(a, torch._C.Value)
        assert isinstance(b, int)
        assert isinstance(c, list)
        assert isinstance(c[0], float)
    ```

    Args:
        arg_descriptors: list of str, where each element is
            a string that specifies the type to convert to. Valid descriptors:
            "v": no conversion, keep torch._C.Value.
            "i": int
            "is": list of int
            "f": float
            "fs": list of float
            "b": bool
            "s": str
            "t": torch.Tensor
            "none": the variable is unused
    """

    def decorator(
        fn: Callable[_Concatenate[_U, _P], _T],
    ) -> Callable[_Concatenate[_U, _P], _T]:
        fn._arg_descriptors = arg_descriptors  # type: ignore[attr-defined]

        @functools.wraps(fn)
        def wrapper(g: _U, *args: _P.args, **kwargs: _P.kwargs) -> _T:
            # some args may be optional, so the length may be smaller
            FILE_BUG_MSG = (
                "If you believe this is not due to custom symbolic implementation within your code or "
                "an external library, please file an issue at "
                "https://github.com/pytorch/pytorch/issues/new?template=bug-report.yml to report this bug."
            )
            assert len(arg_descriptors) >= len(args), (
                f"A mismatch between the number of arguments ({len(args)}) and "
                f"their descriptors ({len(arg_descriptors)}) was found at symbolic function '{fn.__name__}'. "
                f"{FILE_BUG_MSG}"
            )

            try:
                sig = inspect.signature(fn)
                arg_names = list(sig.parameters.keys())[1:]
                fn_name = fn.__name__
            except Exception:
                # FIXME(justinchuby): Avoid catching Exception.
                # Catch a more specific exception instead.
                arg_names = [None] * len(args)  # type: ignore[list-item]
                fn_name = None
            args = [
                _parse_arg(arg, arg_desc, arg_name, fn_name)  # type: ignore[method-assign]
                for arg, arg_desc, arg_name in zip(args, arg_descriptors, arg_names)
            ]
            # only support _outputs in kwargs
            assert len(kwargs) <= 1, (
                f"Symbolic function {fn.__name__}'s '**kwargs' can contain a single "
                f"key/value entry. "
                f"{FILE_BUG_MSG}"
            )

            if len(kwargs) == 1:
                assert "_outputs" in kwargs, (
                    f"Symbolic function {fn.__name__}'s '**kwargs' can only contain "
                    f"'_outputs' key at '**kwargs'. "
                    f"{FILE_BUG_MSG}"
                )
            return fn(g, *args, **kwargs)

        return wrapper

    return decorator
