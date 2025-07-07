import functools

from typing import Any, Callable, Optional, TypeVar
from typing_extensions import ParamSpec

R = TypeVar("R", covariant=True)  # return type (always covariant)
P = ParamSpec("P")

def _script_if_tracing(fn: Callable[P, R]) -> Callable[P, R]:
    @functools.wraps(fn)
    def wrapper(*args: P.args, **kwargs: P.kwargs) -> R:
        if not is_tracing():
            # Not tracing, don't do anything
            return fn(*args, **kwargs)

        compiled_fn: Callable[P, R] = script(wrapper.__original_fn)  # type: ignore[attr-defined]
        return compiled_fn(*args, **kwargs)

    wrapper.__original_fn = fn  # type: ignore[attr-defined]
    wrapper.__script_if_tracing_wrapper = True  # type: ignore[attr-defined]

    return wrapper
