from functools import cached_property
from typing import (
    Any,
    Callable,
    ClassVar,
    final,
    Generic,
    Optional,
    TYPE_CHECKING,
    Union,
)
from typing_extensions import Concatenate, ParamSpec, TypeVar


_T = TypeVar("_T", default=Any)
_P = ParamSpec("_P", default=...)

class OpOverload:
    def __init__(
        self,
        op: Callable[_P, _T],
    ) -> None:
        super().__init__()
        self._op = op
        self._opname = op.__name__
        self._overloadname = "default"

    # it's a no-op since OpOverload object is immutable and must be unique for a given op overload.
    def __deepcopy__(self, memo=None):
        return self

    def __repr__(self):
        return f"<OpOverload(op='{self._opname}', overload='{self._overloadname}')>"

    # Use positional-only argument to avoid naming collision with aten ops arguments
    # that are named "self". This way, all the aten ops can be called by kwargs.
    def __call__(self, /, *args: _P.args, **kwargs: _P.kwargs) -> _T:
        return self._op(*args, **kwargs)

    def __hash__(self):
        return hash(self._op)

    # `my_namespace.my_op_name.overload_name`
    def __str__(self):
        return "{}.{}.{}".format(*self._schema.name.split("::"), self._overloadname)

    @property
    def op(self):
        return self._op

