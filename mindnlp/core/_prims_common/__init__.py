from typing import (
    Any,
    Callable,
    cast,
    NamedTuple,
    Optional,
    overload,
    TYPE_CHECKING,
    TypeVar,
    Union,
)

from typing_extensions import deprecated, TypeAlias

from mindnlp import core

ShapeType: TypeAlias = Union[core.Size, list[int], tuple[int, ...]]
DeviceLikeType: TypeAlias = Union[str, core.device, int]
