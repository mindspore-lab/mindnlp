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

import mindtorch

ShapeType: TypeAlias = Union[mindtorch.Size, list[int], tuple[int, ...]]
DeviceLikeType: TypeAlias = Union[str, mindtorch.device, int]
