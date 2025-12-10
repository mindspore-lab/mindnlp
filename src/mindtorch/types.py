from builtins import (  # noqa: F401
    bool as _bool,
    bytes as _bytes,
    complex as _complex,
    float as _float,
    int as _int,
    str as _str,
)
import mindspore
from typing import Any, IO, TYPE_CHECKING, Union, Dict, Sequence
from typing_extensions import Self, TypeAlias

import mindtorch
from mindtorch import Tensor
from ._dtype import dtype
from ._C import device as _device

_TensorOrTensors: TypeAlias = Union[Tensor, Sequence[Tensor]]  # noqa: PYI047

Device: TypeAlias = Union[_device, str, int, None]

# Meta-type for "numeric" things; matches our docs
Number: TypeAlias = Union[int, float, bool]
# tuple for isinstance(x, Number) checks.
# FIXME: refactor once python 3.9 support is dropped.
_Number = (int, float, bool)

# Storage protocol implemented by ${Type}StorageBase classes
class Storage:
    _cdata: int
    device: _device
    dtype: dtype
    _torch_load_uninitialized: bool

    def __deepcopy__(self, memo: Dict[int, Any]) -> "Storage":
        raise NotImplementedError

    def _new_shared(self, size: int) -> "Storage":
        raise NotImplementedError

    def _write_file(
        self,
        f: Any,
        is_real_file: bool,
        save_size: bool,
        element_size: int,
    ) -> None:
        raise NotImplementedError

    def element_size(self) -> int:
        raise NotImplementedError

    def is_shared(self) -> bool:
        raise NotImplementedError

    def share_memory_(self) -> "Storage":
        raise NotImplementedError

    def nbytes(self) -> int:
        raise NotImplementedError

    def cpu(self) -> "Storage":
        raise NotImplementedError

    def data_ptr(self) -> int:
        raise NotImplementedError

    def from_file(
        self,
        filename: str,
        shared: bool = False,
        nbytes: int = 0,
    ) -> "Storage":
        raise NotImplementedError

    def _new_with_file(
        self,
        f: Any,
        element_size: int,
    ) -> "Storage":
        raise NotImplementedError

_dtype = dtype
_size = tuple