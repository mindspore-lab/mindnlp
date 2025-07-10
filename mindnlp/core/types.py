from builtins import (  # noqa: F401
    bool as _bool,
    bytes as _bytes,
    complex as _complex,
    float as _float,
    int as _int,
    str as _str,
)
from typing import Any, IO, TYPE_CHECKING, Union, Dict
from typing_extensions import Self, TypeAlias

from ._dtype import dtype

class device():
    def __init__(self, type=None, index=None):
        if type is not None:
            if isinstance(type, str):
                if ':' in type:
                    if index is not None:
                        raise ValueError("`type` must not include an index because index was "
                                         f"passed explicitly: {type}")
                    _target, _id = type.split(':')
                    _id = int(_id)
                else:
                    _target = type
                    _id = None if _target == 'cpu' else 0
            elif isinstance(type, device):
                if index is not None:
                    raise ValueError("core.device(): When input is core.device, `index` can not be set.")
                _target = type.type
                _id = type.index
            else:
                raise TypeError("core.device(): `type` must be type of 'str' or 'core.device'.")
        else:
            raise ValueError("core.device(): `type` can not be None")

        self.type = _target
        self.index = _id

    def __repr__(self):
        if self.index is None:
            return f"device(type={self.type})"
        return f"device(type={self.type}, index={self.index})"

    def __eq__(self, __value):
        if not isinstance(__value, device):
            return False
        return hash(self) == hash(__value)

    def __hash__(self):
        return hash(self.type) ^ hash(self.index)

    def __enter__(self):
        # self.prev_idx = torch.cuda._exchange_device(self.idx)
        pass

    def __exit__(self, type: Any, value: Any, traceback: Any):
        # self.idx = torch.cuda._maybe_exchange_device(self.prev_idx)
        return False


# Meta-type for "numeric" things; matches our docs
Number: TypeAlias = Union[int, float, bool]
# tuple for isinstance(x, Number) checks.
# FIXME: refactor once python 3.9 support is dropped.
_Number = (int, float, bool)

# Storage protocol implemented by ${Type}StorageBase classes
class Storage:
    _cdata: int
    device: device
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

_device = device
_dtype = dtype
_size = tuple