from typing import Union, List, Optional
import numpy as np
from mindnlp._legacy.hypercomplex.tensor_decomposition.linal.basics.vector import Vector as AbstractVector
from mindnlp._legacy.hypercomplex.tensor_decomposition.linal.basics.real.real_scalar import Scalar

NoneType = type(None)


class Vector(AbstractVector):

    def __init__(self,
                 size: Optional[int] = None,
                 items: Union[np.ndarray, List, NoneType] = None) -> None:
        if items is None:
            size = self._init_default(size)
        elif isinstance(items, np.ndarray):
            size = self._init_ndarray(size, items)
        elif isinstance(items, List):
            size = self._init_list
        else:
            raise TypeError(f"'items' must be one of: None, ndarray or List of scalars, but got: {items}")
        super(Vector, self).__init__(size)

    def __getitem__(self, key):
        x = self._items.__getitem__(key)
        if len(x.shape) == 1:
            return Vector(x.shape[0], x)
        else:
            return Scalar(x)

    def __neg__(self) -> 'Vector':
        return Vector(self._size, np.negative(self._items))

    def add_scalar(self, that: Union[Scalar, float, np.float64]) -> 'Vector':
        num = that.as_number() if isinstance(that, Scalar) else that
        return Vector(self._size, np.add(self._items, num))

    def sub_scalar(self, that: Union[Scalar, float, np.float64]) -> 'Vector':
        num = that.as_number() if isinstance(that, Scalar) else that
        return Vector(self._size, np.subtract(self._items, num))

    def mul_scalar(self, that: Union[Scalar, float, np.float64]) -> 'Vector':
        num = that.as_number() if isinstance(that, Scalar) else that
        return Vector(self._size, np.multiply(self._items, num))

    def div_scalar(self, that: Union[Scalar, float, np.float64]) -> 'Vector':
        num = that.as_number() if isinstance(that, Scalar) else that
        return Vector(self._size, np.divide(self._items, num))

    def conjugate(self) -> 'Vector':
        return self

    def as_array(self) -> np.ndarray:
        return np.array(self._items)

    def _init_default(self, size: Optional[int]) -> int:
        if size is None:
            raise ValueError(f"Either 'items' or 'size' must not be None")
        self._items = np.zeros((size,), dtype=np.float64)
        return size

    def _init_ndarray(self, size: Optional[int], items: np.ndarray) -> int:
        if size is None:
            size = len(items)
        self._items = np.array(items).astype(np.float64)
        return size

    def _init_list(self, size: Optional[int], items: List[Scalar]) -> int:
        if size is None:
            size = len(items)
            if size <= 0:
                raise ValueError(f"'items' must be a list of some scalars, but got: {items}")
        elif len(items) != size:
            raise ValueError(f"'items' must be a list of {size} scalars, but got: {items}")
        if any(not isinstance(s, Scalar) for s in items):
            raise ValueError(f"'items' must be a list of {size} scalars, but got: {items}")
        self._items = np.array([item.as_number() for item in items])
        return size

    def _add(self, that: 'Vector') -> 'Vector':
        return Vector(self._size, np.add(self._items, that._items))

    def _sub(self, that: 'Vector') -> 'Vector':
        return Vector(self._size, np.subtract(self._items, that._items))

    def _mul(self, that: 'Vector') -> 'Vector':
        return Vector(self._size, np.multiply(self._items, that._items))

    def _dot_product(self, that: 'Vector') -> Scalar:
        return Scalar(np.dot(self._items, that._items).item())

    def _div(self, that: 'Vector') -> 'Vector':
        return Vector(self._size, np.divide(self._items, that._items))

    def _full_like(self, items: Union[np.ndarray, List]) -> 'Vector':
        return Vector(items=items)

    def _ones_like(self) -> 'Vector':
        return Vector(items=[Scalar.one()] * self._size)

    def _zeros_like(self) -> 'Vector':
        return Vector(items=[Scalar.zero()] * self._size)
