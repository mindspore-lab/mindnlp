from typing import List, Union, Optional, Tuple
import numpy as np
from mindnlp._legacy.hypercomplex.tensor_decomposition.linal.basics.matrix import Matrix as AbstractMatrix
from mindnlp._legacy.hypercomplex.tensor_decomposition.linal.basics.real.real_scalar import Scalar
from mindnlp._legacy.hypercomplex.tensor_decomposition.linal.basics.real.real_vector import Vector

NoneType = type(None)


class Matrix(AbstractMatrix):

    def __init__(self,
                 height: Optional[int] = None,
                 width: Optional[int] = None,
                 items: Union[np.ndarray, List, NoneType] = None) -> None:
        if items is None:
            height, width = self._init_default(height, width)
        elif isinstance(items, np.ndarray):
            height, width = self._init_ndarray(height, width, items)
        elif isinstance(items, List):
            height, width = self._init_list(height, width, items)
        else:
            raise TypeError(f"'items' must be one of: None, ndarray, or list of vectors, but got: {items}")
        super(Matrix, self).__init__(height, width)

    def __getitem__(self, key):
        x = self._items.__getitem__(key)
        if len(x.shape) == 2:
            return Matrix(x.shape[0], x.shape[1], x)
        elif len(x.shape) == 1:
            return Vector(x.shape[0], x)
        else:
            return Scalar(x)

    def __neg__(self) -> 'Matrix':
        return Matrix(self._height, self._width, np.negative(self._items))

    def add_scalar(self, that: Union[Scalar, float, np.float64]) -> 'Matrix':
        num = that.as_number() if isinstance(that, Scalar) else that
        return Matrix(self._height, self._width, np.add(self._items, num))

    def sub_scalar(self, that: Union[Scalar, float, np.float64]) -> 'Matrix':
        num = that.as_number() if isinstance(that, Scalar) else that
        return Matrix(self._height, self._width, np.subtract(self._items, num))

    def mul_scalar(self, that: Union[Scalar, float, np.float64]) -> 'Matrix':
        num = that.as_number() if isinstance(that, Scalar) else that
        return Matrix(self._height, self._width, np.multiply(self._items, num))

    def transpose(self) -> 'Matrix':
        return Matrix(self._width, self._height, np.transpose(self._items))

    def transpose_conjugate(self) -> 'Matrix':
        return self.transpose()

    def div_scalar(self, that: Union[Scalar, float, np.float64]) -> 'Matrix':
        num = that.as_number() if isinstance(that, Scalar) else that
        return Matrix(self._height, self._width, np.divide(self._items, num))

    def as_array(self) -> np.ndarray:
        return np.array(self._items)

    def _init_default(self,
                      height: Optional[int],
                      width: Optional[int]) -> Tuple[int, int]:
        if height is None or width is None:
            raise ValueError(f"Either 'items' or both 'height' or 'width' must not be None")
        self._items = np.zeros((height, width), dtype=np.float64)
        return height, width

    def _init_ndarray(self,
                      height: Optional[int],
                      width: Optional[int],
                      items: np.ndarray) -> Tuple[int, int]:
        if height is None:
            height = items.shape[0]
        if width is None:
            width = items.shape[1]
        self._items = np.array(items).astype(np.float64)
        return height, width

    def _init_list(self,
                   height: Optional[int],
                   width: Optional[int],
                   items: List[Vector]) -> Tuple[int, int]:
        if height is None:
            height = len(items)
            if height <= 0:
                raise ValueError(f"'items' must be a list of some vectors, but got: {items}")
        elif len(items) != height:
            raise ValueError(f"'items' must be a list of {height} vectors, but got: {items}")
        if width is None:
            if not isinstance(items[0], Vector):
                raise ValueError(f"'items' must be a list of {height} vectors, but got: {items}")
            width = items[0].get_size()
        if len(items) != height or any(not isinstance(v, Vector) or v.get_size() != width for v in items):
            raise ValueError(f"'items' must be a list of {height} vectors, each of size {width}, but got: {items}")
        self._items = np.reshape(
            np.concatenate([v.as_array() for v in items], axis=0),
            (height, width)
        ).astype(np.float64)
        return height, width

    def _add(self, that: 'Matrix') -> 'Matrix':
        return Matrix(self._height, self._width, np.add(self._items, that._items))

    def _sub(self, that: 'Matrix') -> 'Matrix':
        return Matrix(self._height, self._width, np.subtract(self._items, that._items))

    def _mul(self, that: 'Matrix') -> 'Matrix':
        return Matrix(self._height, self._width, np.multiply(self._items, that._items))

    def _matmul(self, that: 'Matrix') -> 'Matrix':
        return Matrix(self._height, that._width, np.matmul(self._items, that._items))

    def _div(self, that: 'Matrix') -> 'Matrix':
        return Matrix(self._height, self._width, np.divide(self._items, that._items))

    def _full_like(self, items: Union[np.ndarray, List]) -> 'Matrix':
        return Matrix(items=items)

    def _ones_like(self) -> 'Matrix':
        return Matrix(items=[Vector(items=[Scalar.one()] * self._width)] * self._height)

    def _zeros_like(self) -> 'Matrix':
        return Matrix(items=[Vector(items=[Scalar.zero()] * self._width)] * self._height)

    def _identity_like(self) -> 'Matrix':
        items = [
            Vector(items=[Scalar.zero()] * i + [Scalar.one()] + [Scalar.zero()] * (self._width - i - 1))
            for i in range(self._height)
        ]
        return Matrix(items=items)

    def _concat_rows(self, that: 'Matrix') -> 'Matrix':
        items = np.concatenate((self._items, that._items), axis=0)
        return Matrix(height=self._height + that._height, width=self._width, items=items)

    def _concat_cols(self, that: 'Matrix') -> 'Matrix':
        items = np.concatenate((self._items, that._items), axis=1)
        return Matrix(height=self._height, width=self._width + that._width, items=items)
