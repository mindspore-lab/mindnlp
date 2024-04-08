from typing import Type, TypeVar, Tuple, Union, Optional, List
import numpy as np
from mindnlp._legacy.hypercomplex.tensor_decomposition.linal.basics.vector import Vector as AbstractVector
from mindnlp._legacy.hypercomplex.tensor_decomposition.linal.basics.hypercomplex.hc_algebra_factory import AlgebraFactory
from mindnlp._legacy.hypercomplex.tensor_decomposition.linal.basics.hypercomplex.hc_scalar import Scalar

TAlgebraFactory = TypeVar('TAlgebraFactory', bound=AlgebraFactory)
NoneType = type(None)


class Vector(AbstractVector):

    def __init__(self,
                 algebra_factory: Type[TAlgebraFactory],
                 size: Optional[int] = None,
                 items: Union[Tuple, np.ndarray, List, NoneType] = None) -> None:
        self._alg_factory = algebra_factory
        self._impl = algebra_factory().get_vector_impl()
        if items is None:
            size = self._init_default(size)
        elif isinstance(items, Tuple):
            size = self._init_tuple(size, items)
        elif isinstance(items, np.ndarray):
            size = self._init_ndarray(size, items)
        elif isinstance(items, List):
            size = self._init_list(size, items)
        else:
            raise TypeError(f"'items' must be one of: None, ndarray, list of scalars or Tuple, but got: {items}")
        super(Vector, self).__init__(size)

    def __getitem__(self, key):
        x = self._items_x.__getitem__(key)
        y = self._items_y.__getitem__(key)
        if len(x.shape) == 1:
            return Vector(self._alg_factory, x.shape[0], (x, y))
        else:
            return Scalar(self._alg_factory, x, y)

    def __neg__(self) -> 'Vector':
        return Vector(
            self._alg_factory, self._size,
            (np.negative(self._items_x), np.negative(self._items_y))
        )

    def get_algebra_type(self) -> Type[AlgebraFactory]:
        return self._alg_factory

    def add_scalar(self, that: Union[Scalar, Tuple[np.float64, np.float64]]) -> 'Vector':
        if isinstance(that, Scalar) and that.get_algebra_type() != self._alg_factory:
            raise TypeError(
                f'It is only possible to add a scalar of the same data type {self._alg_factory}, '
                f'but got: {that.get_algebra_type()}'
            )
        return Vector(
            self._alg_factory, self._size,
            (np.add(self._items_x, that[0]), np.add(self._items_y, that[1]))
        )

    def sub_scalar(self, that: Union[Scalar, Tuple[np.float64, np.float64]]) -> 'Vector':
        if isinstance(that, Scalar) and that.get_algebra_type() != self._alg_factory:
            raise TypeError(
                f'It is only possible to subtract a scalar of the same data type {self._alg_factory}, '
                f'but got: {that.get_algebra_type()}'
            )
        return Vector(
            self._alg_factory, self._size,
            (np.subtract(self._items_x, that[0]), np.subtract(self._items_y, that[1]))
        )

    def mul_scalar(self, that: Union[Scalar, Tuple[np.float64, np.float64]]) -> 'Vector':
        if isinstance(that, Scalar) and that.get_algebra_type() != self._alg_factory:
            raise TypeError(
                f'It is only possible to multiply by a scalar of the same data type {self._alg_factory}, '
                f'but got: {that.get_algebra_type()}'
            )
        items = self._impl.mul(self._items_x, self._items_y, that[0], that[1])
        return Vector(self._alg_factory, self._size, items)

    def div_scalar(self, that: Union[Scalar, Tuple[np.float64, np.float64]]) -> 'Vector':
        if isinstance(that, Scalar) and that.get_algebra_type() != self._alg_factory:
            raise TypeError(
                f'It is only possible to divide over a scalar of the same data type {self._alg_factory}, '
                f'but got: {that.get_algebra_type()}'
            )
        items = self._impl.div(self._items_x, self._items_y, that[0], that[1])
        return Vector(self._alg_factory, self._size, items)

    def conjugate(self) -> 'Vector':
        return Vector(self._alg_factory, self._size, (self._items_x, np.negative(self._items_y)))

    def as_array(self) -> np.ndarray:
        return np.array(list(zip(self._items_x, self._items_y)))

    def _init_default(self, size: Optional[int]) -> int:
        if size is None:
            raise ValueError(f"Either 'items' or 'size' must not be None")
        self._items_x = np.zeros((size,), dtype=np.float64)
        self._items_y = np.zeros((size,), dtype=np.float64)
        return size

    def _init_tuple(self, size: Optional[int], items: Tuple[np.ndarray, np.ndarray]) -> int:
        if len(items) != 2:
            raise ValueError(f"'items' must be a 2-element tuple, but got: {items}")
        self._items_x = items[0]
        self._items_y = items[1]
        if not isinstance(self._items_x, np.ndarray) or len(self._items_x.shape) != 1:
            raise ValueError(
                f"elements of 'items' must be 1d arrays, but got: {items}"
            )
        if not isinstance(self._items_y, np.ndarray) or len(self._items_y.shape) != 1:
            raise ValueError(
                f"elements of 'items' must be 1d arrays, but got: {items}"
            )
        if size is None:
            size = len(self._items_x)
        if len(self._items_x) != size or len(self._items_y) != size:
            raise ValueError(
                f"elements of 'items' must be 1d arrays of size {size}, but got: {items}"
            )
        return size

    def _init_ndarray(self, size: Optional[int], items: np.ndarray) -> int:
        if size is None:
            size = len(items)
        super(Vector, self).__init__(size)
        all_items = np.ravel(items)
        self._items_x = np.reshape(all_items[::2], (size,)).astype(np.float64)
        self._items_y = np.reshape(all_items[1::2], (size,)).astype(np.float64)
        return size

    def _init_list(self, size: Optional[int], items: List[Scalar]) -> int:
        if size is None:
            size = len(items)
            if size <= 0:
                raise ValueError(f"'items' must be a list of some scalars, but got: {items}")
        elif len(items) != size:
            raise ValueError(f"'items' must be a list of {size} scalars, but got: {items}")
        if any(
                not isinstance(s, Scalar) or s.get_algebra_type() != self._alg_factory for s in items
        ):
            raise ValueError(f"'items' must be a list of {size} scalars, but got: {items}")
        all_items = np.ravel(np.concatenate([s.as_array() for s in items], axis=0))
        self._items_x = np.reshape(all_items[::2], (size,)).astype(np.float64)
        self._items_y = np.reshape(all_items[1::2], (size,)).astype(np.float64)
        return size

    def _add(self, that: 'Vector') -> 'Vector':
        if self._alg_factory != that._alg_factory:
            raise TypeError(
                f'It is only possible to add a vector of the same data type {self._alg_factory}, '
                f'but got: {that._alg_factory}'
            )
        return Vector(
            self._alg_factory, self._size,
            (np.add(self._items_x, that._items_x), np.add(self._items_y, that._items_y))
        )

    def _sub(self, that: 'Vector') -> 'Vector':
        if self._alg_factory != that._alg_factory:
            raise TypeError(
                f'It is only possible to subtract a vector of the same data type {self._alg_factory}, '
                f'but got: {that._alg_factory}'
            )
        return Vector(
            self._alg_factory, self._size,
            (np.subtract(self._items_x, that._items_x), np.subtract(self._items_y, that._items_y))
        )

    def _mul(self, that: 'Vector') -> 'Vector':
        if self._alg_factory != that._alg_factory:
            raise TypeError(
                f'It is only possible to multiply by a vector of the same data type {self._alg_factory}, '
                f'but got: {that._alg_factory}'
            )
        items = self._impl.mul(self._items_x, self._items_y, that._items_x, that._items_y)
        return Vector(self._alg_factory, self._size, items)

    def _dot_product(self, that: 'Vector') -> Scalar:
        if self._alg_factory != that._alg_factory:
            raise TypeError(
                'It is only possible to calculate dot product with a vector'
                f' of the same data type {self._alg_factory}, but got: {that._alg_factory}'
            )
        x, y = self._impl.dot_product(self._items_x, self._items_y, that._items_x, that._items_y)
        return Scalar(self._alg_factory, x, y)

    def _div(self, that: 'Vector') -> 'Vector':
        if self._alg_factory != that._alg_factory:
            raise TypeError(
                f'It is only possible to divide over a vector of the same data type {self._alg_factory}, '
                f'but got: {that._alg_factory}'
            )
        items = self._impl.div(self._items_x, self._items_y, that._items_x, that._items_y)
        return Vector(self._alg_factory, self._size, items)

    def _full_like(self, items: Union[Tuple, np.ndarray, List]) -> 'Vector':
        return Vector(self._alg_factory, items=items)

    def _ones_like(self) -> 'Vector':
        return Vector(
            self._alg_factory, items=[Scalar.one(self._alg_factory)] * self._size
        )

    def _zeros_like(self) -> 'Vector':
        return Vector(
            self._alg_factory, items=[Scalar.zero(self._alg_factory)] * self._size
        )
