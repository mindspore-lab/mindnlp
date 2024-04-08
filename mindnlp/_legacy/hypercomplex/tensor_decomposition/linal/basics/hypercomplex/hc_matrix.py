from typing import Type, TypeVar, Tuple, List, Union, Optional
import numpy as np
from mindnlp._legacy.hypercomplex.tensor_decomposition.linal.basics.matrix import Matrix as AbstractMatrix
from mindnlp._legacy.hypercomplex.tensor_decomposition.linal.basics.hypercomplex.hc_algebra_factory import AlgebraFactory
from mindnlp._legacy.hypercomplex.tensor_decomposition.linal.basics.hypercomplex.hc_scalar import Scalar
from mindnlp._legacy.hypercomplex.tensor_decomposition.linal.basics.hypercomplex.hc_vector import Vector

TAlgebraFactory = TypeVar('TAlgebraFactory', bound=AlgebraFactory)
NoneType = type(None)


class Matrix(AbstractMatrix):

    def __init__(self,
                 algebra_factory: Type[TAlgebraFactory],
                 height: Optional[int] = None,
                 width: Optional[int] = None,
                 items: Union[Tuple, np.ndarray, List, NoneType] = None) -> None:
        self._alg_factory = algebra_factory
        self._impl = algebra_factory().get_matrix_impl()
        if items is None:
            height, width = self._init_default(height, width)
        elif isinstance(items, Tuple):
            height, width = self._init_tuple(height, width, items)
        elif isinstance(items, np.ndarray):
            height, width = self._init_ndarray(height, width, items)
        elif isinstance(items, List):
            height, width = self._init_list(height, width, items)
        else:
            raise TypeError(f"'items' must be one of: None, ndarray, list of vectors, or Tuple, but got: {items}")
        super(Matrix, self).__init__(height, width)

    def __getitem__(self, key):
        x = self._items_x.__getitem__(key)
        y = self._items_y.__getitem__(key)
        if len(x.shape) == 2:
            return Matrix(self._alg_factory, x.shape[0], x.shape[1], (x, y))
        elif len(x.shape) == 1:
            return Vector(self._alg_factory, x.shape[0], (x, y))
        else:
            return Scalar(self._alg_factory, x, y)

    def __neg__(self) -> 'Matrix':
        return Matrix(
            self._alg_factory, self._height, self._width,
            (np.negative(self._items_x), np.negative(self._items_y))
        )

    def get_algebra_type(self) -> Type[AlgebraFactory]:
        return self._alg_factory

    def add_scalar(self, that: Union[Scalar, Tuple[Union[float, np.float64], Union[float, np.float64]]]) -> 'Matrix':
        if isinstance(that, Scalar) and that.get_algebra_type() != self._alg_factory:
            raise TypeError(
                f'It is only possible to add a scalar of the same data type {self._alg_factory}, '
                f'but got: {that.get_algebra_type()}'
            )
        return Matrix(
            self._alg_factory, self._height, self._width,
            (np.add(self._items_x, that[0]), np.add(self._items_y, that[1]))
        )

    def sub_scalar(self, that: Union[Scalar, Tuple[Union[float, np.float64], Union[float, np.float64]]]) -> 'Matrix':
        if isinstance(that, Scalar) and that.get_algebra_type() != self._alg_factory:
            raise TypeError(
                f'It is only possible to subtract a scalar of the same data type {self._alg_factory}, '
                f'but got: {that.get_algebra_type()}'
            )
        return Matrix(
            self._alg_factory, self._height, self._width,
            (np.subtract(self._items_x, that[0]), np.subtract(self._items_y, that[1]))
        )

    def mul_scalar(self, that: Union[Scalar, Tuple[Union[float, np.float64], Union[float, np.float64]]]) -> 'Matrix':
        if isinstance(that, Scalar) and that.get_algebra_type() != self._alg_factory:
            raise TypeError(
                f'It is only possible to multiply by a scalar of the same data type {self._alg_factory}, '
                f'but got: {that.get_algebra_type()}'
            )
        items = self._impl.mul(self._items_x, self._items_y, that[0], that[1])
        return Matrix(self._alg_factory, self._height, self._width, items)

    def transpose(self) -> 'Matrix':
        return Matrix(
            self._alg_factory, self._width, self._height,
            (np.transpose(self._items_x), np.transpose(self._items_y))
        )

    def transpose_conjugate(self) -> 'Matrix':
        return Matrix(
            self._alg_factory, self._width, self._height,
            (np.transpose(self._items_x), np.negative(np.transpose(self._items_y)))
        )

    def div_scalar(self, that: Union[Scalar, Tuple[Union[float, np.float64], Union[float, np.float64]]]) -> 'Matrix':
        if isinstance(that, Scalar) and that.get_algebra_type() != self._alg_factory:
            raise TypeError(
                f'It is only possible to divide over a scalar of the same data type {self._alg_factory}, '
                f'but got: {that.get_algebra_type()}'
            )
        items = self._impl.div(self._items_x, self._items_y, that[0], that[1])
        return Matrix(self._alg_factory, self._height, self._width, items)

    def as_array(self) -> np.ndarray:
        return np.reshape(
            np.array(list(zip(np.ravel(self._items_x), np.ravel(self._items_y)))),
            (self._height, self._width, 2)
        )

    def _init_default(self,
                      height: Optional[int],
                      width: Optional[int]) -> Tuple[int, int]:
        if height is None or width is None:
            raise ValueError(f"Either 'items' or both 'height' or 'width' must not be None")
        self._items_x = np.zeros((height, width), dtype=np.float64)
        self._items_y = np.zeros((height, width), dtype=np.float64)
        return height, width

    def _init_tuple(self,
                    height: Optional[int],
                    width: Optional[int],
                    items: Tuple[np.ndarray, np.ndarray]) -> Tuple[int, int]:
        if len(items) != 2:
            raise ValueError(f"'items' must be a 2-element tuple, but got: {items}")
        self._items_x = items[0]
        self._items_y = items[1]
        if not isinstance(self._items_x, np.ndarray) or len(self._items_x.shape) != 2:
            raise ValueError(
                f"elements of 'items' must be 2d arrays, but got: {items}"
            )
        if not isinstance(self._items_y, np.ndarray) or len(self._items_y.shape) != 2:
            raise ValueError(
                f"elements of 'items' must be 2d arrays, but got: {items}"
            )
        if height is None:
            height = self._items_x.shape[0]
        if width is None:
            width = self._items_x.shape[1]
        if self._items_x.shape[0] != height or self._items_x.shape[1] != width:
            raise ValueError(
                f"elements of 'items' must be 2d arrays of dimensions {height}x{width}, but got: {items}"
            )
        if self._items_y.shape[0] != height or self._items_y.shape[1] != width:
            raise ValueError(
                f"elements of 'items' must be 2d arrays of dimensions {height}x{width}, but got: {items}"
            )
        return height, width

    def _init_ndarray(self,
                      height: Optional[int],
                      width: Optional[int],
                      items: np.ndarray) -> Tuple[int, int]:
        if height is None:
            height = items.shape[0]
        if width is None:
            width = items.shape[1]
        all_items = np.ravel(items)
        self._items_x = np.reshape(all_items[::2], (height, width)).astype(np.float64)
        self._items_y = np.reshape(all_items[1::2], (height, width)).astype(np.float64)
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
        if any(
            not isinstance(v, Vector) or v.get_algebra_type() != self._alg_factory or v.get_size() != width
            for v in items
        ):
            raise ValueError(f"'items' must be a list of {height} vectors, each of size {width}, but got: {items}")
        all_items = np.ravel(np.concatenate([v.as_array() for v in items], axis=0))
        self._items_x = np.reshape(all_items[::2], (height, width)).astype(np.float64)
        self._items_y = np.reshape(all_items[1::2], (height, width)).astype(np.float64)
        return height, width

    def _add(self, that: 'Matrix') -> 'Matrix':
        if self._alg_factory != that._alg_factory:
            raise TypeError(
                f'It is only possible to add a matrix of the same data type {self._alg_factory}, '
                f'but got: {that._alg_factory}'
            )
        return Matrix(
            self._alg_factory, self._height, self._width,
            (np.add(self._items_x, that._items_x), np.add(self._items_y, that._items_y))
        )

    def _sub(self, that: 'Matrix') -> 'Matrix':
        if self._alg_factory != that._alg_factory:
            raise TypeError(
                f'It is only possible to subtract a matrix of the same data type {self._alg_factory}, '
                f'but got: {that._alg_factory}'
            )
        return Matrix(
            self._alg_factory, self._height, self._width,
            (np.subtract(self._items_x, that._items_x), np.subtract(self._items_y, that._items_y))
        )

    def _mul(self, that: 'Matrix') -> 'Matrix':
        if self._alg_factory != that._alg_factory:
            raise TypeError(
                f'It is only possible to multiply by a matrix of the same data type {self._alg_factory}, '
                f'but got: {that._alg_factory}'
            )
        items = self._impl.mul(self._items_x, self._items_y, that._items_x, that._items_y)
        return Matrix(self._alg_factory, self._height, self._width, items)

    def _matmul(self, that: 'Matrix') -> 'Matrix':
        if self._alg_factory != that._alg_factory:
            raise TypeError(
                f'It is only possible to multiply by a matrix of the same data type {self._alg_factory}, '
                f'but got: {that._alg_factory}'
            )
        items = self._impl.matmul(self._items_x, self._items_y, that._items_x, that._items_y)
        return Matrix(self._alg_factory, self._height, that._width, items)

    def _div(self, that: 'Matrix') -> 'Matrix':
        if self._alg_factory != that._alg_factory:
            raise TypeError(
                f'It is only possible to divide over a matrix of the same data type {self._alg_factory}, '
                f'but got: {that._alg_factory}'
            )
        items = self._impl.div(self._items_x, self._items_y, that._items_x, that._items_y)
        return Matrix(self._alg_factory, self._height, self._width, items)

    def _full_like(self, items: Union[Tuple, np.ndarray, List]) -> 'Matrix':
        return Matrix(self._alg_factory, items=items)

    def _ones_like(self) -> 'Matrix':
        return Matrix(
            self._alg_factory,
            items=[Vector(self._alg_factory, items=[Scalar.one(self._alg_factory)] * self._width)] * self._height
        )

    def _zeros_like(self) -> 'Matrix':
        return Matrix(
            self._alg_factory,
            items=[Vector(self._alg_factory, items=[Scalar.zero(self._alg_factory)] * self._width)] * self._height
        )

    def _identity_like(self) -> 'Matrix':
        items = [
            Vector(
                self._alg_factory,
                items=[Scalar.zero(self._alg_factory)] * i
                + [Scalar.one(self._alg_factory)]
                + [Scalar.zero(self._alg_factory)] * (self._width - i - 1)
            )
            for i in range(self._height)
        ]
        return Matrix(self._alg_factory, items=items)

    def _concat_rows(self, that: 'Matrix') -> 'Matrix':
        if self._alg_factory != that._alg_factory:
            raise TypeError(
                f'It is only possible to concatenate a matrix of the same data type {self._alg_factory}, '
                f'but got: {that._alg_factory}'
            )
        items_x = np.concatenate((self._items_x, that._items_x), axis=0)
        items_y = np.concatenate((self._items_y, that._items_y), axis=0)
        return Matrix(
            self._alg_factory,
            height=self._height + that._height,
            width=self._width,
            items=(items_x, items_y)
        )

    def _concat_cols(self, that: 'Matrix') -> 'Matrix':
        if self._alg_factory != that._alg_factory:
            raise TypeError(
                f'It is only possible to concatenate a matrix of the same data type {self._alg_factory}, '
                f'but got: {that._alg_factory}'
            )
        items_x = np.concatenate((self._items_x, that._items_x), axis=1)
        items_y = np.concatenate((self._items_y, that._items_y), axis=1)
        return Matrix(
            self._alg_factory,
            height=self._height,
            width=self._width + that._width,
            items=(items_x, items_y)
        )
