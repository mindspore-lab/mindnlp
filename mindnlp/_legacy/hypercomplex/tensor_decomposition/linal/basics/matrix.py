from abc import ABC, abstractmethod
from typing import Any, Union
import numpy as np
from mindnlp._legacy.hypercomplex.tensor_decomposition.linal.basics.scalar import Scalar


class Matrix(ABC):

    def __init__(self, height: int, width: int) -> None:
        if height <= 0 or width <= 0:
            raise ValueError(f"'height' and 'width' must be positive integers, but got {height} and {width}")
        self._height = height
        self._width = width

    def __matmul__(self, that: 'Matrix') -> 'Matrix':
        return self.matmul(that)

    def __mul__(self, that: Union['Matrix', Scalar, Any]) -> 'Matrix':
        if isinstance(that, Matrix):
            return self.mul(that)
        else:
            return self.mul_scalar(that)

    def __add__(self, that: Union['Matrix', Scalar, Any]) -> 'Matrix':
        if isinstance(that, Matrix):
            return self.add(that)
        else:
            return self.add_scalar(that)

    def __sub__(self, that: Union['Matrix', Scalar, Any]) -> 'Matrix':
        if isinstance(that, Matrix):
            return self.sub(that)
        else:
            return self.sub_scalar(that)

    def __truediv__(self, that: Union['Matrix', Scalar, Any]) -> 'Matrix':
        if isinstance(that, Matrix):
            return self.div(that)
        else:
            return self.div_scalar(that)

    def __str__(self) -> str:
        ret = f'{self._height}x{self._width} '
        if self._height <= 4:
            items = [self[i] for i in range(self._height)]
            ret += '[' + ', '.join([str(item) for item in items]) + ']'
        else:
            items_top = [self[i] for i in range(2)]
            items_bottom = [self[-i - 1] for i in range(2)]
            ret += '[' + ', '.join([str(item) for item in items_top]) + ', ... ,' \
                + ', '.join([str(item) for item in items_bottom]) + ']'
        return ret

    @staticmethod
    def full_like(m: 'Matrix', items: Any) -> 'Matrix':
        return m._full_like(items)

    @staticmethod
    def ones_like(m: 'Matrix') -> 'Matrix':
        return m._ones_like()

    @staticmethod
    def zeros_like(m: 'Matrix') -> 'Matrix':
        return m._zeros_like()

    @staticmethod
    def identity_like(m: 'Matrix') -> 'Matrix':
        if m._width != m._height:
            raise ValueError(
                'It is only possible to make an identity out of a matrix of the same dimensions, '
                f'but got: {m._height}x{m._width}'
            )
        return m._identity_like()

    @abstractmethod
    def __getitem__(self, key):
        pass

    @abstractmethod
    def __neg__(self) -> 'Matrix':
        pass

    def get_height(self) -> int:
        return self._height

    def get_width(self) -> int:
        return self._width

    def add(self, that: 'Matrix') -> 'Matrix':
        if self._height != that._height or self._width != that._width:
            raise ValueError(
                f'It is only possible to add a matrix of the same dimensions {self._height}x{self._width},'
                f' but got: {that._height}x{that._width}'
            )
        return self._add(that)

    @abstractmethod
    def add_scalar(self, that: Union[Scalar, Any]) -> 'Matrix':
        pass

    def sub(self, that: 'Matrix') -> 'Matrix':
        if self._height != that._height or self._width != that._width:
            raise ValueError(
                f'It is only possible to subtract a matrix of the same dimensions {self._height}x{self._width},'
                f' but got: {that._height}x{that._width}'
            )
        return self._sub(that)

    @abstractmethod
    def sub_scalar(self, that: Union[Scalar, Any]) -> 'Matrix':
        pass

    def mul(self, that: 'Matrix') -> 'Matrix':
        if self._height != that._height or self._width != that._width:
            raise ValueError(
                f'It is only possible to multiply by a matrix of the same dimensions {self._height}x{self._width},'
                f' but got: {that._height}x{that._width}'
            )
        return self._mul(that)

    @abstractmethod
    def mul_scalar(self, that: Union[Scalar, Any]) -> 'Matrix':
        pass

    def div(self, that: 'Matrix') -> 'Matrix':
        if self._height != that._height or self._width != that._width:
            raise ValueError(
                f'It is only possible to divide over a matrix of the same dimensions {self._height}x{self._width},'
                f' but got: {that._height}x{that._width}'
            )
        return self._div(that)

    @abstractmethod
    def div_scalar(self, that: Union[Scalar, Any]) -> 'Matrix':
        pass

    def matmul(self, that: 'Matrix') -> 'Matrix':
        if self._width != that._height:
            raise ValueError(
                f'It is only possible to multiply a matrix of height {self._width}, but got: {that._height}'
            )
        return self._matmul(that)

    def concat_rows(self, that: 'Matrix') -> 'Matrix':
        if self._width != that._width:
            raise ValueError(
                f'It is only possible to concat a matrix of width {self._width}, but got: {that._width}'
            )
        return self._concat_rows(that)

    def concat_cols(self, that: 'Matrix') -> 'Matrix':
        if self._height != that._height:
            raise ValueError(
                f'It is only possible to concat a matrix of height {self._height}, but got: {that._height}'
            )
        return self._concat_cols(that)

    @abstractmethod
    def transpose(self) -> 'Matrix':
        pass

    @abstractmethod
    def transpose_conjugate(self) -> 'Matrix':
        pass

    @abstractmethod
    def _full_like(self, items: Any) -> 'Matrix':
        pass

    @abstractmethod
    def _ones_like(self) -> 'Matrix':
        pass

    @abstractmethod
    def _zeros_like(self) -> 'Matrix':
        pass

    @abstractmethod
    def _identity_like(self) -> 'Matrix':
        pass

    @abstractmethod
    def as_array(self) -> np.ndarray:
        pass

    @abstractmethod
    def _add(self, that: 'Matrix') -> 'Matrix':
        pass

    @abstractmethod
    def _sub(self, that: 'Matrix') -> 'Matrix':
        pass

    @abstractmethod
    def _mul(self, that: 'Matrix') -> 'Matrix':
        pass

    @abstractmethod
    def _div(self, that: 'Matrix') -> 'Matrix':
        pass

    @abstractmethod
    def _matmul(self, that: 'Matrix') -> 'Matrix':
        pass

    @abstractmethod
    def _concat_rows(self, that: 'Matrix') -> 'Matrix':
        pass

    @abstractmethod
    def _concat_cols(self, that: 'Matrix') -> 'Matrix':
        pass

