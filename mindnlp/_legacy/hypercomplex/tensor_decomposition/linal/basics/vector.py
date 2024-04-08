from abc import ABC, abstractmethod
from typing import Any, Union
import numpy as np
from mindnlp._legacy.hypercomplex.tensor_decomposition.linal.basics.scalar import Scalar


class Vector(ABC):

    def __init__(self, size: int) -> None:
        if size <= 0:
            raise ValueError(f"'size' must be a positive integer, but got {size}")
        self._size = size

    def __matmul__(self, that: 'Vector') -> Scalar:
        return self.dot_product(that)

    def __mul__(self, that: Union['Vector', Scalar, Any]) -> 'Vector':
        if isinstance(that, Vector):
            return self.mul(that)
        else:
            return self.mul_scalar(that)

    def __add__(self, that: Union['Vector', Scalar, Any]) -> 'Vector':
        if isinstance(that, Vector):
            return self.add(that)
        else:
            return self.add_scalar(that)

    def __sub__(self, that: Union['Vector', Scalar, Any]) -> 'Vector':
        if isinstance(that, Vector):
            return self.sub(that)
        else:
            return self.sub_scalar(that)

    def __truediv__(self, that: Union['Vector', Scalar, Any]) -> 'Vector':
        if isinstance(that, Vector):
            return self.div(that)
        else:
            return self.div_scalar(that)

    def __str__(self) -> str:
        if self._size <= 6:
            items = [self[i] for i in range(self._size)]
            return '[' + ', '.join([str(item) for item in items]) + ']'
        else:
            items_left = [self[i] for i in range(3)]
            items_right = [self[-i - 1] for i in range(3)]
            return '[' + ', '.join([str(item) for item in items_left]) + ', ... ,' \
                + ', '.join([str(item) for item in items_right]) + ']'

    @staticmethod
    def full_like(m: 'Vector', items: Any) -> 'Vector':
        return m._full_like(items)

    @staticmethod
    def ones_like(m: 'Vector') -> 'Vector':
        return m._ones_like()

    @staticmethod
    def zeros_like(m: 'Vector') -> 'Vector':
        return m._zeros_like()

    @abstractmethod
    def __getitem__(self, key):
        pass

    @abstractmethod
    def __neg__(self) -> 'Vector':
        pass

    def get_size(self) -> int:
        return self._size

    def add(self, that: 'Vector') -> 'Vector':
        if self._size != that._size:
            raise ValueError(
                f'It is only possible to add a vector of the same size {self._size}, but got: {that._size}'
            )
        return self._add(that)

    @abstractmethod
    def add_scalar(self, that: Union[Scalar, Any]) -> 'Vector':
        pass

    def sub(self, that: 'Vector') -> 'Vector':
        if self._size != that._size:
            raise ValueError(
                f'It is only possible to subtract a vector of the same size {self._size}, but got: {that._size}'
            )
        return self._sub(that)

    @abstractmethod
    def sub_scalar(self, that: Union[Scalar, Any]) -> 'Vector':
        pass

    def mul(self, that: 'Vector') -> 'Vector':
        if self._size != that._size:
            raise ValueError(
                f'It is only possible to multiply by a vector of the same size {self._size},'
                f' but got: {that._size}'
            )
        return self._mul(that)

    @abstractmethod
    def mul_scalar(self, that: Union[Scalar, Any]) -> 'Vector':
        pass

    def div(self, that: 'Vector') -> 'Vector':
        if self._size != that._size:
            raise ValueError(
                f'It is only possible to divide over a vector of the same size {self._size},'
                f' but got: {that._size}'
            )
        return self._div(that)

    @abstractmethod
    def div_scalar(self, that: Union[Scalar, Any]) -> 'Vector':
        pass
    
    @abstractmethod
    def as_array(self) -> np.ndarray:
        pass

    @abstractmethod
    def conjugate(self) -> 'Vector':
        pass

    def dot_product(self, that: 'Vector') -> Scalar:
        if self._size != that._size:
            raise ValueError(
                'It is only possible to calculate dot product with a vector'
                f'of the same size {self._size}, but got: {that._size}'
            )
        return self._dot_product(that)

    @abstractmethod
    def _full_like(self, items: Any) -> 'Vector':
        pass

    @abstractmethod
    def _ones_like(self) -> 'Vector':
        pass

    @abstractmethod
    def _zeros_like(self) -> 'Vector':
        pass

    @abstractmethod
    def _add(self, that: 'Vector') -> 'Vector':
        pass

    @abstractmethod
    def _sub(self, that: 'Vector') -> 'Vector':
        pass

    @abstractmethod
    def _mul(self, that: 'Vector') -> 'Vector':
        pass

    @abstractmethod
    def _div(self, that: 'Vector') -> 'Vector':
        pass

    @abstractmethod
    def _dot_product(self, that: 'Vector') -> Scalar:
        pass

