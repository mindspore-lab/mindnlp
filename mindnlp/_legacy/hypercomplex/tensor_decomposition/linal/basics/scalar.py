from abc import ABC, abstractmethod
from typing import Any
import numpy as np


class Scalar(ABC):

    _EQUALITY_TOLERANCE: float = 1e-10

    @staticmethod
    def make_like(m: 'Scalar', val: Any) -> 'Scalar':
        return m._make_like(val)

    @staticmethod
    def zero_like(m: 'Scalar') -> 'Scalar':
        return m._zero_like()

    @staticmethod
    def one_like(m: 'Scalar') -> 'Scalar':
        return m._one_like()

    @abstractmethod
    def __neg__(self) -> 'Scalar':
        pass

    @abstractmethod
    def __add__(self, that: 'Scalar') -> 'Scalar':
        pass

    @abstractmethod
    def __sub__(self, that: 'Scalar') -> 'Scalar':
        pass

    @abstractmethod
    def __mul__(self, that: 'Scalar') -> 'Scalar':
        pass

    @abstractmethod
    def __truediv__(self, that: 'Scalar') -> 'Scalar':
        pass

    @abstractmethod
    def sqrt(self) -> 'Scalar':
        pass

    @abstractmethod
    def visit(self, visitor, *args, **kwargs) -> None:
        pass

    @abstractmethod
    def _make_like(self, val: Any) -> 'Scalar':
        pass

    @abstractmethod
    def _zero_like(self) -> 'Scalar':
        pass

    @abstractmethod
    def _one_like(self) -> 'Scalar':
        pass

    @abstractmethod
    def as_array(self) -> np.ndarray:
        pass

    @abstractmethod
    def is_zero(self) -> bool:
        pass

    @abstractmethod
    def get_real(self) -> np.float64:
        pass
