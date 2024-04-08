from typing import Union
import numpy as np
from mindnlp._legacy.hypercomplex.tensor_decomposition.linal.basics.scalar import Scalar as AbstractScalar


class Scalar(AbstractScalar):

    def __init__(self, x: Union[float, np.float64]) -> None:
        self._x = np.float64(x)

    def __str__(self) -> str:
        return str(self._x)

    def __neg__(self) -> 'Scalar':
        return Scalar(-self._x)

    def __add__(self, that: 'Scalar') -> 'Scalar':
        return Scalar(self._x + that._x)

    def __sub__(self, that: 'Scalar') -> 'Scalar':
        return Scalar(self._x - that._x)

    def __mul__(self, that: 'Scalar') -> 'Scalar':
        return Scalar(self._x * that._x)

    def __truediv__(self, that: 'Scalar') -> 'Scalar':
        return Scalar(self._x / that._x)

    @staticmethod
    def one() -> 'Scalar':
        return Scalar(1.)

    @staticmethod
    def zero() -> 'Scalar':
        return Scalar(0.)

    def sqrt(self) -> 'Scalar':
        return Scalar(np.sqrt(self._x))

    def as_array(self) -> np.ndarray:
        return np.array(self._x, dtype=np.float64)

    def as_number(self) -> np.float64:
        return self._x

    def is_zero(self) -> bool:
        return -Scalar._EQUALITY_TOLERANCE < self._x < Scalar._EQUALITY_TOLERANCE

    def get_real(self) -> np.float64:
        return self._x

    def visit(self, visitor, *args, **kwargs) -> None:
        visitor.visit_real(self, *args, **kwargs)

    def _make_like(self, val: np.float64) -> 'Scalar':
        return Scalar(val)

    def _zero_like(self) -> 'Scalar':
        return Scalar.zero()

    def _one_like(self) -> 'Scalar':
        return Scalar.one()
