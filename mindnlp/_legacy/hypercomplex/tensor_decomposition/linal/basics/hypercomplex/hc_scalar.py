from typing import TypeVar, Type, Tuple, Union
import numpy as np
from mindnlp._legacy.hypercomplex.tensor_decomposition.linal.basics.scalar import Scalar as AbstractScalar
from mindnlp._legacy.hypercomplex.tensor_decomposition.linal.basics.hypercomplex.hc_algebra_factory import AlgebraFactory

TAlgebraFactory = TypeVar('TAlgebraFactory', bound=AlgebraFactory)


class Scalar(AbstractScalar):

    def __init__(self,
                 algebra_factory: Type[TAlgebraFactory],
                 x: Union[float, np.float64],
                 y: Union[float, np.float64]) -> None:
        self._alg_factory = algebra_factory
        self._impl = algebra_factory().get_scalar_impl()
        self._x = np.float64(x)
        self._y = np.float64(y)

    def __str__(self) -> str:
        return str(self._x) + ('+' if self._y >= 0. else '-') + self._impl.special_element() + str(abs(self._y))

    def __neg__(self) -> 'Scalar':
        return Scalar(self._alg_factory, -self._x, -self._y)

    def __add__(self, that: 'Scalar') -> 'Scalar':
        return Scalar(self._alg_factory, self._x + that._x, self._y + that._y)

    def __sub__(self, that: 'Scalar') -> 'Scalar':
        return Scalar(self._alg_factory, self._x - that._x, self._y - that._y)

    def __mul__(self, that: 'Scalar') -> 'Scalar':
        product = self._impl.mul(self._x, self._y, that._x, that._y)
        return Scalar(self._alg_factory, *product)

    def __truediv__(self, that: 'Scalar') -> 'Scalar':
        div = self._impl.div(self._x, self._y, that._x, that._y)
        return Scalar(self._alg_factory, *div)

    def __getitem__(self, key):
        if key == 0:
            return self._x
        if key == 1:
            return self._y
        raise KeyError('The key is supposed to be 0 or 1')

    @staticmethod
    def one(algebra_factory: Type[TAlgebraFactory]) -> 'Scalar':
        return Scalar(algebra_factory, 1., 0.)

    @staticmethod
    def zero(algebra_factory: Type[TAlgebraFactory]) -> 'Scalar':
        return Scalar(algebra_factory, 0., 0.)

    def get_algebra_type(self) -> Type[AlgebraFactory]:
        return self._alg_factory

    def sqrt(self) -> 'Scalar':
        x, y = self._impl.sqrt(self._x, self._y)
        return Scalar(self._alg_factory, x, y)

    def as_array(self) -> np.ndarray:
        return np.array([self._x, self._y], dtype=np.float64)

    def is_zero(self) -> bool:
        return -Scalar._EQUALITY_TOLERANCE < self._x < Scalar._EQUALITY_TOLERANCE \
            and -Scalar._EQUALITY_TOLERANCE < self._y < Scalar._EQUALITY_TOLERANCE

    def get_real(self) -> np.float64:
        return self._x

    def visit(self, visitor, *args, **kwargs) -> None:
        self._impl.visit(self, visitor, *args, **kwargs)

    def _make_like(self, val: Tuple[np.float64, np.float64]) -> 'Scalar':
        return Scalar(self._alg_factory, *val)

    def _zero_like(self) -> 'Scalar':
        return Scalar.zero(self._alg_factory)

    def _one_like(self) -> 'Scalar':
        return Scalar.one(self._alg_factory)
