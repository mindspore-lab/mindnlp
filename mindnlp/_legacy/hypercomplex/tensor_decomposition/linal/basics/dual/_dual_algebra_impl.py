from typing import Tuple, Union
import numpy as np
from mindnlp._legacy.hypercomplex.tensor_decomposition.linal.basics.hypercomplex._hc_matrix_impl import _MatrixImpl
from mindnlp._legacy.hypercomplex.tensor_decomposition.linal.basics.hypercomplex._hc_scalar_impl import _ScalarImpl
from mindnlp._legacy.hypercomplex.tensor_decomposition.linal.basics.hypercomplex._hc_vector_impl import _VectorImpl


class _DualAlgebraImpl(_MatrixImpl, _VectorImpl, _ScalarImpl):

    def mul(self,
            a_x: np.ndarray,
            a_y: np.ndarray,
            b_x: Union[np.ndarray, np.float64],
            b_y: Union[np.ndarray, np.float64]) -> Tuple[np.ndarray, np.ndarray]:
        return (
            a_x * b_x,
            a_x * b_y + a_y * b_x
        )

    def div(self,
            a_x: np.ndarray,
            a_y: np.ndarray,
            b_x: Union[np.ndarray, np.float64],
            b_y: Union[np.ndarray, np.float64]) -> Tuple[np.ndarray, np.ndarray]:
        return (
            a_x / b_x,
            (a_y * b_x - a_x * b_y) / np.square(b_x)
        )

    def matmul(self,
               a_x: np.ndarray,
               a_y: np.ndarray,
               b_x: np.ndarray,
               b_y: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        return (
            a_x @ b_x,
            a_x @ b_y + a_y @ b_x
        )

    def dot_product(self,
                    a_x: np.ndarray,
                    a_y: np.ndarray,
                    b_x: np.ndarray,
                    b_y: np.ndarray) -> Tuple[np.float64, np.float64]:
        return (
            np.float64(np.dot(a_x, b_x)),
            np.float64(np.dot(a_x, b_y) + np.dot(a_y, b_x))
        )

    def sqrt(self, x: np.float64, y: np.float64) -> Tuple[np.float64, np.float64]:
        x_sqrt = np.sqrt(x)
        y_sqrt = np.divide(y, x) / 2
        return x_sqrt, y_sqrt

    def special_element(self) -> str:
        return u'\u03b5'

    def visit(self, scalar, visitor, *args, **kwargs) -> None:
        visitor.visit_dual(scalar, *args, **kwargs)

    def mul_scalar(self,
                   x1: np.float64,
                   y1: np.float64,
                   x2: np.float64,
                   y2: np.float64) -> Tuple[np.float64, np.float64]:
        return (
            np.float64(x1 * x2),
            np.float64(x1 * y2 + y1 * x2)
        )

    def div_scalar(self,
                   x1: np.float64,
                   y1: np.float64,
                   x2: np.float64,
                   y2: np.float64) -> Tuple[np.float64, np.float64]:
        return (
            np.float64(x1 / x2),
            (y1 * x2 - x1 * y2) / np.square(x2)
        )
