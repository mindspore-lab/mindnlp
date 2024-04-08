from typing import Tuple, Union
import numpy as np
from mindnlp._legacy.hypercomplex.tensor_decomposition.linal.basics.hypercomplex._hc_matrix_impl import _MatrixImpl
from mindnlp._legacy.hypercomplex.tensor_decomposition.linal.basics.hypercomplex._hc_scalar_impl import _ScalarImpl
from mindnlp._legacy.hypercomplex.tensor_decomposition.linal.basics.hypercomplex._hc_vector_impl import _VectorImpl


class _ComplexAlgebraImpl(_MatrixImpl, _VectorImpl, _ScalarImpl):

    def mul(self,
            a_x: np.ndarray,
            a_y: np.ndarray,
            b_x: Union[np.ndarray, np.float64],
            b_y: Union[np.ndarray, np.float64]) -> Tuple[np.ndarray, np.ndarray]:
        return (
            a_x * b_x - a_y * b_y,
            a_x * b_y + a_y * b_x
        )

    def div(self,
            a_x: np.ndarray,
            a_y: np.ndarray,
            b_x: Union[np.ndarray, np.float64],
            b_y: Union[np.ndarray, np.float64]) -> Tuple[np.ndarray, np.ndarray]:
        square_norm = np.square(b_x) + np.square(b_y)
        return (
            (a_x * b_x + a_y * b_y) / square_norm,
            (a_y * b_x - a_x * b_y) / square_norm
        )

    def matmul(self,
               a_x: np.ndarray,
               a_y: np.ndarray,
               b_x: np.ndarray,
               b_y: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        return (
            a_x @ b_x - a_y @ b_y,
            a_x @ b_y + a_y @ b_x
        )

    def dot_product(self,
                    a_x: np.ndarray,
                    a_y: np.ndarray,
                    b_x: np.ndarray,
                    b_y: np.ndarray) -> Tuple[np.float64, np.float64]:
        return (
            np.float64(np.dot(a_x, b_x) - np.dot(a_y, b_y)),
            np.float64(np.dot(a_x, b_y) + np.dot(a_y, b_x))
        )

    def sqrt(self, x: np.float64, y: np.float64) -> Tuple[np.float64, np.float64]:
        x_sqr = np.square(x)
        y_sqr = np.square(y)
        square_norm = x_sqr + y_sqr
        norm = np.sqrt(square_norm)
        x_abs = np.fabs(x)
        x_sqrt = np.sqrt((norm + x_abs) / 2)
        y_sqrt = y / np.sqrt(2. * (norm + x_abs))
        if np.isnan(y_sqrt):
            y_sqrt = np.float64(0.)
        return x_sqrt, y_sqrt

    def special_element(self) -> str:
        return 'i'

    def visit(self, scalar, visitor, *args, **kwargs) -> None:
        visitor.visit_complex(scalar, *args, **kwargs)

    def mul_scalar(self,
                   x1: np.float64,
                   y1: np.float64,
                   x2: np.float64,
                   y2: np.float64) -> Tuple[np.float64, np.float64]:
        return (
            np.float64(x1 * x2 - y1 * y2),
            np.float64(x1 * y2 + y1 * x2)
        )

    def div_scalar(self,
                   x1: np.float64,
                   y1: np.float64,
                   x2: np.float64,
                   y2: np.float64) -> Tuple[np.float64, np.float64]:
        square_norm = np.square(x2) + np.square(y2)
        return (
            (x1 * x2 + y1 * y2) / square_norm,
            (y1 * x2 - x1 * y2) / square_norm
        )
