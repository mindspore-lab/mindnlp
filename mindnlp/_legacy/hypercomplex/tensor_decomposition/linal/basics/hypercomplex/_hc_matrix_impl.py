from abc import ABC, abstractmethod
from typing import Tuple
import numpy as np
from mindnlp._legacy.hypercomplex.tensor_decomposition.linal.basics.hypercomplex._hc_math_obj_impl import _MathObjImpl


class _MatrixImpl(_MathObjImpl, ABC):

    @abstractmethod
    def matmul(self,
               a_x: np.ndarray,
               a_y: np.ndarray,
               b_x: np.ndarray,
               b_y: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        pass
