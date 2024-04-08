from abc import ABC, abstractmethod
from typing import Tuple
import numpy as np
from mindnlp._legacy.hypercomplex.tensor_decomposition.linal.basics.hypercomplex._hc_math_obj_impl import _MathObjImpl


class _VectorImpl(_MathObjImpl, ABC):

    @abstractmethod
    def dot_product(self,
                    a_x: np.ndarray,
                    a_y: np.ndarray,
                    b_x: np.ndarray,
                    b_y: np.ndarray) -> Tuple[np.float64, np.float64]:
        pass
