from typing import List, Tuple
import numpy as np
from mindnlp._legacy.hypercomplex.tensor_decomposition.linal.basics.matrix import Matrix
from mindnlp._legacy.hypercomplex.tensor_decomposition.linal.algorithm.qr import QR
from mindnlp._legacy.hypercomplex.tensor_decomposition.linal.basics.scalar import Scalar
from mindnlp._legacy.hypercomplex.tensor_decomposition.linal.basics.hypercomplex.hc_matrix import Matrix as HCMatrix
from mindnlp._legacy.hypercomplex.tensor_decomposition.linal.basics.dual.dual_algebra_factory import DualAlgebraFactory
from mindnlp._legacy.hypercomplex.tensor_decomposition.linal.algorithm.dual_svd import DualSVD


class StarSVD(DualSVD):

    @staticmethod
    def decompose(matrix: Matrix) -> Tuple[Matrix, Matrix, Matrix]:
        return DualSVD._decompose(matrix, 1)
