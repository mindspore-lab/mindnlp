from typing import List, Tuple
import numpy as np
from mindnlp._legacy.hypercomplex.tensor_decomposition.linal.basics.matrix import Matrix
from mindnlp._legacy.hypercomplex.tensor_decomposition.linal.algorithm.qr import QR
from mindnlp._legacy.hypercomplex.tensor_decomposition.linal.basics.scalar import Scalar
from mindnlp._legacy.hypercomplex.tensor_decomposition.linal.basics.hypercomplex.hc_matrix import Matrix as HCMatrix
from mindnlp._legacy.hypercomplex.tensor_decomposition.linal.basics.dual.dual_algebra_factory import DualAlgebraFactory
from mindnlp._legacy.hypercomplex.tensor_decomposition.linal.algorithm.dual_svd import DualSVD


class StarSVD(DualSVD):

    r"""
    StarSVD is a Python class that represents a specialized version of the Singular Value Decomposition (SVD) algorithm. 
    This class inherits functionality from the DualSVD class and provides a method to decompose a given matrix using a specific approach. 
    The decompose method within StarSVD returns three matrices that represent the decomposition result. 
    """
    @staticmethod
    def decompose(matrix: Matrix) -> Tuple[Matrix, Matrix, Matrix]:
        r"""
        This method decomposes the input matrix into three matrices using Singular Value Decomposition (SVD).
        
        Args:
            matrix (Matrix): The input matrix to be decomposed.
                It should be a valid Matrix object representing the data to be decomposed.
        
        Returns:
            Tuple[Matrix, Matrix, Matrix]: A tuple containing three matrices resulting from the decomposition:
                - The left singular matrix.
                - The singular values as a diagonal matrix.
                - The right singular matrix.
        
        Raises:
            None
        """
        return DualSVD._decompose(matrix, 1)
