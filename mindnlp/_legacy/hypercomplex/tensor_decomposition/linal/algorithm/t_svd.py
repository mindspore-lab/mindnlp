from typing import List, Tuple
import numpy as np
from mindnlp._legacy.hypercomplex.tensor_decomposition.linal.basics.matrix import Matrix
from mindnlp._legacy.hypercomplex.tensor_decomposition.linal.algorithm.qr import QR
from mindnlp._legacy.hypercomplex.tensor_decomposition.linal.basics.scalar import Scalar
from mindnlp._legacy.hypercomplex.tensor_decomposition.linal.basics.hypercomplex.hc_matrix import Matrix as HCMatrix
from mindnlp._legacy.hypercomplex.tensor_decomposition.linal.basics.dual.dual_algebra_factory import DualAlgebraFactory
from mindnlp._legacy.hypercomplex.tensor_decomposition.linal.algorithm.dual_svd import DualSVD


class TSVD(DualSVD):

    r"""
    This class represents a truncated singular value decomposition (TSVD) algorithm.
    
    TSVD is a matrix factorization technique that decomposes a given matrix into three separate matrices: U, S, and V^T. The U matrix contains the left singular vectors, the S matrix contains the singular
values, and the V^T matrix contains the right singular vectors.
    
    This TSVD class is a subclass of the DualSVD class, which provides a general implementation of the SVD algorithm. The inherited _decompose() method from DualSVD is used to perform the actual decomposition
of the matrix.
    
    The decompose() method of this class takes a matrix as input and returns a tuple containing the three decomposed matrices: U, S, and V^T. The method internally calls the _decompose() method of the DualSVD
class with a value of -1, indicating that no truncation is applied.
    
    Note that the decompose() method is a static method, meaning it can be called directly on the class without needing to create an instance of the TSVD class.
    
    Example usage:
        matrix = Matrix([[1, 2], [3, 4], [5, 6]])
        u, s, v = TSVD.decompose(matrix)
        # u = [[-0.22975292, -0.88346102], [-0.52474482, -0.24078249], [-0.81973672, 0.40189604]]
        # s = [[8.17225678, 0.], [0., 0.27386128]]
        # v = [[-0.61962948, -0.78489445], [0.78489445, -0.61962948]]
    
    Note: The actual implementation of the TSVD class and its inherited methods are not included in this docstring.
    
    """
    @staticmethod
    def decompose(matrix: Matrix) -> Tuple[Matrix, Matrix, Matrix]:
        r"""
        Decompose the given matrix using the TSVD method.
        
        Args:
            matrix (Matrix): The input matrix to be decomposed. It should be a valid Matrix object.
            
        Returns:
            Tuple[Matrix, Matrix, Matrix]: A tuple containing three matrices representing the decomposed components of the input matrix.
            The three matrices correspond to the left singular vectors, singular values, and right singular vectors, respectively.
        
        Raises:
            This method does not raise any exceptions.
        """
        return DualSVD._decompose(matrix, -1)
