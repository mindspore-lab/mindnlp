from typing import Tuple
import numpy as np
from mindnlp._legacy.hypercomplex.tensor_decomposition.linal.basics.hypercomplex.hc_scalar import Scalar as HCScalar
from mindnlp._legacy.hypercomplex.tensor_decomposition.linal.basics.matrix import Matrix
from mindnlp._legacy.hypercomplex.tensor_decomposition.linal.basics.real.real_scalar import Scalar as RealScalar
from mindnlp._legacy.hypercomplex.tensor_decomposition.linal.basics.scalar import Scalar
from mindnlp._legacy.hypercomplex.tensor_decomposition.linal.basics.scalar_visitor import ScalarVisitor


class QR:

    r"""
    The QR class provides methods for performing QR decomposition on matrices using Gram-Schmidt and Householder transformations.
    
    The decompose_gram_schmidt method takes a matrix as input and returns the Q and R matrices obtained from the QR decomposition using the Gram-Schmidt method.
    
    The decompose_householder method takes a matrix as input and returns the Q and R matrices obtained from the QR decomposition using the Householder transformation method.
    
    The _householder method is a private method that performs QR decomposition using the Householder transformation method.
    
    The _gram_schmidt method is a private method that performs QR decomposition using the Gram-Schmidt method.
    """

    class _HouseholderAlphaCalculator(ScalarVisitor):

        def visit_real(self, s: RealScalar, *args, **kwargs) -> None:
            output = kwargs.get('output')
            f = s.as_number()
            alpha = Scalar.one_like(s)
            if f < 0:
                alpha = -alpha
            output.append(alpha)

        def visit_complex(self, s: HCScalar, *args, **kwargs) -> None:
            output = kwargs.get('output')
            x, y = s.as_array()
            z = np.complex(x, y)
            angle = np.angle(z)
            alpha = Scalar.make_like(s, (np.cos(angle), np.sin(angle)))
            alpha = -alpha
            output.append(alpha)

        def visit_dual(self, s: HCScalar, *args, **kwargs) -> None:
            output = kwargs.get('output')
            x, y = s.as_array()
            y_ = y / np.abs(x)
            if np.isnan(y_):
                y_ = np.float64(0.)
            x_ = np.float64(1. if x < 0 else -1.)
            alpha = Scalar.make_like(s, (x_, y_))
            output.append(alpha)

        def calculate_alpha(self, x: Scalar, norm: Scalar) -> Scalar:
            output = []
            kwargs = {
                'output': output,
            }
            x.visit(self, **kwargs)
            alpha = output[0]
            alpha = alpha * norm
            return alpha

    @staticmethod
    def decompose_gram_schmidt(matrix: Matrix) -> Tuple[Matrix, Matrix]:

        r"""
        This method decomposes a square matrix using the Gram-Schmidt process into an orthogonal matrix 'q' and an upper triangular matrix 'r'.
        
        Args:
            matrix (Matrix): The square matrix to be decomposed using the Gram-Schmidt process.
            
        Returns:
            Tuple[Matrix, Matrix]: A tuple containing the orthogonal matrix 'q' and the upper triangular matrix 'r' resulting from the QR decomposition.
        
        Raises:
            ValueError: If the input matrix is not square, a ValueError is raised indicating that only square matrices can be QR decomposed.
        """
        if matrix.get_height() != matrix.get_width():
            raise ValueError(
                'Only square matrices can be QR decomposed, but got: '
                f'height={matrix.get_height()}, width={matrix.get_width()}'
            )
        q = QR._gram_schmidt(matrix)
        q_inv = q.transpose_conjugate()
        r = q_inv @ matrix
        return q, r

    @staticmethod
    def decompose_householder(matrix: Matrix) -> Tuple[Matrix, Matrix]:

        r"""
        Decompose the given square matrix into its QR decomposition using Householder reflection.
        
        Args:
            matrix (Matrix): The square matrix to be decomposed into its QR representation.
                It must be a square matrix, i.e., the number of rows must be equal to the number of columns.
        
        Returns:
            Tuple[Matrix, Matrix]: A tuple containing two matrices representing the QR decomposition of the input matrix.
                The first matrix ('q') is the orthogonal matrix, and the second matrix ('r') is the upper triangular matrix.
        
        Raises:
            ValueError: Raised if the input matrix is not square. In such cases, the error message specifies that only square matrices
                can be QR decomposed, along with the dimensions of the provided matrix (height and width).
        """
        if matrix.get_height() != matrix.get_width():
            raise ValueError(
                'Only square matrices can be QR decomposed, but got: '
                f'height={matrix.get_height()}, width={matrix.get_width()}'
            )
        q, r = QR._householder(matrix)
        return q, r

    @staticmethod
    def _householder(m: Matrix):

        r"""
        Performs the Householder transformation on a given matrix.
        
        Args:
            m (Matrix): The matrix on which the Householder transformation is applied.
        
        Returns:
            None
        
        Raises:
            None
        
        This method applies the Householder transformation to the input matrix 'm'. The Householder transformation is used in 
        numerical linear algebra to reduce a matrix to upper triangular form through a series of orthogonal transformations.
        
        The Householder transformation is computed by iterating over the columns of the matrix and performing the following steps:
        1. Calculate the reflection vector 'v' using the Householder algorithm.
        2. Construct the Householder matrix 'q_k' using the reflection vector 'v'.
        3. Update the matrix 'q' by multiplying it with 'q_k'.
        4. Update the matrix 'r' by multiplying it with 'q_k'.
        
        The Householder transformation is useful in various applications such as solving systems of linear equations, 
        least squares problems, and eigenvalue computations.
        
        Note:
        - The input matrix 'm' should be a valid instance of the 'Matrix' class.
        - The method does not modify the input matrix 'm' directly, but instead returns two transformed matrices 'q' and 'r'.
        - The matrices 'q' and 'r' represent the orthogonal and upper triangular forms respectively.
        
        Example:
            m = Matrix([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
            qr = QR()
            q, r = qr._householder(m)
            # q:
            # Matrix([[...]])
            # r:
            # Matrix([[...]])
        """
        r = m
        q = identity = Matrix.identity_like(m)
        one = identity[0][0]
        two = one + one
        hh_alpha = QR._HouseholderAlphaCalculator()
        for k in range(m.get_width()):
            x = r[k:, k:(k + 1)]
            e = identity[k:, k:(k + 1)]
            x_conj_t = x.transpose_conjugate()
            x_norm_sqr = (x_conj_t @ x)[0][0]
            x_norm = x_norm_sqr.sqrt()
            alpha = hh_alpha.calculate_alpha(x[0][0], x_norm)
            u = x - e * alpha
            u_conj_t = u.transpose_conjugate()
            u_norm_sqr = (u_conj_t @ u)[0][0]
            u_norm = u_norm_sqr.sqrt()
            v = u / u_norm
            v_items = v.as_array()
            if np.any(np.isnan(v_items)) or np.any(np.isinf(v_items)):
                v = e
            v_conj_t = v.transpose_conjugate()
            supplementary = v @ v_conj_t
            q_k = identity[k:, k:] - supplementary * two
            if k > 0:
                q_k = identity[k:, :k].concat_cols(q_k)
                q_k = identity[:k].concat_rows(q_k)
            q = q @ q_k
            r = q_k @ r
        return q, r

    @staticmethod
    def _gram_schmidt(m: Matrix) -> Matrix:

        r"""
        Performs the Gram-Schmidt process on a given matrix.
        
        Args:
            m (Matrix): The matrix on which the Gram-Schmidt process is to be performed. The matrix should be a 2D matrix.
        
        Returns:
            Matrix: A matrix containing the normalized vectors after the Gram-Schmidt process.
        
        Raises:
            None.
        
        The Gram-Schmidt process is used to orthogonalize a set of vectors in a given matrix. This process takes the input matrix and produces a new matrix where each column vector is orthogonal to all the
previous column vectors. The resulting vectors are normalized to have a length of 1.
        
        The method starts by initializing a normalized matrix and a projections matrix. It then iterates over each column in the input matrix. For each column, it calculates the corresponding vector and
performs the following steps:
        - If it's not the first column, it calculates the projection of the vector onto the previous normalized vectors.
        - It then calculates the squared length of the vector and its actual length.
        - The vector is then divided by its length to normalize it.
        - The normalized vector is concatenated to the normalized matrix.
        - If there are more columns, it calculates the dot products between the normalized vector and the remaining columns in the input matrix.
        - The dot products are adjusted and concatenated to the projections matrix.
        
        Finally, the method returns the resulting normalized matrix after the Gram-Schmidt process.
        
        Note: This method is a static method and should be called using the class name, followed by the method name.
        """
        normalized = None
        projections = Matrix.ones_like(m[0:1])
        v_zero = Matrix.zeros_like(projections)
        for j in range(m.get_width()):
            v = m[:, j:(j + 1)]
            if j > 0:
                t = v.concat_cols(normalized)
                v = t @ projections[:, j:(j + 1)]
            v_conj = v.transpose_conjugate()
            v_sqr_len = (v_conj @ v)[0, 0]
            v_len = v_sqr_len.sqrt()
            v = v / v_len
            normalized = v if j == 0 else normalized.concat_cols(v)
            if j + 1 < m.get_width():
                v_conj = v.transpose_conjugate()
                dot_products = v_conj @ m[:, j + 1:]
                dot_products = -dot_products
                dot_products = v_zero[..., :(j + 1)].concat_cols(dot_products)
                projections = projections.concat_rows(dot_products)
        return normalized
