from typing import List, Tuple, Callable
import numpy as np
from mindnlp._legacy.hypercomplex.tensor_decomposition.linal.basics.matrix import Matrix
from mindnlp._legacy.hypercomplex.tensor_decomposition.linal.algorithm.qr import QR
from mindnlp._legacy.hypercomplex.tensor_decomposition.linal.basics.scalar import Scalar


class SVD:

    r"""
    The SVD class provides functionalities for performing Singular Value Decomposition (SVD) on matrices. It includes methods for decomposing a matrix into its singular value, left singular vectors, and right
singular vectors, as well as auxiliary methods for sorting singular values, dividing rows by singular values, and removing columns from a matrix.
    
    The decompose method takes a matrix and optional parameters for the number of iterations and the QR algorithm to use, and returns the left singular vectors, singular values, and the conjugate transpose of
the right singular vectors. The _divide_rows_by_singular_values method divides the rows of the given matrix by the corresponding singular values. The _sort_singular_values method sorts the singular values and
the corresponding rows of the input matrix. The _remove_column method removes a specified column from a matrix. The _find_singular_values method calculates the singular values and the left or right singular
vectors using the specified QR algorithm and a specified number of iterations.
    
    The SVD class provides essential functionality for performing SVD and manipulating the resulting singular value, left singular vectors, and right singular vectors. It serves as a valuable tool for matrix
analysis and decomposition.
    """

    @staticmethod
    def decompose(matrix: Matrix,
                  iterations: int = 10,
                  qr_alg: Callable[[Matrix], Tuple[Matrix, Matrix]] = QR.decompose_householder
                  ) -> Tuple[Matrix, Matrix, Matrix]:
        r"""
        This method decomposes a given matrix using Singular Value Decomposition (SVD).
        
        Args:
            matrix (Matrix): The input matrix to be decomposed.
            iterations (int, optional): The number of iterations for the decomposition process. Defaults to 10.
            qr_alg (Callable[[Matrix], Tuple[Matrix, Matrix]], optional): The QR decomposition algorithm to be used. 
                Defaults to QR.decompose_householder.
        
        Returns:
            Tuple[Matrix, Matrix, Matrix]: A tuple containing three matrices representing the SVD decomposition:
                - The left singular vectors matrix (u).
                - The singular values matrix (sigma).
                - The right singular vectors matrix (v_conj_t).
        
        Raises:
            None
        """
        height = matrix.get_height()
        width = matrix.get_width()
        matrix_t = matrix.transpose_conjugate()
        m = matrix @ matrix_t if height <= width else matrix_t @ matrix
        sigma, d = SVD._find_singular_values(m, iterations, qr_alg)

        if height <= width:
            u = d
            u_conj_t = u.transpose_conjugate()
            sigma, u_conj_t = SVD._sort_singular_values(sigma, u_conj_t)
            u = u_conj_t.transpose_conjugate()
            v_conj_t = u_conj_t @ matrix
            v_conj_t = SVD._divide_rows_by_singular_values(sigma, v_conj_t)
        else:
            v = d
            v_conj_t = v.transpose_conjugate()
            sigma, v_conj_t = SVD._sort_singular_values(sigma, v_conj_t)
            u_conj_t = v_conj_t @ matrix_t
            u_conj_t = SVD._divide_rows_by_singular_values(sigma, u_conj_t)
            u = u_conj_t.transpose_conjugate()

        sigma_items = np.zeros_like(m[:len(sigma), :len(sigma)].as_array())
        for i in range(v_conj_t.get_height()):
            sigma_items[i, i] = sigma[i].as_array()
        sigma = Matrix.full_like(m, sigma_items)
        return u, sigma, v_conj_t

    @staticmethod
    def _divide_rows_by_singular_values(sigma: List[Scalar],
                                        conj_transposed_u_or_v: Matrix) -> Matrix:
        r"""
        Method _divide_rows_by_singular_values in the SVD class.
        
        Args:
            sigma (List[Scalar]): A list of singular values to divide the rows by.
                Type: List
                Purpose: Represents the singular values used in the division operation.
                Restrictions: Must be a list of Scalar values.
            
            conj_transposed_u_or_v (Matrix): The matrix whose rows are divided by the singular values.
                Type: Matrix
                Purpose: Represents the matrix containing the rows to be divided.
                Restrictions: Should be a Matrix object.
        
        Returns:
            Matrix: A new Matrix object with rows divided by the corresponding singular values.
                Type: Matrix
                Purpose: Represents the result of dividing the rows.
                
        Raises:
            None
        """
        rows = [
            conj_transposed_u_or_v[i] / sigma[i]
            for i in range(conj_transposed_u_or_v.get_height())
        ]
        return Matrix.full_like(conj_transposed_u_or_v, rows)

    @staticmethod
    def _sort_singular_values(sigma: List[Scalar],
                              conj_transposed_u_or_v: Matrix) -> Tuple[List[Scalar], Matrix]:
        r"""
        This method sorts the singular values and corresponding columns of U or V in the Singular Value Decomposition (SVD).
        
        Args:
            sigma (List[Scalar]): A list of singular values.
                The singular values to be sorted.
            conj_transposed_u_or_v (Matrix): A matrix representing the conjugate transpose of U or V.
                The corresponding columns of U or V to be sorted along with the singular values.
        
        Returns:
            Tuple[List[Scalar], Matrix]: A tuple containing the sorted singular values and the corresponding sorted columns of U or V.
        
        Raises:
            None: This method does not explicitly raise any exceptions.
        """
        perm = list(zip(sigma, conj_transposed_u_or_v))
        perm.sort(key=lambda item: item[0].get_real(), reverse=True)
        sigma, rows = list(zip(*perm))
        conj_transposed_u_or_v = Matrix.full_like(conj_transposed_u_or_v, list(rows))
        return list(sigma), conj_transposed_u_or_v

    @staticmethod
    def _remove_column(m: Matrix, idx: int) -> Matrix:
        r"""
        Method to remove a specific column from a matrix.
        
        Args:
            m (Matrix): The input matrix from which a column will be removed.
                It must be an instance of the Matrix class.
            idx (int): The index of the column to be removed.
                It must be an integer representing the index of the column to be removed from the matrix.
        
        Returns:
            Matrix: Returns a new matrix with the specified column removed based on the provided index.
                It will be an instance of the Matrix class.
        
        Raises:
            IndexError: If the specified index is out of range or invalid for the given matrix.
            TypeError: If the input matrix is not an instance of the Matrix class.
        """
        return m[:, 1:] if idx == 0 \
            else m[:, :-1] if idx in [m.get_width() - 1, -1] \
            else m[:, :idx].concat_cols(m[:, (idx + 1):])

    @staticmethod
    def _find_singular_values(m: Matrix,
                              iterations: int,
                              qr_alg: Callable[[Matrix], Tuple[Matrix, Matrix]]) -> Tuple[List[Scalar], Matrix]:
        r"""
        Finds the singular values of a given matrix using the specified QR algorithm.
        
        Args:
            m (Matrix): The matrix for which the singular values are to be found.
            iterations (int): The number of iterations to perform in the QR algorithm.
            qr_alg (Callable[[Matrix], Tuple[Matrix, Matrix]]): A callable function that performs the QR algorithm on a matrix.
                The function should take a matrix as input and return a tuple of the resulting matrices (Q, R).
        
        Returns:
            Tuple[List[Scalar], Matrix]: A tuple containing the singular values and the transformed matrix.
                - singular_values (List[Scalar]): A list of scalar values representing the singular values of the matrix.
                - transformed_matrix (Matrix): The matrix that has been transformed using the QR algorithm.
        
        Raises:
            ValueError: If an error occurs during the calculation of the singular values.
                This can happen if the matrix contains NaN or infinite values, which cannot be square rooted.
        
        Note:
            This method uses the QR algorithm to iteratively transform the matrix and calculate the singular values.
            The QR algorithm performs a series of QR factorizations and updates the matrix at each iteration.
            The singular values are then extracted from the updated matrix by taking the square root of the diagonal elements.
            If any NaN or infinite values are encountered during the calculation, the corresponding singular value is removed.
        
        Example:
            m = Matrix([[1, 2], [3, 4]])
            iterations = 10
            qr_alg = qr_decomposition  # Example QR algorithm function
            singular_values, transformed_matrix = SVD._find_singular_values(m, iterations, qr_alg)
        """
        u_or_v = None
        for _ in range(iterations):
            q, r = qr_alg(m)
            u_or_v = q if u_or_v is None else u_or_v @ q
            m = r @ q
        sigma_items = []
        j = 0
        for i in range(m.get_height()):
            try:
                sigma = m[i, i].sqrt()
                sigma_val = sigma.as_array()
                if np.any(np.isnan(sigma_val)) or np.any(np.isinf(sigma_val)):
                    u_or_v = SVD._remove_column(u_or_v, j)
                else:
                    sigma_items.append(sigma)
                    j += 1
            except ValueError:
                u_or_v = SVD._remove_column(u_or_v, j)
        return sigma_items, u_or_v
