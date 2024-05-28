from abc import ABC, abstractmethod
from typing import Tuple
import numpy as np
from mindnlp._legacy.hypercomplex.tensor_decomposition.linal.basics.hypercomplex._hc_math_obj_impl import _MathObjImpl


class _MatrixImpl(_MathObjImpl, ABC):

    r"""
    _MatrixImpl is a Python class that represents an implementation of matrix operations. It inherits from _MathObjImpl and is designed to provide functionality for matrix multiplication.
    
    This class provides an abstract method matmul that must be implemented by subclasses. The matmul method takes four NumPy arrays as input parameters and returns a tuple of two NumPy arrays as output,
representing the result of the matrix multiplication operation.
    
    Subclasses of _MatrixImpl are expected to provide concrete implementations of the matmul method to support specific matrix multiplication operations.
    
    Note: This docstring does not include signatures or any other code to maintain consistency.
    """

    @abstractmethod
    def matmul(self,
               a_x: np.ndarray,
               a_y: np.ndarray,
               b_x: np.ndarray,
               b_y: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:

        r"""
        This method performs matrix multiplication by taking two pairs of input arrays and returning the resulting matrix product.
        
        Args:
            self (_MatrixImpl): The instance of the _MatrixImpl class.
            a_x (np.ndarray): The input array representing the x-coordinates of the first matrix.
            a_y (np.ndarray): The input array representing the y-coordinates of the first matrix.
            b_x (np.ndarray): The input array representing the x-coordinates of the second matrix.
            b_y (np.ndarray): The input array representing the y-coordinates of the second matrix.
        
        Returns:
            Tuple[np.ndarray, np.ndarray]: A tuple containing two numpy arrays representing the x and y coordinates of the resulting matrix product.
        
        Raises:
            ValueError: If the input arrays are not compatible for matrix multiplication.
            TypeError: If the input arrays are not of type np.ndarray.
            IndexError: If the input arrays do not have compatible dimensions for matrix multiplication.
            Exception: Any other unforeseen exception may also be raised during the matrix multiplication process.
        """
        pass
