from typing import Tuple, Union
import numpy as np
from mindnlp._legacy.hypercomplex.tensor_decomposition.linal.basics.hypercomplex._hc_matrix_impl import _MatrixImpl
from mindnlp._legacy.hypercomplex.tensor_decomposition.linal.basics.hypercomplex._hc_scalar_impl import _ScalarImpl
from mindnlp._legacy.hypercomplex.tensor_decomposition.linal.basics.hypercomplex._hc_vector_impl import _VectorImpl


class _DualAlgebraImpl(_MatrixImpl, _VectorImpl, _ScalarImpl):

    r"""
    The _DualAlgebraImpl class represents a collection of methods for performing algebraic operations on dual numbers. 
    These operations include multiplication, division, matrix multiplication, dot product, square root, handling special elements, visiting elements, and performing scalar operations. 
    This class inherits from _MatrixImpl, _VectorImpl, _ScalarImpl, and provides an implementation for each method to handle dual numbers and perform the corresponding algebraic operations on them.
    """

    def mul(self,
            a_x: np.ndarray,
            a_y: np.ndarray,
            b_x: Union[np.ndarray, np.float64],
            b_y: Union[np.ndarray, np.float64]) -> Tuple[np.ndarray, np.ndarray]:

        r"""
        Multiply two arrays or an array and a scalar element-wise.
        
        Args:
            self: The instance of the '_DualAlgebraImpl' class.
            a_x: An ndarray of shape (N,) representing the x-values of the first array.
            a_y: An ndarray of shape (N,) representing the y-values of the first array.
            b_x: Either an ndarray of shape (N,) or a scalar of type np.float64 representing the x-values of the second array or scalar.
            b_y: Either an ndarray of shape (N,) or a scalar of type np.float64 representing the y-values of the second array or scalar.
        
        Returns:
            A tuple of two ndarrays representing the result of the element-wise multiplication:
            - The first ndarray has the same shape as the input arrays and contains the element-wise product of a_x and b_x.
            - The second ndarray has the same shape as the input arrays and contains the element-wise product of a_x and b_y, added to the element-wise product of a_y and b_x.
        
        Raises:
            None.
        """
        return (
            a_x * b_x,
            a_x * b_y + a_y * b_x
        )

    def div(self,
            a_x: np.ndarray,
            a_y: np.ndarray,
            b_x: Union[np.ndarray, np.float64],
            b_y: Union[np.ndarray, np.float64]) -> Tuple[np.ndarray, np.ndarray]:

        r"""
        Divides two arrays and returns the result as a tuple of two arrays.
        
        Args:
            self: An instance of the _DualAlgebraImpl class.
            a_x (np.ndarray): The first input array representing the x-values of the dividend.
            a_y (np.ndarray): The second input array representing the y-values of the dividend.
            b_x (Union[np.ndarray, np.float64]): The third input representing the x-value or an array of the divisor.
            b_y (Union[np.ndarray, np.float64]): The fourth input representing the y-value or an array of the divisor.
        
        Returns:
            Tuple[np.ndarray, np.ndarray]: A tuple containing two arrays.
                - The first array represents the result of dividing a_x by b_x element-wise.
                - The second array represents the result of the formula (a_y * b_x - a_x * b_y) / b_x^2 element-wise.
        
        Raises:
            None.
        """
        return (
            a_x / b_x,
            (a_y * b_x - a_x * b_y) / np.square(b_x)
        )

    def matmul(self,
               a_x: np.ndarray,
               a_y: np.ndarray,
               b_x: np.ndarray,
               b_y: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:

        r"""
        Performs matrix multiplication between two pairs of arrays.
        
        Args:
            self: The instance of the '_DualAlgebraImpl' class.
            a_x: A NumPy array representing the coefficients for the first axis of the first matrix.
            a_y: A NumPy array representing the coefficients for the second axis of the first matrix.
            b_x: A NumPy array representing the coefficients for the first axis of the second matrix.
            b_y: A NumPy array representing the coefficients for the second axis of the second matrix.
        
        Returns:
            A tuple containing two NumPy arrays:
            1. The result of the matrix multiplication between a_x and b_x.
            2. The sum of the matrix multiplication between a_x and b_y, and the matrix multiplication between a_y and b_x.
        
        Raises:
            - No specific exceptions are explicitly raised by this method.
        """
        return (
            a_x @ b_x,
            a_x @ b_y + a_y @ b_x
        )

    def dot_product(self,
                    a_x: np.ndarray,
                    a_y: np.ndarray,
                    b_x: np.ndarray,
                    b_y: np.ndarray) -> Tuple[np.float64, np.float64]:

        r"""
        Calculates the dot product of two vectors in a dual algebra.
        
        Args:
            self: An instance of the _DualAlgebraImpl class.
            a_x (np.ndarray): The x-component of the first vector.
            a_y (np.ndarray): The y-component of the first vector.
            b_x (np.ndarray): The x-component of the second vector.
            b_y (np.ndarray): The y-component of the second vector.
        
        Returns:
            Tuple[np.float64, np.float64]: A tuple containing two values.
            - The first value is the dot product of the x-components of the two vectors.
            - The second value is the dot product of the x-component of the first vector with the y-component of the second vector,
              plus the dot product of the y-component of the first vector with the x-component of the second vector.
        
        Raises:
            None.
        
        """
        return (
            np.float64(np.dot(a_x, b_x)),
            np.float64(np.dot(a_x, b_y) + np.dot(a_y, b_x))
        )

    def sqrt(self, x: np.float64, y: np.float64) -> Tuple[np.float64, np.float64]:

        r""" 
        Calculate the square root of a number and its division with another number.
        
        Args:
            self: The instance of the _DualAlgebraImpl class.
            x (np.float64): The number for which the square root will be calculated.
            y (np.float64): The number which will be divided by x and then by 2.
        
        Returns:
            Tuple[np.float64, np.float64]: A tuple containing the square root of x and the result of y divided by x and then by 2.
        
        Raises:
            None.
        """
        x_sqrt = np.sqrt(x)
        y_sqrt = np.divide(y, x) / 2
        return x_sqrt, y_sqrt

    def special_element(self) -> str:

        r"""
        This method 'special_element' in the class '_DualAlgebraImpl' returns a special element 'ε'.
        
        Args:
            self: (_DualAlgebraImpl) The instance of the class invoking the method.
        
        Returns:
            str: The special element 'ε' as a string.
        
        Raises:
            None
        """
        return u'\u03b5'

    def visit(self, scalar, visitor, *args, **kwargs) -> None:

        r"""
        This method 'visit' in the class '_DualAlgebraImpl' processes a scalar value by invoking a corresponding method in the provided visitor object.
        
        Args:
            self (object): The instance of the '_DualAlgebraImpl' class.
            scalar (any): The scalar value to be processed.
            visitor (object): An object that implements the 'visit_dual' method to handle the scalar value processing.
            
        Returns:
            None. This method does not return any value.
        
        Raises:
            No specific exceptions are raised within this method. However, exceptions may be raised by the 'visit_dual' method of the 'visitor' object.
        """
        visitor.visit_dual(scalar, *args, **kwargs)

    def mul_scalar(self,
                   x1: np.float64,
                   y1: np.float64,
                   x2: np.float64,
                   y2: np.float64) -> Tuple[np.float64, np.float64]:

        r"""
        Multiplies two scalar values and returns the result as a tuple.
        
        Args:
            self: An instance of the '_DualAlgebraImpl' class.
            x1: A scalar value of type 'np.float64'. Represents the first component of the first scalar.
            y1: A scalar value of type 'np.float64'. Represents the second component of the first scalar.
            x2: A scalar value of type 'np.float64'. Represents the first component of the second scalar.
            y2: A scalar value of type 'np.float64'. Represents the second component of the second scalar.
        
        Returns:
            A tuple of two scalar values of type 'np.float64'.
            - The first element represents the result of multiplying the first component of the first scalar with the first component of the second scalar.
            - The second element represents the result of multiplying the first component of the first scalar with the second component of the second scalar, 
              and adding it to the result of multiplying the second component of the first scalar with the first component of the second scalar.
        
        Raises:
            None.
        
        """
        return (
            np.float64(x1 * x2),
            np.float64(x1 * y2 + y1 * x2)
        )

    def div_scalar(self,
                   x1: np.float64,
                   y1: np.float64,
                   x2: np.float64,
                   y2: np.float64) -> Tuple[np.float64, np.float64]:

        r"""
        div_scalar(self, x1: np.float64, y1: np.float64, x2: np.float64, y2: np.float64) -> Tuple[np.float64, np.float64]
        
        Divides two scalar values and performs a mathematical operation on them.
        
        Args:
            self (object): The instance of the '_DualAlgebraImpl' class.
            x1 (np.float64): The first scalar value to be divided.
            y1 (np.float64): The second scalar value to be divided.
            x2 (np.float64): The third scalar value to be divided.
            y2 (np.float64): The fourth scalar value to be divided.
        
        Returns:
            Tuple[np.float64, np.float64]: A tuple containing two np.float64 values - the result of the division operation between 'x1' and 'x2', and the result of the mathematical operation performed on 'y1',
'x2', 'x1', and 'y2'.
        
        Raises:
            None.
        
        Note:
            The division operation is performed by dividing 'x1' by 'x2'.
            The mathematical operation is performed by subtracting the product of 'y1' and 'x2' from the product of 'x1' and 'y2', and then dividing the result by the square of 'x2'.
        """
        return (
            np.float64(x1 / x2),
            (y1 * x2 - x1 * y2) / np.square(x2)
        )
