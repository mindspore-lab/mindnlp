from typing import Tuple, Union
import numpy as np
from mindnlp._legacy.hypercomplex.tensor_decomposition.linal.basics.hypercomplex._hc_matrix_impl import _MatrixImpl
from mindnlp._legacy.hypercomplex.tensor_decomposition.linal.basics.hypercomplex._hc_scalar_impl import _ScalarImpl
from mindnlp._legacy.hypercomplex.tensor_decomposition.linal.basics.hypercomplex._hc_vector_impl import _VectorImpl


class _ComplexAlgebraImpl(_MatrixImpl, _VectorImpl, _ScalarImpl):

    """
    Represents a class for implementing complex algebra operations.
    
    This class inherits from _MatrixImpl, _VectorImpl, _ScalarImpl and provides methods for performing various complex algebra operations 
    such as multiplication, division, matrix multiplication, dot product, square root, handling special elements, visiting complex numbers, 
    and operations with scalar values.
    
    The class includes methods for:
    - Multiplying complex numbers
    - Dividing complex numbers
    - Performing matrix multiplication with complex numbers
    - Calculating the dot product of complex vectors
    - Calculating the square root of a complex number
    - Returning a special element 'i'
    - Visiting complex numbers with a visitor pattern
    - Multiplying and dividing complex numbers by scalar values
    
    For detailed information on each method's parameters and return values, refer to the method docstrings within the class.
    """
    def mul(self,
            a_x: np.ndarray,
            a_y: np.ndarray,
            b_x: Union[np.ndarray, np.float64],
            b_y: Union[np.ndarray, np.float64]) -> Tuple[np.ndarray, np.ndarray]:
        r"""
        Performs complex multiplication of two numbers.
        
        Args:
            self: The instance of the '_ComplexAlgebraImpl' class.
            a_x (np.ndarray): A numpy array representing the real part of the first complex number.
            a_y (np.ndarray): A numpy array representing the imaginary part of the first complex number.
            b_x (Union[np.ndarray, np.float64]): A numpy array or a float representing the real part of the second complex number.
            b_y (Union[np.ndarray, np.float64]): A numpy array or a float representing the imaginary part of the second complex number.
        
        Returns:
            Tuple[np.ndarray, np.ndarray]: A tuple containing two numpy arrays. The first array represents the real part of the result, and the second array represents the imaginary part of the result.
        
        Raises:
            None.
        
        Note:
            The method assumes that both a_x and a_y have the same shape, and b_x and b_y also have the same shape. If b_x or b_y is a float, it will be broadcasted to match the shape of a_x and a_y.
        
        Example:
            obj = _ComplexAlgebraImpl()
            result_real, result_imag = obj.mul(a_x, a_y, b_x, b_y)
        """
        return (
            a_x * b_x - a_y * b_y,
            a_x * b_y + a_y * b_x
        )

    def div(self,
            a_x: np.ndarray,
            a_y: np.ndarray,
            b_x: Union[np.ndarray, np.float64],
            b_y: Union[np.ndarray, np.float64]) -> Tuple[np.ndarray, np.ndarray]:
        r"""
        Performs division of complex numbers based on the given inputs.
        
        Args:
            self: An instance of the '_ComplexAlgebraImpl' class.
            a_x: An array of type 'np.ndarray' representing the real part of the dividend complex number(s).
            a_y: An array of type 'np.ndarray' representing the imaginary part of the dividend complex number(s).
            b_x: Either an array of type 'np.ndarray' or a single value of type 'np.float64' representing the real part of the divisor complex number(s).
            b_y: Either an array of type 'np.ndarray' or a single value of type 'np.float64' representing the imaginary part of the divisor complex number(s).
        
        Returns:
            A tuple of two 'np.ndarray' arrays representing the real and imaginary parts of the result obtained by dividing the dividend complex number(s) by the divisor complex number(s).
        
        Raises:
            - TypeError: If any of the following conditions are not met:
                - 'a_x', 'a_y', 'b_x', and 'b_y' are not of type 'np.ndarray'.
                - 'b_x' or 'b_y' is not of type 'np.ndarray' or 'np.float64'.
            - ValueError: If the shapes of the arrays 'a_x', 'a_y', 'b_x', and 'b_y' are not compatible for element-wise division.
            - ZeroDivisionError: If any of the divisor complex number(s) has a square norm of zero, i.e., if any of the elements of 'b_x' or 'b_y' squared are zero.
        
        Note:
            Division of complex numbers is performed using the following formula:
                - real part of the result = (a_x * b_x + a_y * b_y) / (b_x^2 + b_y^2)
                - imaginary part of the result = (a_y * b_x - a_x * b_y) / (b_x^2 + b_y^2)
            The square norm is calculated as the sum of squares of the real and imaginary parts of the divisor complex number(s).
        """
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
        r"""
        Performs matrix multiplication for complex algebra.
        
        Args:
            self: An instance of the '_ComplexAlgebraImpl' class.
            a_x (np.ndarray): A 2D array representing the real part of matrix A.
            a_y (np.ndarray): A 2D array representing the imaginary part of matrix A.
            b_x (np.ndarray): A 2D array representing the real part of matrix B.
            b_y (np.ndarray): A 2D array representing the imaginary part of matrix B.
        
        Returns:
            Tuple[np.ndarray, np.ndarray]: A tuple containing two 2D arrays resulting from the matrix multiplication.
                The first array is the real part of the resulting matrix and the second array is the imaginary part.
        
        Raises:
            None.
        """
        return (
            a_x @ b_x - a_y @ b_y,
            a_x @ b_y + a_y @ b_x
        )

    def dot_product(self,
                    a_x: np.ndarray,
                    a_y: np.ndarray,
                    b_x: np.ndarray,
                    b_y: np.ndarray) -> Tuple[np.float64, np.float64]:
        r"""
        Calculates the dot product of two complex vectors.
        
        Args:
            self: An instance of the '_ComplexAlgebraImpl' class.
            a_x: An array of type np.ndarray representing the real part of vector 'a'.
            a_y: An array of type np.ndarray representing the imaginary part of vector 'a'.
            b_x: An array of type np.ndarray representing the real part of vector 'b'.
            b_y: An array of type np.ndarray representing the imaginary part of vector 'b'.
        
        Returns:
            A tuple of type 'Tuple[np.float64, np.float64]' containing the real and imaginary parts of the dot product.
        
        Raises:
            None.
        
        Note:
            The dot product of two complex vectors 'a' and 'b' is calculated as follows:
            - The real part of the dot product is obtained by subtracting the product of the real parts of 'a' and 'b' from the product of the imaginary parts of 'a' and 'b'.
            - The imaginary part of the dot product is obtained by adding the product of the real part of 'a' and the imaginary part of 'b' to the product of the imaginary part of 'a' and the real part of 'b'.
        """
        return (
            np.float64(np.dot(a_x, b_x) - np.dot(a_y, b_y)),
            np.float64(np.dot(a_x, b_y) + np.dot(a_y, b_x))
        )

    def sqrt(self, x: np.float64, y: np.float64) -> Tuple[np.float64, np.float64]:
        r"""
        This method calculates the square root of a complex number given its real and imaginary parts.
        
        Args:
            self: Instance of the _ComplexAlgebraImpl class.
            x (np.float64): The real part of the complex number.
            y (np.float64): The imaginary part of the complex number.
        
        Returns:
            Tuple[np.float64, np.float64]: A tuple containing the square root of the complex number, where the first element represents the real part and the second element represents the imaginary part.
        
        Raises:
            None
        """
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
        r"""
        This method returns a special element as a string.
        
        Args:
            self: An instance of the _ComplexAlgebraImpl class. This parameter is required to access the method within the class.
        
        Returns:
            str: A string representing the special element.
        
        Raises:
            This method does not raise any exceptions.
        """
        return 'i'

    def visit(self, scalar, visitor, *args, **kwargs) -> None:
        r"""
        This method 'visit' in the class '_ComplexAlgebraImpl' is used to perform a visit operation on a scalar value using a visitor object.
        
        Args:
            self (_ComplexAlgebraImpl): The instance of the _ComplexAlgebraImpl class.
            scalar (Any): The scalar value to be visited.
            visitor (Visitor): The visitor object used to perform the visit operation on the scalar.
            
        Returns:
            None: This method does not return any value.
        
        Raises:
            No specific exceptions are raised by this method under normal circumstances.
        """
        visitor.visit_complex(scalar, *args, **kwargs)

    def mul_scalar(self,
                   x1: np.float64,
                   y1: np.float64,
                   x2: np.float64,
                   y2: np.float64) -> Tuple[np.float64, np.float64]:
        """
        Performs scalar multiplication of two complex numbers.
        
        Args:
            self: The instance of the _ComplexAlgebraImpl class.
            x1 (np.float64): The real part of the first complex number.
            y1 (np.float64): The imaginary part of the first complex number.
            x2 (np.float64): The real part of the second complex number.
            y2 (np.float64): The imaginary part of the second complex number.
        
        Returns:
            Tuple[np.float64, np.float64]: A tuple containing the real and imaginary parts of the resulting complex number.
        
        Raises:
            None
        """
        return (
            np.float64(x1 * x2 - y1 * y2),
            np.float64(x1 * y2 + y1 * x2)
        )

    def div_scalar(self,
                   x1: np.float64,
                   y1: np.float64,
                   x2: np.float64,
                   y2: np.float64) -> Tuple[np.float64, np.float64]:
        r"""
        The method 'div_scalar' in the class '_ComplexAlgebraImpl' divides the complex number represented by (x1, y1) by the complex number represented by (x2, y2).
        
        Args:
            self: The instance of the class '_ComplexAlgebraImpl'.
            x1 (np.float64): The real part of the first complex number.
            y1 (np.float64): The imaginary part of the first complex number.
            x2 (np.float64): The real part of the second complex number.
            y2 (np.float64): The imaginary part of the second complex number.
        
        Returns:
            Tuple[np.float64, np.float64]: A tuple containing the real and imaginary parts of the result of dividing the first complex number by the second complex number.
        
        Raises:
            This method does not explicitly raise any exceptions.
        """
        square_norm = np.square(x2) + np.square(y2)
        return (
            (x1 * x2 + y1 * y2) / square_norm,
            (y1 * x2 - x1 * y2) / square_norm
        )
