from typing import Union
import numpy as np
from mindnlp._legacy.hypercomplex.tensor_decomposition.linal.basics.scalar import Scalar as AbstractScalar


class Scalar(AbstractScalar):

    r"""
    Represents a scalar value in a mathematical context. This class provides methods to perform arithmetic operations and transformations on scalar values.
    
    Methods:
    - __init__(self, x: Union[float, np.float64]) -> None: Initializes a Scalar object with the specified value.
    - __str__(self) -> str: Returns a string representation of the Scalar object.
    - __neg__(self) -> 'Scalar': Returns the negation of the Scalar object.
    - __add__(self, that: 'Scalar') -> 'Scalar': Adds two Scalar objects together.
    - __sub__(self, that: 'Scalar') -> 'Scalar': Subtracts one Scalar object from another.
    - __mul__(self, that: 'Scalar') -> 'Scalar': Multiplies two Scalar objects.
    - __truediv__(self, that: 'Scalar') -> 'Scalar': Divides one Scalar object by another.
    - one() -> 'Scalar': Returns a Scalar object with a value of 1.0.
    - zero() -> 'Scalar': Returns a Scalar object with a value of 0.0.
    - sqrt(self) -> 'Scalar': Returns the square root of the Scalar object.
    - as_array(self) -> np.ndarray: Returns the Scalar value as a NumPy array.
    - as_number(self) -> np.float64: Returns the Scalar value as a NumPy float64 number.
    - is_zero(self) -> bool: Checks if the Scalar value is close to zero within a tolerance.
    - get_real(self) -> np.float64: Returns the real part of the Scalar value.
    - visit(self, visitor, *args, **kwargs) -> None: Allows visiting the Scalar value with a visitor object.
    - _make_like(self, val: np.float64) -> 'Scalar': Creates a new Scalar object similar to the given value.
    - _zero_like(self) -> 'Scalar': Creates a Scalar object with a value of zero.
    - _one_like(self) -> 'Scalar': Creates a Scalar object with a value of one.
    """

    def __init__(self, x: Union[float, np.float64]) -> None:

        r"""
        Initializes a new instance of the Scalar class.
        
        Args:
            self: The instance of the Scalar class.
            x (Union[float, np.float64]): The value to be assigned to the _x attribute of the Scalar instance. It can be either a float or an np.float64.
        
        Returns:
            None. This method does not return any value.
        
        Raises:
            N/A
        """
        self._x = np.float64(x)

    def __str__(self) -> str:

        r"""
        Converts the Scalar object to a string representation.
        
        Args:
            self (Scalar): The instance of the Scalar class.
        
        Returns:
            str: A string representation of the Scalar object.
        
        Raises:
            None.
        """
        return str(self._x)

    def __neg__(self) -> 'Scalar':

        r"""
        Method '__neg__' in the class 'Scalar'.
        
        Args:
            self (Scalar): The instance of the Scalar class on which the negation operation is performed.
                This parameter is automatically passed when the method is called.
                It represents the scalar value to be negated.
        
        Returns:
            Scalar: Returns a new Scalar object that is the negation of the original scalar value.
                The negation operation changes the sign of the scalar value.
        
        Raises:
            No specific exceptions are documented to be raised by this method.
        """
        return Scalar(-self._x)

    def __add__(self, that: 'Scalar') -> 'Scalar':

        r"""
        Method '__add__' in the class 'Scalar' adds the value of 'that' to the value of 'self' and returns a new instance of 'Scalar'.
        
        Args:
            self (Scalar): The current instance of the 'Scalar' class.
            that (Scalar): Another instance of the 'Scalar' class that will be added to 'self'.
        
        Returns:
            Scalar: A new instance of the 'Scalar' class, representing the sum of 'self' and 'that'.
        
        Raises:
            None.
        """
        return Scalar(self._x + that._x)

    def __sub__(self, that: 'Scalar') -> 'Scalar':

        r"""
        __sub__
        
        Method in the class named 'Scalar' that performs subtraction operation.
        
        Args:
            self (Scalar): The current instance of Scalar.
            that (Scalar): The Scalar object to be subtracted from the current instance.
        
        Returns:
            Scalar: A new Scalar object resulting from the subtraction operation.
        
        Raises:
            TypeError: If the 'that' parameter is not of type Scalar.
        """
        return Scalar(self._x - that._x)

    def __mul__(self, that: 'Scalar') -> 'Scalar':

        r"""
        Multiply two Scalar objects.
        
        Args:
            self (Scalar): The first Scalar object.
                This parameter represents the current Scalar object on which the method is called.
            that (Scalar): The second Scalar object.
                This parameter represents the Scalar object that will be multiplied with the current Scalar object.
        
        Returns:
            Scalar: A new Scalar object representing the result of the multiplication.
                The new Scalar object will have a value equal to the product of the two input Scalar objects.
        
        Raises:
            None.
        
        Note:
            The multiplication operation is performed by multiplying the values of the two Scalar objects.
            The result is returned as a new Scalar object.
        """
        return Scalar(self._x * that._x)

    def __truediv__(self, that: 'Scalar') -> 'Scalar':

        r"""
        Performs the true division operation on two Scalar objects.
        
        Args:
            self (Scalar): The first Scalar object.
            that (Scalar): The second Scalar object.
        
        Returns:
            Scalar: A new Scalar object representing the result of the true division operation.
        
        Raises:
            TypeError: If either self or that is not an instance of the Scalar class.
        
        This method divides the value of the first Scalar object (self) by the value of the second Scalar object (that). The true division operation returns a new Scalar object that represents the result of the division.
        
        Note that both self and that must be instances of the Scalar class. If either of them is not, a TypeError is raised.
        
        Example:
            s1 = Scalar(10)
            s2 = Scalar(2)
            result = s1.__truediv__(s2)
            # result will be a new Scalar object with the value 5
        """
        return Scalar(self._x / that._x)

    @staticmethod
    def one() -> 'Scalar':

        r"""
        This method, named 'one', is a static method belonging to the 'Scalar' class.
        
        Args:
            This method does not take any parameters.
        
        Returns:
            Returns a value of type 'Scalar'. The returned value is a scalar object representing the value 1.0.
        
        Raises:
            This method does not raise any exceptions.
        """
        return Scalar(1.)

    @staticmethod
    def zero() -> 'Scalar':

        r"""
        Method 'zero' in the class 'Scalar' returns a Scalar object with a value of 0.0.
        
        Args:
            No parameters.
        
        Returns:
            Scalar: A Scalar object with a value of 0.0.
        
        Raises:
            No exceptions are raised by this method.
        """
        return Scalar(0.)

    def sqrt(self) -> 'Scalar':

        r"""
        This method calculates the square root of the value contained within the Scalar object.
        
        Args:
            self (Scalar): The Scalar object for which the square root will be calculated.
        
        Returns:
            Scalar: A new Scalar object containing the square root of the original value.
        
        Raises:
            None
        """
        return Scalar(np.sqrt(self._x))

    def as_array(self) -> np.ndarray:

        r"""
        Converts the scalar value to a NumPy array.
        
        Args:
            self (Scalar): An instance of the Scalar class.
                The scalar value to be converted into a NumPy array.
        
        Returns:
            np.ndarray: A NumPy array representation of the scalar value.
                The array is of type np.float64.
        
        Raises:
            None.
        
        Example:
            >>> s = Scalar(3.14)
            >>> s.as_array()
            array([3.14])
        
        Note:
            The dtype of the resulting NumPy array is np.float64 to ensure
            consistency with the scalar value.
        """
        return np.array(self._x, dtype=np.float64)

    def as_number(self) -> np.float64:

        r"""
        Method to convert the scalar value to a numpy float64 number.
        
        Args:
            self (Scalar): The instance of the Scalar class.
                Represents the scalar value that needs to be converted to a numpy float64 number.
        
        Returns:
            np.float64: Returns the scalar value converted to a numpy float64 number.
        
        Raises:
            None
        """
        return self._x

    def is_zero(self) -> bool:

        r"""
        Check if the Scalar object is approximately zero.
        
        Args:
            self (Scalar): The Scalar object to be checked for being zero.
                This parameter is required and represents the instance of the Scalar class.
        
        Returns:
            bool: Returns True if the value of the Scalar object is within the equality tolerance range defined by the class attributes Scalar._EQUALITY_TOLERANCE.
                Returns False if the value of the Scalar object is outside the equality tolerance range.
        
        Raises:
            None
        """
        return -Scalar._EQUALITY_TOLERANCE < self._x < Scalar._EQUALITY_TOLERANCE

    def get_real(self) -> np.float64:

        r"""
        This method returns the real value of a scalar.
        
        Args:
            self: An instance of the Scalar class.
        
        Returns:
            np.float64: The real value of the scalar.
        
        Raises:
            None.
        
        This method retrieves the real value of the scalar object and returns it as a np.float64 data type.
        """
        return self._x

    def visit(self, visitor, *args, **kwargs) -> None:

        r"""
        This method 'visit' is defined in the class 'Scalar' and is used to accept a visitor and invoke the visitor's method 'visit_real' with the current instance as the first argument.
        
        Args:
            self (Scalar): The current instance of the Scalar class.
            visitor (object): The visitor object that will be used to invoke the 'visit_real' method.
        
        Returns:
            None: This method does not return any value.
        
        Raises:
            No specific exceptions are documented for this method.
        """
        visitor.visit_real(self, *args, **kwargs)

    def _make_like(self, val: np.float64) -> 'Scalar':

        r"""
        Method _make_like in class Scalar.
        
        Args:
            self: The instance of the class Scalar.
            val (np.float64): The value used to create a new instance of Scalar. Should be of type np.float64.
        
        Returns:
            Scalar: A new instance of Scalar created with the provided value.
        
        Raises:
            None.
        """
        return Scalar(val)

    def _zero_like(self) -> 'Scalar':

        r"""
        Returns a new instance of 'Scalar' with all values set to zero.
        
        Args:
            self: The current instance of the 'Scalar' class.
        
        Returns:
            A new instance of 'Scalar' with all values set to zero.
        
        Raises:
            None.
        """
        return Scalar.zero()

    def _one_like(self) -> 'Scalar':

        r"""
        Method _one_like in class Scalar.
        
        Args:
            self: The instance of the class on which the method is called.
                Type: Scalar.
                Purpose: Represents the current instance of the Scalar class.
                Restrictions: None.
        
        Returns:
            Scalar: Returns a Scalar instance representing the value one.
        
        Raises:
            None.
        """
        return Scalar.one()
