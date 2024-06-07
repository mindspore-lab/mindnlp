from abc import ABC, abstractmethod
from typing import Any
import numpy as np


class Scalar(ABC):

    r"""
    Represents a scalar value and provides abstract methods for mathematical operations and transformations.
    
    This class inherits from ABC and defines abstract methods for arithmetic operations, square root, visitor pattern, creating new instances with the same type, getting the scalar as a NumPy array, checking
if the scalar is zero, and getting the real value as a NumPy float64.
    
    Subclasses of 'Scalar' must implement these abstract methods to define the behavior of scalar values in their specific contexts.
    """
    _EQUALITY_TOLERANCE: float = 1e-10

    @staticmethod
    def make_like(m: 'Scalar', val: Any) -> 'Scalar':
        r"""
        This method creates a new 'Scalar' object by utilizing the '_make_like' method of the 'Scalar' class.
        
        Args:
            m (Scalar): An instance of the 'Scalar' class that serves as a template for creating the new object.
                        It is used to determine the structure and properties of the new object.
            val (Any): The value that will be used to initialize the new 'Scalar' object. It can be of any data type.
        
        Returns:
            Scalar: A new 'Scalar' object that is created based on the template provided by the 'm' parameter and initialized with the 'val' parameter.
        
        Raises:
            This method does not explicitly raise any exceptions. However, exceptions may be raised if the '_make_like' method called internally on the 'm' object raises any exceptions.
        """
        return m._make_like(val)

    @staticmethod
    def zero_like(m: 'Scalar') -> 'Scalar':
        r"""
        This method creates a new instance of the class 'Scalar' with values initialized to zero, based on the provided 'Scalar' instance 'm'.
        
        Args:
            m (Scalar): The input 'Scalar' instance from which the new instance will be created with values initialized to zero.
        
        Returns:
            Scalar: A new instance of the class 'Scalar' with values set to zero.
        
        Raises:
            This method does not raise any exceptions.
        """
        return m._zero_like()

    @staticmethod
    def one_like(m: 'Scalar') -> 'Scalar':
        r"""
        Method 'one_like' in class 'Scalar' creates a new instance of 'Scalar' object with the same shape and type as the input.
        
        Args:
            m (Scalar): The input 'Scalar' object for which we want to create a new instance with the same shape and type. It is of type 'Scalar'.
        
        Returns:
            Scalar: A new 'Scalar' object that is created to be identical in shape and type to the input 'Scalar' object.
        
        Raises:
            This method does not raise any exceptions.
        """
        return m._one_like()

    @abstractmethod
    def __neg__(self) -> 'Scalar':
        r"""
        This method '__neg__' in the class 'Scalar' implements the negation operation for a scalar object.
        
        Args:
            self (Scalar): The instance of the Scalar class on which the negation operation will be performed.
        
        Returns:
            Scalar: A new instance of the Scalar class representing the negated value of the original scalar.
        
        Raises:
            This method does not raise any exceptions.
        """
        pass

    @abstractmethod
    def __add__(self, that: 'Scalar') -> 'Scalar':
        r"""
        __add__
        
        This method is used to perform addition on two Scalar objects.
        
        Args:
            self (Scalar): The first Scalar object participating in the addition operation.
            that (Scalar): The second Scalar object to be added to the first Scalar.
        
        Returns:
            Scalar: The result of the addition operation, which is a new Scalar object.
        
        Raises:
            - TypeError: If the 'that' parameter is not of type Scalar, a TypeError is raised.
        """
        pass

    @abstractmethod
    def __sub__(self, that: 'Scalar') -> 'Scalar':
        r"""
        This method performs subtraction between two Scalar objects.
        
        Args:
            self (Scalar): The current Scalar object on which the subtraction operation is being performed.
            that (Scalar): The Scalar object to be subtracted from the current object.
        
        Returns:
            Scalar: A new Scalar object resulting from the subtraction operation.
        
        Raises:
            NotImplementedError: If the method is not implemented in a subclass that inherits from Scalar.
        """
        pass

    @abstractmethod
    def __mul__(self, that: 'Scalar') -> 'Scalar':
        r"""
        Multiplies two instances of the 'Scalar' class.
        
        Args:
            self (Scalar): The first 'Scalar' instance to be multiplied.
            that (Scalar): The second 'Scalar' instance to be multiplied.
        
        Returns:
            Scalar: A new 'Scalar' instance resulting from the multiplication of the two input instances.
        
        Raises:
            None.
        
        This method multiplies two 'Scalar' instances together and returns a new 'Scalar' instance as the result.
        The 'self' parameter represents the first 'Scalar' instance to be multiplied, while the 'that' parameter represents the second 'Scalar' instance.
        Both 'self' and 'that' should be instances of the 'Scalar' class.
        The method does not modify any of the input instances and returns a new 'Scalar' instance that represents the product of the two input instances.
        """
        pass

    @abstractmethod
    def __truediv__(self, that: 'Scalar') -> 'Scalar':
        r"""
        This method performs the true division operation on the 'self' object and the 'that' object of type 'Scalar'.
        
        Args:
            self (Scalar): The current Scalar object on which the true division operation is performed.
            that (Scalar): The Scalar object to be divided from the current Scalar object.
        
        Returns:
            Scalar: The result of the true division operation, which is a new Scalar object.
        
        Raises:
            TypeError: If the 'that' parameter is not of type Scalar.
            ZeroDivisionError: If the 'that' parameter is zero, causing division by zero.
        """
        pass

    @abstractmethod
    def sqrt(self) -> 'Scalar':
        r"""
        Calculate the square root of the Scalar object.
        
        Args:
            self: Represents the instance of the Scalar class.
            
        Returns:
            Scalar: A Scalar object representing the square root of the original Scalar value.
        
        Raises:
            This method does not raise any exceptions.
        """
        pass

    @abstractmethod
    def visit(self, visitor, *args, **kwargs) -> None:
        r"""
        This method 'visit' in the class 'Scalar' is an abstract method that must be implemented by subclasses. It is used to allow a visitor to interact with the Scalar object.
        
        Args:
            self (Scalar): The instance of the Scalar class invoking the method.
            visitor: The visitor object that will interact with the Scalar object. It is expected to have specific methods to handle the visitation.
        
        Returns:
            None: This method does not return any value.
        
        Raises:
            This method is an abstract method and does not raise any exceptions itself. However, subclasses implementing this method may raise exceptions based on their specific implementations.
        """
        pass

    @abstractmethod
    def _make_like(self, val: Any) -> 'Scalar':
        r"""
        Make a new instance of the 'Scalar' class with the same attributes as the current instance, except for the 'val' attribute which will be updated to the specified value.
        
        Args:
            self (Scalar): The current instance of the 'Scalar' class.
            val (Any): The new value to be assigned to the 'val' attribute of the new instance. 
        
        Returns:
            Scalar: A new instance of the 'Scalar' class with the same attributes as the current instance, except for the 'val' attribute which is updated to the specified value.
        
        Raises:
            None.
        
        """
        pass

    @abstractmethod
    def _zero_like(self) -> 'Scalar':
        r"""
        This method creates a new instance of 'Scalar' with zero values.
        
        Args:
            self: The instance of the class 'Scalar' on which this method is called.
        
        Returns:
            'Scalar': A new instance of 'Scalar' with zero values.
        
        Raises:
            None
        """
        pass

    @abstractmethod
    def _one_like(self) -> 'Scalar':
        r"""
        This method, _one_like, returns a value of type 'Scalar'.
        
        Args:
            self (object): The instance of the Scalar class.
                Purpose: Represents the current instance of the Scalar class.
                Restrictions: N/A
        
        Returns:
            Scalar: The return value represents a Scalar type object.
                Purpose: The value returned by the method.
        
        Raises:
            N/A
        """
        pass

    @abstractmethod
    def as_array(self) -> np.ndarray:
        r"""
        Converts the scalar value to a NumPy array.
        
        Args:
            self (Scalar): The instance of the Scalar class.
        
        Returns:
            np.ndarray: A NumPy array representing the scalar value.
        
        Raises:
            None.
        
        Examples:
            >>> scalar = Scalar()
            >>> scalar.value = 5
            >>> scalar.as_array()
            array(5)
        
        Note:
            The as_array method converts a scalar value to a NumPy array. This can be useful when performing mathematical operations or when working with libraries that require NumPy arrays as input.
        """
        pass

    @abstractmethod
    def is_zero(self) -> bool:
        r"""
        Method to check if the Scalar object is zero.
        
        Args:
            self (Scalar): The instance of the Scalar class.
                Represents the Scalar object for which the zero check is performed.
                Must be an instance of the Scalar class.
        
        Returns:
            bool: Returns a boolean value indicating whether the Scalar object is zero.
                True if the Scalar object is zero, False otherwise.
        
        Raises:
            This method does not raise any exceptions.
        """
        pass

    @abstractmethod
    def get_real(self) -> np.float64:
        r"""
        Returns the real part of the scalar value.
        
        Args:
            self (Scalar): The instance of the Scalar class.
        
        Returns:
            np.float64: The real part of the scalar value.
        
        Raises:
            None.
        """
        pass
