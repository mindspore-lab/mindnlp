from abc import ABC, abstractmethod
from typing import Tuple, Union
import numpy as np


class _MathObjImpl(ABC):

    r"""
    Mathematical operations implementation class.
    
    This class serves as a template for implementing mathematical operations such as multiplication and division. 
    It defines abstract methods mul() and div() which must be implemented by subclasses to perform the actual calculations.
    
    Inherits from ABC (Abstract Base Class) to define abstract methods that subclasses must implement.
    """

    @abstractmethod
    def mul(self,
            a_x: np.ndarray,
            a_y: np.ndarray,
            b_x: Union[np.ndarray, np.float64],
            b_y: Union[np.ndarray, np.float64]) -> Tuple[np.ndarray, np.ndarray]:
        r"""
        This method performs multiplication of given input arrays and scalar values.
        
        Args:
            self: The instance of the _MathObjImpl class.
            a_x (np.ndarray): The input array for x-coordinates.
            a_y (np.ndarray): The input array for y-coordinates.
            b_x (Union[np.ndarray, np.float64]): The scalar value or array for x-coordinates.
            b_y (Union[np.ndarray, np.float64]): The scalar value or array for y-coordinates.
        
        Returns:
            Tuple[np.ndarray, np.ndarray]: A tuple containing two numpy arrays representing the result of the multiplication for x and y coordinates.
        
        Raises:
            - ValueError: If the input arrays and scalar values are not compatible for multiplication.
            - TypeError: If the input parameters are not of the expected types.
            - Any other exceptions specific to the numpy operations performed within the method.
        """
        pass

    @abstractmethod
    def div(self,
            a_x: np.ndarray,
            a_y: np.ndarray,
            b_x: Union[np.ndarray, np.float64],
            b_y: Union[np.ndarray, np.float64]) -> Tuple[np.ndarray, np.ndarray]:
        r"""
        This method performs division on the input arrays and returns the result as a tuple of numpy arrays.
        
        Args:
            self: The instance of the _MathObjImpl class.
            a_x: An input numpy array representing the x-coordinates of the dividend.
            a_y: An input numpy array representing the y-coordinates of the dividend.
            b_x: An input numpy array or a scalar of type np.float64 representing the x-coordinate of the divisor.
            b_y: An input numpy array or a scalar of type np.float64 representing the y-coordinate of the divisor.
        
        Returns:
            A tuple of two numpy arrays representing the x and y coordinates of the division result.
        
        Raises:
            This method does not raise any specific exceptions.
        """
        pass
