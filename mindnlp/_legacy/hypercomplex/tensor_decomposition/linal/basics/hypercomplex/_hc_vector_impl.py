from abc import ABC, abstractmethod
from typing import Tuple
import numpy as np
from mindnlp._legacy.hypercomplex.tensor_decomposition.linal.basics.hypercomplex._hc_math_obj_impl import _MathObjImpl


class _VectorImpl(_MathObjImpl, ABC):

    r"""
    This class represents an implementation of a vector object. It inherits from the _MathObjImpl class and is an abstract base class (ABC).
    
    The _VectorImpl class provides functionality for performing operations on vectors, such as calculating the dot product between two vectors.
    
    Attributes:
        None
    
    Methods:
        dot_product(a_x: np.ndarray, a_y: np.ndarray, b_x: np.ndarray, b_y: np.ndarray) -> Tuple[np.float64, np.float64]:
            Calculates the dot product between two vectors defined by their x and y components.
            Args:
                a_x (np.ndarray): The x component of the first vector.
                a_y (np.ndarray): The y component of the first vector.
                b_x (np.ndarray): The x component of the second vector.
                b_y (np.ndarray): The y component of the second vector.
            Returns:
                Tuple[np.float64, np.float64]: A tuple containing the dot product of the two vectors.
        
        Note:
            This method is an abstract method and must be implemented by any concrete subclasses of _VectorImpl.
    """
    @abstractmethod
    def dot_product(self,
                    a_x: np.ndarray,
                    a_y: np.ndarray,
                    b_x: np.ndarray,
                    b_y: np.ndarray) -> Tuple[np.float64, np.float64]:
        r"""
        Method to calculate the dot product of two vectors.
        
        Args:
            self (_VectorImpl): An instance of the _VectorImpl class.
            a_x (np.ndarray): The x-coordinates of the first vector.
            a_y (np.ndarray): The y-coordinates of the first vector.
            b_x (np.ndarray): The x-coordinates of the second vector.
            b_y (np.ndarray): The y-coordinates of the second vector.
        
        Returns:
            Tuple[np.float64, np.float64]: A tuple containing the dot product values for the two vectors.
        
        Raises:
            None.
        """
        pass
