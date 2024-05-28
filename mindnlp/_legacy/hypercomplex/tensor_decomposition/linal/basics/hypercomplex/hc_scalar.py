from typing import TypeVar, Type, Tuple, Union
import numpy as np
from mindnlp._legacy.hypercomplex.tensor_decomposition.linal.basics.scalar import Scalar as AbstractScalar
from mindnlp._legacy.hypercomplex.tensor_decomposition.linal.basics.hypercomplex.hc_algebra_factory import AlgebraFactory

TAlgebraFactory = TypeVar('TAlgebraFactory', bound=AlgebraFactory)


class Scalar(AbstractScalar):

    r"""
    The Scalar class represents a scalar value in an algebraic system. It provides methods for arithmetic operations such as addition, subtraction, multiplication, and division, as well as functionality for
obtaining the square root, converting to an array, checking if the value is zero, and accessing the real component. The class also includes static methods for creating scalar instances representing zero and
one.
    
    Attributes:
        _alg_factory (Type[TAlgebraFactory]): The algebra factory used to create the scalar instance.
        _impl: The implementation obtained from the algebra factory for performing scalar operations.
        _x (np.float64): The real component of the scalar value.
        _y (np.float64): The imaginary component of the scalar value.
    
    Methods:
        __str__(): Returns a string representation of the scalar value.
        __neg__(): Returns the negation of the scalar.
        __add__(that: 'Scalar'): Returns the result of adding another scalar to the current scalar.
        __sub__(that: 'Scalar'): Returns the result of subtracting another scalar from the current scalar.
        __mul__(that: 'Scalar'): Returns the result of multiplying the current scalar by another scalar.
        __truediv__(that: 'Scalar'): Returns the result of dividing the current scalar by another scalar.
        __getitem__(key): Returns the real or imaginary component of the scalar based on the provided key.
        sqrt(): Returns the square root of the scalar value.
        as_array(): Converts the scalar to a NumPy array.
        is_zero(): Checks if the scalar value is effectively zero within a tolerance.
        get_real(): Returns the real component of the scalar value.
        visit(visitor, *args, **kwargs): Invokes a visitor function with optional arguments and keyword arguments.
        _make_like(val: Tuple[np.float64, np.float64]): Creates a new scalar instance with the same algebra factory as the current instance based on the provided values.
        _zero_like(): Creates a new scalar instance representing zero with the same algebra factory as the current instance.
        _one_like(): Creates a new scalar instance representing one with the same algebra factory as the current instance.
    
    Static Methods:
        one(algebra_factory: Type[TAlgebraFactory]): Returns a scalar instance representing the value 1.0.
        zero(algebra_factory: Type[TAlgebraFactory]): Returns a scalar instance representing the value 0.0.
    
    Properties:
        algebra_type (Type[AlgebraFactory]): The type of algebra factory used to create the scalar instance.
    """

    def __init__(self,
                 algebra_factory: Type[TAlgebraFactory],
                 x: Union[float, np.float64],
                 y: Union[float, np.float64]) -> None:

        r"""
        Initializes an instance of the Scalar class.
        
        Args:
            self: The instance of the Scalar class.
            algebra_factory (Type[TAlgebraFactory]): The algebra factory used to create the scalar implementation.
            x (Union[float, np.float64]): The value of 'x' to be assigned to the Scalar instance. It can be either a float or np.float64.
            y (Union[float, np.float64]): The value of 'y' to be assigned to the Scalar instance. It can be either a float or np.float64.
        
        Returns:
            None. This method does not return any value.
        
        Raises:
            None.
        """
        self._alg_factory = algebra_factory
        self._impl = algebra_factory().get_scalar_impl()
        self._x = np.float64(x)
        self._y = np.float64(y)

    def __str__(self) -> str:

        r"""
        The '__str__' method in the 'Scalar' class converts the object to a string representation.
        
        Args:
            self (Scalar): The instance of the Scalar class for which the string representation is generated.
        
        Returns:
            str: A string representing the Scalar object with the format 'x +|- special_element() + |y|', where:
                 - 'x' is the value of the '_x' attribute in the Scalar object.
                 - If 'y' is greater than or equal to 0.0, it is represented as '+', otherwise as '-'.
                 - 'special_element()' is the result of calling the 'special_element' method of the '_impl' attribute.
                 - '|y|' is the absolute value of the '_y' attribute in the Scalar object.
        
        Raises:
            No specific exceptions are raised by this method under normal operation.
        """
        return str(self._x) + ('+' if self._y >= 0. else '-') + self._impl.special_element() + str(abs(self._y))

    def __neg__(self) -> 'Scalar':

        r"""
        This method negates the current Scalar object by returning a new Scalar object with negative coordinates.
        
        Args:
            self (Scalar): The current Scalar object to be negated.
        
        Returns:
            Scalar: A new Scalar object with negated coordinates.
        
        Raises:
            None
        """
        return Scalar(self._alg_factory, -self._x, -self._y)

    def __add__(self, that: 'Scalar') -> 'Scalar':

        r"""
        Method '__add__' in the class 'Scalar' performs addition operation on two Scalar objects.
        
        Args:
            self (Scalar): The current Scalar object on which the method is called.
                Represents the first operand of the addition operation.
            that (Scalar): Another Scalar object that is being added to the current Scalar object.
                Represents the second operand of the addition operation.
        
        Returns:
            Scalar: A new Scalar object representing the result of adding the two input Scalar objects.
                The new object has the same algebraic factory as the current object and its x and y values
                are the sum of the corresponding values of the input Scalar objects.
        
        Raises:
            This method does not explicitly raise any exceptions. However, the addition operation may result in
            overflow or other arithmetic errors if the resulting x or y values exceed the limits of the data type.
        """
        return Scalar(self._alg_factory, self._x + that._x, self._y + that._y)

    def __sub__(self, that: 'Scalar') -> 'Scalar':

        
        """
        Subtracts the components of the current Scalar object from another Scalar object.
        
        Args:
            self (Scalar): The current Scalar object.
            that (Scalar): The Scalar object to be subtracted from the current Scalar object.
        
        Returns:
            Scalar: A new Scalar object resulting from the subtraction operation.
        
        Raises:
            TypeError: If the 'that' parameter is not of type 'Scalar'.
        """
        
        return Scalar(self._alg_factory, self._x - that._x, self._y - that._y)

    def __mul__(self, that: 'Scalar') -> 'Scalar':

        r"""
        Method '__mul__' in the class 'Scalar'.
        
        Args:
            self (Scalar): The current instance of the Scalar class.
                Represents the first operand in the multiplication operation.
            that (Scalar): Another instance of the Scalar class.
                Represents the second operand in the multiplication operation.
        
        Returns:
            Scalar: A new instance of the Scalar class that represents the product of the two operands.
        
        Raises:
            - TypeError: If the 'that' parameter is not of type 'Scalar'.
            - ValueError: If the multiplication operation encounters an error or invalid input.
        """
        product = self._impl.mul(self._x, self._y, that._x, that._y)
        return Scalar(self._alg_factory, *product)

    def __truediv__(self, that: 'Scalar') -> 'Scalar':

        
        """
        Perform division operation between two Scalar objects.
        
        Args:
            self (Scalar): The first Scalar object involved in the division operation.
            that (Scalar): The second Scalar object to divide the first Scalar by.
        
        Returns:
            Scalar: A new Scalar object representing the result of the division operation.
        
        Raises:
            TypeError: If the input parameters are not of type Scalar.
            ZeroDivisionError: If the second Scalar object has a value of zero, causing division by zero.
        """
        
        div = self._impl.div(self._x, self._y, that._x, that._y)
        return Scalar(self._alg_factory, *div)

    def __getitem__(self, key):

        r"""
        This method allows accessing the x and y coordinates of a Scalar object using the [] operator.
        
        Args:
            self (Scalar): The instance of the Scalar class.
            key (int): The index representing the coordinate to be accessed. Must be 0 or 1.
        
        Returns:
            None. This method returns the value of the specified coordinate.
        
        Raises:
            KeyError: If the key provided is not 0 or 1, a KeyError is raised to indicate an invalid key value.
        """
        if key == 0:
            return self._x
        if key == 1:
            return self._y
        raise KeyError('The key is supposed to be 0 or 1')

    @staticmethod
    def one(algebra_factory: Type[TAlgebraFactory]) -> 'Scalar':

        r"""
        This method creates a scalar instance with a value of 1.0 and an error of 0.0 using the specified algebra factory.
        
        Args:
            algebra_factory (Type[TAlgebraFactory]): The algebra factory used to create the scalar instance.
        
        Returns:
            Scalar: A scalar instance with a value of 1.0 and an error of 0.0.
        
        Raises:
            None
        """
        return Scalar(algebra_factory, 1., 0.)

    @staticmethod
    def zero(algebra_factory: Type[TAlgebraFactory]) -> 'Scalar':

        r"""
        This method initializes a Scalar object with a value of zero.
        
        Args:
            algebra_factory (Type[TAlgebraFactory]): The type of algebra factory used to create the Scalar object. 
                This parameter is mandatory and must be provided to instantiate the Scalar object.
        
        Returns:
            Scalar: A Scalar object initialized with a value of zero. 
        
        Raises:
            None
        """
        return Scalar(algebra_factory, 0., 0.)

    def get_algebra_type(self) -> Type[AlgebraFactory]:

        r"""
        get_algebra_type method in the Scalar class.
        
        Args:
            self: The instance of the Scalar class.
                Type: Scalar
                Purpose: Represents the current instance of the Scalar class.
                Restrictions: None
        
        Returns:
            Type[AlgebraFactory]: 
                The type of algebra factory associated with the Scalar instance.
        
        Raises:
            None
        """
        return self._alg_factory

    def sqrt(self) -> 'Scalar':

        r"""
        This method calculates the square root of the current Scalar instance.
        
        Args:
            self: Scalar - The current Scalar instance.
        
        Returns:
            Scalar: A new Scalar instance representing the square root of the current instance.
        
        Raises:
            None
        """
        x, y = self._impl.sqrt(self._x, self._y)
        return Scalar(self._alg_factory, x, y)

    def as_array(self) -> np.ndarray:

        r"""
        Converts the scalar to a numpy array.
        
        Args:
            self: Scalar - The scalar object to be converted to a numpy array.
        
        Returns:
            np.ndarray - A numpy array containing the x and y components of the scalar.
        
        Raises:
            None
        """
        return np.array([self._x, self._y], dtype=np.float64)

    def is_zero(self) -> bool:

        r"""
        Check if the Scalar instance is effectively zero.
        
        Args:
            self (Scalar): The Scalar instance on which the method is called.
                This parameter represents the current Scalar instance for which the zero check is performed.
                It is a required parameter and should be an instance of the Scalar class.
        
        Returns:
            bool: A boolean value indicating whether the Scalar instance is effectively zero.
                Returns True if both the x and y components of the Scalar instance fall within the equality tolerance range around zero;
                otherwise, returns False.
        
        Raises:
            None
        """
        return -Scalar._EQUALITY_TOLERANCE < self._x < Scalar._EQUALITY_TOLERANCE \
            and -Scalar._EQUALITY_TOLERANCE < self._y < Scalar._EQUALITY_TOLERANCE

    def get_real(self) -> np.float64:

        r"""
        Method to retrieve the real component of a Scalar object.
        
        Args:
            self (Scalar): The Scalar object itself.
                This parameter represents the instance of the Scalar class for which the real component is being retrieved.
        
        Returns:
            np.float64: The real component of the Scalar object.
                This method returns the real component of the Scalar object as a NumPy float64 data type.
        
        Raises:
            None
        """
        return self._x

    def visit(self, visitor, *args, **kwargs) -> None:

        r""" 
        Visit method in the Scalar class.
        
        This method is used to visit the Scalar object with a visitor object.
        
        Args:
            self (Scalar): The current instance of the Scalar object.
            visitor: The visitor object that will perform operations on the Scalar object.
            
        Returns:
            None. This method does not return any value.
            
        Raises:
            None.
        """
        self._impl.visit(self, visitor, *args, **kwargs)

    def _make_like(self, val: Tuple[np.float64, np.float64]) -> 'Scalar':

        r"""
        Create a new instance of the 'Scalar' class that is similar to the current instance.
        
        Args:
            self (Scalar): The current instance of the 'Scalar' class.
            val (Tuple[np.float64, np.float64]): A tuple containing two 'np.float64' values.
                These values will be used to create the new instance of the 'Scalar' class.
        
        Returns:
            Scalar: A new instance of the 'Scalar' class that has been created using the provided values.
        
        Raises:
            None.
        """
        return Scalar(self._alg_factory, *val)

    def _zero_like(self) -> 'Scalar':

        r"""
        This method returns a new Scalar instance that is zero-like, based on the current Scalar instance.
        
        Args:
            self (Scalar): The current Scalar instance.
                - type: Scalar
                - purpose: Represents the current Scalar instance for which a zero-like instance is to be created.
        
        Returns:
            Scalar: A new Scalar instance that is zero-like, based on the current Scalar instance.
                - type: Scalar
                - purpose: Represents a zero-like instance based on the current Scalar instance.
        
        Raises:
            None
        """
        return Scalar.zero(self._alg_factory)

    def _one_like(self) -> 'Scalar':

        r"""
        This method returns a new instance of Scalar with the same algebraic structure as the current instance.
        
        Args:
            self (Scalar): The current instance of Scalar.
            
        Returns:
            Scalar: A new instance of Scalar with the same algebraic structure as the current instance.
        
        Raises:
            None
        """
        return Scalar.one(self._alg_factory)
