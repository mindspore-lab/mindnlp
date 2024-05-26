from typing import Union, List, Optional
import numpy as np
from mindnlp._legacy.hypercomplex.tensor_decomposition.linal.basics.vector import Vector as AbstractVector
from mindnlp._legacy.hypercomplex.tensor_decomposition.linal.basics.real.real_scalar import Scalar

NoneType = type(None)


class Vector(AbstractVector):

    r"""
    Represents a vector in a mathematical space.
    
    This class inherits from AbstractVector, providing functionality for vector operations and manipulations. The Vector class allows for the creation and manipulation of vectors, and supports operations such as addition, subtraction, multiplication, and division with scalars and other vectors.
    
    Attributes:
        _size (int): The size of the vector.
        _items (np.ndarray): An array representing the vector's elements.
    
    Methods:
        __init__: Initializes the Vector object with the specified size and items.
        __getitem__: Returns the element at the given index as a Vector or Scalar object.
        __neg__: Returns the negation of the vector.
        add_scalar: Adds a scalar value to the vector and returns a new Vector object.
        sub_scalar: Subtracts a scalar value from the vector and returns a new Vector object.
        mul_scalar: Multiplies the vector by a scalar value and returns a new Vector object.
        div_scalar: Divides the vector by a scalar value and returns a new Vector object.
        conjugate: Returns the conjugate of the vector.
        as_array: Returns the vector elements as a numpy array.
        _init_default: Initializes the vector with default values.
        _init_ndarray: Initializes the vector with values from a numpy array.
        _init_list: Initializes the vector with values from a list of scalars.
        _add: Adds another Vector to the current Vector and returns a new Vector object.
        _sub: Subtracts another Vector from the current Vector and returns a new Vector object.
        _mul: Multiplies the current Vector by another Vector and returns a new Vector object.
        _dot_product: Computes the dot product with another Vector and returns a Scalar object.
        _div: Divides the current Vector by another Vector and returns a new Vector object.
        _full_like: Returns a new Vector object with the same shape and type as the specified array or list.
        _ones_like: Returns a new Vector object filled with Scalar.one() values, matching the size of the current Vector.
        _zeros_like: Returns a new Vector object filled with Scalar.zero() values, matching the size of the current Vector.
    
    Raises:
        TypeError: If the 'items' argument is not None, a numpy array, or a list of scalars.
        ValueError: If the size or contents of the 'items' argument are invalid.
    
    """

    def __init__(self,
                 size: Optional[int] = None,
                 items: Union[np.ndarray, List, NoneType] = None) -> None:

        r"""
        Initializes a Vector object.
        
        Args:
            self (Vector): The instance of the Vector class itself.
            size (Optional[int]): The size of the vector. Default is None.
                If items is None, size is initialized using the _init_default method.
            items (Union[np.ndarray, List, NoneType]): The items to populate the vector.
                It must be one of: None, ndarray, or List of scalars.
                If items is an ndarray, size is initialized using the _init_ndarray method.
                If items is a List, size is initialized using the _init_list method.
        
        Returns:
            None: This method does not return any value.
        
        Raises:
            TypeError: If items is not None, ndarray, or List of scalars.
        """
        if items is None:
            size = self._init_default(size)
        elif isinstance(items, np.ndarray):
            size = self._init_ndarray(size, items)
        elif isinstance(items, List):
            size = self._init_list
        else:
            raise TypeError(f"'items' must be one of: None, ndarray or List of scalars, but got: {items}")
        super(Vector, self).__init__(size)

    def __getitem__(self, key):

        r"""
        Method to retrieve an item from the Vector instance.
        
        Args:
            self (Vector): The Vector instance from which an item is to be retrieved.
            key: The key representing the item to be retrieved from the Vector instance.
        
        Returns:
            None: This method does not return a value. Instead, it initializes and returns a new Vector or Scalar object based on the retrieved item.
        
        Raises:
            IndexError: If the key is invalid or out of range.
        """
        x = self._items.__getitem__(key)
        if len(x.shape) == 1:
            return Vector(x.shape[0], x)
        else:
            return Scalar(x)

    def __neg__(self) -> 'Vector':

        r"""
        Method '__neg__' in the 'Vector' class negates the vector items.
        
        Args:
            self (Vector): The instance of the Vector class.
                Represents the vector on which the negation operation will be performed.
        
        Returns:
            Vector: A new Vector instance after negating the items of the original vector.
                The size of the new vector remains the same as the original vector.
                The items of the new vector are the negation of the items in the original vector.
        
        Raises:
            None.
        """
        return Vector(self._size, np.negative(self._items))

    def add_scalar(self, that: Union[Scalar, float, np.float64]) -> 'Vector':

        r"""
        Adds a scalar value to each element of the vector.
        
        Args:
            self (Vector): An instance of the Vector class.
            that (Union[Scalar, float, np.float64]): The value to be added to the vector. It can be either a Scalar object,
                a float, or a np.float64 value. If that is a Scalar object, its numerical value will be extracted and used.
        
        Returns:
            Vector: A new Vector object with the same size as the original vector, where each element is the sum of the corresponding
            element in the original vector and the scalar value.
        
        Raises:
            None.
        
        Example:
            >>> v = Vector([1, 2, 3])
            >>> v.add_scalar(5)
            Vector([6, 7, 8])
            >>> v.add_scalar(Scalar(2))
            Vector([3, 4, 5])
        """
        num = that.as_number() if isinstance(that, Scalar) else that
        return Vector(self._size, np.add(self._items, num))

    def sub_scalar(self, that: Union[Scalar, float, np.float64]) -> 'Vector':

        r"""
        Method to subtract a scalar value from each element of the Vector.
        
        Args:
            self (Vector): The Vector object itself.
                - Type: Vector
                - Purpose: Represents the Vector from which the scalar value will be subtracted.
            that (Union[Scalar, float, np.float64]): The scalar value or Scalar object to subtract.
                - Type: Union[Scalar, float, np.float64]
                - Purpose: Represents the scalar value to subtract from each element of the Vector.
                - Restrictions: It can be a Scalar object, float, or np.float64.
        
        Returns:
            Vector: A new Vector object created after subtracting the scalar value from each element of the original Vector.
                - Type: Vector
                - Purpose: Represents the resulting Vector after the subtraction operation.
        
        Raises:
            None
        """
        num = that.as_number() if isinstance(that, Scalar) else that
        return Vector(self._size, np.subtract(self._items, num))

    def mul_scalar(self, that: Union[Scalar, float, np.float64]) -> 'Vector':

        r"""
        Calculates the result of multiplying the Vector by a scalar value.
        
        Args:
            self (Vector): The Vector object on which the operation is performed.
            that (Union[Scalar, float, np.float64]): The scalar value to multiply the Vector by. 
                It can be an instance of the Scalar class, a Python float, or a NumPy float64. 
                If 'that' is an instance of Scalar, it will be converted to a number using the 'as_number()' method.
        
        Returns:
            Vector: A new Vector object of the same size as the original Vector, 
            containing the result of element-wise multiplication of the original Vector by the scalar value.
        
        Raises:
            - TypeError: If 'that' is not of type Scalar, float, or np.float64.
            - ValueError: If the operation results in an invalid value that cannot be represented.
        """
        num = that.as_number() if isinstance(that, Scalar) else that
        return Vector(self._size, np.multiply(self._items, num))

    def div_scalar(self, that: Union[Scalar, float, np.float64]) -> 'Vector':

        r"""
        Divides the elements of the Vector by a scalar value.
        
        Args:
            self (Vector): The Vector instance on which the method is called.
            that (Union[Scalar, float, np.float64]): The scalar value to divide the Vector by. 
                It can be an instance of the Scalar class, a float, or a np.float64.
                If that is an instance of Scalar, its numerical value will be used. Otherwise, that will be used directly.
        
        Returns:
            Vector: A new Vector instance with the elements of the original Vector divided by the scalar value.
        
        Raises:
            None.
        
        Note:
            - The scalar value must not be zero, as division by zero is undefined.
        
        Example:
            vector = Vector([1, 2, 3, 4])
            scalar = Scalar(2)
            result = vector.div_scalar(scalar)
            print(result)
            Output: Vector([0.5, 1.0, 1.5, 2.0])
        """
        num = that.as_number() if isinstance(that, Scalar) else that
        return Vector(self._size, np.divide(self._items, num))

    def conjugate(self) -> 'Vector':

        r"""
        Conjugates the current vector.
        
        Args:
            self (Vector): The vector instance on which the conjugate operation is performed.
        
        Returns:
            Vector: A new vector instance representing the conjugate of the input vector.
        
        Raises:
            None.
        """
        return self

    def as_array(self) -> np.ndarray:

        r"""
        Converts the Vector object into a NumPy array.
        
        Args:
            self (Vector): The Vector object to be converted into a NumPy array.
        
        Returns:
            np.ndarray: A NumPy array representation of the Vector object.
        
        Raises:
            None
        """
        return np.array(self._items)

    def _init_default(self, size: Optional[int]) -> int:

        
        """
        Initializes the Vector with a default size and returns the size of the Vector.
        
        Args:
            self (Vector): The Vector instance.
            size (Optional[int]): The size of the Vector. If size is None, a ValueError is raised.
            
        Returns:
            int: The size of the Vector.
        
        Raises:
            ValueError: If size is None, indicating that either 'size' or 'items' must not be None.
        """
        
        if size is None:
            raise ValueError(f"Either 'items' or 'size' must not be None")
        self._items = np.zeros((size,), dtype=np.float64)
        return size

    def _init_ndarray(self, size: Optional[int], items: np.ndarray) -> int:

        r"""
        Initializes an ndarray in the Vector class.
        
        Args:
            self: An instance of the Vector class.
            size (Optional[int]): The size of the ndarray. If None, it is set to the length of the items parameter.
            items (np.ndarray): An ndarray containing the items to be stored in the Vector.
        
        Returns:
            int: The size of the ndarray after initialization.
        
        Raises:
            None.
        """
        if size is None:
            size = len(items)
        self._items = np.array(items).astype(np.float64)
        return size

    def _init_list(self, size: Optional[int], items: List[Scalar]) -> int:

        r"""
        Initializes the list of items for the Vector class.
        
        Args:
            self (Vector): The instance of the Vector class.
            size (Optional[int]): The size of the list. If None, it defaults to the length of the 'items' list. Defaults to None.
            items (List[Scalar]): A list of scalar items to be stored in the vector.
        
        Returns:
            int: The size of the list.
        
        Raises:
            ValueError: If 'size' is None and the length of 'items' is less than or equal to 0.
            ValueError: If the length of 'items' does not match the specified 'size'.
            ValueError: If any item in 'items' is not of type Scalar.
        """
        if size is None:
            size = len(items)
            if size <= 0:
                raise ValueError(f"'items' must be a list of some scalars, but got: {items}")
        elif len(items) != size:
            raise ValueError(f"'items' must be a list of {size} scalars, but got: {items}")
        if any(not isinstance(s, Scalar) for s in items):
            raise ValueError(f"'items' must be a list of {size} scalars, but got: {items}")
        self._items = np.array([item.as_number() for item in items])
        return size

    def _add(self, that: 'Vector') -> 'Vector':

        r"""
        Adds the elements of the current Vector with another Vector provided as input.
        
        Args:
            self (Vector): The current Vector object to perform addition on.
            that (Vector): The Vector object to be added to the current Vector. It must have the same size as the current Vector.
        
        Returns:
            Vector: A new Vector object representing the result of adding the elements of the two Vectors element-wise.
        
        Raises:
            TypeError: If the 'that' parameter is not of type Vector.
            ValueError: If the size of the 'that' Vector is different from the size of the current Vector.
        """
        return Vector(self._size, np.add(self._items, that._items))

    def _sub(self, that: 'Vector') -> 'Vector':

        r"""
        Args:
            self (Vector): The first vector used for subtraction. It represents the instance of the Vector class.
            that (Vector): The second vector used for subtraction. It represents another instance of the Vector class.
        
        Returns:
            Vector: A new Vector instance resulting from the subtraction operation.
        
        Raises:
            TypeError: If the 'that' parameter is not an instance of the Vector class.
            ValueError: If the size of the vectors is not compatible for subtraction.
        """
        return Vector(self._size, np.subtract(self._items, that._items))

    def _mul(self, that: 'Vector') -> 'Vector':

        r"""
        This method performs element-wise multiplication of two Vector objects.
        
        Args:
            self (Vector): The Vector object on which the method is called.
                Represents the first Vector for the multiplication operation.
            that (Vector): The Vector object passed as a parameter.
                Represents the second Vector to be multiplied with 'self'.
                It should have the same size as 'self'.
        
        Returns:
            Vector: A new Vector object resulting from the element-wise multiplication of 'self' and 'that'.
        
        Raises:
            ValueError: If the sizes of the two Vector objects are not equal, an exception is raised.
        """
        return Vector(self._size, np.multiply(self._items, that._items))

    def _dot_product(self, that: 'Vector') -> Scalar:

        r"""
        Calculate the dot product between two vectors.
        
        Args:
            self (Vector): The instance of the Vector class calling the method.
                Represents the first vector for the dot product calculation.
            that (Vector): Another instance of the Vector class representing the second vector
                to be used in the dot product calculation.
        
        Returns:
            Scalar: A Scalar object representing the dot product of the two vectors.
        
        Raises:
            ValueError: If the lengths of the vectors are not equal, a ValueError is raised.
            TypeError: If the input vectors are not instances of the Vector class,
                a TypeError is raised.
        """
        return Scalar(np.dot(self._items, that._items).item())

    def _div(self, that: 'Vector') -> 'Vector':

        r"""
        Method _div in class Vector.
        
        Args:
            self (Vector): The current Vector instance on which the division operation is performed.
            that (Vector): The Vector instance to divide the current Vector by.
        
        Returns:
            Vector: A new Vector instance resulting from element-wise division of the two input Vectors.
        
        Raises:
            TypeError: If the 'that' parameter is not of type Vector.
            ValueError: If the sizes of the two Vectors are not the same, making element-wise division impossible.
            ZeroDivisionError: If any element in 'that' Vector is zero, causing division by zero.
        """
        return Vector(self._size, np.divide(self._items, that._items))

    def _full_like(self, items: Union[np.ndarray, List]) -> 'Vector':

        r"""
        Method _full_like in class Vector.
        
        Args:
            self: The instance of the Vector class.
            items (Union[np.ndarray, List]): An np.ndarray or a List containing items to create the Vector from.
        
        Returns:
            Vector: A new Vector instance created using the provided items.
        
        Raises:
            - TypeError: If items is not of type np.ndarray or List.
        """
        return Vector(items=items)

    def _ones_like(self) -> 'Vector':

        r"""
        This method creates a new Vector object with elements that are all set to the value of Scalar.one().
        
        Args:
            self (Vector): The current Vector object for which to create a new Vector with elements set to Scalar.one().
            
        Returns:
            Vector: A new Vector object with elements that are all set to the value of Scalar.one().
        
        Raises:
            None.
        """
        return Vector(items=[Scalar.one()] * self._size)

    def _zeros_like(self) -> 'Vector':

        r"""Creates a new Vector object with the same size as the current Vector object, where all elements are set to zero.
        
        Args:
            self (Vector): The current Vector object.
            
        Returns:
            Vector: A new Vector object with the same size as the current object, where all elements are set to zero.
            
        Raises:
            None.
        """
        return Vector(items=[Scalar.zero()] * self._size)
