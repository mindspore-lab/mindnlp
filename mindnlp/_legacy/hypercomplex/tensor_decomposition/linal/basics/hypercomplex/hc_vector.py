from typing import Type, TypeVar, Tuple, Union, Optional, List
import numpy as np
from mindnlp._legacy.hypercomplex.tensor_decomposition.linal.basics.vector import Vector as AbstractVector
from mindnlp._legacy.hypercomplex.tensor_decomposition.linal.basics.hypercomplex.hc_algebra_factory import AlgebraFactory
from mindnlp._legacy.hypercomplex.tensor_decomposition.linal.basics.hypercomplex.hc_scalar import Scalar

TAlgebraFactory = TypeVar('TAlgebraFactory', bound=AlgebraFactory)
NoneType = type(None)


class Vector(AbstractVector):

    r"""
    The `Vector` class represents a vector in a mathematical algebra. It is a subclass of `AbstractVector` and provides various operations and functionalities for vector manipulation.
    
    Attributes:
        - `_alg_factory`: The algebra factory used to create the vector implementation.
        - `_impl`: The vector implementation object.
        - `_items_x`: The x-coordinate values of the vector.
        - `_items_y`: The y-coordinate values of the vector.
    
    Methods:
        - `__init__(self, algebra_factory: Type[TAlgebraFactory], size: Optional[int] = None, items: Union[Tuple, np.ndarray, List, NoneType] = None) -> None`: Initializes a new instance of `Vector` with the
given algebra factory, size, and items.
        - `__getitem__(self, key)`: Returns the value at the specified key in the vector.
        - `__neg__(self) -> 'Vector'`: Returns the negation of the vector.
        - `get_algebra_type(self) -> Type[AlgebraFactory]`: Returns the algebra factory type used by the vector.
        - `add_scalar(self, that: Union[Scalar, Tuple[np.float64, np.float64]]) -> 'Vector'`: Adds a scalar value to the vector.
        - `sub_scalar(self, that: Union[Scalar, Tuple[np.float64, np.float64]]) -> 'Vector'`: Subtracts a scalar value from the vector.
        - `mul_scalar(self, that: Union[Scalar, Tuple[np.float64, np.float64]]) -> 'Vector'`: Multiplies the vector by a scalar value.
        - `div_scalar(self, that: Union[Scalar, Tuple[np.float64, np.float64]]) -> 'Vector'`: Divides the vector by a scalar value.
        - `conjugate(self) -> 'Vector'`: Returns the conjugate of the vector.
        - `as_array(self) -> np.ndarray`: Returns the vector as a NumPy array.
        - `_init_default(self, size: Optional[int]) -> int`: Initializes the vector with a default size.
        - `_init_tuple(self, size: Optional[int], items: Tuple[np.ndarray, np.ndarray]) -> int`: Initializes the vector with a tuple of items.
        - `_init_ndarray(self, size: Optional[int], items: np.ndarray) -> int`: Initializes the vector with a NumPy array of items.
        - `_init_list(self, size: Optional[int], items: List[Scalar]) -> int`: Initializes the vector with a list of scalar items.
        - `_add(self, that: 'Vector') -> 'Vector'`: Adds another vector to the current vector.
        - `_sub(self, that: 'Vector') -> 'Vector'`: Subtracts another vector from the current vector.
        - `_mul(self, that: 'Vector') -> 'Vector'`: Multiplies the current vector by another vector.
        - `_dot_product(self, that: 'Vector') -> Scalar`: Calculates the dot product of the current vector and another vector.
        - `_div(self, that: 'Vector') -> 'Vector'`: Divides the current vector by another vector.
        - `_full_like(self, items: Union[Tuple, np.ndarray, List]) -> 'Vector'`: Creates a new vector with the same size as the given items.
        - `_ones_like(self) -> 'Vector'`: Creates a new vector with all elements set to one.
        - `_zeros_like(self) -> 'Vector'`: Creates a new vector with all elements set to zero.
    """

    def __init__(self,
                 algebra_factory: Type[TAlgebraFactory],
                 size: Optional[int] = None,
                 items: Union[Tuple, np.ndarray, List, NoneType] = None) -> None:
        r"""
        Initializes a Vector object.
        
        Args:
            self: The instance of the Vector class.
            algebra_factory (Type[TAlgebraFactory]): The type of algebra factory to be used for creating the vector implementation.
            size (Optional[int]): The optional size of the vector. If not provided, it will be initialized with a default value.
            items (Union[Tuple, np.ndarray, List, NoneType]): The items used to initialize the vector. It can be a tuple, numpy array, list of scalars, or None. 
        
        Returns:
            None: This method does not return any value.
        
        Raises:
            TypeError: If 'items' is not one of: None, ndarray, list of scalars or Tuple.
        
        """
        self._alg_factory = algebra_factory
        self._impl = algebra_factory().get_vector_impl()
        if items is None:
            size = self._init_default(size)
        elif isinstance(items, Tuple):
            size = self._init_tuple(size, items)
        elif isinstance(items, np.ndarray):
            size = self._init_ndarray(size, items)
        elif isinstance(items, List):
            size = self._init_list(size, items)
        else:
            raise TypeError(f"'items' must be one of: None, ndarray, list of scalars or Tuple, but got: {items}")
        super(Vector, self).__init__(size)

    def __getitem__(self, key):
        r"""
        Retrieve an item from the Vector object using the given key.
        
        Args:
            self (Vector): The Vector object itself.
            key: The key used to retrieve the item. It can be an index or a slice.
        
        Returns:
            None
        
        Raises:
            IndexError: If the key is out of range or invalid.
        """
        x = self._items_x.__getitem__(key)
        y = self._items_y.__getitem__(key)
        if len(x.shape) == 1:
            return Vector(self._alg_factory, x.shape[0], (x, y))
        else:
            return Scalar(self._alg_factory, x, y)

    def __neg__(self) -> 'Vector':
        r"""
        Negates the vector.
        
        Args:
            self (Vector): The vector instance.
        
        Returns:
            Vector: A new vector instance with all elements negated.
        
        Raises:
            None.
        """
        return Vector(
            self._alg_factory, self._size,
            (np.negative(self._items_x), np.negative(self._items_y))
        )

    def get_algebra_type(self) -> Type[AlgebraFactory]:
        r"""
        Retrieve the type of algebra factory associated with the Vector instance.
        
        Args:
            self (Vector): The Vector instance for which the algebra factory type is being retrieved.
        
        Returns:
            Type[AlgebraFactory]: The type of algebra factory associated with the Vector instance. 
            This type indicates the specific algebraic structure to be used for vector operations.
        
        Raises:
            None.
        """
        return self._alg_factory

    def add_scalar(self, that: Union[Scalar, Tuple[np.float64, np.float64]]) -> 'Vector':
        r"""
        Adds a scalar value to the vector.
        
        Args:
            self (Vector): The vector instance to which the scalar is being added.
            that (Union[Scalar, Tuple[np.float64, np.float64]]): The scalar value or tuple of float64 values to be added to the vector. If a Scalar object is provided, it must have the same data type as the
vector's algebra type.
        
        Returns:
            Vector: A new Vector instance resulting from adding the scalar value to the original vector.
        
        Raises:
            TypeError: If the provided scalar is of a different data type than the vector's algebra type.
        """
        if isinstance(that, Scalar) and that.get_algebra_type() != self._alg_factory:
            raise TypeError(
                f'It is only possible to add a scalar of the same data type {self._alg_factory}, '
                f'but got: {that.get_algebra_type()}'
            )
        return Vector(
            self._alg_factory, self._size,
            (np.add(self._items_x, that[0]), np.add(self._items_y, that[1]))
        )

    def sub_scalar(self, that: Union[Scalar, Tuple[np.float64, np.float64]]) -> 'Vector':
        r"""
        sub_scalar
        
        This method subtracts a scalar from the vector.
        
        Args:
            self (Vector): The current vector object.
            that (Union[Scalar, Tuple[np.float64, np.float64]]): The scalar value or a tuple of two np.float64 values to be subtracted from the vector. If 'that' is a Scalar, it must have the same data type as
the current vector's algebra type.
        
        Returns:
            Vector: A new Vector object resulting from subtracting the scalar from the current vector.
        
        Raises:
            TypeError: If 'that' is a Scalar with a different data type than the current vector's algebra type.
        """
        if isinstance(that, Scalar) and that.get_algebra_type() != self._alg_factory:
            raise TypeError(
                f'It is only possible to subtract a scalar of the same data type {self._alg_factory}, '
                f'but got: {that.get_algebra_type()}'
            )
        return Vector(
            self._alg_factory, self._size,
            (np.subtract(self._items_x, that[0]), np.subtract(self._items_y, that[1]))
        )

    def mul_scalar(self, that: Union[Scalar, Tuple[np.float64, np.float64]]) -> 'Vector':
        r"""
        mul_scalar method in the Vector class multiplies the Vector by a scalar.
        
        Args:
            self (Vector): The Vector instance on which the multiplication operation is performed.
            that (Union[Scalar, Tuple[np.float64, np.float64]]): The scalar value or a tuple of two np.float64 values. 
              If a Scalar instance is provided, it should have the same data type as the Vector's algebra type.
        
        Returns:
            Vector: A new Vector resulting from the multiplication operation.
        
        Raises:
            TypeError: If the provided scalar is not of the same data type as the Vector's algebra type.
        """
        if isinstance(that, Scalar) and that.get_algebra_type() != self._alg_factory:
            raise TypeError(
                f'It is only possible to multiply by a scalar of the same data type {self._alg_factory}, '
                f'but got: {that.get_algebra_type()}'
            )
        items = self._impl.mul(self._items_x, self._items_y, that[0], that[1])
        return Vector(self._alg_factory, self._size, items)

    def div_scalar(self, that: Union[Scalar, Tuple[np.float64, np.float64]]) -> 'Vector':
        r"""
        Divides the vector by a scalar value.
        
        Args:
            self (Vector): The current Vector object.
            that (Union[Scalar, Tuple[np.float64, np.float64]]): The scalar value to divide the vector by. 
                It can either be a Scalar object or a tuple of two np.float64 values.
        
        Returns:
            Vector: A new Vector object with the result of the division.
        
        Raises:
            TypeError: If 'that' is a Scalar object and its algebra type is different from the current Vector's algebra type.
        
        Note:
            The division operation is performed element-wise on the vector. If 'that' is a Scalar object, it must have the same 
            algebra type as the current Vector object.
        
        Example:
            >>> vec = Vector(alg_factory, size, items)
            >>> scalar = Scalar(alg_factory, value)
            >>> result = vec.div_scalar(scalar)
        """
        if isinstance(that, Scalar) and that.get_algebra_type() != self._alg_factory:
            raise TypeError(
                f'It is only possible to divide over a scalar of the same data type {self._alg_factory}, '
                f'but got: {that.get_algebra_type()}'
            )
        items = self._impl.div(self._items_x, self._items_y, that[0], that[1])
        return Vector(self._alg_factory, self._size, items)

    def conjugate(self) -> 'Vector':
        r"""
        Conjugates the vector by negating the imaginary part of each element.
        
        Args:
            self (Vector): The Vector object on which the conjugate operation is to be performed.
        
        Returns:
            Vector: A new Vector object resulting from the conjugate operation, where the imaginary part of each element is negated.
        
        Raises:
            None
        """
        return Vector(self._alg_factory, self._size, (self._items_x, np.negative(self._items_y)))

    def as_array(self) -> np.ndarray:
        r"""
        Converts the Vector object into a NumPy array.
        
        Args:
            self (Vector): The Vector object to be converted into an array.
            
        Returns:
            np.ndarray: A NumPy array containing the items of the Vector object grouped by their x and y coordinates.
        
        Raises:
            None
        """
        return np.array(list(zip(self._items_x, self._items_y)))

    def _init_default(self, size: Optional[int]) -> int:
        r"""
        Initializes the default values for the Vector class.
        
        Args:
            self (Vector): The instance of the Vector class.
            size (Optional[int]): The size of the vector. It can be None if 'items' parameter is provided.
                                  If 'size' is None, a ValueError will be raised.
        
        Returns:
            int: The size of the vector.
        
        Raises:
            ValueError: If either 'items' or 'size' parameter is None, this exception is raised.
        
        """
        if size is None:
            raise ValueError(f"Either 'items' or 'size' must not be None")
        self._items_x = np.zeros((size,), dtype=np.float64)
        self._items_y = np.zeros((size,), dtype=np.float64)
        return size

    def _init_tuple(self, size: Optional[int], items: Tuple[np.ndarray, np.ndarray]) -> int:
        r"""Initialize the Vector class with a tuple of numpy arrays.
        
        Args:
            size (Optional[int]): The size of the arrays. If None, it defaults to the length of the first array.
            items (Tuple[np.ndarray, np.ndarray]): A 2-element tuple containing two 1d numpy arrays representing x and y coordinates.
        
        Returns:
            int: The size of the arrays.
        
        Raises:
            ValueError: If 'items' is not a 2-element tuple, if the elements of 'items' are not 1d arrays, if the sizes of the arrays do not match the specified size.
        """
        if len(items) != 2:
            raise ValueError(f"'items' must be a 2-element tuple, but got: {items}")
        self._items_x = items[0]
        self._items_y = items[1]
        if not isinstance(self._items_x, np.ndarray) or len(self._items_x.shape) != 1:
            raise ValueError(
                f"elements of 'items' must be 1d arrays, but got: {items}"
            )
        if not isinstance(self._items_y, np.ndarray) or len(self._items_y.shape) != 1:
            raise ValueError(
                f"elements of 'items' must be 1d arrays, but got: {items}"
            )
        if size is None:
            size = len(self._items_x)
        if len(self._items_x) != size or len(self._items_y) != size:
            raise ValueError(
                f"elements of 'items' must be 1d arrays of size {size}, but got: {items}"
            )
        return size

    def _init_ndarray(self, size: Optional[int], items: np.ndarray) -> int:
        r"""
        Initializes a Vector object with the provided size and items.
        
        Args:
            self (Vector): The instance of the Vector class.
            size (Optional[int]): The size of the Vector. If None, the size is determined based on the length of the items.
            items (np.ndarray): The array of items to be stored in the Vector.
        
        Returns:
            int: The size of the initialized Vector.
        
        Raises:
            None.
        """
        if size is None:
            size = len(items)
        super(Vector, self).__init__(size)
        all_items = np.ravel(items)
        self._items_x = np.reshape(all_items[::2], (size,)).astype(np.float64)
        self._items_y = np.reshape(all_items[1::2], (size,)).astype(np.float64)
        return size

    def _init_list(self, size: Optional[int], items: List[Scalar]) -> int:
        r"""
        Initializes a list of vectors in the class 'Vector'.
        
        Args:
            self (Vector): The instance of the Vector class.
            size (Optional[int]): The size of the list. If not provided, it will be inferred from the 'items' parameter.
            items (List[Scalar]): A list of scalars representing the vector elements.
        
        Returns:
            int: The size of the list after initialization.
        
        Raises:
            ValueError: If 'items' is an empty list or not a list of scalars.
            ValueError: If the length of 'items' is not equal to the specified 'size'.
            ValueError: If 'items' contains scalars that are not of the same algebra type as the instance's algebra factory.
        
        Note:
            - The 'items' parameter must be a list of scalars.
            - The 'size' parameter is optional. If not provided, the size will be determined based on the length of 'items'.
            - The 'size' parameter is required when specifying a non-empty 'items' list.
            - The 'items' list must contain scalars of the same algebra type as the instance's algebra factory.
            - The method stores the x and y components of each scalar in separate arrays '_items_x' and '_items_y'.
              These arrays are created by reshaping and converting the 'items' list to numpy arrays.
        
        Example:
            # Create an instance of the Vector class
            v = Vector()
        
            # Initialize a list of vectors with a specified size and items
            size = 3
            items = [Scalar(1, 2), Scalar(3, 4), Scalar(5, 6)]
            v._init_list(size, items)
        
            # Initialize a list of vectors without specifying the size
            items = [Scalar(7, 8), Scalar(9, 10), Scalar(11, 12)]
            v._init_list(items=items)
        """
        if size is None:
            size = len(items)
            if size <= 0:
                raise ValueError(f"'items' must be a list of some scalars, but got: {items}")
        elif len(items) != size:
            raise ValueError(f"'items' must be a list of {size} scalars, but got: {items}")
        if any(
                not isinstance(s, Scalar) or s.get_algebra_type() != self._alg_factory for s in items
        ):
            raise ValueError(f"'items' must be a list of {size} scalars, but got: {items}")
        all_items = np.ravel(np.concatenate([s.as_array() for s in items], axis=0))
        self._items_x = np.reshape(all_items[::2], (size,)).astype(np.float64)
        self._items_y = np.reshape(all_items[1::2], (size,)).astype(np.float64)
        return size

    def _add(self, that: 'Vector') -> 'Vector':
        r"""
        This method performs the addition operation between two vectors and returns a new Vector object as the result.
        
        Args:
            self (Vector): The current Vector object on which the addition operation is being performed.
            that (Vector): The Vector object to be added to the current Vector. Both Vectors must have the same data type for successful addition.
        
        Returns:
            Vector: A new Vector object that is the result of adding the 'that' Vector to the 'self' Vector. The data type of the resulting Vector matches the data type of the input Vectors.
        
        Raises:
            TypeError: Raised if the data types of the input Vectors are different, indicating that addition is only possible between Vectors of the same data type.
        """
        if self._alg_factory != that._alg_factory:
            raise TypeError(
                f'It is only possible to add a vector of the same data type {self._alg_factory}, '
                f'but got: {that._alg_factory}'
            )
        return Vector(
            self._alg_factory, self._size,
            (np.add(self._items_x, that._items_x), np.add(self._items_y, that._items_y))
        )

    def _sub(self, that: 'Vector') -> 'Vector':
        r"""
        Subtracts a vector from the current vector.
        
        Args:
            self (Vector): The current vector.
            that (Vector): The vector to be subtracted from the current vector.
        
        Returns:
            Vector: A new vector resulting from the subtraction operation.
        
        Raises:
            TypeError: If the data types of the current vector and the vector to be subtracted are not the same.
        
        This method subtracts the components of the 'that' vector from the components of the 'self' vector. The resulting vector is returned as a new 'Vector' object.
        
        Note that the 'that' vector must have the same data type as the 'self' vector, otherwise a 'TypeError' is raised with a descriptive error message.
        """
        if self._alg_factory != that._alg_factory:
            raise TypeError(
                f'It is only possible to subtract a vector of the same data type {self._alg_factory}, '
                f'but got: {that._alg_factory}'
            )
        return Vector(
            self._alg_factory, self._size,
            (np.subtract(self._items_x, that._items_x), np.subtract(self._items_y, that._items_y))
        )

    def _mul(self, that: 'Vector') -> 'Vector':
        """
        Performs multiplication operation with another vector.
        
        Args:
            self (Vector): The current Vector object.
            that (Vector): The Vector object to be multiplied with.
        
        Returns:
            Vector: A new Vector object resulting from the multiplication operation.
        
        Raises:
            TypeError: If the data type of the current Vector and the input Vector are different.
        """
        
        if self._alg_factory != that._alg_factory:
            raise TypeError(
                f'It is only possible to multiply by a vector of the same data type {self._alg_factory}, '
                f'but got: {that._alg_factory}'
            )
        items = self._impl.mul(self._items_x, self._items_y, that._items_x, that._items_y)
        return Vector(self._alg_factory, self._size, items)

    def _dot_product(self, that: 'Vector') -> Scalar:
        r"""
        Calculate the dot product between two vectors.
        
        Args:
            self (Vector): The first vector for which the dot product will be calculated.
            that (Vector): The second vector with which the dot product will be calculated. Both vectors must be of the same data type as specified by the 'alg_factory' attribute.
            
        Returns:
            Scalar: A scalar value representing the dot product of the two input vectors.
        
        Raises:
            TypeError: If the two vectors are not of the same data type, an exception is raised indicating that the dot product can only be calculated with vectors of the same data type.
        """
        if self._alg_factory != that._alg_factory:
            raise TypeError(
                'It is only possible to calculate dot product with a vector'
                f' of the same data type {self._alg_factory}, but got: {that._alg_factory}'
            )
        x, y = self._impl.dot_product(self._items_x, self._items_y, that._items_x, that._items_y)
        return Scalar(self._alg_factory, x, y)

    def _div(self, that: 'Vector') -> 'Vector':
        r"""
        Method '_div' in the class 'Vector'.
        
        Args:
            self (Vector): The current Vector instance on which the division operation is being performed.
                            It represents the numerator vector in the division.
            that (Vector): The Vector instance to divide by. It must be of the same data type as 'self'.
                            It represents the denominator vector in the division.
        
        Returns:
            Vector: A new Vector instance resulting from the division operation between 'self' and 'that'.
        
        Raises:
            TypeError: If the data type of 'self' and 'that' vectors are not the same, a TypeError is raised.
        """
        if self._alg_factory != that._alg_factory:
            raise TypeError(
                f'It is only possible to divide over a vector of the same data type {self._alg_factory}, '
                f'but got: {that._alg_factory}'
            )
        items = self._impl.div(self._items_x, self._items_y, that._items_x, that._items_y)
        return Vector(self._alg_factory, self._size, items)

    def _full_like(self, items: Union[Tuple, np.ndarray, List]) -> 'Vector':
        r"""
        _full_like Method
        
        This method creates a new Vector instance with the same algebraic properties as the current Vector instance, using the provided items.
        
        Args:
            self (Vector): The current Vector instance.
            items (Union[Tuple, np.ndarray, List]): The items used to create the new Vector instance. The items can be a Tuple, numpy array, or List.
        
        Returns:
            Vector: A new Vector instance with the same algebraic properties as the current Vector instance.
        
        Raises:
            N/A
        """
        return Vector(self._alg_factory, items=items)

    def _ones_like(self) -> 'Vector':
        r"""
        Method _ones_like in class Vector.
        
        Args:
            self: This parameter represents the instance of the Vector class invoking the method. It is used to access attributes and methods of the class instance.
        
        Returns:
            'Vector': A new Vector object that is created with the same algebraic factory as the invoking Vector instance, where each element in the new Vector is set to the scalar value of 1.
        
        Raises:
            None.
        """
        return Vector(
            self._alg_factory, items=[Scalar.one(self._alg_factory)] * self._size
        )

    def _zeros_like(self) -> 'Vector':
        r"""
        Create a new Vector object with all elements initialized to zero.
        
        Args:
            self (Vector): The current instance of the Vector class.
        
        Returns:
            Vector: A new Vector object with the same algebra factory as the current instance, 
                    and all elements initialized to zero.
        
        Raises:
            None.
        
        Note:
            This method creates a new Vector object that has the same algebra factory as the current instance.
            The size of the new Vector object will be the same as the size of the current instance.
            Each element in the new Vector object will be initialized to zero using the Scalar.zero() method.
        
        Example:
            >>> v = Vector(alg_factory, items=[1, 2, 3])
            >>> v._zeros_like()
            Vector(alg_factory, items=[0, 0, 0])
        """
        return Vector(
            self._alg_factory, items=[Scalar.zero(self._alg_factory)] * self._size
        )
