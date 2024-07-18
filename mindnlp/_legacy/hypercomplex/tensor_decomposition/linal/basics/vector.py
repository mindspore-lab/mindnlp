from abc import ABC, abstractmethod
from typing import Any, Union
import numpy as np
from mindnlp._legacy.hypercomplex.tensor_decomposition.linal.basics.scalar import Scalar


class Vector(ABC):

    r""" 
    This class represents a vector and defines various mathematical operations that can be performed on vectors. 
    It inherits from ABC (Abstract Base Class) and provides abstract methods that must be implemented by subclasses.
    The Vector class supports operations such as addition, subtraction, multiplication, division, dot product, and more. 
    It also includes methods for creating vectors with specific values (full, ones, zeros), accessing elements, and converting to numpy arrays. 
    Instances of this class should have a positive integer size representing the length of the vector.
    Subclasses must implement the abstract methods to define specific behavior for vector operations.
    """
    def __init__(self, size: int) -> None:
        r"""
        Initialize a new instance of the Vector class.
        
        Args:
            self (object): The instance of the Vector class.
            size (int): The size of the vector to be initialized. Must be a positive integer greater than 0.
        
        Returns:
            None: This method does not return any value.
        
        Raises:
            ValueError: If the provided size is less than or equal to 0. An exception is raised with an error message indicating that 'size' must be a positive integer.
        """
        if size <= 0:
            raise ValueError(f"'size' must be a positive integer, but got {size}")
        self._size = size

    def __matmul__(self, that: 'Vector') -> Scalar:
        r"""
        __matmul__
        
        Performs matrix multiplication between two vectors.
        
        Args:
            self (Vector): The first vector participating in the matrix multiplication.
            that (Vector): The second vector participating in the matrix multiplication.
        
        Returns:
            Scalar: The result of the matrix multiplication, represented as a scalar value.
        
        Raises:
            TypeError: If the 'that' parameter is not an instance of the 'Vector' class.
        """
        return self.dot_product(that)

    def __mul__(self, that: Union['Vector', Scalar, Any]) -> 'Vector':
        r"""
        Perform multiplication operation between a Vector object and another Vector, Scalar, or Any object.
        
        Args:
            self (Vector): The Vector object on which the multiplication operation is performed.
            that (Union[Vector, Scalar, Any]): The object with which the multiplication operation is performed. 
                It can be a Vector object, a Scalar value, or any other object.
        
        Returns:
            Vector: A new Vector object resulting from the multiplication operation. 
                If 'that' is a Vector object, the result is the element-wise multiplication of the two Vectors.
                If 'that' is a Scalar value, each element of the Vector is multiplied by that Scalar value.
        
        Raises:
            TypeError: If 'that' is neither a Vector object nor a Scalar value.
        """
        if isinstance(that, Vector):
            return self.mul(that)
        else:
            return self.mul_scalar(that)

    def __add__(self, that: Union['Vector', Scalar, Any]) -> 'Vector':
        r"""
        __add__ method in the Vector class.
        
        Args:
            self (Vector): The current Vector object that the method is being called on.
                Represents the Vector instance to which elements will be added.
            that (Union[Vector, Scalar, Any]): The object to be added to the current Vector.
                Can be a Vector object, a Scalar value, or any other type.
                If a Vector object, it will be added element-wise to the current Vector.
                If a Scalar value, it will be added to all elements of the current Vector.
        
        Returns:
            Vector: A new Vector object resulting from the addition operation.
                If 'that' is a Vector object, the result will be the element-wise addition of both Vectors.
                If 'that' is a Scalar value, it will be added to all elements of the current Vector
                to generate the resulting Vector.
        
        Raises:
            None.
        """
        if isinstance(that, Vector):
            return self.add(that)
        else:
            return self.add_scalar(that)

    def __sub__(self, that: Union['Vector', Scalar, Any]) -> 'Vector':
        r"""
        Subtracts the given value from the vector.
        
        Args:
            self (Vector): The vector from which the value will be subtracted.
            that (Union[Vector, Scalar, Any]): The value to be subtracted from the vector. It can be either a Vector object, a Scalar object, or any other data type.
        
        Returns:
            Vector: A new Vector object that represents the result of the subtraction.
        
        Raises:
            TypeError: If the 'that' parameter is not of type Vector or Scalar.
        
        Note:
            - If 'that' is a Vector object, the subtraction is performed element-wise between the two vectors.
            - If 'that' is a Scalar object, the subtraction is performed by subtracting the scalar value from each element of the vector.
            - If 'that' is of any other data type, a TypeError is raised.
        
        Examples:
            # Subtracting two vectors
            v1 = Vector([1, 2, 3])
            v2 = Vector([4, 5, 6])
            result = v1.__sub__(v2)  # returns a new Vector object [1-4, 2-5, 3-6]
        
            # Subtracting a scalar value
            v3 = Vector([1, 2, 3])
            scalar = Scalar(2)
            result = v3.__sub__(scalar)  # returns a new Vector object [1-2, 2-2, 3-2]
        
            # Trying to subtract with an invalid type
            v4 = Vector([1, 2, 3])
            result = v4.__sub__('invalid')  # raises a TypeError
        
        """
        if isinstance(that, Vector):
            return self.sub(that)
        else:
            return self.sub_scalar(that)

    def __truediv__(self, that: Union['Vector', Scalar, Any]) -> 'Vector':
        r"""
        Perform the true division operation on the Vector object.
        
        Args:
            self (Vector): The Vector object on which the division operation is performed.
            that (Union[Vector, Scalar, Any]): The second operand for the division operation. It can be a Vector object, a Scalar value, or any other data type.
        
        Returns:
            Vector: A new Vector object resulting from the division operation. If the second operand is a Vector object, the division is element-wise. If the second operand is a Scalar value, the division is
scalar division.
        
        Raises:
            None: This method does not raise any exceptions.
        """
        if isinstance(that, Vector):
            return self.div(that)
        else:
            return self.div_scalar(that)

    def __str__(self) -> str:
        r"""
        Method '__str__' in the class 'Vector' converts the Vector object into a string representation.
        
        Args:
        - self (object): The Vector object itself.
        
        Returns:
        - str: A string representation of the Vector object. If the size of the Vector is less than or equal to 6, it returns a string containing all elements of the Vector enclosed in square brackets. If the
size is greater than 6, it returns a string containing the first 3 elements, followed by '...,', and then the last 3 elements, all enclosed in square brackets.
        
        Raises:
        This method does not raise any exceptions.
        """
        if self._size <= 6:
            items = [self[i] for i in range(self._size)]
            return '[' + ', '.join([str(item) for item in items]) + ']'
        else:
            items_left = [self[i] for i in range(3)]
            items_right = [self[-i - 1] for i in range(3)]
            return '[' + ', '.join([str(item) for item in items_left]) + ', ... ,' \
                + ', '.join([str(item) for item in items_right]) + ']'

    @staticmethod
    def full_like(m: 'Vector', items: Any) -> 'Vector':
        r"""
        This method creates a new Vector with the same shape and type as the input Vector m, filled with the specified items.
        
        Args:
            m (Vector): The input Vector from which the shape and type information will be used to create the new Vector.
            items (Any): The value or object used to fill the new Vector.
        
        Returns:
            Vector: A new Vector with the same shape and type as the input Vector m, filled with the specified items.
        
        Raises:
            This method does not raise any specific exceptions.
        """
        return m._full_like(items)

    @staticmethod
    def ones_like(m: 'Vector') -> 'Vector':
        r"""
        Generate a new Vector object with all elements set to 1.
        
        Args:
            m (Vector): The Vector object for which a new Vector object with all elements set to 1 will be generated.
        
        Returns:
            Vector: A new Vector object with the same shape as the input Vector 'm', but with all elements set to 1.
        
        Raises:
            This method does not raise any exceptions.
        """
        return m._ones_like()

    @staticmethod
    def zeros_like(m: 'Vector') -> 'Vector':
        r"""
        Returns a new Vector object with the same shape as the input Vector 'm', but with all elements initialized to zero.
        
        Args:
            m (Vector): The input Vector object to be used as a template for creating a new Vector object.
                It must be an instance of the Vector class.
        
        Returns:
            Vector: A new Vector object with the same shape as 'm', containing all zero elements.
        
        Raises:
            None.
        
        Note:
            The 'zeros_like' method is a static method of the Vector class. It can be called directly on the class itself, without the need for an instance of the class.
            This method is useful when you need to create a new Vector object with the same shape as an existing Vector object, but with all elements initialized to zero.
            It is particularly helpful in scenarios where you want to allocate memory for a new Vector object and initialize it to zero without explicitly specifying the shape of the new Vector object.
        
        Example:
            m = Vector([1, 2, 3])
            zeros_vector = Vector.zeros_like(m)
            # zeros_vector is a new Vector object with the same shape as m, but with all elements initialized to zero.
        """
        return m._zeros_like()

    @abstractmethod
    def __getitem__(self, key):
        r"""
        __getitem__ method
        
        This method allows accessing an element of the Vector object using the square bracket notation.
        
        Args:
            self (Vector): The Vector object itself.
            key (Any): The index or key to access the element in the Vector.
            
        Returns:
            None: This method does not return any value.
        
        Raises:
            None: This method does not raise any exceptions.
        
        """
        pass

    @abstractmethod
    def __neg__(self) -> 'Vector':
        r"""
        __neg__
        
        This method represents the negation of the Vector object.
        
        Args:
            self (Vector): The Vector object on which the negation operation is performed.
        
        Returns:
            Vector: A new Vector object resulting from the negation operation.
        
        Raises:
            This method does not raise any specific exceptions.
        """
        pass

    def get_size(self) -> int:
        r"""Get the size of the Vector.
        
        This method takes no additional parameters besides the 'self' parameter.
        
        Args:
            self: An instance of the Vector class.
        
        Returns:
            int: The size of the Vector.
        
        Raises:
            None.
        """
        return self._size

    def add(self, that: 'Vector') -> 'Vector':
        r"""
        Add a vector to another vector.
        
        Args:
            self (Vector): The current vector.
            that (Vector): The vector to be added to the current vector. It should have the same size as the current vector.
        
        Returns:
            Vector: A new vector resulting from the addition of the current vector and the provided vector.
        
        Raises:
            ValueError: If the size of the provided vector is different from the size of the current vector.
        """
        if self._size != that._size:
            raise ValueError(
                f'It is only possible to add a vector of the same size {self._size}, but got: {that._size}'
            )
        return self._add(that)

    @abstractmethod
    def add_scalar(self, that: Union[Scalar, Any]) -> 'Vector':
        r"""
        Adds a scalar value to the vector.
        
        Args:
            self (Vector): The current Vector object.
            that (Union[Scalar, Any]): The scalar value to be added to the vector. It can be of type Scalar or any other compatible type.
        
        Returns:
            Vector: A new Vector object that is the result of adding the scalar value to the current vector.
        
        Raises:
            None.
        
        Note:
            The add_scalar method performs element-wise addition of the scalar value to each element of the vector. If the scalar value is of a different type than the vector elements, it will be automatically
converted to the appropriate type before the addition takes place.
        
        Example:
            >>> v = Vector([1, 2, 3])
            >>> v.add_scalar(2)
            Vector([3, 4, 5])
        """
        pass

    def sub(self, that: 'Vector') -> 'Vector':
        r"""
        Subtracts a vector from the current vector and returns the result.
        
        Args:
            self (Vector): The current vector instance.
            that (Vector): The vector to be subtracted from the current vector. It must have the same size as the current vector.
            
        Returns:
            Vector: The resulting vector after subtracting the given vector from the current vector.
            
        Raises:
            ValueError: If the size of the given vector is different from the size of the current vector.
        """
        if self._size != that._size:
            raise ValueError(
                f'It is only possible to subtract a vector of the same size {self._size}, but got: {that._size}'
            )
        return self._sub(that)

    @abstractmethod
    def sub_scalar(self, that: Union[Scalar, Any]) -> 'Vector':
        r"""
        This method performs scalar subtraction on the Vector object.
        
        Args:
            self (Vector): The Vector object on which the scalar subtraction operation will be performed.
            that (Union[Scalar, Any]): The scalar value or another data type for the subtraction operation.
        
        Returns:
            Vector: A new Vector object resulting from the scalar subtraction operation.
        
        Raises:
            - TypeError: If the input parameter 'that' is not a valid scalar or compatible data type for the subtraction operation.
        """
        pass

    def mul(self, that: 'Vector') -> 'Vector':
        r"""
        This method multiplies two vectors element-wise and returns a new vector.
        
        Args:
            self (Vector): The current vector instance.
            that (Vector): The vector to be multiplied with the current vector.
                It must have the same size as the current vector.
        
        Returns:
            Vector: A new vector resulting from the element-wise multiplication of the current vector and the input vector.
        
        Raises:
            ValueError: If the size of the input vector is different from the size of the current vector.
                This exception is raised to indicate that the multiplication can only be performed with vectors of the same size.
        """
        if self._size != that._size:
            raise ValueError(
                f'It is only possible to multiply by a vector of the same size {self._size},'
                f' but got: {that._size}'
            )
        return self._mul(that)

    @abstractmethod
    def mul_scalar(self, that: Union[Scalar, Any]) -> 'Vector':
        r"""
        This method multiplies the vector by a scalar value.
        
        Args:
            self (Vector): The Vector object on which the scalar multiplication operation is performed.
            that (Union[Scalar, Any]): The scalar value by which the vector will be multiplied. It can be a Scalar object or any other type.
        
        Returns:
            Vector: A new Vector object resulting from the scalar multiplication operation.
        
        Raises:
            This method does not raise any exceptions.
        """
        pass

    def div(self, that: 'Vector') -> 'Vector':
        r"""
        Divides the current Vector object by another Vector object of the same size.
        
        Args:
            self (Vector): The current Vector object being operated on.
            that (Vector): The Vector object to divide the current Vector by. It must be of the same size as self.
        
        Returns:
            Vector: A new Vector object resulting from the element-wise division of self by that.
        
        Raises:
            ValueError: If the size of the Vector objects self and that are not equal, a ValueError is raised indicating that division is only possible with Vectors of the same size.
        """
        if self._size != that._size:
            raise ValueError(
                f'It is only possible to divide over a vector of the same size {self._size},'
                f' but got: {that._size}'
            )
        return self._div(that)

    @abstractmethod
    def div_scalar(self, that: Union[Scalar, Any]) -> 'Vector':
        r"""
        Divides the vector by a scalar value.
        
        Args:
            self (Vector): The vector object on which the division operation is performed.
            that (Union[Scalar, Any]): The scalar value by which the vector is divided. Can be of type Scalar or any other type that supports division with the vector.
            
        Returns:
            Vector: A new vector object resulting from the division operation.
        
        Raises:
            None: This method does not raise any exceptions.
        """
        pass
    
    @abstractmethod
    def as_array(self) -> np.ndarray:
        r"""
        Converts the Vector to a numpy array.
        
        Args:
            self: The instance of the Vector class.
        
        Returns:
            np.ndarray: A numpy array representing the Vector.
        
        Raises:
            This method does not raise any exceptions.
        """
        pass

    @abstractmethod
    def conjugate(self) -> 'Vector':
        r"""
        Method to calculate the conjugate of a Vector.
        
        Args:
            self (Vector): The Vector object on which the conjugate operation is performed.
        
        Returns:
            Vector: A new Vector object representing the conjugate of the input Vector.
        
        Raises:
            This method does not raise any exceptions.
        """
        pass

    def dot_product(self, that: 'Vector') -> Scalar:
        """
        Calculates the dot product between this Vector and another Vector.
        
        Args:
            self (Vector): The first Vector for the dot product calculation.
            that (Vector): The second Vector for the dot product calculation. It must have the same size as the first Vector.
        
        Returns:
            Scalar: The result of the dot product calculation, which is a scalar value.
        
        Raises:
            ValueError: If the size of the provided 'that' Vector is different from the size of the 'self' Vector, a ValueError is raised.
        """
        if self._size != that._size:
            raise ValueError(
                'It is only possible to calculate dot product with a vector'
                f'of the same size {self._size}, but got: {that._size}'
            )
        return self._dot_product(that)

    @abstractmethod
    def _full_like(self, items: Any) -> 'Vector':
        r"""
        This method, '_full_like', is a member of the 'Vector' class and is used to create a new 'Vector' object with the same shape and data type as the input 'items', but with all elements set to the value
specified by the 'items' parameter.
        
        Args:
            self: An instance of the 'Vector' class. This parameter is automatically passed when calling the method on an object of the class. It is used to access the attributes and methods of the object.
        
            items: The input parameter representing the source object used to determine the shape and data type of the output 'Vector' object. This parameter can have any type. It is used to specify the
desired value for all elements in the output 'Vector' object.
        
        Returns:
            A new 'Vector' object with the same shape and data type as the input 'items', but with all elements set to the value specified by the 'items' parameter. The output 'Vector' object is returned as
the result of the method.
        
        Raises:
            This method does not raise any exceptions.
        """
        pass

    @abstractmethod
    def _ones_like(self) -> 'Vector':
        r"""
        Returns a new Vector object with all elements set to 1, having the same shape as the calling Vector object.
        
        Args:
            self (Vector): The calling Vector object.
        
        Returns:
            Vector: A new Vector object with the same shape as the calling Vector object, where each element is set to 1.
        
        Raises:
            None.
        
        Note:
            This method is an abstract method and must be implemented by any concrete subclass of the Vector class.
        """
        pass

    @abstractmethod
    def _zeros_like(self) -> 'Vector':
        r"""
        This method '_zeros_like' creates a new Vector object with the same shape as the current Vector object but filled with zeros.
        
        Args:
            self (Vector): The current Vector object instance.
            
        Returns:
            Vector: A new Vector object with the same shape as the current Vector object but filled with zeros.
            
        Raises:
            This method does not raise any exceptions.
        """
        pass

    @abstractmethod
    def _add(self, that: 'Vector') -> 'Vector':
        r"""
        This method adds the components of the current Vector instance with the components of another Vector instance.
        
        Args:
            self (Vector): The current Vector instance.
            that (Vector): The Vector instance to be added to the current instance. It should be of the same dimension as the current Vector instance.
        
        Returns:
            Vector: A new Vector instance resulting from the addition of the current Vector instance and the input Vector instance.
        
        Raises:
            TypeError: If the input 'that' is not a Vector instance.
            ValueError: If the dimensions of the input 'that' Vector instance do not match the dimensions of the current Vector instance.
        """
        pass

    @abstractmethod
    def _sub(self, that: 'Vector') -> 'Vector':
        r"""
        This method performs subtraction operation between two Vector objects.
        
        Args:
            self (Vector): The Vector object on which the subtraction operation is performed.
            that (Vector): The Vector object to be subtracted from the self Vector.
        
        Returns:
            Vector: A new Vector object resulting from the subtraction operation.
        
        Raises:
            NotImplementedError: If the method is not implemented in the derived class.
            ValueError: If the dimensions of the two vectors are not equal and subtraction cannot be performed.
        """
        pass

    @abstractmethod
    def _mul(self, that: 'Vector') -> 'Vector':
        r"""
        Method that multiplies this Vector with another Vector.
        
        Args:
            self (Vector): The current Vector object to be multiplied with.
            that (Vector): The Vector object to be multiplied with self.
        
        Returns:
            Vector: A new Vector resulting from the multiplication of self and that.
        
        Raises:
            This method does not raise any exceptions.
        """
        pass

    @abstractmethod
    def _div(self, that: 'Vector') -> 'Vector':
        r"""
        Divides the vector by another vector.
        
        Args:
            self (Vector): The current vector object.
            that (Vector): The vector to divide by.
        
        Returns:
            Vector: A new vector resulting from the division.
        
        Raises:
            TypeError: If the 'that' parameter is not of type Vector.
        
        This method divides the current vector by another vector. It creates a new vector object that contains the result of the division. The 'self' parameter represents the current vector object, while the
'that' parameter represents the vector to divide by.
        
        The division operation is performed element-wise between the two vectors. Each element of the current vector is divided by the corresponding element of the 'that' vector to produce the corresponding
element of the resulting vector.
        
        The resulting vector is returned as the output of this method.
        
        Note:
            The division operation is only defined for vectors of the same length. If the lengths of the vectors differ, a TypeError is raised.
        """
        pass

    @abstractmethod
    def _dot_product(self, that: 'Vector') -> Scalar:
        r"""
        Calculates the dot product between this vector and another vector.
        
        Args:
            self (Vector): The current Vector instance.
            that (Vector): The vector to calculate the dot product with.
        
        Returns:
            Scalar: The dot product of the two vectors.
        
        Raises:
            TypeError: If either self or that is not an instance of the Vector class.
        
        This method calculates the dot product between this vector and another vector. The dot product is a scalar value that represents the similarity between the two vectors. It is calculated by multiplying
the corresponding elements of the two vectors and summing the results.
        
        Note that both self and that must be instances of the Vector class. If either parameter is not of the correct type, a TypeError will be raised.
        
        Example:
            v1 = Vector([1, 2, 3])
            v2 = Vector([4, 5, 6])
            dot_product = v1._dot_product(v2)
            # dot_product = 32
        """
        pass

