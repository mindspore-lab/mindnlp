from typing import List, Union, Optional, Tuple
import numpy as np
from mindnlp._legacy.hypercomplex.tensor_decomposition.linal.basics.matrix import Matrix as AbstractMatrix
from mindnlp._legacy.hypercomplex.tensor_decomposition.linal.basics.real.real_scalar import Scalar
from mindnlp._legacy.hypercomplex.tensor_decomposition.linal.basics.real.real_vector import Vector

NoneType = type(None)


class Matrix(AbstractMatrix):

    r""" 
    Represents a matrix object that can perform various operations such as addition, subtraction, multiplication, and more.
    
    This class inherits from AbstractMatrix and provides functionality for initializing matrices, performing arithmetic operations with scalars and other matrices, transposing, and concatenating rows and columns.
    
    Attributes:
        height (Optional[int]): The height of the matrix.
        width (Optional[int]): The width of the matrix.
        items (Union[np.ndarray, List, NoneType]): The elements of the matrix stored as a NumPy array or a list of vectors.
    
    Methods:
        __init__: Initializes the matrix with optional height, width, and items.
        __getitem__: Retrieves a submatrix, vector, or scalar from the matrix.
        __neg__: Returns the negation of the matrix.
        add_scalar: Adds a scalar value to the matrix.
        sub_scalar: Subtracts a scalar value from the matrix.
        mul_scalar: Multiplies the matrix by a scalar value.
        transpose: Transposes the matrix.
        transpose_conjugate: Returns the conjugate transpose of the matrix.
        div_scalar: Divides the matrix by a scalar value.
        as_array: Returns the matrix elements as a NumPy array.
        _init_default: Initializes the matrix with default values.
        _init_ndarray: Initializes the matrix with a NumPy array.
        _init_list: Initializes the matrix with a list of vectors.
        _add: Adds another matrix to the current matrix.
        _sub: Subtracts another matrix from the current matrix.
        _mul: Multiplies the matrix element-wise by another matrix.
        _matmul: Performs matrix multiplication with another matrix.
        _div: Divides the matrix element-wise by another matrix.
        _full_like: Creates a matrix with the same shape as the given array.
        _ones_like: Creates a matrix filled with ones.
        _zeros_like: Creates a matrix filled with zeros.
        _identity_like: Creates an identity matrix with the same dimensions.
        _concat_rows: Concatenates another matrix below the current matrix.
        _concat_cols: Concatenates another matrix to the right of the current matrix.
    """

    def __init__(self,
                 height: Optional[int] = None,
                 width: Optional[int] = None,
                 items: Union[np.ndarray, List, NoneType] = None) -> None:

        r"""
        Initializes a Matrix object with the provided parameters.
        
        Args:
            self (Matrix): The instance of the Matrix class.
            height (Optional[int]): The height dimension of the matrix. Defaults to None.
            width (Optional[int]): The width dimension of the matrix. Defaults to None.
            items (Union[np.ndarray, List, NoneType]): The items to initialize the matrix with. It can be a numpy ndarray, a list, or None. Defaults to None.
        
        Returns:
            None: This method does not return any value.
        
        Raises:
            TypeError: If 'items' is not one of: None, ndarray, or a list of vectors.
        """
        if items is None:
            height, width = self._init_default(height, width)
        elif isinstance(items, np.ndarray):
            height, width = self._init_ndarray(height, width, items)
        elif isinstance(items, List):
            height, width = self._init_list(height, width, items)
        else:
            raise TypeError(f"'items' must be one of: None, ndarray, or list of vectors, but got: {items}")
        super(Matrix, self).__init__(height, width)

    def __getitem__(self, key):

        r"""
        This method allows retrieving items from the Matrix object using the specified key.
        
        Args:
            self (Matrix): The Matrix object itself.
            key (int): The key used to access the item in the Matrix.
        
        Returns:
            None: This method does not return a value directly. Instead, it returns a Matrix, Vector, or Scalar object based on the shape of the retrieved item.
        
        Raises:
            IndexError: If the key is out of bounds for the Matrix object.
            TypeError: If the retrieved item is not a valid shape for creating a Matrix, Vector, or Scalar object.
        """
        x = self._items.__getitem__(key)
        if len(x.shape) == 2:
            return Matrix(x.shape[0], x.shape[1], x)
        elif len(x.shape) == 1:
            return Vector(x.shape[0], x)
        else:
            return Scalar(x)

    def __neg__(self) -> 'Matrix':

        r"""
        __neg__
        
        Args:
            self (Matrix): The Matrix object on which the method is called.
            
        Returns:
            Matrix: Returns a new Matrix object with the negation applied to the original Matrix.
        
        Raises:
            N/A
        """
        return Matrix(self._height, self._width, np.negative(self._items))

    def add_scalar(self, that: Union[Scalar, float, np.float64]) -> 'Matrix':

        r"""
        Adds a scalar value to the current matrix.
        
        Args:
            self (Matrix): The Matrix object to which the scalar value will be added.
            that (Union[Scalar, float, np.float64]): The scalar value to be added to the matrix. 
                It can be of type Scalar, float, or np.float64. If 'that' is of type Scalar, its numeric value will be used.
        
        Returns:
            Matrix: A new Matrix object resulting from adding the scalar value to the current matrix. 
                The dimensions of the resulting matrix will be the same as the original matrix.
        
        Raises:
            None
        """
        num = that.as_number() if isinstance(that, Scalar) else that
        return Matrix(self._height, self._width, np.add(self._items, num))

    def sub_scalar(self, that: Union[Scalar, float, np.float64]) -> 'Matrix':

        
        """
        Subtracts a scalar value from each element of the matrix.
        
        Args:
            self (Matrix): The Matrix object itself.
            that (Union[Scalar, float, np.float64]): The scalar value to subtract from the matrix elements. Can be either a Scalar instance, a float, or a np.float64.
        
        Returns:
            Matrix: A new Matrix object with the scalar value subtracted from each element.
        
        Raises:
            - TypeError: If 'that' is not an instance of Scalar, float, or np.float64.
            - ValueError: If the operation results in an invalid matrix shape or data type.
        """
          
        num = that.as_number() if isinstance(that, Scalar) else that
        return Matrix(self._height, self._width, np.subtract(self._items, num))

    def mul_scalar(self, that: Union[Scalar, float, np.float64]) -> 'Matrix':

        
        """
        Multiplies the Matrix by a scalar value.
        
        Args:
            self (Matrix): The instance of the Matrix class.
            that (Union[Scalar, float, np.float64]): The scalar value to multiply the Matrix by. 
                It can be an instance of Scalar, a float, or a np.float64.
        
        Returns:
            Matrix: A new Matrix resulting from the element-wise multiplication of the original Matrix by the scalar value.
        
        Raises:
            None
        """
        
        num = that.as_number() if isinstance(that, Scalar) else that
        return Matrix(self._height, self._width, np.multiply(self._items, num))

    def transpose(self) -> 'Matrix':

        r"""
        Transposes the matrix by swapping its rows with columns.
        
        Args:
            self (Matrix): The current matrix object.
        
        Returns:
            Matrix: A new matrix object with the rows and columns transposed.
        
        Raises:
            None.
        """
        return Matrix(self._width, self._height, np.transpose(self._items))

    def transpose_conjugate(self) -> 'Matrix':

        r"""
        Method to transpose and conjugate the current matrix.
        
        Args:
            self (Matrix): The current Matrix object to perform the transpose and conjugate operation on.
        
        Returns:
            Matrix: A new Matrix object which is the result of transposing and conjugating the current matrix.
        
        Raises:
            None.
        """
        return self.transpose()

    def div_scalar(self, that: Union[Scalar, float, np.float64]) -> 'Matrix':

        r"""
        Divides each element of the Matrix by a scalar value.
        
        Args:
            self (Matrix): The Matrix object on which the operation is performed.
            that (Union[Scalar, float, np.float64]): The scalar value to divide each element of the Matrix.
                If 'that' is of type Scalar, the value of 'that' is extracted using the as_number() method.
                The scalar value can also be of type float or np.float64.
        
        Returns:
            Matrix: A new Matrix object with the resulting values after dividing each element of the original Matrix by the scalar value.
        
        Raises:
            None.
        
        Note:
            The division operation is performed element-wise between the Matrix and the scalar value.
        
        Example:
            >>> matrix = Matrix([[1, 2, 3], [4, 5, 6]])
            >>> scalar = 2
            >>> result = matrix.div_scalar(scalar)
            >>> print(result)
            Matrix([[0.5, 1.0, 1.5], [2.0, 2.5, 3.0]])
        
            In the above example, each element of the Matrix is divided by the scalar value 2, resulting in a new Matrix with the updated values.
        """
        num = that.as_number() if isinstance(that, Scalar) else that
        return Matrix(self._height, self._width, np.divide(self._items, num))

    def as_array(self) -> np.ndarray:

        r"""
        Converts the Matrix object to a NumPy array.
        
        Args:
            self (Matrix): The Matrix object to be converted.
        
        Returns:
            np.ndarray: The NumPy array representation of the Matrix object.
        
        Raises:
            None.
        
        This method takes a single parameter, `self`, which is an instance of the Matrix class. The purpose of this parameter is to provide access to the current instance of the Matrix object. There are no restrictions on the type of the `self` parameter.
        
        The method returns a NumPy array (`np.ndarray`) that represents the Matrix object. The returned array has the same shape and contents as the Matrix object. The purpose of this return value is to enable further manipulation and analysis using the powerful NumPy library.
        
        This method does not raise any exceptions.
        """
        return np.array(self._items)

    def _init_default(self,
                      height: Optional[int],
                      width: Optional[int]) -> Tuple[int, int]:

        r"""
        Initializes a Matrix with default values based on the provided height and width.
        
        Args:
            self (Matrix): The instance of the Matrix class.
            height (Optional[int]): The height of the Matrix. Must be a non-negative integer.
            width (Optional[int]): The width of the Matrix. Must be a non-negative integer.
            
        Returns:
            Tuple[int, int]: A tuple containing the height and width of the Matrix after initialization.
        
        Raises:
            ValueError: If either 'height' or 'width' is None.
        """
        if height is None or width is None:
            raise ValueError(f"Either 'items' or both 'height' or 'width' must not be None")
        self._items = np.zeros((height, width), dtype=np.float64)
        return height, width

    def _init_ndarray(self,
                      height: Optional[int],
                      width: Optional[int],
                      items: np.ndarray) -> Tuple[int, int]:

        
        """
        Initializes the ndarray for the Matrix class.
        
        Args:
            self (Matrix): The instance of the Matrix class.
            height (Optional[int]): The height of the ndarray. If None, it defaults to the first dimension of the 'items' array.
            width (Optional[int]): The width of the ndarray. If None, it defaults to the second dimension of the 'items' array.
            items (np.ndarray): The input array to be converted to a numpy ndarray.
        
        Returns:
            Tuple[int, int]: A tuple containing the height and width of the ndarray after initialization.
        
        Raises:
            TypeError: If the input array 'items' is not a valid numpy ndarray.
        """
        
        if height is None:
            height = items.shape[0]
        if width is None:
            width = items.shape[1]
        self._items = np.array(items).astype(np.float64)
        return height, width

    def _init_list(self,
                   height: Optional[int],
                   width: Optional[int],
                   items: List[Vector]) -> Tuple[int, int]:

        r"""
        This method initializes a Matrix object with the provided height, width, and a list of vectors.
        
        Args:
            self: The instance of the Matrix class.
            height (Optional[int]): The height of the matrix. If None, it will be inferred from the length of the items list.
            width (Optional[int]): The width of the matrix. If None, it will be inferred from the first vector in the items list.
            items (List[Vector]): A list of vectors to populate the matrix.
        
        Returns:
            Tuple[int, int]: A tuple containing the height and width of the initialized matrix.
        
        Raises:
            ValueError: 
                - If the items list is empty or contains invalid vectors when height is inferred.
                - If the length of the items list does not match the provided height.
                - If the items list contains vectors of varying sizes or types inconsistent with the provided height and width.
        """
        if height is None:
            height = len(items)
            if height <= 0:
                raise ValueError(f"'items' must be a list of some vectors, but got: {items}")
        elif len(items) != height:
            raise ValueError(f"'items' must be a list of {height} vectors, but got: {items}")
        if width is None:
            if not isinstance(items[0], Vector):
                raise ValueError(f"'items' must be a list of {height} vectors, but got: {items}")
            width = items[0].get_size()
        if len(items) != height or any(not isinstance(v, Vector) or v.get_size() != width for v in items):
            raise ValueError(f"'items' must be a list of {height} vectors, each of size {width}, but got: {items}")
        self._items = np.reshape(
            np.concatenate([v.as_array() for v in items], axis=0),
            (height, width)
        ).astype(np.float64)
        return height, width

    def _add(self, that: 'Matrix') -> 'Matrix':

        r"""
        Method _add in the class Matrix.
        
        Args:
            self (Matrix): The current Matrix object on which the method is called.
                Purpose: Represents the Matrix object to which another Matrix object will be added.
                Restrictions: Must be an instance of the Matrix class.
        
            that (Matrix): The Matrix object that will be added to the current Matrix object.
                Purpose: Represents the Matrix object that will be added to the current Matrix object.
                Restrictions: Must be an instance of the Matrix class.
        
        Returns:
            Matrix: A new Matrix object that is the result of adding the elements of the current Matrix object
                with the elements of the provided Matrix object.
                Purpose: Represents the result of adding the two Matrix objects together.
        
        Raises:
            No specific exceptions are documented to be raised by this method.
        """
        return Matrix(self._height, self._width, np.add(self._items, that._items))

    def _sub(self, that: 'Matrix') -> 'Matrix':

        r"""
        Subtracts the values of a given Matrix object from the calling Matrix object and returns a new Matrix object with the result.
        
        Args:
            self (Matrix): The calling Matrix object.
            that (Matrix): The Matrix object to be subtracted from the calling Matrix object.
            
        Returns:
            Matrix: A new Matrix object with the values resulting from the subtraction of the two matrices.
            
        Raises:
            None.
        """
        return Matrix(self._height, self._width, np.subtract(self._items, that._items))

    def _mul(self, that: 'Matrix') -> 'Matrix':

        r"""
        Method for multiplying two matrices.
        
        Args:
            self (Matrix): The first matrix to be multiplied.
                This is an instance of the Matrix class representing the first matrix.
            that (Matrix): The second matrix to be multiplied with self.
                This is an instance of the Matrix class representing the second matrix.
        
        Returns:
            Matrix: A new Matrix instance representing the result of multiplying self with that.
                The resulting Matrix has the same height as self and the same width as that.
        
        Raises:
            TypeError: If the 'that' parameter is not an instance of the Matrix class.
            ValueError: If the dimensions of the matrices are not compatible for multiplication.
        """
        return Matrix(self._height, self._width, np.multiply(self._items, that._items))

    def _matmul(self, that: 'Matrix') -> 'Matrix':

        r"""
        Performs matrix multiplication between the current matrix and another matrix.
        
        Args:
            self (Matrix): The current matrix instance.
            that (Matrix): The matrix to be multiplied with the current matrix.
            
        Returns:
            Matrix: A new matrix resulting from the multiplication of the current matrix and the given matrix.
        
        Raises:
            None.
        
        Note:
            - The number of columns in the current matrix must be equal to the number of rows in the given matrix.
            - If the above condition is not satisfied, a ValueError will be raised.
        """
        return Matrix(self._height, that._width, np.matmul(self._items, that._items))

    def _div(self, that: 'Matrix') -> 'Matrix':

        r"""
        Method _div in the class Matrix performs element-wise division between two Matrix objects.
        
        Args:
            self (Matrix): The first Matrix object to perform division on. It is the current instance of the Matrix class.
            
            that (Matrix): The second Matrix object to divide the first Matrix object by. It must have the same dimensions as the current Matrix object.
        
        Returns:
            Matrix: A new Matrix object resulting from element-wise division of the current Matrix object by the provided Matrix object 'that'.
        
        Raises:
            ValueError: If the dimensions of the two Matrix objects are not compatible for element-wise division.
            TypeError: If the 'that' parameter is not of type Matrix.
            ZeroDivisionError: If any element in the 'that' Matrix object is 0, causing division by zero.
        """
        return Matrix(self._height, self._width, np.divide(self._items, that._items))

    def _full_like(self, items: Union[np.ndarray, List]) -> 'Matrix':

        r"""
        Performs a deep copy of the given items and creates a new 'Matrix' object with the same shape and data type as the original matrix.
        
        Args:
            self (Matrix): The instance of the 'Matrix' class that this method is called on.
            items (Union[np.ndarray, List]): The input items which can be either a numpy array or a list. The items represent the data that will be used to populate the new 'Matrix' object. If a numpy array is provided, a deep copy of it will be made. If a list is provided, a numpy array will be created from it.
        
        Returns:
            Matrix: A new 'Matrix' object that has the same shape and data type as the original matrix, but with the values from the input items.
        
        Raises:
            None.
        
        Note:
            This method is useful for creating a new 'Matrix' object with the same shape and data type as an existing matrix, while providing new data values.
        
        Example:
            
            original_matrix = Matrix([[1, 2], [3, 4]])
            new_matrix = original_matrix._full_like([[5, 6], [7, 8]])
            print(new_matrix)
            
            Output:
            
            Matrix([[5, 6], [7, 8]])
            
        
            In the example above, the '_full_like' method is called on the 'original_matrix' object to create a new 'Matrix' object with the same shape and data type. The new 'Matrix' object is then printed, showing the updated values.
        """
        return Matrix(items=items)

    def _ones_like(self) -> 'Matrix':

        r"""
        Returns a new Matrix object with the same dimensions as the calling Matrix object, where all elements are set to a value of 1.
        
        Args:
            self (Matrix): The calling Matrix object.
        
        Returns:
            Matrix: A new Matrix object with the same dimensions as the calling Matrix object, where all elements are set to 1.
        
        Raises:
            None.
        """
        return Matrix(items=[Vector(items=[Scalar.one()] * self._width)] * self._height)

    def _zeros_like(self) -> 'Matrix':

        
        """
        Method _zeros_like in the class Matrix.
        
        Args:
            self (Matrix): The current Matrix object.
            
        Returns:
            Matrix: A new Matrix object with the same dimensions as the current Matrix, where all elements are initialized to zero.
            
        Raises:
            None
        """
        
        return Matrix(items=[Vector(items=[Scalar.zero()] * self._width)] * self._height)

    def _identity_like(self) -> 'Matrix':

        r"""
        Creates a new Matrix object that has the same dimensions as the current Matrix object, with the diagonal elements set to 1 and all other elements set to 0.
        
        Args:
            self (Matrix): The current Matrix object.
        
        Returns:
            Matrix: A new Matrix object with the same dimensions as the current Matrix object, where the diagonal elements are set to 1 and all other elements are set to 0.
        
        Raises:
            None.
        
        """
        items = [
            Vector(items=[Scalar.zero()] * i + [Scalar.one()] + [Scalar.zero()] * (self._width - i - 1))
            for i in range(self._height)
        ]
        return Matrix(items=items)

    def _concat_rows(self, that: 'Matrix') -> 'Matrix':

        r"""
        Concatenates the rows of the current matrix with the rows of another matrix.
        
        Args:
            self (Matrix): The current matrix object.
            that (Matrix): Another matrix object to concatenate the rows with.
        
        Returns:
            Matrix: A new matrix object resulting from the concatenation of rows.
        
        Raises:
            None.
        
        This method takes two parameters: self, which represents the current matrix object, and that, which represents another matrix object. The method concatenates the rows of self with the rows of that and returns a new matrix object.
        
        The self parameter is of type Matrix and is used to access the current matrix object.
        
        The that parameter is also of type Matrix and is used to provide the matrix object to concatenate the rows with.
        
        The method returns a new matrix object of type Matrix, resulting from the concatenation of rows. The height of the resulting matrix is the sum of the heights of self and that, while the width remains the same as self. The items of the resulting matrix are obtained by concatenating the items of self and that along the row axis.
        
        No exceptions are raised by this method.
        """
        items = np.concatenate((self._items, that._items), axis=0)
        return Matrix(height=self._height + that._height, width=self._width, items=items)

    def _concat_cols(self, that: 'Matrix') -> 'Matrix':

        r"""
        Concatenates the columns of the current Matrix with another Matrix.
        
        Args:
            self (Matrix): The current Matrix object.
            that (Matrix): The Matrix object to be concatenated with the current Matrix.
        
        Returns:
            Matrix: A new Matrix resulting from the concatenation of columns.
        
        Raises:
            TypeError: If the 'that' parameter is not of type Matrix.
            ValueError: If the heights of the two matrices do not match, preventing column-wise concatenation.
        """
        items = np.concatenate((self._items, that._items), axis=1)
        return Matrix(height=self._height, width=self._width + that._width, items=items)
