from typing import Type, TypeVar, Tuple, List, Union, Optional
import numpy as np
from mindnlp._legacy.hypercomplex.tensor_decomposition.linal.basics.matrix import Matrix as AbstractMatrix
from mindnlp._legacy.hypercomplex.tensor_decomposition.linal.basics.hypercomplex.hc_algebra_factory import AlgebraFactory
from mindnlp._legacy.hypercomplex.tensor_decomposition.linal.basics.hypercomplex.hc_scalar import Scalar
from mindnlp._legacy.hypercomplex.tensor_decomposition.linal.basics.hypercomplex.hc_vector import Vector

TAlgebraFactory = TypeVar('TAlgebraFactory', bound=AlgebraFactory)
NoneType = type(None)


class Matrix(AbstractMatrix):

    r"""
    The Matrix class represents a matrix using a given algebra factory. It provides methods for various matrix operations such as addition, subtraction, multiplication, division, transpose, conjugate
transpose, and more. The class supports initialization from various input types including tuples, NumPy arrays, and lists of vectors. Additionally, it provides methods for creating matrices with specific
properties such as ones, zeros, and identity matrices, as well as concatenating matrices along rows and columns.
    
    This class inherits from the AbstractMatrix class and provides an implementation for the matrix-specific functionality.
    
    """
    def __init__(self,
                 algebra_factory: Type[TAlgebraFactory],
                 height: Optional[int] = None,
                 width: Optional[int] = None,
                 items: Union[Tuple, np.ndarray, List, NoneType] = None) -> None:
        r"""
        Initializes a Matrix object.
        
        Args:
            algebra_factory (Type[TAlgebraFactory]): The algebra factory used to create matrix implementations.
            height (Optional[int]): The height of the matrix. Defaults to None.
            width (Optional[int]): The width of the matrix. Defaults to None.
            items (Union[Tuple, np.ndarray, List, NoneType]): The initial items to populate the matrix. 
                It can be a Tuple, np.ndarray, List, or None. Defaults to None.
        
        Returns:
            None. This method does not return any value explicitly.
        
        Raises:
            TypeError: If 'items' is not None, ndarray, list of vectors, or Tuple.
        """
        self._alg_factory = algebra_factory
        self._impl = algebra_factory().get_matrix_impl()
        if items is None:
            height, width = self._init_default(height, width)
        elif isinstance(items, Tuple):
            height, width = self._init_tuple(height, width, items)
        elif isinstance(items, np.ndarray):
            height, width = self._init_ndarray(height, width, items)
        elif isinstance(items, List):
            height, width = self._init_list(height, width, items)
        else:
            raise TypeError(f"'items' must be one of: None, ndarray, list of vectors, or Tuple, but got: {items}")
        super(Matrix, self).__init__(height, width)

    def __getitem__(self, key):
        """
        __getitem__
        
        Args:
            self (Matrix): The Matrix instance on which the method is called.
            key: The key used to access the elements in the matrix.
        
        Returns:
            None: This method does not return a value directly, but it returns an instance of Matrix, Vector, or Scalar based on the shape of the accessed elements.
        
        Raises:
            IndexError: If the key is out of range for the matrix dimensions.
            ValueError: If the key is not valid or if the shape of the accessed elements is not supported.
            TypeError: If the accessed elements are not compatible with the Matrix, Vector, or Scalar types.
        """
        x = self._items_x.__getitem__(key)
        y = self._items_y.__getitem__(key)
        if len(x.shape) == 2:
            return Matrix(self._alg_factory, x.shape[0], x.shape[1], (x, y))
        elif len(x.shape) == 1:
            return Vector(self._alg_factory, x.shape[0], (x, y))
        else:
            return Scalar(self._alg_factory, x, y)

    def __neg__(self) -> 'Matrix':
        r"""
        Negates the elements of the matrix.
        
        Args:
            self (Matrix): The matrix object itself.
                This parameter is used to access the matrix attributes and perform the negation operation.
        
        Returns:
            Matrix: A new Matrix object with the negated elements.
                The method returns a new Matrix object with the negated values of the original Matrix.
        
        Raises:
            None.
        """
        return Matrix(
            self._alg_factory, self._height, self._width,
            (np.negative(self._items_x), np.negative(self._items_y))
        )

    def get_algebra_type(self) -> Type[AlgebraFactory]:
        r"""
        This method returns the type of algebra factory associated with the Matrix object.
        
        Args:
            self: The instance of the Matrix class.
            
        Returns:
            Type[AlgebraFactory]: The type of algebra factory associated with the Matrix object.
        
        Raises:
            No specific exceptions are raised by this method.
        """
        return self._alg_factory

    def add_scalar(self, that: Union[Scalar, Tuple[Union[float, np.float64], Union[float, np.float64]]]) -> 'Matrix':
        r"""
        This method 'add_scalar' in the class 'Matrix' adds a scalar value to the matrix.
        
        Args:
            self (Matrix): The matrix object to which a scalar value will be added.
            that (Union[Scalar, Tuple[Union[float, np.float64], Union[float, np.float64]]]): The scalar value to be added to the matrix. It can either be a Scalar object or a tuple containing two elements,
each being a float or np.float64. If 'that' is a Scalar object, it must have the same data type as the matrix defined by 'self._alg_factory'.
        
        Returns:
            Matrix: A new Matrix object resulting from adding the scalar value to the original matrix. The new matrix has the same algebra factory, height, and width as the original matrix, with each element
being the sum of the corresponding elements of the original matrix and the scalar value.
        
        Raises:
            TypeError: If 'that' is a Scalar object with a different data type than the matrix defined by 'self._alg_factory', a TypeError is raised with a message indicating that only scalars of the same data
type are allowed for addition.
        """
        if isinstance(that, Scalar) and that.get_algebra_type() != self._alg_factory:
            raise TypeError(
                f'It is only possible to add a scalar of the same data type {self._alg_factory}, '
                f'but got: {that.get_algebra_type()}'
            )
        return Matrix(
            self._alg_factory, self._height, self._width,
            (np.add(self._items_x, that[0]), np.add(self._items_y, that[1]))
        )

    def sub_scalar(self, that: Union[Scalar, Tuple[Union[float, np.float64], Union[float, np.float64]]]) -> 'Matrix':
        r"""
        Subtracts a scalar value from each element in the Matrix.
        
        Args:
            self (Matrix): The Matrix object performing the subtraction.
            that (Union[Scalar, Tuple[Union[float, np.float64], Union[float, np.float64]]]): The value to be subtracted from each element of the Matrix. 
                It can be either a Scalar object or a tuple of two values representing the x and y components respectively. 
                If the value is a Scalar, it must have the same algebra type as the Matrix. 
        
        Returns:
            Matrix: A new Matrix object with the same dimensions as the original Matrix, where each element has been subtracted by the given scalar value.
        
        Raises:
            TypeError: If the 'that' parameter is a Scalar object and its algebra type is different from the algebra type of the Matrix. 
                In this case, the subtraction operation is only possible with a scalar of the same data type as the Matrix.
        
        """
        if isinstance(that, Scalar) and that.get_algebra_type() != self._alg_factory:
            raise TypeError(
                f'It is only possible to subtract a scalar of the same data type {self._alg_factory}, '
                f'but got: {that.get_algebra_type()}'
            )
        return Matrix(
            self._alg_factory, self._height, self._width,
            (np.subtract(self._items_x, that[0]), np.subtract(self._items_y, that[1]))
        )

    def mul_scalar(self, that: Union[Scalar, Tuple[Union[float, np.float64], Union[float, np.float64]]]) -> 'Matrix':
        """
        Perform scalar multiplication on the Matrix.
        
        Args:
            self (Matrix): The Matrix object on which the scalar multiplication operation is performed.
            that (Union[Scalar, Tuple[Union[float, np.float64], Union[float, np.float64]]]): The scalar value or a tuple containing two scalar values to be multiplied with the Matrix. 
                If 'that' is a Scalar instance, it should have the same data type as the Matrix object's algebra type (self._alg_factory).
        
        Returns:
            Matrix: A new Matrix object resulting from the scalar multiplication operation.
        
        Raises:
            TypeError: If 'that' is a Scalar instance with a different data type than the Matrix object's algebra type (self._alg_factory).
        """
        if isinstance(that, Scalar) and that.get_algebra_type() != self._alg_factory:
            raise TypeError(
                f'It is only possible to multiply by a scalar of the same data type {self._alg_factory}, '
                f'but got: {that.get_algebra_type()}'
            )
        items = self._impl.mul(self._items_x, self._items_y, that[0], that[1])
        return Matrix(self._alg_factory, self._height, self._width, items)

    def transpose(self) -> 'Matrix':
        r"""
        Transposes the current matrix by swapping its rows with columns.
        
        Args:
            self (Matrix): The current Matrix object.
        
        Returns:
            Matrix: A new Matrix object representing the transposed matrix.
        
        Raises:
            None.
        
        This method takes no additional parameters besides 'self'. It returns a new Matrix object that has the same elements as the current matrix, but with its rows and columns interchanged. The transposed
matrix will have dimensions 'width' x 'height', where 'width' is the number of columns in the original matrix and 'height' is the number of rows.
        
        Example:
            # Create a 2x3 matrix
            matrix = Matrix([[1, 2, 3], [4, 5, 6]])
            
            # Transpose the matrix
            transposed = matrix.transpose()
            
            # The original matrix:
            # 1 2 3
            # 4 5 6
            
            # The transposed matrix:
            # 1 4
            # 2 5
            # 3 6
        """
        return Matrix(
            self._alg_factory, self._width, self._height,
            (np.transpose(self._items_x), np.transpose(self._items_y))
        )

    def transpose_conjugate(self) -> 'Matrix':
        r"""
        Method to compute the transpose conjugate of the matrix.
        
        Args:
            self (Matrix): The Matrix object on which the transpose conjugate operation is performed.
                It is an instance of the Matrix class.
                
        Returns:
            Matrix: A new Matrix object that represents the transpose conjugate of the original matrix.
            The dimensions of the new matrix will be swapped compared to the original matrix,
            and the elements will be conjugated.
        
        Raises:
            None
        """
        return Matrix(
            self._alg_factory, self._width, self._height,
            (np.transpose(self._items_x), np.negative(np.transpose(self._items_y)))
        )

    def div_scalar(self, that: Union[Scalar, Tuple[Union[float, np.float64], Union[float, np.float64]]]) -> 'Matrix':
        r"""
        This method performs scalar division on the matrix.
        
        Args:
            self (Matrix): The instance of the Matrix class.
            that (Union[Scalar, Tuple[Union[float, np.float64], Union[float, np.float64]]]): The scalar value or tuple of scalar values with which the matrix will be divided. If a Scalar object is provided,
its algebra type must match the algebra type of the matrix.
        
        Returns:
            Matrix: A new Matrix object resulting from the division operation.
        
        Raises:
            TypeError: If the provided scalar has a different algebra type than the matrix.
        """
        if isinstance(that, Scalar) and that.get_algebra_type() != self._alg_factory:
            raise TypeError(
                f'It is only possible to divide over a scalar of the same data type {self._alg_factory}, '
                f'but got: {that.get_algebra_type()}'
            )
        items = self._impl.div(self._items_x, self._items_y, that[0], that[1])
        return Matrix(self._alg_factory, self._height, self._width, items)

    def as_array(self) -> np.ndarray:
        r"""
        Converts the Matrix object to a NumPy ndarray.
        
        Args:
            self (Matrix): The Matrix object to be converted.
        
        Returns:
            np.ndarray: The converted Matrix object as a NumPy ndarray.
        
        Raises:
            None.
        """
        return np.reshape(
            np.array(list(zip(np.ravel(self._items_x), np.ravel(self._items_y)))),
            (self._height, self._width, 2)
        )

    def _init_default(self,
                      height: Optional[int],
                      width: Optional[int]) -> Tuple[int, int]:
        r"""
        Initialize the default matrix with given height and width.
        
        Args:
            self (Matrix): The instance of the Matrix class.
            height (Optional[int]): The height of the matrix. Must not be None.
            width (Optional[int]): The width of the matrix. Must not be None.
        
        Returns:
            Tuple[int, int]: A tuple containing the height and width of the initialized matrix.
        
        Raises:
            ValueError: If either 'height' or 'width' is None, a ValueError is raised.
        """
        if height is None or width is None:
            raise ValueError(f"Either 'items' or both 'height' or 'width' must not be None")
        self._items_x = np.zeros((height, width), dtype=np.float64)
        self._items_y = np.zeros((height, width), dtype=np.float64)
        return height, width

    def _init_tuple(self,
                    height: Optional[int],
                    width: Optional[int],
                    items: Tuple[np.ndarray, np.ndarray]) -> Tuple[int, int]:
        r"""
        Initializes the Matrix object with the given parameters.
        
        Args:
            self (Matrix): The current instance of the Matrix class.
            height (Optional[int]): The height of the matrix. If not provided, it is determined by the height of the 'items' array.
            width (Optional[int]): The width of the matrix. If not provided, it is determined by the width of the 'items' array.
            items (Tuple[np.ndarray, np.ndarray]): A tuple containing two numpy arrays representing the elements of the matrix. The first array represents the x-coordinate values, and the second array
represents the y-coordinate values.
        
        Returns:
            Tuple[int, int]: A tuple containing the height and width of the initialized matrix.
        
        Raises:
            ValueError: If the 'items' tuple does not contain exactly two elements.
            ValueError: If the elements of the 'items' tuple are not 2-dimensional arrays.
            ValueError: If the dimensions of the 'items' arrays do not match the specified height and width.
        
        """
        if len(items) != 2:
            raise ValueError(f"'items' must be a 2-element tuple, but got: {items}")
        self._items_x = items[0]
        self._items_y = items[1]
        if not isinstance(self._items_x, np.ndarray) or len(self._items_x.shape) != 2:
            raise ValueError(
                f"elements of 'items' must be 2d arrays, but got: {items}"
            )
        if not isinstance(self._items_y, np.ndarray) or len(self._items_y.shape) != 2:
            raise ValueError(
                f"elements of 'items' must be 2d arrays, but got: {items}"
            )
        if height is None:
            height = self._items_x.shape[0]
        if width is None:
            width = self._items_x.shape[1]
        if self._items_x.shape[0] != height or self._items_x.shape[1] != width:
            raise ValueError(
                f"elements of 'items' must be 2d arrays of dimensions {height}x{width}, but got: {items}"
            )
        if self._items_y.shape[0] != height or self._items_y.shape[1] != width:
            raise ValueError(
                f"elements of 'items' must be 2d arrays of dimensions {height}x{width}, but got: {items}"
            )
        return height, width

    def _init_ndarray(self,
                      height: Optional[int],
                      width: Optional[int],
                      items: np.ndarray) -> Tuple[int, int]:
        r"""
        Initializes the ndarray for the Matrix class.
        
        Args:
            self (Matrix): The Matrix instance.
            height (Optional[int]): The height of the ndarray. If None, it will be inferred from the items parameter.
            width (Optional[int]): The width of the ndarray. If None, it will be inferred from the items parameter.
            items (np.ndarray): The input ndarray from which the height and width are inferred, and the items_x and items_y arrays are forwarded.
        
        Returns:
            Tuple[int, int]: A tuple containing the height and width of the ndarray.
        
        Raises:
            - ValueError: If the dimensions of the items ndarray do not match the specified height and width.
            - TypeError: If the items parameter is not a valid numpy ndarray.
        """
        if height is None:
            height = items.shape[0]
        if width is None:
            width = items.shape[1]
        all_items = np.ravel(items)
        self._items_x = np.reshape(all_items[::2], (height, width)).astype(np.float64)
        self._items_y = np.reshape(all_items[1::2], (height, width)).astype(np.float64)
        return height, width

    def _init_list(self,
                   height: Optional[int],
                   width: Optional[int],
                   items: List[Vector]) -> Tuple[int, int]:
        r"""
        Initializes the list of vectors for the Matrix class.
        
        Args:
            self: The instance of the Matrix class.
            height (Optional[int]): The height of the matrix. If None, it will be determined based on the length of the 'items' list.
            width (Optional[int]): The width of the matrix. If None, it will be determined based on the size of the vectors in the 'items' list.
            items (List[Vector]): A list of vectors to be used as the elements of the matrix.
        
        Returns:
            Tuple[int, int]: A tuple containing the height and width of the initialized matrix.
        
        Raises:
            ValueError: 
                - If 'items' is an empty list or contains vectors of inconsistent sizes when 'height' is None.
                - If the length of 'items' does not match the specified 'height'.
                - If the vectors in 'items' do not match the specified 'height' or 'width'.
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
        if any(
            not isinstance(v, Vector) or v.get_algebra_type() != self._alg_factory or v.get_size() != width
            for v in items
        ):
            raise ValueError(f"'items' must be a list of {height} vectors, each of size {width}, but got: {items}")
        all_items = np.ravel(np.concatenate([v.as_array() for v in items], axis=0))
        self._items_x = np.reshape(all_items[::2], (height, width)).astype(np.float64)
        self._items_y = np.reshape(all_items[1::2], (height, width)).astype(np.float64)
        return height, width

    def _add(self, that: 'Matrix') -> 'Matrix':
        r"""
        Method _add in the class Matrix performs element-wise addition between two matrices of the same data type.
        
        Args:
            self (Matrix): The instance of the Matrix class on which the method is called.
                It represents the first matrix involved in the addition operation.
            that (Matrix): The Matrix object to be added to the self matrix.
                It must be of the same data type as self for the addition operation to be valid.
        
        Returns:
            Matrix: A new Matrix object resulting from the element-wise addition of the two input matrices.
                The dimensions of the resulting matrix will be the same as the dimensions of the input matrices.
        
        Raises:
            TypeError: Raised if the data types of the two input matrices are different.
                The message will indicate that only matrices of the same data type can be added.
        """
        if self._alg_factory != that._alg_factory:
            raise TypeError(
                f'It is only possible to add a matrix of the same data type {self._alg_factory}, '
                f'but got: {that._alg_factory}'
            )
        return Matrix(
            self._alg_factory, self._height, self._width,
            (np.add(self._items_x, that._items_x), np.add(self._items_y, that._items_y))
        )

    def _sub(self, that: 'Matrix') -> 'Matrix':
        r"""
        Subtracts another Matrix from the current Matrix.
        
        Args:
            self (Matrix): The current Matrix object.
            that (Matrix): Another Matrix object to be subtracted from the current Matrix. It should have the same data type as the current Matrix.
        
        Returns:
            Matrix: A new Matrix object resulting from the subtraction of the two matrices.
        
        Raises:
            TypeError: If the data type of the current Matrix and the provided Matrix are different, a TypeError is raised. The error message will indicate that it is only possible to subtract a matrix of the
same data type.
        
        """
        if self._alg_factory != that._alg_factory:
            raise TypeError(
                f'It is only possible to subtract a matrix of the same data type {self._alg_factory}, '
                f'but got: {that._alg_factory}'
            )
        return Matrix(
            self._alg_factory, self._height, self._width,
            (np.subtract(self._items_x, that._items_x), np.subtract(self._items_y, that._items_y))
        )

    def _mul(self, that: 'Matrix') -> 'Matrix':
        r"""
        This method multiplies the current Matrix object with another Matrix object.
        
        Args:
            self (Matrix): The current Matrix object that will be multiplied with the 'that' Matrix object.
            that (Matrix): The Matrix object that will be multiplied with the current Matrix object. It must be of the same data type as the current Matrix object.
        
        Returns:
            Matrix: The resulting Matrix object after the multiplication operation.
        
        Raises:
            TypeError: If the 'that' Matrix object is not of the same data type as the current Matrix object.
        """
        if self._alg_factory != that._alg_factory:
            raise TypeError(
                f'It is only possible to multiply by a matrix of the same data type {self._alg_factory}, '
                f'but got: {that._alg_factory}'
            )
        items = self._impl.mul(self._items_x, self._items_y, that._items_x, that._items_y)
        return Matrix(self._alg_factory, self._height, self._width, items)

    def _matmul(self, that: 'Matrix') -> 'Matrix':
        r"""
        Performs matrix multiplication between two Matrix objects and returns a new Matrix object.
        
        Args:
            self (Matrix): The calling Matrix object.
            that (Matrix): The Matrix object to be multiplied with self.
        
        Returns:
            Matrix: A new Matrix object resulting from the matrix multiplication.
        
        Raises:
            TypeError: If the data types of self and that do not match.
        
        Note:
            The matrix multiplication operation can only be performed between Matrix objects of the same data type. 
            If the data types of self and that do not match, a TypeError is raised.
        
        Example:
            >>> m1 = Matrix(alg_factory, height, width, items)
            >>> m2 = Matrix(alg_factory, height, width, items)
            >>> result = m1._matmul(m2)
        """
        if self._alg_factory != that._alg_factory:
            raise TypeError(
                f'It is only possible to multiply by a matrix of the same data type {self._alg_factory}, '
                f'but got: {that._alg_factory}'
            )
        items = self._impl.matmul(self._items_x, self._items_y, that._items_x, that._items_y)
        return Matrix(self._alg_factory, self._height, that._width, items)

    def _div(self, that: 'Matrix') -> 'Matrix':
        r"""
        Performs division operation between two matrices.
        
        Args:
            self (Matrix): The current Matrix instance.
            that (Matrix): The Matrix instance to divide self with.
        
        Returns:
            Matrix: A new Matrix instance resulting from the division operation.
        
        Raises:
            TypeError: If the data types of self and that matrices are different. Only matrices of the same data type can be divided.
        
        """
        if self._alg_factory != that._alg_factory:
            raise TypeError(
                f'It is only possible to divide over a matrix of the same data type {self._alg_factory}, '
                f'but got: {that._alg_factory}'
            )
        items = self._impl.div(self._items_x, self._items_y, that._items_x, that._items_y)
        return Matrix(self._alg_factory, self._height, self._width, items)

    def _full_like(self, items: Union[Tuple, np.ndarray, List]) -> 'Matrix':
        r"""
        Performs a deep copy of the current matrix with the same shape as the provided 'items'.
        
        Args:
            self: The current 'Matrix' instance.
            items (Union[Tuple, np.ndarray, List]): The data structure containing the shape information for the new matrix. 
                It can be a tuple, a NumPy array, or a list. If a tuple is provided, it should contain the dimensions 
                for the new matrix. If a NumPy array or a list is provided, the shape will be inferred from the provided 
                data structure.
        
        Returns:
            Matrix: A new 'Matrix' object with the same shape as the provided 'items', containing a deep copy of the 
            current matrix data.
        
        Raises:
            None.
        
        Note:
            This method is useful when you need to create a new 'Matrix' instance with the same shape as the provided 
            'items', without altering the original matrix data.
        
        Example:
            >>> m = Matrix(alg_factory)
            >>> m.set_data([[1, 2, 3], [4, 5, 6]])
            >>> m2 = m._full_like((2, 3))
            >>> print(m2)
            Matrix([[0, 0, 0],
                    [0, 0, 0]])
        """
        return Matrix(self._alg_factory, items=items)

    def _ones_like(self) -> 'Matrix':
        r"""
        Args:
            self (Matrix): The Matrix instance on which the method is called.
                Represents the matrix for which the ones_like operation is to be performed.
        
        Returns:
            Matrix: A new Matrix instance that has the same dimensions as the original matrix, 
                with all elements set to 1.
        
        Raises:
            None
        """
        return Matrix(
            self._alg_factory,
            items=[Vector(self._alg_factory, items=[Scalar.one(self._alg_factory)] * self._width)] * self._height
        )

    def _zeros_like(self) -> 'Matrix':
        r"""
        Returns a new Matrix object with the same dimensions as the current Matrix object, but with all elements initialized to zero.
        
        Args:
            self (Matrix): The current Matrix object.
        
        Returns:
            Matrix: A new Matrix object with the same dimensions as the current Matrix object, but with all elements initialized to zero.
        
        Raises:
            None.
        
        Note:
            - The method creates a new Matrix object using the same algebra factory as the current Matrix object.
            - The dimensions of the new Matrix object are determined by the height and width of the current Matrix object.
            - The new Matrix object is initialized with a grid of Scalar.zero values, where each Scalar.zero value is wrapped within a Vector object.
            - The grid is forwarded using list comprehension to create a list of Vectors, and then multiplying this list by the height of the current Matrix object.
            - The resulting list of Vectors is used to initialize the items parameter of the new Matrix object.
        
        Example:
            >>> matrix = Matrix(alg_factory, items=[[1, 2], [3, 4]])
            >>> zeros_matrix = matrix._zeros_like()
            >>> zeros_matrix
            [[0, 0], [0, 0]]
            >>> zeros_matrix.height
            2
            >>> zeros_matrix.width
            2
        """
        return Matrix(
            self._alg_factory,
            items=[Vector(self._alg_factory, items=[Scalar.zero(self._alg_factory)] * self._width)] * self._height
        )

    def _identity_like(self) -> 'Matrix':
        r"""
        Method _identity_like in the class Matrix creates a new Matrix object that is an identity matrix of the same dimensions as the calling instance.
        
        Args:
            self (Matrix): The calling instance of the Matrix class.
            
        Returns:
            Matrix: A new Matrix object representing an identity matrix with the same dimensions as the calling instance.
        
        Raises:
            None
        """
        items = [
            Vector(
                self._alg_factory,
                items=[Scalar.zero(self._alg_factory)] * i
                + [Scalar.one(self._alg_factory)]
                + [Scalar.zero(self._alg_factory)] * (self._width - i - 1)
            )
            for i in range(self._height)
        ]
        return Matrix(self._alg_factory, items=items)

    def _concat_rows(self, that: 'Matrix') -> 'Matrix':
        r"""
        The _concat_rows method concatenates the rows of the current Matrix instance with another Matrix instance.
        
        Args:
            self (Matrix): The current Matrix instance.
            that (Matrix): The Matrix instance to be concatenated with the current Matrix. It should be of the same data type as the current Matrix.
        
        Returns:
            Matrix: A new Matrix instance resulting from the concatenation of the rows of the current Matrix and the input Matrix.
        
        Raises:
            TypeError: If the input Matrix instance 'that' does not have the same data type as the current Matrix instance.
        """
        if self._alg_factory != that._alg_factory:
            raise TypeError(
                f'It is only possible to concatenate a matrix of the same data type {self._alg_factory}, '
                f'but got: {that._alg_factory}'
            )
        items_x = np.concatenate((self._items_x, that._items_x), axis=0)
        items_y = np.concatenate((self._items_y, that._items_y), axis=0)
        return Matrix(
            self._alg_factory,
            height=self._height + that._height,
            width=self._width,
            items=(items_x, items_y)
        )

    def _concat_cols(self, that: 'Matrix') -> 'Matrix':
        r"""Concatenates the columns of the current matrix with another matrix.
        
        Args:
            self (Matrix): The current matrix object.
            that (Matrix): The matrix object to concatenate with the current matrix. It must be of the same data type as the current matrix.
        
        Returns:
            Matrix: A new matrix resulting from the concatenation of the columns of the current matrix with the columns of the 'that' matrix.
        
        Raises:
            TypeError: If the 'that' matrix is not of the same data type as the current matrix.
        """
        if self._alg_factory != that._alg_factory:
            raise TypeError(
                f'It is only possible to concatenate a matrix of the same data type {self._alg_factory}, '
                f'but got: {that._alg_factory}'
            )
        items_x = np.concatenate((self._items_x, that._items_x), axis=1)
        items_y = np.concatenate((self._items_y, that._items_y), axis=1)
        return Matrix(
            self._alg_factory,
            height=self._height,
            width=self._width + that._width,
            items=(items_x, items_y)
        )
