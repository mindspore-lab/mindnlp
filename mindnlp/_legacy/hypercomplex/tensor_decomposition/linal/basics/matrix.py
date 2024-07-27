from abc import ABC, abstractmethod
from typing import Any, Union
import numpy as np
from mindnlp._legacy.hypercomplex.tensor_decomposition.linal.basics.scalar import Scalar


class Matrix(ABC):

    r"""
    The 'Matrix' class represents a matrix object with a specified height and width. This class inherits from the ABC (Abstract Base Class).
    
    Attributes:
        _height (int): The height of the matrix.
        _width (int): The width of the matrix.
    
    Methods:
        __init__(self, height: int, width: int) -> None:
            Initializes a new instance of the 'Matrix' class with the specified height and width.
    
        __matmul__(self, that: 'Matrix') -> 'Matrix':
            Performs matrix multiplication with another matrix.
    
        __mul__(self, that: Union['Matrix', Scalar, Any]) -> 'Matrix':
            Performs matrix multiplication or scalar multiplication with another matrix or scalar value.
    
        __add__(self, that: Union['Matrix', Scalar, Any]) -> 'Matrix':
            Performs matrix addition or scalar addition with another matrix or scalar value.
    
        __sub__(self, that: Union['Matrix', Scalar, Any]) -> 'Matrix':
            Performs matrix subtraction or scalar subtraction with another matrix or scalar value.
    
        __truediv__(self, that: Union['Matrix', Scalar, Any]) -> 'Matrix':
            Performs matrix division or scalar division with another matrix or scalar value.
    
        __str__(self) -> str:
            Returns a string representation of the matrix.
    
        @staticmethod
        full_like(m: 'Matrix', items: Any) -> 'Matrix':
            Creates a new matrix with the same shape as the input matrix, filled with the specified items.
    
        @staticmethod
        ones_like(m: 'Matrix') -> 'Matrix':
            Creates a new matrix with the same shape as the input matrix, filled with ones.
    
        @staticmethod
        zeros_like(m: 'Matrix') -> 'Matrix':
            Creates a new matrix with the same shape as the input matrix, filled with zeros.
    
        @staticmethod
        identity_like(m: 'Matrix') -> 'Matrix':
            Creates a new identity matrix with the same dimensions as the input matrix.
    
        __getitem__(self, key):
            Abstract method for accessing elements of the matrix.
    
        __neg__(self) -> 'Matrix':
            Abstract method for negating the matrix.
    
        get_height(self) -> int:
            Returns the height of the matrix.
    
        get_width(self) -> int:
            Returns the width of the matrix.
    
        add(self, that: 'Matrix') -> 'Matrix':
            Adds another matrix to the current matrix.
    
        add_scalar(self, that: Union[Scalar, Any]) -> 'Matrix':
            Abstract method for adding a scalar value to the matrix.
    
        sub(self, that: 'Matrix') -> 'Matrix':
            Subtracts another matrix from the current matrix.
    
        sub_scalar(self, that: Union[Scalar, Any]) -> 'Matrix':
            Abstract method for subtracting a scalar value from the matrix.
    
        mul(self, that: 'Matrix') -> 'Matrix':
            Multiplies the current matrix by another matrix.
    
        mul_scalar(self, that: Union[Scalar, Any]) -> 'Matrix':
            Abstract method for multiplying the matrix by a scalar value.
    
        div(self, that: 'Matrix') -> 'Matrix':
            Divides the current matrix by another matrix.
    
        div_scalar(self, that: Union[Scalar, Any]) -> 'Matrix':
            Abstract method for dividing the matrix by a scalar value.
    
        matmul(self, that: 'Matrix') -> 'Matrix':
            Performs matrix multiplication with another matrix.
    
        concat_rows(self, that: 'Matrix') -> 'Matrix':
            Concatenates another matrix to the current matrix along the rows.
    
        concat_cols(self, that: 'Matrix') -> 'Matrix':
            Concatenates another matrix to the current matrix along the columns.
    
        transpose(self) -> 'Matrix':
            Abstract method for transposing the matrix.
    
        transpose_conjugate(self) -> 'Matrix':
            Abstract method for transposing and conjugating the matrix.
    
        _full_like(self, items: Any) -> 'Matrix':
            Abstract method for creating a new matrix with the same shape as the input matrix, filled with the specified items.
    
        _ones_like(self) -> 'Matrix':
            Abstract method for creating a new matrix with the same shape as the input matrix, filled with ones.
    
        _zeros_like(self) -> 'Matrix':
            Abstract method for creating a new matrix with the same shape as the input matrix, filled with zeros.
    
        _identity_like(self) -> 'Matrix':
            Abstract method for creating a new identity matrix with the same dimensions as the input matrix.
    
        as_array(self) -> np.ndarray:
            Abstract method for converting the matrix to a NumPy array.
    
        _add(self, that: 'Matrix') -> 'Matrix':
            Abstract method for adding another matrix to the current matrix.
    
        _sub(self, that: 'Matrix') -> 'Matrix':
            Abstract method for subtracting another matrix from the current matrix.
    
        _mul(self, that: 'Matrix') -> 'Matrix':
            Abstract method for multiplying the current matrix by another matrix.
    
        _div(self, that: 'Matrix') -> 'Matrix':
            Abstract method for dividing the current matrix by another matrix.
    
        _matmul(self, that: 'Matrix') -> 'Matrix':
            Abstract method for performing matrix multiplication with another matrix.
    
        _concat_rows(self, that: 'Matrix') -> 'Matrix':
            Abstract method for concatenating another matrix to the current matrix along the rows.
    
        _concat_cols(self, that: 'Matrix') -> 'Matrix':
            Abstract method for concatenating another matrix to the current matrix along the columns.
    """
    def __init__(self, height: int, width: int) -> None:
        r"""
        Initializes a Matrix object with the given height and width.
        
        Args:
            self (object): The instance of the Matrix class.
            height (int): The height of the matrix. Must be a positive integer.
            width (int): The width of the matrix. Must be a positive integer.
        
        Returns:
            None: This method does not return any value.
        
        Raises:
            ValueError: If the provided height or width is less than or equal to 0.
        """
        if height <= 0 or width <= 0:
            raise ValueError(f"'height' and 'width' must be positive integers, but got {height} and {width}")
        self._height = height
        self._width = width

    def __matmul__(self, that: 'Matrix') -> 'Matrix':
        r"""
        __matmul__
        
        Performs matrix multiplication with another Matrix object.
        
        Args:
            self (Matrix): The first matrix operand for the multiplication.
            that (Matrix): The second matrix operand for the multiplication.
        
        Returns:
            Matrix: A new Matrix object resulting from the matrix multiplication of self and that.
        
        Raises:
            TypeError: If the 'that' parameter is not of type Matrix.
            ValueError: If the dimensions of the matrices are not compatible for multiplication.
        """
        return self.matmul(that)

    def __mul__(self, that: Union['Matrix', Scalar, Any]) -> 'Matrix':
        """
        Performs multiplication with another matrix or a scalar.
        
        Args:
            self (Matrix): The current matrix instance.
            that (Union[Matrix, Scalar, Any]): The value to multiply with the current matrix. It can be another Matrix instance, a Scalar value, or any other type.
            
        Returns:
            Matrix: A new Matrix instance resulting from the multiplication operation.
            
        Raises:
            - TypeError: If 'that' is not a Matrix or a Scalar.
            - ValueError: If the dimensions of the matrices are not compatible for multiplication.
        """
        if isinstance(that, Matrix):
            return self.mul(that)
        else:
            return self.mul_scalar(that)

    def __add__(self, that: Union['Matrix', Scalar, Any]) -> 'Matrix':
        r"""
        Addition method for Matrix class.
        
        Args:
            self (Matrix): The Matrix instance on which the method is called.
            that (Union[Matrix, Scalar, Any]): The object to be added to the Matrix instance. It can be another Matrix object,
                a Scalar value, or any object. If 'that' is a Matrix object, it will be added element-wise to the calling Matrix instance.
                If 'that' is a Scalar value, it will be added to all elements of the Matrix instance.
        
        Returns:
            Matrix: A new Matrix instance resulting from the addition operation. If 'that' is a Matrix object, the addition is
            performed element-wise; if 'that' is a Scalar value, it is added to all elements of the Matrix instance.
        
        Raises:
            - TypeError: If 'that' is not a Matrix object, Scalar value, or any other invalid type.
            - ValueError: If the dimensions of the Matrix object and 'that' Matrix object do not match for element-wise addition.
        """
        if isinstance(that, Matrix):
            return self.add(that)
        else:
            return self.add_scalar(that)

    def __sub__(self, that: Union['Matrix', Scalar, Any]) -> 'Matrix':
        r"""
        Method '__sub__' in class 'Matrix' performs subtraction operation between the current Matrix object and another object.
        
        Args:
            self (Matrix): The Matrix object itself.
                - Type: Matrix
                - Purpose: Represents the current Matrix object participating in the subtraction operation.
        
            that (Union['Matrix', Scalar, Any]): The object to be subtracted from the current Matrix object.
                - Type: Union['Matrix', Scalar, Any]
                - Purpose: Represents the object that will be subtracted from the current Matrix object.
                - Restrictions: Accepts either a Matrix object, a Scalar value, or any other data type.
        
        Returns:
            Matrix: A new Matrix object resulting from the subtraction operation.
                - Type: Matrix
                - Purpose: Represents the result of subtracting the 'that' object from the current Matrix object.
        
        Raises:
            - None
        """
        if isinstance(that, Matrix):
            return self.sub(that)
        else:
            return self.sub_scalar(that)

    def __truediv__(self, that: Union['Matrix', Scalar, Any]) -> 'Matrix':
        r"""
        Performs the division operation using the '__truediv__' operator on the 'Matrix' class.
        
        Args:
            self (Matrix): The current 'Matrix' instance on which the division operation is being performed.
            that (Union[Matrix, Scalar, Any]): The second operand of the division operation. It can be a 'Matrix' instance, a 'Scalar' value, or any other object.
            
        Returns:
            Matrix: A new 'Matrix' instance resulting from the division operation. If 'that' is a 'Matrix' instance, element-wise division is performed. If 'that' is a 'Scalar' value, each element in the
'Matrix' instance is divided by 'that'.
        
        Raises:
            None: This method does not raise any exceptions.
        """
        if isinstance(that, Matrix):
            return self.div(that)
        else:
            return self.div_scalar(that)

    def __str__(self) -> str:
        r"""
        Args:
            self (Matrix): The Matrix object on which the __str__ method is called.
        
        Returns:
            str: A string representation of the Matrix object. The string contains the dimensions of the matrix in the format 'height x width', followed by the elements of the matrix enclosed in square
brackets. If the height of the matrix is less than or equal to 4, the elements are listed row-wise within the brackets. If the height is greater than 4, only the first 2 and last 2 elements of the matrix are
listed, separated by an ellipsis '...'.
        
        Raises:
            None
        """
        ret = f'{self._height}x{self._width} '
        if self._height <= 4:
            items = [self[i] for i in range(self._height)]
            ret += '[' + ', '.join([str(item) for item in items]) + ']'
        else:
            items_top = [self[i] for i in range(2)]
            items_bottom = [self[-i - 1] for i in range(2)]
            ret += '[' + ', '.join([str(item) for item in items_top]) + ', ... ,' \
                + ', '.join([str(item) for item in items_bottom]) + ']'
        return ret

    @staticmethod
    def full_like(m: 'Matrix', items: Any) -> 'Matrix':
        r"""
        This method creates a new Matrix object filled with the specified items, matching the shape of the input Matrix object.
        
        Args:
            m (Matrix): The input Matrix object that serves as a template for the shape of the new Matrix. It must be an instance of the Matrix class.
            items (Any): The items to fill the new Matrix with. It can be of any type.
        
        Returns:
            Matrix: A new Matrix object filled with the specified items, matching the shape of the input Matrix object.
        
        Raises:
            None
        """
        return m._full_like(items)

    @staticmethod
    def ones_like(m: 'Matrix') -> 'Matrix':
        r"""
        This method creates a new Matrix object with the same shape and data type as the input Matrix.
        
        Args:
            m (Matrix): The input Matrix object for which the new Matrix object will be created to have the same shape and data type.
            
        Returns:
            Matrix: A new Matrix object with the same shape and data type as the input Matrix.
        
        Raises:
            None
        """
        return m._ones_like()

    @staticmethod
    def zeros_like(m: 'Matrix') -> 'Matrix':
        r"""
        This method creates a new Matrix object with the same shape and data type as the input Matrix.
        
        Args:
            m (Matrix): The input Matrix object for which the zeros_like method will create a new Matrix with the same shape and data type.
        
        Returns:
            Matrix: Returns a new Matrix object with the same shape and data type as the input Matrix.
        
        Raises:
            No specific exceptions are documented to be raised by this method.
        """
        return m._zeros_like()

    @staticmethod
    def identity_like(m: 'Matrix') -> 'Matrix':
        r"""
        This method creates an identity matrix with the same dimensions as the input matrix.
        
        Args:
            m (Matrix): The input matrix for which an identity matrix is to be created.
                        It should be a square matrix with the same width and height.
                        Type: Matrix object.
                        Purpose: To use the dimensions of the input matrix for creating the identity matrix.
        
        Returns:
            Matrix: Returns a new Matrix object representing an identity matrix with the same dimensions as the input matrix.
            Purpose: To provide an identity matrix based on the dimensions of the input matrix.
        
        Raises:
            ValueError: Raised if the input matrix is not a square matrix (i.e., width is not equal to height).
                        The exception message will indicate that an identity matrix can only be created from a matrix with the same dimensions.
                        Exception message format: 'It is only possible to make an identity out of a matrix of the same dimensions, but got: {height}x{width}'.
        """
        if m._width != m._height:
            raise ValueError(
                'It is only possible to make an identity out of a matrix of the same dimensions, '
                f'but got: {m._height}x{m._width}'
            )
        return m._identity_like()

    @abstractmethod
    def __getitem__(self, key):
        r"""
        Docstring for method '__getitem__' in the class 'Matrix':
        
        Args:
            self: Matrix
                The instance of the Matrix class.
            
            key: Any
                The key parameter used to access an item in the Matrix.
        
        Returns:
            None
                This method does not return any value.
        
        Raises:
            NotImplementedError
                If the method is not implemented in a subclass of Matrix.
        """
        pass

    @abstractmethod
    def __neg__(self) -> 'Matrix':
        r"""
        Negates the matrix by changing the sign of each element.
        
        Args:
            self (Matrix): The current instance of the Matrix class.
        
        Returns:
            Matrix: A new Matrix object with the sign of each element negated.
        
        Raises:
            None.
        """
        pass

    def get_height(self) -> int:
        r"""
        This method retrieves the height attribute of a Matrix object.
        
        Args:
            self (Matrix): The instance of the Matrix class.
                Parameter to access the height attribute of the Matrix object.
        
        Returns:
            int: The height of the Matrix object.
                Returns the integer value representing the height of the Matrix.
        
        Raises:
            None.
        """
        return self._height

    def get_width(self) -> int:
        r"""
        Retrieve the width of the matrix.
        
        Args:
            self (Matrix): The instance of the Matrix class.
                It represents the current matrix object for which the width needs to be retrieved.
        
        Returns:
            int: The width of the matrix.
                It indicates the number of columns in the matrix.
        
        Raises:
            No specific exceptions are raised by this method.
        """
        return self._width

    def add(self, that: 'Matrix') -> 'Matrix':
        r"""
        Add a matrix of the same dimensions to this matrix.
        
        Args:
            self (Matrix): The Matrix instance calling the method.
                Represents the matrix to which another matrix will be added.
            that (Matrix): The Matrix instance being added to the current matrix.
                Represents the matrix that will be added to self. It must have the same dimensions as self.
        
        Returns:
            Matrix: A new Matrix instance representing the result of adding the two matrices together.
        
        Raises:
            ValueError: If the dimensions of the matrix being added (that) are not the same as the dimensions of the current matrix (self).
                The exception message will indicate that only matrices of the same dimensions can be added.
        """
        if self._height != that._height or self._width != that._width:
            raise ValueError(
                f'It is only possible to add a matrix of the same dimensions {self._height}x{self._width},'
                f' but got: {that._height}x{that._width}'
            )
        return self._add(that)

    @abstractmethod
    def add_scalar(self, that: Union[Scalar, Any]) -> 'Matrix':
        r"""
        Adds a scalar value to the matrix.
        
        Args:
            self (Matrix): The Matrix object itself.
            that (Union[Scalar, Any]): The scalar value to be added to the matrix. It can be of type Scalar or any other type.
        
        Returns:
            Matrix: A new Matrix object resulting from adding the scalar value to the original matrix.
        
        Raises:
            - NotImplementedError: If the method is not implemented in the derived class.
            - TypeError: If the provided scalar value is not compatible for addition with the matrix.
        """
        pass

    def sub(self, that: 'Matrix') -> 'Matrix':
        r"""
        Method 'sub' in the class 'Matrix'.
        
        Args:
            self (Matrix): The current Matrix object that the operation is being performed on.
            that (Matrix): The Matrix object to be subtracted from the current Matrix. Both matrices must have the same dimensions.
        
        Returns:
            Matrix: A new Matrix object resulting from the subtraction operation.
        
        Raises:
            ValueError: If the dimensions of the two matrices are not the same, a ValueError is raised with a message indicating the mismatch.
        """
        if self._height != that._height or self._width != that._width:
            raise ValueError(
                f'It is only possible to subtract a matrix of the same dimensions {self._height}x{self._width},'
                f' but got: {that._height}x{that._width}'
            )
        return self._sub(that)

    @abstractmethod
    def sub_scalar(self, that: Union[Scalar, Any]) -> 'Matrix':
        r"""
        This method subtracts a scalar value from each element of the matrix.
        
        Args:
            self (Matrix): The matrix from which the scalar value will be subtracted.
            that (Union[Scalar, Any]): The scalar value or another object that can be converted to a scalar, which will be subtracted from each element of the matrix.
        
        Returns:
            Matrix: A new Matrix object resulting from subtracting the scalar value from each element of the original matrix.
        
        Raises:
            TypeError: If the 'that' parameter is not a scalar value or cannot be converted to a scalar.
        """
        pass

    def mul(self, that: 'Matrix') -> 'Matrix':
        r"""
        This method performs matrix multiplication between the current Matrix object and another Matrix object.
        
        Args:
            self (Matrix): The Matrix object on which the multiplication operation is performed.
            that (Matrix): The Matrix object that is being multiplied with the current Matrix. It should have the same dimensions (height and width) as the current Matrix. 
        
        Returns:
            Matrix: A new Matrix object resulting from the multiplication operation.
        
        Raises:
            ValueError: If the dimensions of the current Matrix and the 'that' Matrix do not match, a ValueError is raised indicating that matrix multiplication is only possible between matrices with the same
dimensions.
        """
        if self._height != that._height or self._width != that._width:
            raise ValueError(
                f'It is only possible to multiply by a matrix of the same dimensions {self._height}x{self._width},'
                f' but got: {that._height}x{that._width}'
            )
        return self._mul(that)

    @abstractmethod
    def mul_scalar(self, that: Union[Scalar, Any]) -> 'Matrix':
        r"""
        This method multiplies the Matrix object by a scalar value.
        
        Args:
            self (Matrix): The Matrix object on which the scalar multiplication operation is performed.
            that (Union[Scalar, Any]): The scalar value to multiply the Matrix by. It can be of type Scalar or any other type. 
        
        Returns:
            Matrix: A new Matrix object resulting from the scalar multiplication operation.
        
        Raises:
            None
        """
        pass

    def div(self, that: 'Matrix') -> 'Matrix':
        r"""
        Divides the current matrix by another matrix.
        
        Args:
            self (Matrix): The current matrix.
            that (Matrix): The matrix to divide by.
        
        Returns:
            Matrix: A new matrix resulting from the division.
        
        Raises:
            ValueError: If the dimensions of the current matrix and the given matrix are not the same.
        
        Note:
            The division is only possible when the dimensions of both matrices are the same.
        
        Example:
            >>> matrix1 = Matrix([[1, 2], [3, 4]])
            >>> matrix2 = Matrix([[2, 2], [2, 2]])
            >>> result = matrix1.div(matrix2)
            >>> print(result)
            Matrix([[0.5, 1.0], [1.5, 2.0]])
        """
        if self._height != that._height or self._width != that._width:
            raise ValueError(
                f'It is only possible to divide over a matrix of the same dimensions {self._height}x{self._width},'
                f' but got: {that._height}x{that._width}'
            )
        return self._div(that)

    @abstractmethod
    def div_scalar(self, that: Union[Scalar, Any]) -> 'Matrix':
        r"""
        Divides the elements of the current matrix by a scalar value.
        
        Args:
            self (Matrix): The current instance of the Matrix class.
            that (Union[Scalar, Any]): The scalar value to be divided by. It can either be a Scalar object or any other data type.
        
        Returns:
            Matrix: A new Matrix object containing the result of the division operation.
        
        Raises:
            None
        
        Note:
            The division operation is performed element-wise between the current matrix and the scalar value. If the scalar value is of type Scalar, it will be divided by each element of the matrix. If the
scalar value is of any other data type, it will be divided by each corresponding element of the matrix based on their positions.
        
        Example:
            Consider a matrix 'A' with the following elements:
                A = [[1, 2],
                     [3, 4]]
            If we call the 'div_scalar' method on 'A' with the scalar value 2, the resulting matrix 'B' would be:
                B = [[0.5, 1.0],
                     [1.5, 2.0]]
        """
        pass

    def matmul(self, that: 'Matrix') -> 'Matrix':
        r"""
        This method performs matrix multiplication between two Matrix objects.
        
        Args:
            self (Matrix): The Matrix object calling the method.
                - Type: Matrix
                - Purpose: Represents the first matrix participating in the multiplication.
            that (Matrix): The Matrix object passed as an argument for multiplication.
                - Type: Matrix
                - Purpose: Represents the second matrix to be multiplied with the first.
                - Restrictions: The width of the first matrix (self) must be equal to the height of the second matrix (that).
        
        Returns:
            Matrix: A new Matrix resulting from the multiplication operation.
                - Type: Matrix
                - Purpose: Represents the product of the multiplication operation between the two input matrices.
        
        Raises:
            ValueError: Raised if the width of the first matrix (self) does not match the height of the second matrix (that).
                - Exception message format: 
                    'It is only possible to multiply a matrix of height {self._width}, but got: {that._height}'
        """
        if self._width != that._height:
            raise ValueError(
                f'It is only possible to multiply a matrix of height {self._width}, but got: {that._height}'
            )
        return self._matmul(that)

    def concat_rows(self, that: 'Matrix') -> 'Matrix':
        r"""
        Concatenates the rows of the current matrix with another matrix.
        
        Args:
            self (Matrix): The current matrix.
            that (Matrix): The matrix to be concatenated with the current matrix.
        
        Returns:
            Matrix: A new matrix resulting from concatenating the rows of the current matrix and the 'that' matrix.
        
        Raises:
            ValueError: If the width of the 'that' matrix is not equal to the width of the current matrix.
        
        """
        if self._width != that._width:
            raise ValueError(
                f'It is only possible to concat a matrix of width {self._width}, but got: {that._width}'
            )
        return self._concat_rows(that)

    def concat_cols(self, that: 'Matrix') -> 'Matrix':
        r"""
        Concatenates the columns of this Matrix with another Matrix.
        
        Args:
            self (Matrix): The current Matrix object.
                Represents the Matrix whose columns will be concatenated with another Matrix.
            that (Matrix): The Matrix object to be concatenated with the current Matrix.
                Represents the Matrix with columns that will be concatenated to the columns of the current Matrix.
        
        Returns:
            Matrix: A new Matrix resulting from the concatenation of the columns of the current Matrix with the columns of the input Matrix 'that'.
        
        Raises:
            ValueError: If the height of the current Matrix is not equal to the height of the input Matrix 'that'.
                The concatenation operation is only possible when both matrices have the same height.
        """
        if self._height != that._height:
            raise ValueError(
                f'It is only possible to concat a matrix of height {self._height}, but got: {that._height}'
            )
        return self._concat_cols(that)

    @abstractmethod
    def transpose(self) -> 'Matrix':
        r"""
        Transposes the matrix.
        
        Args:
            self (Matrix): The matrix object on which the transpose operation is performed.
        
        Returns:
            Matrix: The transposed matrix.
        
        Raises:
            None.
        
        This method transposes the matrix by switching its rows with columns. The transpose of a matrix is obtained by reflecting the elements along the main diagonal. The resulting matrix will have the rows
of the original matrix as its columns and the columns of the original matrix as its rows.
        
        Example:
            Given a matrix object 'matrix' with the following elements:
            [[1, 2],
             [3, 4],
             [5, 6]]
            
            Calling the 'transpose' method on the 'matrix' object will result in the following transposed matrix:
            [[1, 3, 5],
             [2, 4, 6]]
        """
        pass

    @abstractmethod
    def transpose_conjugate(self) -> 'Matrix':
        r"""
        Method to calculate the transpose conjugate of the current Matrix instance.
        
        Args:
            self (Matrix): The Matrix instance on which the transpose conjugate operation will be performed.
        
        Returns:
            Matrix: A new Matrix instance representing the transpose conjugate of the input Matrix.
        
        Raises:
            This method does not raise any exceptions.
        """
        pass

    @abstractmethod
    def _full_like(self, items: Any) -> 'Matrix':
        r"""
        This method creates a new Matrix object with the same shape and data type as the input 'items'.
        
        Args:
            self (Matrix): The instance of the Matrix class.
            items (Any): The input items used to determine the shape and data type of the new Matrix object. It can be any valid input that can be used to forward a Matrix object.
        
        Returns:
            Matrix: A new Matrix object that has the same shape and data type as the input 'items'.
        
        Raises:
            NotImplementedError: If the method is not implemented in the subclass.
        """
        pass

    @abstractmethod
    def _ones_like(self) -> 'Matrix':
        r"""
        Method _ones_like in class Matrix.
        
        Args:
            self: The instance of the Matrix class.
                   Purpose: To access the attributes and methods of the current Matrix instance.
                   Restrictions: None.
        
        Returns:
            'Matrix': A new Matrix object filled with ones, having the same shape as the original Matrix object.
        
        Raises:
            None.
        """
        pass

    @abstractmethod
    def _zeros_like(self) -> 'Matrix':
        r"""
        This method creates a new Matrix object with the same shape and dtype as the current Matrix object, but with all elements set to zero.
        
        Args:
            self (Matrix): The current Matrix object. It serves as a template for creating the new Matrix object. Must be an instance of the Matrix class.
        
        Returns:
            Matrix: A new Matrix object that has the same shape and dtype as the current Matrix object, but with all elements initialized to zero.
        
        Raises:
            This method does not raise any exceptions.
        """
        pass

    @abstractmethod
    def _identity_like(self) -> 'Matrix':
        r"""
        This method creates a new matrix that is an identity matrix with the same dimensions as the original matrix.
        
        Args:
            self (Matrix): The instance of the Matrix class.
            
        Returns:
            Matrix: A new Matrix instance that is an identity matrix with the same dimensions as the original matrix.
        
        Raises:
            This method does not raise any exceptions.
        """
        pass

    @abstractmethod
    def as_array(self) -> np.ndarray:
        r"""
        Converts the Matrix object into a NumPy array.
        
        Args:
            self (Matrix): The Matrix object to be converted into a NumPy array.
        
        Returns:
            np.ndarray: The NumPy array representation of the Matrix object.
        
        Raises:
            None.
        
        Note:
            This method does not modify the original Matrix object. It creates a new NumPy array that is a copy of the Matrix data.
        
        Example:
            >>> matrix = Matrix([[1, 2], [3, 4]])
            >>> array = matrix.as_array()
            >>> print(array)
            [[1 2]
             [3 4]]
        """
        pass

    @abstractmethod
    def _add(self, that: 'Matrix') -> 'Matrix':
        r"""
        This method, named '_add', is a private abstract method in the class 'Matrix'. It is used to perform addition between two matrix objects.
        
        Args:
            self (Matrix): The first matrix object to be added. It is of type 'Matrix' and is required for the method to work.
            that (Matrix): The second matrix object to be added. It is of type 'Matrix' and is required for the method to work.
        
        Returns:
            Matrix: The result of the addition operation between the two matrix objects. The result is also a 'Matrix' object.
        
        Raises:
            None: This method does not raise any exceptions.
        
        Note:
            This method is an abstract method and should be overridden in any concrete subclass that inherits from the 'Matrix' class.
        """
        pass

    @abstractmethod
    def _sub(self, that: 'Matrix') -> 'Matrix':
        r"""
        This method performs subtraction between two Matrix objects.
        
        Args:
            self (Matrix): The Matrix object on which the subtraction operation is performed.
            that (Matrix): The Matrix object to be subtracted from the self Matrix.
        
        Returns:
            Matrix: A new Matrix object resulting from the subtraction operation.
        
        Raises:
            NotImplementedError: If the method is not implemented in the subclass.
        """
        pass

    @abstractmethod
    def _mul(self, that: 'Matrix') -> 'Matrix':
        r"""
        This method performs matrix multiplication between two matrices.
        
        Args:
            self (Matrix): The calling matrix object.
            that (Matrix): The matrix object to be multiplied with the calling matrix.
        
        Returns:
            Matrix: A new matrix object representing the result of matrix multiplication.
        
        Raises:
            None.
        
        Note:
            - Both matrices must have compatible dimensions for matrix multiplication.
            - The number of columns in the calling matrix must be equal to the number of rows in the 'that' matrix.
            - If any of the matrices is not valid or does not meet the requirements, the behavior is undefined.
            - The original matrices remain unchanged after the multiplication operation.
        """
        pass

    @abstractmethod
    def _div(self, that: 'Matrix') -> 'Matrix':
        r"""
        This method performs division operation between two Matrix objects.
        
        Args:
            self (Matrix): The Matrix object on which the division operation is performed.
            that (Matrix): The Matrix object by which the division operation is performed.
        
        Returns:
            Matrix: A new Matrix object resulting from the division operation.
        
        Raises:
            - TypeError: If the input parameters are not of type Matrix.
            - ValueError: If the matrices are not compatible for division (e.g., different dimensions).
            - ZeroDivisionError: If the division by zero occurs during the operation.
        """
        pass

    @abstractmethod
    def _matmul(self, that: 'Matrix') -> 'Matrix':
        r"""
        Perform matrix multiplication with another Matrix object.
        
        Args:
            self (Matrix): The Matrix object on which the method is called.
            that (Matrix): The Matrix object to multiply with self.
        
        Returns:
            Matrix: A new Matrix resulting from the matrix multiplication operation.
        
        Raises:
            NotImplementedError: If the method is not implemented in the subclass.
        """
        pass

    @abstractmethod
    def _concat_rows(self, that: 'Matrix') -> 'Matrix':
        r"""
        This method concatenates the rows of the current Matrix instance with another Matrix instance.
        
        Args:
            self (Matrix): The current Matrix instance.
            that (Matrix): The Matrix instance to be concatenated with the current instance. It must have the same number of columns as the current Matrix.
        
        Returns:
            Matrix: A new Matrix instance resulting from the concatenation of the rows of the current Matrix with the rows of the 'that' Matrix instance.
        
        Raises:
            - TypeError: If 'that' parameter is not a Matrix instance.
            - ValueError: If the number of columns in the 'that' Matrix instance is not equal to the number of columns in the current Matrix instance.
        """
        pass

    @abstractmethod
    def _concat_cols(self, that: 'Matrix') -> 'Matrix':
        r"""
        Concatenates the columns of the current Matrix with the columns of another Matrix.
        
        Args:
            self (Matrix): The current Matrix object.
            that (Matrix): The Matrix object to concatenate columns with.
        
        Returns:
            Matrix: A new Matrix object resulting from the concatenation of columns.
        
        Raises:
            TypeError: If the 'that' parameter is not of type Matrix.
            ValueError: If the number of rows in the current Matrix is not equal to the number of rows in the 'that' Matrix.
        """
        pass

