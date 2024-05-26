from abc import ABC, abstractmethod
from mindnlp._legacy.hypercomplex.tensor_decomposition.linal.basics.hypercomplex._hc_matrix_impl import _MatrixImpl
from mindnlp._legacy.hypercomplex.tensor_decomposition.linal.basics.hypercomplex._hc_scalar_impl import _ScalarImpl
from mindnlp._legacy.hypercomplex.tensor_decomposition.linal.basics.hypercomplex._hc_vector_impl import _VectorImpl


class AlgebraFactory(ABC):

    r"""
    AlgebraFactory is an abstract base class (ABC) representing a factory for creating matrix, vector, and scalar implementations.
    
    This class defines abstract methods for retrieving implementations of matrices, vectors, and scalars. Subclasses of AlgebraFactory are expected to provide concrete implementations for these methods.
    
    Subclasses:
        - Subclasses implementing AlgebraFactory must provide concrete implementations for the abstract methods defined in this class.
    
    Abstract Methods:
        - get_matrix_impl: Returns the implementation for matrices.
        - get_vector_impl: Returns the implementation for vectors.
        - get_scalar_impl: Returns the implementation for scalars.
    
    Note:
        - Implementing classes must provide concrete implementations for the abstract methods to instantiate specific implementations of matrices, vectors, and scalars.
    
    """

    @abstractmethod
    def get_matrix_impl(self) -> _MatrixImpl:

        r"""
        Returns the implementation of the matrix class used by the AlgebraFactory.
        
        Args:
            self: An instance of the AlgebraFactory class.
        
        Returns:
            An object of type _MatrixImpl, which is the implementation of the matrix class used by the AlgebraFactory.
        
        Raises:
            None.
        
        Note:
            The returned _MatrixImpl object is responsible for providing the functionality related to matrices within the AlgebraFactory. This method is abstract and must be implemented in any subclass of AlgebraFactory. The returned implementation may vary depending on the specific requirements and configuration of the subclass.
        """
        pass

    @abstractmethod
    def get_vector_impl(self) -> _VectorImpl:

        r"""
        Retrieve the vector implementation for the AlgebraFactory.
        
        Args:
            self: An instance of the AlgebraFactory class.
        
        Returns:
            _VectorImpl: An instance of the _VectorImpl class representing the vector implementation.
        
        Raises:
            This method does not raise any exceptions.
        """
        pass

    @abstractmethod
    def get_scalar_impl(self) -> _ScalarImpl:

        r"""
        This method retrieves the implementation of a scalar from the AlgebraFactory.
        
        Args:
            self: Instance of the AlgebraFactory class.
        
        Returns:
            _ScalarImpl: An object representing the implementation of a scalar.
        
        Raises:
            This method is expected to be implemented by subclasses of AlgebraFactory and should raise a NotImplementedError if called directly.
        """
        pass
