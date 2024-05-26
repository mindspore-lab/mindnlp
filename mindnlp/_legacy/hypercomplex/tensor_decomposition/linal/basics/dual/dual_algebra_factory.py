from mindnlp._legacy.hypercomplex.tensor_decomposition.linal.basics.hypercomplex.hc_algebra_factory import AlgebraFactory
from mindnlp._legacy.hypercomplex.tensor_decomposition.linal.basics.dual._dual_algebra_impl import _DualAlgebraImpl
from mindnlp._legacy.hypercomplex.tensor_decomposition.linal.basics.hypercomplex._hc_matrix_impl import _MatrixImpl
from mindnlp._legacy.hypercomplex.tensor_decomposition.linal.basics.hypercomplex._hc_scalar_impl import _ScalarImpl
from mindnlp._legacy.hypercomplex.tensor_decomposition.linal.basics.hypercomplex._hc_vector_impl import _VectorImpl


class DualAlgebraFactory(AlgebraFactory):

    r"""
    The DualAlgebraFactory class represents a factory for creating instances of dual algebra objects including matrices, vectors, and scalars. It inherits from the AlgebraFactory class.
    
    This class provides methods to obtain implementations of dual algebra objects, such as matrices, vectors, and scalars. The get_matrix_impl method returns an instance of _DualAlgebraImpl for matrices, the get_vector_impl method returns an instance of _DualAlgebraImpl for vectors, and the get_scalar_impl method returns an instance of _DualAlgebraImpl for scalars.
    
    Note: The actual implementations of _DualAlgebraImpl for matrices, vectors, and scalars are returned by the respective methods.
    """

    def get_matrix_impl(self) -> _MatrixImpl:

        r"""
        This method returns an instance of _MatrixImpl created by _DualAlgebraImpl.
        
        Args:
            self: The instance of the DualAlgebraFactory class.
        
        Returns:
            _MatrixImpl: An instance of _MatrixImpl representing the matrix implementation.
        
        Raises:
            None.
        """
        return _DualAlgebraImpl()

    def get_vector_impl(self) -> _VectorImpl:

        r"""
        This method returns an instance of the _VectorImpl class.
        
        Args:
            self: An instance of the DualAlgebraFactory class.
                This parameter refers to the current instance of the DualAlgebraFactory class.
        
        Returns:
            _VectorImpl: An instance of the _VectorImpl class.
                This method returns an object of type _VectorImpl that represents a vector implementation.
        
        Raises:
            No specific exceptions are documented to be raised by this method.
        """
        return _DualAlgebraImpl()

    def get_scalar_impl(self) -> _ScalarImpl:

        r"""
        Retrieve the scalar implementation for the DualAlgebraFactory.
        
        Args:
            self: The instance of the DualAlgebraFactory class.
        
        Returns:
            _ScalarImpl: An instance of the _ScalarImpl class representing the scalar implementation.
        
        Raises:
            None.
        """
        return _DualAlgebraImpl()
