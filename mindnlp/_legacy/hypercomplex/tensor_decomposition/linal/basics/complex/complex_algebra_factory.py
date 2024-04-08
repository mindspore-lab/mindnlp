from mindnlp._legacy.hypercomplex.tensor_decomposition.linal.basics.hypercomplex.hc_algebra_factory import AlgebraFactory
from mindnlp._legacy.hypercomplex.tensor_decomposition.linal.basics.complex._complex_algebra_impl import _ComplexAlgebraImpl
from mindnlp._legacy.hypercomplex.tensor_decomposition.linal.basics.hypercomplex._hc_matrix_impl import _MatrixImpl
from mindnlp._legacy.hypercomplex.tensor_decomposition.linal.basics.hypercomplex._hc_scalar_impl import _ScalarImpl
from mindnlp._legacy.hypercomplex.tensor_decomposition.linal.basics.hypercomplex._hc_vector_impl import _VectorImpl


class ComplexAlgebraFactory(AlgebraFactory):

    def get_matrix_impl(self) -> _MatrixImpl:
        return _ComplexAlgebraImpl()

    def get_vector_impl(self) -> _VectorImpl:
        return _ComplexAlgebraImpl()

    def get_scalar_impl(self) -> _ScalarImpl:
        return _ComplexAlgebraImpl()
