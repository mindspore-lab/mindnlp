from abc import ABC, abstractmethod
from mindnlp._legacy.hypercomplex.tensor_decomposition.linal.basics.hypercomplex._hc_matrix_impl import _MatrixImpl
from mindnlp._legacy.hypercomplex.tensor_decomposition.linal.basics.hypercomplex._hc_scalar_impl import _ScalarImpl
from mindnlp._legacy.hypercomplex.tensor_decomposition.linal.basics.hypercomplex._hc_vector_impl import _VectorImpl


class AlgebraFactory(ABC):

    @abstractmethod
    def get_matrix_impl(self) -> _MatrixImpl:
        pass

    @abstractmethod
    def get_vector_impl(self) -> _VectorImpl:
        pass

    @abstractmethod
    def get_scalar_impl(self) -> _ScalarImpl:
        pass
