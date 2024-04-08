from abc import ABC, abstractmethod
from mindnlp._legacy.hypercomplex.tensor_decomposition.linal.basics.real.real_scalar import Scalar as RealScalar
from mindnlp._legacy.hypercomplex.tensor_decomposition.linal.basics.hypercomplex.hc_scalar import Scalar as HCScalar


class ScalarVisitor(ABC):

    @abstractmethod
    def visit_real(self, s: RealScalar, *args, **kwargs) -> None:
        pass

    @abstractmethod
    def visit_complex(self, s: HCScalar, *args, **kwargs) -> None:
        pass

    @abstractmethod
    def visit_dual(self, s: HCScalar, *args, **kwargs) -> None:
        pass
