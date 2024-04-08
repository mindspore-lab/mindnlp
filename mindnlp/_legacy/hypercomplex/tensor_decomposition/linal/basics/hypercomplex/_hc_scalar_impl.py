from abc import ABC, abstractmethod
from typing import Tuple
import numpy as np


class _ScalarImpl(ABC):

    @abstractmethod
    def sqrt(self, x: np.float64, y: np.float64) -> Tuple[np.float64, np.float64]:
        pass

    @abstractmethod
    def special_element(self) -> str:
        pass

    @abstractmethod
    def visit(self, scalar, visitor, *args, **kwargs) -> None:
        pass

    @abstractmethod
    def mul_scalar(self,
                   x1: np.float64,
                   y1: np.float64,
                   x2: np.float64,
                   y2: np.float64) -> Tuple[np.float64, np.float64]:
        pass

    @abstractmethod
    def div_scalar(self,
                   x1: np.float64,
                   y1: np.float64,
                   x2: np.float64,
                   y2: np.float64) -> Tuple[np.float64, np.float64]:
        pass
