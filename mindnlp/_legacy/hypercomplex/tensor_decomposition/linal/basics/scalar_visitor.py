from abc import ABC, abstractmethod
from mindnlp._legacy.hypercomplex.tensor_decomposition.linal.basics.real.real_scalar import Scalar as RealScalar
from mindnlp._legacy.hypercomplex.tensor_decomposition.linal.basics.hypercomplex.hc_scalar import Scalar as HCScalar


class ScalarVisitor(ABC):

    r"""
    This class represents a visitor that can visit different types of scalar values and perform specific operations on them.
    
    The ScalarVisitor class is an abstract base class (ABC) that provides a framework for implementing concrete scalar visitors. Concrete classes inheriting from this class must implement the visit_real,
visit_complex, and visit_dual methods.
    
    The visit_real method is responsible for handling a real scalar value. It takes a RealScalar object as input and performs the necessary operations on it.
    
    The visit_complex method is responsible for handling a complex scalar value. It takes an HCScalar object as input and performs the necessary operations on it.
    
    The visit_dual method is responsible for handling a dual scalar value. It also takes an HCScalar object as input and performs the necessary operations on it.
    
    Concrete classes that inherit from ScalarVisitor can define additional methods as needed to handle other types of scalar values.
    
    Note that this class does not provide any implementation details for the visit_real, visit_complex, and visit_dual methods, as they are meant to be implemented by the subclasses.
    
    It is important to note that instances of ScalarVisitor cannot be created directly, as it is an abstract base class. Instead, concrete subclasses should be created and used.
    
    Example usage:
    
    class MyScalarVisitor(ScalarVisitor):
        def visit_real(self, s: RealScalar, *args, **kwargs) -> None:
            # Implementation for visiting real scalar values
    
        def visit_complex(self, s: HCScalar, *args, **kwargs) -> None:
            # Implementation for visiting complex scalar values
    
        def visit_dual(self, s: HCScalar, *args, **kwargs) -> None:
            # Implementation for visiting dual scalar values
    
    # Create an instance of MyScalarVisitor
    visitor = MyScalarVisitor()
    
    # Use the visitor to perform operations on scalar values
    visitor.visit_real(real_scalar)
    visitor.visit_complex(complex_scalar)
    visitor.visit_dual(dual_scalar)
    
    """

    @abstractmethod
    def visit_real(self, s: RealScalar, *args, **kwargs) -> None:

        r"""
        This method is responsible for visiting a RealScalar object.
        
        Args:
            self (object): The instance of the ScalarVisitor class.
            s (RealScalar): The RealScalar object to be visited.
        
        Returns:
            None. This method does not return any value.
        
        Raises:
            This method does not raise any exceptions.
        """
        pass

    @abstractmethod
    def visit_complex(self, s: HCScalar, *args, **kwargs) -> None:

        r"""
        visit_complex method in ScalarVisitor class.
        
        Args:
            self: Represents the instance of the ScalarVisitor class.
            s (HCScalar): An input parameter representing a complex scalar value to be visited.
                It should be an instance of the HCScalar class.
            
        Returns:
            None: This method does not return any value.
        
        Raises:
            This method does not raise any exceptions.
        """
        pass

    @abstractmethod
    def visit_dual(self, s: HCScalar, *args, **kwargs) -> None:

        r"""
        This method 'visit_dual' is defined in the 'ScalarVisitor' class and is an abstract method that must be implemented in subclasses.
        
        Args:
            self: Instance of the class invoking the method.
            s (HCScalar): The scalar object to be visited by the method.
            
        Returns:
            None: This method does not return any value.
        
        Raises:
            This method is expected to be implemented in subclasses. If not implemented, a NotImplementedError will be raised when invoked.
        """
        pass
