"""Operator level amp"""
import functools
from typing import Any

import mindspore

CELL_WHITE_LIST = [
    'Dense',
    'Conv1d',
    'Conv2d',
    'Conv3d',
]

OP_WHITE_LIST = [
    'MatMul',
    'BatchMatMul',
    'Dense',
    'Conv2D',
    'Conv2DTranspose',
    'Conv3D',
    'Conv3DTranspose',
    'LSTM',
    'CudnnGRU',
    'PReLU'
]

OP_BLACK_LIST = [
    'Asin',
    'Acos',
    'BCEWithLogitsLoss',
    'BinaryCrossEntropy',
    'Cosh',
    'Cdis',
    'CumProd',
    'CumSum',
    'Div',
    'Erfinv',
    'Exp',
    'Expm1',
    'KLDivLoss',
    'LayerNorm',
    'Log',
    'LogSoftmax',
    'Log10',
    'Log1p',
    'Log2',
    'MultilabelMarginLoss',
    'MultiMarginLoss',
    'NLLLoss',
    'LpNorm',
    'L2Normalize',
    'Pdist',
    'Pow',
    'RealDiv',
    'ReduceProd',
    'Reciprocal',
    'Rsqrt',
    'Renorm',
    'Sinh',
    'Sum',
    'Softplus',
    'Softmax',
    'Softmin',
    'SoftMarginLoss',
    'SoftmaxCrossEntropyWithLogits',
    'SparseSoftmaxCrossEntropyWithLogits',
    'SmoothL1Loss',
    'Tan',
    'TripletMarginLoss'
]

GLOBAL_AMP = False
GLOBAL_AMP_DTYPE = mindspore.float32

def _set_amp(mode, dtype):
    r"""
    Sets the global amplifier mode and data type.
    
    Args:
        mode (str): The mode to set the global amplifier to. Valid values are 'on' or 'off'.
        dtype (type): The data type to set for the global amplifier. This can be any valid Python data type.
    
    Returns:
        None: This function does not return any value.
    
    Raises:
        None: This function does not raise any exceptions.
    """
    global GLOBAL_AMP
    global GLOBAL_AMP_DTYPE
    GLOBAL_AMP = mode
    GLOBAL_AMP_DTYPE = dtype

def get_global_amp():
    r"""
    Returns the global amplitude and its data type.
    
    Returns:
        tuple: A tuple containing the global amplitude and its data type.
    """
    return GLOBAL_AMP, GLOBAL_AMP_DTYPE


def autocast_decorator(autocast_instance, func):
    r"""
    Decorator function that applies an autocast instance to a given function.
    
    Args:
        autocast_instance (Autocast): An instance of the Autocast class.
            The Autocast class provides a context manager that automatically casts inputs to a specified data type.
            This autocast instance will be used to automatically cast the inputs of the decorated function.
        func (function): The function to be decorated.
            This function will be called within the context of the autocast instance.
    
    Returns:
        None
    
    Raises:
        None
    """
    @functools.wraps(func)
    def decorate_autocast(*args, **kwargs):
        with autocast_instance:
            return func(*args, **kwargs)

    return decorate_autocast

class autocast:

    r"""
    The 'autocast' class represents a context manager for automatic mixed precision (AMP) in Python. It provides functionality for enabling or disabling automatic mixed precision for a specific code block and
specifying the data type for computations.
    
    Upon entering the context, the 'autocast' class sets the enabled state and data type for AMP. Upon exiting the context, it restores the original data type. Additionally, the class can be used as a
decorator for functions to apply automatic mixed precision to the decorated function.
    
    This class is designed to be used in conjunction with the MindSpore framework for deep learning and neural network computations.
    """
    def __init__(
        self,
        enabled: bool = True,
        dtype = mindspore.float16,
    ):
        r"""
        Initialize the autocast object.
        
        Args:
            self (object): The instance of the autocast class.
            enabled (bool, optional): A flag indicating whether autocast is enabled. Defaults to True.
            dtype (dtype, optional): The data type for autocasting. Defaults to mindspore.float16.
        
        Returns:
            None. This method does not return any value.
        
        Raises:
            None.
        """
        self.enabled = enabled
        self.dtype = dtype
        self.old_dtype = GLOBAL_AMP_DTYPE

    def __enter__(self):
        r"""
        Method '__enter__' in the class 'autocast'.
        
        Args:
            self: autocast instance.
                Represents the current instance of the autocast class.
        
        Returns:
            None.
            The method does not explicitly return any value.
        
        Raises:
            No specific exceptions are raised by this method.
        """
        _set_amp(self.enabled, self.dtype)
        return self

    def __exit__(self, exc_type: Any, exc_val: Any, exc_tb: Any):
        r"""
        This method is called when exiting a context managed by the 'autocast' class.
        
        Args:
            self: Instance of the 'autocast' class.
            exc_type: Type of the exception being handled.
            exc_val: Value of the exception being handled.
            exc_tb: Traceback of the exception being handled.
        
        Returns:
            None. This method does not return any value.
        
        Raises:
            This method does not raise any exceptions explicitly. 
            However, exceptions may be raised during the execution of '_set_amp' function called within this method.
        """
        _set_amp(self.enabled, self.old_dtype)
        return False

    def __call__(self, func):
        r"""
        Executes the '__call__' method of the 'autocast' class.
        
        Args:
            self (autocast): An instance of the 'autocast' class.
            func (function): The function to be decorated.
        
        Returns:
            None: This method does not return any value.
        
        Raises:
            None: This method does not raise any exceptions.
        """
        return autocast_decorator(self, func)
