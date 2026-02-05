"""Parameter class - a Tensor that is a module parameter."""

from .._tensor import Tensor


class Parameter(Tensor):
    """A Tensor that is automatically registered as a parameter when assigned to a Module.

    Parameters are Tensor subclasses that have requires_grad=True by default.
    When assigned as a Module attribute, they are automatically added to the
    list of module parameters.
    """

    def __new__(cls, data=None, requires_grad=True):
        if data is None:
            raise ValueError("Parameter requires data")

        if isinstance(data, Tensor):
            # Create new Parameter from existing Tensor
            instance = object.__new__(cls)
            instance._storage = data._storage
            instance._shape = data._shape
            instance._stride = data._stride
            instance._storage_offset = data._storage_offset
            instance._dtype = data._dtype
            instance._device = data._device
            instance._requires_grad = requires_grad
            instance._grad_fn = None
            instance._grad = None
            instance._version = 0
            instance._hooks = {}
            instance._hook_counter = 0
            return instance
        else:
            # Create from raw data
            tensor = Tensor(data, requires_grad=requires_grad)
            instance = object.__new__(cls)
            instance._storage = tensor._storage
            instance._shape = tensor._shape
            instance._stride = tensor._stride
            instance._storage_offset = tensor._storage_offset
            instance._dtype = tensor._dtype
            instance._device = tensor._device
            instance._requires_grad = requires_grad
            instance._grad_fn = None
            instance._grad = None
            instance._version = 0
            instance._hooks = {}
            instance._hook_counter = 0
            return instance

    def __init__(self, data=None, requires_grad=True):
        # All initialization done in __new__
        pass

    def __repr__(self):
        return f"Parameter containing:\n{super().__repr__()}"
