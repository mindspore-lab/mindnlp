"""new Parameter"""
import uuid
import copy
from mindspore import Tensor
from mindspore._c_expression import ParamInfo # pylint: disable=no-name-in-module
from mindspore.common._stub_tensor import StubTensor

class Parameter(Tensor):
    grad = None
    requires_grad = False
    _grad_fn = None

    def __init__(self, input_data=None, requires_grad=True, **kwargs):
        super().__init__(input_data)
        self.meta = False
        self.param_info = ParamInfo()
        self.param_info.name = str(uuid.uuid4())
        self.param_info.parameter_shape = self._shape
        self.param_info.requires_grad = requires_grad
        self._requires_grad = requires_grad

    def __deepcopy__(self, memodict):
        new_obj = Parameter(self)
        return new_obj

    def clone(self):
        return copy.deepcopy(self)

    def __parameter__(self): # only for O2
        """For parse check."""

    @property
    def name(self): # only for O2
        """
        Get the name of the parameter.

        Examples:
            >>> from mindspore import Tensor, Parameter
            >>> import numpy as np
            >>> x = Parameter(Tensor(np.array([1, 2], dtype=np.float32)), name="param")
            >>> x.name = "param1"
            >>> x.name
            'param1'
        """
        return self.param_info.name

    @property
    def data(self):
        return Tensor(self)

    @data.setter
    def data(self, new_value):
        if isinstance(new_value, StubTensor) and new_value.stub is not None:
            new_value = new_value.stub.get_value()
        self.assign_value(new_value)

    @property
    def requires_grad(self):
        return self._requires_grad

    @requires_grad.setter
    def requires_grad(self, value):
        if not isinstance(value, bool):
            raise TypeError("The 'requires_grad' attribute of parameter must be set as bool.")
        self.param_info.requires_grad = value
        self._requires_grad = value
        if value:
            if not hasattr(self, 'handle'):
                self.retain_grad()
        else:
            if hasattr(self, 'handle'):
                self.handle.remove()
                delattr(self, 'handle')


class UninitializedParameter(Parameter):
    def __init__(self, input_data=None, requires_grad=True):
        super().__init__(input_data, requires_grad)

def is_lazy(param):
    return False


class Buffer(Tensor):
    r"""A kind of Tensor that should not be considered a model
    parameter. For example, BatchNorm's ``running_mean`` is not a parameter, but is part of the module's state.

    Buffers are :class:`~torch.Tensor` subclasses, that have a
    very special property when used with :class:`Module` s -- when they're
    assigned as Module attributes they are automatically added to the list of
    its buffers, and will appear e.g. in :meth:`~torch.nn.Module.buffers` iterator.
    Assigning a Tensor doesn't have such effect. One can still assign a Tensor as explicitly by using
    the :meth:`~torch.nn.Module.register_buffer` function.

    Args:
        data (Tensor): buffer tensor.
        persistent (bool, optional): whether the buffer is part of the module's
            :attr:`state_dict`. Default: ``True``
    """

    def __new__(cls, data=None, *, persistent=True):
        if data is None:
            data = core.empty(0)

        t = data.detach().requires_grad_(data.requires_grad)
        t.persistent = persistent
        t._is_buffer = True
        return t