"""new Parameter"""
import uuid
import copy
from mindspore import Tensor
from mindspore._c_expression import ParamInfo # pylint: disable=no-name-in-module

class Parameter(Tensor):
    def __init__(self, input_data=None, requires_grad=True):
        super().__init__(input_data)
        self.meta = False
        self.param_info = ParamInfo()
        self.param_info.name = str(uuid.uuid4())
        self.param_info.parameter_shape = self.shape
        self.param_info.requires_grad = requires_grad
        self.requires_grad = requires_grad

    def __deepcopy__(self, memodict):
        new_obj = Parameter(self)
        return new_obj

    @property
    def data(self):
        return self

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
