"""new Parameter"""
import uuid
from mindspore import Tensor
from mindspore._c_expression import ParamInfo # pylint: disable=no-name-in-module

class Parameter(Tensor):
    def __init__(self, input_data=None, requires_grad=True):
        super().__init__(input_data)
        self.param_info = ParamInfo()
        self.param_info.name = str(uuid.uuid4())
        self.param_info.requires_grad = requires_grad
        self.requires_grad = requires_grad

    @property
    def data(self):
        return self
