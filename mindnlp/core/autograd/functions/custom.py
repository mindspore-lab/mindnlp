from mindspore._c_expression import TensorPy as MSTensor

from mindnlp import core
from core._prims.ascend import cast_npu
from core._prims.cpu import cast_cpu
from ..node import Node


class AccumulateGrad(Node):
    def __init__(self):
        super().__init__('AccumulateGrad')
        self._post_hook = None

    def construct(self, input):
        return input

    def bprop(self, input, output, grad):
        if input.grad is None:
            input.grad = grad
        else:
            input.grad += grad

        if self._post_hook is not None:
            self._post_hook(input)
        return grad

    def register_post_hook(self, hook):
        self._post_hook = hook


class Cast(Node):
    def __init__(self):
        super().__init__('Cast')
        self.used_bprop_inputs = []

    def construct(self, input, dtype, device):
        self.device = input.device
        self.dtype = input.dtype
        if device.type == 'cpu':
            out = cast_cpu(input, dtype).get_value()
        else:
            out = cast_npu(input, dtype).get_value()
        
        output = core.Tensor.__new__(core.Tensor)
        MSTensor.__init__(output, out)
        output.device = device
        return output

    def bprop(self, *args):
        grad = args[-1]
        if self.device.type == 'cpu':
            out = cast_cpu(grad, self.dtype).get_value()
        else:
            out = cast_npu(grad, self.dtype).get_value()

        output = core.Tensor.__new__(core.Tensor)
        MSTensor.__init__(output, out)
        output.device = self.device
        return output, None, None
