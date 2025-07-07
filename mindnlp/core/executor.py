import mindspore
from mindspore._c_expression import TensorNode, SequenceNode, NoneTypeNode, AnyTypeNode, Tensor as MSTensor
import mindspore.common._stub_tensor
from mindspore.common.api import _pynative_executor
from mindspore.common._stub_tensor import _convert_python_data

from mindnlp import core
from ._tensor import Tensor
from .dispatcher import dispatcher

def _convert_stub(stub, device):
    "convert stub to StubNode or Value"
    if isinstance(stub, (MSTensor, TensorNode)):
        return Tensor(stub, device=device)
    if isinstance(stub, tuple):
        return tuple(_convert_stub(e, device) for e in stub)
    if isinstance(stub, SequenceNode):
        elements = stub.get_elements()
        return tuple(_convert_stub(e, device) for e in elements)
    if isinstance(stub, NoneTypeNode):
        val = stub.get_real_value()
        return _convert_python_data(val)
    if isinstance(stub, AnyTypeNode):
        val = stub.get_real_node()
        return _convert_stub(val, device)
    return _convert_python_data(stub)


def execute(func_name, *args, **kwargs):
    requires_grad = kwargs.pop('requires_grad', False)
    user_created = kwargs.pop('user_created', False)
    out, device = dispatcher.dispatch(func_name, *args, **kwargs)
    out_tensor = _convert_stub(out, device=device)
    if requires_grad:
        out_tensor._requires_grad = True
    if user_created:
        out_tensor._user_created = True
        out_tensor.attach_grad()

    return out_tensor

