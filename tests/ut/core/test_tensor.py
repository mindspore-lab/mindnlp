import mindspore
from mindspore import ops
import numpy as np
from mindnlp.core import Tensor, CorePatch

def test_forward():
    x = Tensor(np.random.randn(3, 4), mindspore.float32)
    y = Tensor(np.random.randn(3, 4), mindspore.float32)

    with CorePatch():
        z = ops.add(x, y)
    
    assert type(z) == Tensor

def test_backward():
    x = Tensor(np.random.randn(3, 4), mindspore.float32)
    y = Tensor(np.random.randn(3, 4), mindspore.float32)

    grad_fn = mindspore.value_and_grad(ops.add)
    with CorePatch():
        z, grads = grad_fn(x, y)
    
    assert type(z) == Tensor
    assert type(grads) == mindspore.Tensor
