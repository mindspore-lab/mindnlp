import mindspore
from mindspore import ops
import numpy as np
from mindnlp.core import Tensor, CorePatch
from mindnlp.core.nn import Parameter

def test_forward():
    x = Parameter(Tensor(np.random.randn(3, 4), mindspore.float32))
    assert isinstance(x, Tensor)
    print(type(x))
    y = Tensor(np.random.randn(3, 4), mindspore.float32)

    with CorePatch():
        z = ops.add(x, y)
    
    assert type(z) == Tensor