import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import mindspore.context
import numpy as np
import mindspore
from mindspore import Tensor
from mindnlp.core import nn
from bitsandbytes.nn import Linear8bitLt


np.random.seed(42)
mindspore.set_seed(42)
mindspore.context.set_context(device_target="GPU")

int8_model = Linear8bitLt(4, 2, has_fp16_weights=False)

weight = Tensor([[1, 2, 3, 4], [5, 6, 7, 8]], dtype=mindspore.float16)
bias = Tensor([1, 2], dtype=mindspore.float16)

int8_model.weight.assign_value(weight)
int8_model.bias.assign_value(bias)

int8_model.quant()
for name, param in int8_model.parameters_and_names():
    print(name)
    print(param)
    print(param.data.asnumpy())


input_data = Tensor([[1, 2, 3, 4], [5, 6, 7, 8]], dtype=mindspore.float16)

int8_output = int8_model(input_data)

print(int8_output)
