import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import mindspore
import numpy as np
import time

from bitsandbytes import matmul
from mindspore import Tensor, ops
from mindspore._c_expression import _framework_profiler_step_start
from mindspore._c_expression import _framework_profiler_step_end

mindspore.context.set_context(device_target="GPU")

a = Tensor(np.random.randn(8192, 8192).astype(np.float16))
b = Tensor(np.random.randn(8192, 8192).astype(np.float16))
b_ops = b.t()
# for i in range(5):
#     c_old = ops.matmul(a, b_ops)
c_old = ops.matmul(a, b_ops)

start = time.time()
# # _framework_profiler_step_start()
# # profiler = mindspore.Profiler()
for i in range(10):
    c_old = ops.matmul(a, b_ops)
# # _framework_profiler_step_end()
# c_old = ops.matmul(a, b.t())
tick = time.time()
time_ops = tick - start
# for i in range(5):
#     c_new = matmul(a, b)
c_new = matmul(a, b)
start = time.time()
# _framework_profiler_step_start()
for i in range(10):
    c_new = matmul(a, b)
# c_new = matmul(a, b)
# _framework_profiler_step_end()
tick = time.time()
time_bnb = tick - start
# profiler.analyse()
# print(c_new)
# print(c_old)
print("ops.matmul time: ", time_ops)
print("bnb.matmul time: ", time_bnb)

# while True:
#     # c_old = matmul(a, b)
#     c_old = ops.matmul(a, b_ops)
