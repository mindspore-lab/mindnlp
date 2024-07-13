from mindspore import ops
import mindspore as ms
ms.set_context(pynative_synchronize=True)
n = 12
randperm_op = ops.randperm(n)[:3]
permutation = randperm_op
print("Random permutation:", permutation)
