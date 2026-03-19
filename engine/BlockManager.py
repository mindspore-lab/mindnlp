from mindspore import tensor, Parameter, float32
from mindspore.ops import ScatterNdUpdate, gather, stack, MatMul
from mindspore.common.initializer import initializer, Zero
import torch

class BlockAllocator:
    def __init__(self, num_blocks, block_size, head_dim):
        self.num_blocks = num_blocks
        self.pool = []
        for _ in range(num_blocks):
            self.pool.append(Parameter(initializer(Zero(), [block_size, head_dim], float32)))

    def alloc(self, indices, A):
        for index, a in list(zip(indices, A)):
            self.pool[index[0]] = ScatterNdUpdate()(self.pool[index[0]], tensor([[index[1]]]), tensor(a.detach().numpy()).unsqueeze(0))
    
    def get(self, indices):
        A = []
        for index in indices:
            A.append(gather(self.pool[index[0]], tensor(index[1]), 0))
        return stack(A)

class MetadataEngine:
    def __init__(self, num_blocks, num_heads, block_size):
        self.num_blocks = num_blocks
        self.num_heads = num_heads
        self.block_size = block_size
        self.block_table = []
        self.num_token = 0
        self.i = 0
        self.j = 0

    def alloc(self):
        list = []
        self.num_token = self.num_token + 1
        for _ in range(self.num_heads):
            if self.i == self.num_blocks:
                raise ValueError("Out of Memory")
            list.append([self.i, self.j])
            self.j = self.j + 1
            if self.j == self.block_size:
                self.i = self.i + 1
                self.j = 0
        self.block_table.append(list)
        return list

    def get(self, x):
        list = []
        for k in range(self.num_token):
            list.append(self.block_table[k][x])
        return list

class Multiplication:
    def matmul(x, A, trans):
        matmul = MatMul(transpose_b = trans)
        return torch.from_numpy(matmul(tensor([x.detach().numpy()]), A).asnumpy())