import torch
from BlockManager import BlockAllocator, MetadataEngine, Multiplication

class CacheWithTorch:
    def __init__(self, num_blocks, num_heads, block_size, head_dim):
        self.num_heads = num_heads
        self.head_dim = head_dim
        self.A = [torch.empty(0, head_dim)] * num_heads
    def write(self, A):
        for i in range(0, self.num_heads):
            self.A[i] = torch.cat([self.A[i], A[i].unsqueeze(0)], 0)
    def transmatmul(self, x):
        B = [torch.empty(0)] * self.num_heads
        for i in range(0, self.num_heads):
            B[i] = x[i].matmul(self.A[i].transpose(-1, -2)).unsqueeze(0)
        output = torch.cat(B)
        return output
    def matmul(self, x):
        B = [torch.empty(0)] * self.num_heads
        for i in range(0, self.num_heads):
            B[i] = x[i].matmul(self.A[i]).unsqueeze(0)
        output = torch.cat(B)
        return output

class CacheWithMindspore:
    def __init__(self, num_blocks, num_heads, block_size, head_dim):
        self.block_allocator = BlockAllocator(num_blocks, block_size, head_dim)
        self.meta_data_engine = MetadataEngine(num_blocks, num_heads, block_size)
        self.num_heads = num_heads

    def write(self, A):
        self.block_allocator.alloc(self.meta_data_engine.alloc(), A)

    def transmatmul(self, x):
        B = []
        for i in range(0, self.num_heads):
            B.append(Multiplication.matmul(x[i], self.block_allocator.get(self.meta_data_engine.get(i)), True))
        return torch.cat(B)

    def matmul(self, x):
        B = []
        for i in range(0, self.num_heads):
            B.append(Multiplication.matmul(x[i], self.block_allocator.get(self.meta_data_engine.get(i)), False))
        return torch.cat(B)

class Cache:
    def __init__(self, num_blocks, num_heads, block_size, head_dim):
        self.cache = CacheWithMindspore(num_blocks, num_heads, block_size, head_dim)
    def write(self, A):
        self.cache.write(A)
    def transmatmul(self, x):
        return self.cache.transmatmul(x)
    def matmul(self, x):
        return self.cache.matmul(x)
    def delete(self):
        del self.cache