"""
MindSpore implementation of PyTorch's _c10d functions (distributed communication).
This module provides MindSpore-based implementations of distributed communication operations.
"""
import torch
import mindspore as ms
import mindspore.numpy as mnp
import mindspore.communication as ms_comm
from mindspore import ops

from torch4ms.ops import ops_registry

# Try to initialize distributed environment if available
try:
    # Initialize HCCL/NCCL for GPU/Ascend distributed training
    ms_comm.init()
    RANK_ID = ms_comm.get_rank()
    WORLD_SIZE = ms_comm.get_group_size()
    DISTRIBUTED_ENABLED = True
except:
    # If distributed initialization fails, use single-device mode
    RANK_ID = 0
    WORLD_SIZE = 1
    DISTRIBUTED_ENABLED = False

def op(*aten, **kwargs):
    """
    Decorator to register MindSpore implementations for PyTorch operations.
    
    Args:
        *aten: PyTorch aten operations to register
        **kwargs: Additional registration arguments
        
    Returns:
        Decorator function
    """
    def inner(func):
        for a in aten:
            ops_registry.register_torch_dispatch_op(a, func, **kwargs)
        return func
    return inner


@op(torch.ops._c10d_functional.all_gather_into_tensor)
def _c10d_all_gather(input_tensor, group_size: int, group_name: str):
    """
    All-gather operation implementation using MindSpore.
    
    Args:
        input_tensor: Tensor to gather
        group_size: Number of processes in the group
        group_name: Name of the communication group
        
    Returns:
        Gathered tensor
    """
    if not DISTRIBUTED_ENABLED:
        # In single-device mode, just return the input tensor
        return input_tensor
    
    # Create a list to hold tensors from all devices
    gathered_tensors = []
    
    # In MindSpore, we use all_gather to collect tensors from all processes
    for i in range(WORLD_SIZE):
        # For each rank, create a tensor to receive data
        if i == RANK_ID:
            # For the current rank, use the input tensor
            gathered_tensors.append(input_tensor)
        else:
            # For other ranks, create a zero tensor of the same shape and dtype
            gathered_tensors.append(mnp.zeros_like(input_tensor))
    
    # Use all_gather to collect tensors from all processes
    # Note: MindSpore's all_gather works differently from JAX's
    # In a real implementation, you would use ms_comm.all_gather
    # This is a simplified implementation
    all_gathered = ms_comm.all_gather(input_tensor)
    
    return all_gathered


@op(torch.ops._c10d_functional.all_reduce)
def _c10d_all_reduce(input_tensor, reduceOp: str, group_name: str):
    """
    All-reduce operation implementation using MindSpore.
    
    Args:
        input_tensor: Tensor to reduce
        reduceOp: Reduction operation type (sum, avg, min, max)
        group_name: Name of the communication group
        
    Returns:
        Reduced tensor
    """
    if not DISTRIBUTED_ENABLED:
        # In single-device mode, just return the input tensor
        return input_tensor
    
    # Perform the appropriate reduction operation
    if reduceOp == "sum":
        # Use all_reduce with sum operation
        res = ms_comm.all_reduce(input_tensor, op=ms_comm.ReduceOp.SUM)
    elif reduceOp == "avg":
        # For average, perform sum and then divide by world size
        sum_result = ms_comm.all_reduce(input_tensor, op=ms_comm.ReduceOp.SUM)
        res = sum_result / WORLD_SIZE
    elif reduceOp == "min":
        # Use all_reduce with min operation
        # Note: MindSpore's all_reduce may not support min directly
        # This is a simplified implementation
        res = ms_comm.all_reduce(input_tensor, op=ms_comm.ReduceOp.MIN)
    elif reduceOp == "max":
        # Use all_reduce with max operation
        # Note: MindSpore's all_reduce may not support max directly
        # This is a simplified implementation
        res = ms_comm.all_reduce(input_tensor, op=ms_comm.ReduceOp.MAX)
    else:
        raise RuntimeError(f"Reduce op {reduceOp} not implemented in MindSpore backend")
    
    return res


@op(torch.ops._c10d_functional.broadcast)
def _c10d_broadcast(input_tensor, src: int, group_name: str):
    """
    Broadcast operation implementation using MindSpore.
    
    Args:
        input_tensor: Tensor to broadcast (used only by the source rank)
        src: Source rank
        group_name: Name of the communication group
        
    Returns:
        Broadcasted tensor
    """
    if not DISTRIBUTED_ENABLED:
        # In single-device mode, just return the input tensor
        return input_tensor
    
    # Use broadcast to send tensor from source rank to all other ranks
    # MindSpore's broadcast works by having each rank call the function
    res = ms_comm.broadcast(input_tensor, src)
    
    return res


@op(torch.ops._c10d_functional.wait_tensor)
def _c10d_wait_tensor(tensor):
    """
    Wait for a tensor to complete asynchronous operations.
    
    Args:
        tensor: Tensor to wait for
        
    Returns:
        The same tensor (MindSpore tensors are synchronous by default)
    """
    # MindSpore tensors are synchronous by default,
    # so we don't need to wait for them explicitly
    return tensor


# Additional utility functions for distributed training
def get_rank():
    """
    Get the current process rank.
    
    Returns:
        Current rank ID
    """
    return RANK_ID


def get_world_size():
    """
    Get the total number of processes.
    
    Returns:
        World size
    """
    return WORLD_SIZE


def is_distributed():
    """
    Check if distributed training is enabled.
    
    Returns:
        True if distributed training is enabled, False otherwise
    """
    return DISTRIBUTED_ENABLED


def barrier():
    """
    Synchronize all processes.
    """
    if DISTRIBUTED_ENABLED:
        # MindSpore doesn't have a direct barrier API
        # We can use a simple all_reduce operation as a barrier
        dummy = mnp.array([1])
        ms_comm.all_reduce(dummy, op=ms_comm.ReduceOp.SUM)