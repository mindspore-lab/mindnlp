"""Default communication hooks for DistributedDataParallel.

These hooks provide standard gradient reduction strategies including
compression techniques to reduce communication overhead.
"""

from .... import distributed as dist
from ....futures import Future


def allreduce_hook(process_group, bucket):
    """Default allreduce hook that averages gradients across all ranks.

    This is the default behavior of DDP - performs an allreduce with SUM
    operation and divides by world size to compute the average gradient.

    Args:
        process_group: The process group to use for communication.
                      If None, uses the default world group.
        bucket: GradBucket containing the gradients to reduce.

    Returns:
        Future[Tensor]: A future that resolves to the averaged gradient tensor.

    Example:
        >>> ddp_model.register_comm_hook(process_group, allreduce_hook)
    """
    group_to_use = process_group if process_group is not None else dist.group.WORLD
    world_size = group_to_use.size()

    # Get the gradient buffer
    tensor = bucket.buffer()

    # Divide by world size first to avoid overflow
    from ...._functional import mul
    tensor = mul(tensor, 1.0 / world_size)

    # Perform allreduce
    dist.all_reduce(tensor, op=dist.ReduceOp.SUM, group=group_to_use)

    # Return a completed future
    fut = Future()
    fut.set_result(tensor)
    return fut


def fp16_compress_hook(process_group, bucket):
    """Compress gradients to FP16 before allreduce, then decompress.

    This hook reduces communication bandwidth by 2x by casting gradients
    to float16 before the allreduce operation, then casting back to the
    original dtype after communication.

    Trade-off: Reduces bandwidth but may introduce numerical errors due
    to lower precision during communication.

    Args:
        process_group: The process group to use for communication.
                      If None, uses the default world group.
        bucket: GradBucket containing the gradients to reduce.

    Returns:
        Future[Tensor]: A future that resolves to the decompressed gradient
                       tensor in the original dtype.

    Example:
        >>> ddp_model.register_comm_hook(process_group, fp16_compress_hook)
    """
    import mindtorch_v2 as torch

    group_to_use = process_group if process_group is not None else dist.group.WORLD
    world_size = group_to_use.size()

    # Get the gradient buffer
    buffer = bucket.buffer()
    original_dtype = buffer.dtype

    # Compress: cast to fp16 and divide by world size
    compressed_tensor = buffer.to(torch.float16)
    from ...._functional import mul
    compressed_tensor = mul(compressed_tensor, 1.0 / world_size)

    # Perform allreduce on compressed tensor
    dist.all_reduce(compressed_tensor, op=dist.ReduceOp.SUM, group=group_to_use)

    # Decompress: cast back to original dtype
    decompressed_tensor = compressed_tensor.to(original_dtype)

    # Return a completed future with decompressed result
    fut = Future()
    fut.set_result(decompressed_tensor)
    return fut


def bf16_compress_hook(process_group, bucket):
    """Compress gradients to BF16 before allreduce, then decompress.

    This hook reduces communication bandwidth by 2x by casting gradients
    to bfloat16 before the allreduce operation, then casting back to the
    original dtype after communication.

    BF16 has the same exponent range as FP32 but reduced mantissa precision,
    making it more suitable for gradients than FP16 in some cases.

    Trade-off: Reduces bandwidth but may introduce numerical errors due
    to lower precision during communication.

    Args:
        process_group: The process group to use for communication.
                      If None, uses the default world group.
        bucket: GradBucket containing the gradients to reduce.

    Returns:
        Future[Tensor]: A future that resolves to the decompressed gradient
                       tensor in the original dtype.

    Example:
        >>> ddp_model.register_comm_hook(process_group, bf16_compress_hook)
    """
    import mindtorch_v2 as torch

    group_to_use = process_group if process_group is not None else dist.group.WORLD
    world_size = group_to_use.size()

    # Get the gradient buffer
    buffer = bucket.buffer()
    original_dtype = buffer.dtype

    # Compress: cast to bfloat16 and divide by world size
    compressed_tensor = buffer.to(torch.bfloat16)
    from ...._functional import mul
    compressed_tensor = mul(compressed_tensor, 1.0 / world_size)

    # Perform allreduce on compressed tensor
    dist.all_reduce(compressed_tensor, op=dist.ReduceOp.SUM, group=group_to_use)

    # Decompress: cast back to original dtype
    decompressed_tensor = compressed_tensor.to(original_dtype)

    # Return a completed future with decompressed result
    fut = Future()
    fut.set_result(decompressed_tensor)
    return fut
