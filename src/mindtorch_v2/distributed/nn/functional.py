def all_gather(tensor, group=None, async_op=False):
    """Differentiable all_gather (torch.distributed.nn.functional style).

    Returns a list of tensors gathered from all ranks.
    """
    import mindtorch_v2 as torch
    from .. import all_gather as _all_gather, get_world_size

    world_size = get_world_size(group)
    tensor_list = [torch.zeros_like(tensor) for _ in range(world_size)]
    work = _all_gather(tensor_list, tensor, group=group, async_op=async_op)
    if async_op:
        return work
    return tensor_list


def broadcast(tensor, src=0, group=None, async_op=False):
    from .. import broadcast as _broadcast
    return _broadcast(tensor, src=src, group=group, async_op=async_op)


def all_reduce(tensor, op=None, group=None, async_op=False):
    from .. import all_reduce as _all_reduce, ReduceOp
    if op is None:
        op = ReduceOp.SUM
    return _all_reduce(tensor, op=op, group=group, async_op=async_op)


def reduce_scatter(input, op=None, group=None, async_op=False):
    """Differentiable reduce_scatter (torch.distributed.nn.functional style).

    Takes a single input tensor, reduces and scatters. Returns output tensor.
    """
    from .. import reduce_scatter_tensor as _reduce_scatter_tensor, ReduceOp, get_world_size
    import mindtorch_v2 as torch

    if op is None:
        op = ReduceOp.SUM
    world_size = get_world_size(group)
    numel = input.numel()
    output = torch.zeros(numel // world_size, dtype=input.dtype,
                         device=input.device)
    _reduce_scatter_tensor(output, input, op=op, group=group,
                           async_op=async_op)
    return output
