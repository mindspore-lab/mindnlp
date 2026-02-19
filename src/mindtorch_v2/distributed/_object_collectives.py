import io
import pickle


def _object_to_tensor(obj):
    """Serialize a picklable object to a uint8 tensor."""
    import mindtorch_v2 as torch
    buf = pickle.dumps(obj)
    t = torch.tensor(list(buf), dtype=torch.uint8)
    return t, torch.tensor([len(buf)], dtype=torch.int64)


def _tensor_to_object(tensor, size):
    """Deserialize a uint8 tensor back to a Python object."""
    buf = bytes(tensor[:int(size)].tolist())
    return pickle.loads(buf)


def broadcast_object_list(object_list, src=0, group=None, device=None):
    from . import broadcast, get_rank, get_world_size
    import mindtorch_v2 as torch

    rank = get_rank(group)
    size_list = torch.tensor([0] * len(object_list), dtype=torch.int64)

    if rank == src:
        tensor_list = []
        for i, obj in enumerate(object_list):
            t, s = _object_to_tensor(obj)
            tensor_list.append(t)
            size_list[i] = s[0]

    broadcast(size_list, src=src, group=group)

    if rank == src:
        max_size = int(max(size_list).tolist())
        flat = torch.zeros(len(object_list) * max_size, dtype=torch.uint8)
        for i, t in enumerate(tensor_list):
            flat[i * max_size: i * max_size + t.numel()] = t
    else:
        max_size = int(max(size_list).tolist())
        flat = torch.zeros(len(object_list) * max_size, dtype=torch.uint8)

    broadcast(flat, src=src, group=group)

    if rank != src:
        for i in range(len(object_list)):
            sz = int(size_list[i])
            object_list[i] = _tensor_to_object(
                flat[i * max_size: i * max_size + sz], sz
            )


def all_gather_object(object_list, obj, group=None):
    from . import all_gather, get_rank, get_world_size, broadcast, all_reduce
    import mindtorch_v2 as torch

    world_size = get_world_size(group)
    t, size_tensor = _object_to_tensor(obj)

    # Gather sizes
    size_list = [torch.tensor([0], dtype=torch.int64) for _ in range(world_size)]
    all_gather(size_list, size_tensor, group=group)

    max_size = max(int(s[0]) for s in size_list)
    padded = torch.zeros(max_size, dtype=torch.uint8)
    padded[:t.numel()] = t

    tensor_list = [torch.zeros(max_size, dtype=torch.uint8) for _ in range(world_size)]
    all_gather(tensor_list, padded, group=group)

    for i in range(world_size):
        sz = int(size_list[i][0])
        object_list[i] = _tensor_to_object(tensor_list[i], sz)


def gather_object(obj, object_gather_list=None, dst=0, group=None):
    from . import gather, get_rank, get_world_size, all_gather
    import mindtorch_v2 as torch

    rank = get_rank(group)
    world_size = get_world_size(group)
    t, size_tensor = _object_to_tensor(obj)

    # Use all_gather for sizes since HCCL may not support gather directly
    size_list = [torch.tensor([0], dtype=torch.int64) for _ in range(world_size)]
    all_gather(size_list, size_tensor, group=group)

    max_size = max(int(s[0]) for s in size_list)
    padded = torch.zeros(max_size, dtype=torch.uint8)
    padded[:t.numel()] = t

    tensor_list = [torch.zeros(max_size, dtype=torch.uint8) for _ in range(world_size)]
    all_gather(tensor_list, padded, group=group)

    if rank == dst and object_gather_list is not None:
        for i in range(world_size):
            sz = int(size_list[i][0])
            object_gather_list[i] = _tensor_to_object(tensor_list[i], sz)


def scatter_object_list(scatter_object_output_list, scatter_object_input_list=None,
                        src=0, group=None):
    from . import broadcast, get_rank, get_world_size
    import mindtorch_v2 as torch

    rank = get_rank(group)
    world_size = get_world_size(group)

    if rank == src:
        tensor_sizes = []
        tensors = []
        for obj in scatter_object_input_list:
            t, s = _object_to_tensor(obj)
            tensors.append(t)
            tensor_sizes.append(int(s[0]))
        max_size = max(tensor_sizes)
        size_tensor = torch.tensor(tensor_sizes, dtype=torch.int64)
    else:
        max_size = 0
        size_tensor = torch.tensor([0] * world_size, dtype=torch.int64)

    broadcast(size_tensor, src=src, group=group)
    max_size = int(max(size_tensor).tolist())

    if rank == src:
        flat = torch.zeros(world_size * max_size, dtype=torch.uint8)
        for i, t in enumerate(tensors):
            flat[i * max_size: i * max_size + t.numel()] = t
    else:
        flat = torch.zeros(world_size * max_size, dtype=torch.uint8)

    broadcast(flat, src=src, group=group)

    sz = int(size_tensor[rank])
    scatter_object_output_list[0] = _tensor_to_object(
        flat[rank * max_size: rank * max_size + sz], sz
    )
