import os
from datetime import timedelta

from ._reduce_op import ReduceOp, RedOpType, reduce_op
from ._work import Work
from ._process_group import ProcessGroup, ProcessGroupHCCL
from ._gloo import ProcessGroupGloo
from ._store import TCPStore
from ._backend import (
    Backend, GroupMember, Store, PrefixStore,
    is_nccl_available, is_gloo_available, is_mpi_available, is_ucc_available,
    is_hccl_available, is_backend_available, is_torchelastic_launched,
    get_default_backend_for_device,
)
from ._p2p import P2POp, batch_isend_irecv
from ._object_collectives import (
    broadcast_object_list, all_gather_object,
    gather_object, scatter_object_list,
)
from .device_mesh import init_device_mesh
from . import nn

# ---------------------------------------------------------------------------
# Global state
# ---------------------------------------------------------------------------

_default_pg = None
_pg_map = {}          # ProcessGroup -> (Backend, Store)
_pg_names = {}        # ProcessGroup -> str
_pg_group_ranks = {}  # ProcessGroup -> {global_rank: group_rank}
_group_count = 0

default_pg_timeout = timedelta(minutes=30)

# Placeholder option classes for API compatibility
AllreduceOptions = object
AllreduceCoalescedOptions = object
AllToAllOptions = object
BarrierOptions = object
BroadcastOptions = object
GatherOptions = object
ReduceOptions = object
ReduceScatterOptions = object
ScatterOptions = object
DebugLevel = object


def get_debug_level():
    return 0


# ---------------------------------------------------------------------------
# Availability and initialization
# ---------------------------------------------------------------------------

def is_available():
    return True


def is_initialized():
    return _default_pg is not None


def _get_env_rank():
    for var in ("RANK", "OMPI_COMM_WORLD_RANK", "PMI_RANK", "PMIX_RANK"):
        val = os.environ.get(var)
        if val is not None:
            return int(val)
    return 0


def _get_env_world_size():
    for var in ("WORLD_SIZE", "OMPI_COMM_WORLD_SIZE", "PMI_SIZE"):
        val = os.environ.get(var)
        if val is not None:
            return int(val)
    return 1


def _create_store_from_env(timeout):
    host = os.environ.get("MASTER_ADDR", "127.0.0.1")
    port = int(os.environ.get("MASTER_PORT", "29500"))
    rank = _get_env_rank()
    world_size = _get_env_world_size()
    return TCPStore(host, port, world_size, is_master=(rank == 0),
                    timeout=int(timeout.total_seconds())
                    if isinstance(timeout, timedelta) else int(timeout))


def _parse_timeout(timeout):
    if timeout is None:
        return default_pg_timeout
    if isinstance(timeout, (int, float)):
        return timedelta(seconds=timeout)
    return timeout


def init_process_group(backend=None, init_method=None, timeout=None,
                       world_size=-1, rank=-1, store=None,
                       group_name="", pg_options=None, device_id=None):
    global _default_pg, _group_count

    if _default_pg is not None:
        raise RuntimeError(
            "trying to initialize the default process group twice!"
        )

    timeout = _parse_timeout(timeout)
    timeout_sec = int(timeout.total_seconds())

    if backend is None:
        backend = "hccl" if is_hccl_available() else "gloo"
    backend = str(backend).lower()

    if rank < 0:
        rank = _get_env_rank()
    if world_size < 0:
        world_size = _get_env_world_size()

    if store is None:
        if init_method is not None and init_method.startswith("tcp://"):
            parts = init_method[len("tcp://"):].split(":")
            host = parts[0]
            port = int(parts[1]) if len(parts) > 1 else 29500
            store = TCPStore(host, port, world_size,
                             is_master=(rank == 0), timeout=timeout_sec)
        else:
            store = _create_store_from_env(timeout_sec)

    dev_id = None
    if device_id is not None:
        dev_id = int(getattr(device_id, "index", device_id))

    # Backend dispatch
    if backend == "gloo":
        pg = ProcessGroupGloo(store, rank, world_size,
                              group_name=group_name,
                              group_ranks=list(range(world_size)))
    elif backend in ("hccl", "nccl"):
        pg = ProcessGroupHCCL(store, rank, world_size, device_id=dev_id,
                              group_name=group_name,
                              group_ranks=list(range(world_size)))
    else:
        raise ValueError(f"Unsupported backend: {backend}")

    _default_pg = pg
    GroupMember.WORLD = pg
    _pg_map[pg] = (Backend(backend), store)
    _pg_names[pg] = group_name or "default_pg"
    _pg_group_ranks[pg] = {i: i for i in range(world_size)}
    _group_count += 1


def destroy_process_group(group=None):
    global _default_pg, _group_count

    if group is None or group is _default_pg:
        # Destroy all groups
        for pg in list(_pg_map.keys()):
            pg.destroy()
        _pg_map.clear()
        _pg_names.clear()
        _pg_group_ranks.clear()
        _default_pg = None
        GroupMember.WORLD = None
        _group_count = 0
    else:
        group.destroy()
        _pg_map.pop(group, None)
        _pg_names.pop(group, None)
        _pg_group_ranks.pop(group, None)


# ---------------------------------------------------------------------------
# Utility functions
# ---------------------------------------------------------------------------

def get_rank(group=None):
    pg = group or _default_pg
    if pg is None:
        return 0
    return pg.rank()


def get_world_size(group=None):
    pg = group or _default_pg
    if pg is None:
        return 1
    return pg.size()


def get_backend(group=None):
    pg = group or _default_pg
    if pg in _pg_map:
        return _pg_map[pg][0]
    return Backend("hccl")


def get_backend_config(group=None):
    pg = group or _default_pg
    if pg in _pg_map:
        backend_name = _pg_map[pg][0]
        if backend_name == "gloo":
            return "cpu:gloo"
        elif backend_name in ("hccl", "nccl"):
            return "npu:hccl"
    return "npu:hccl"


def get_group_rank(group, global_rank):
    rank_map = _pg_group_ranks.get(group)
    if rank_map is None:
        raise ValueError("The given group is not registered")
    if global_rank not in rank_map:
        raise ValueError(
            f"The global rank {global_rank} is not part of the group"
        )
    return rank_map[global_rank]


def get_global_rank(group, group_rank):
    rank_map = _pg_group_ranks.get(group)
    if rank_map is None:
        raise ValueError("The given group is not registered")
    for g, l in rank_map.items():
        if l == group_rank:
            return g
    raise ValueError(
        f"The group rank {group_rank} is not part of the group"
    )


def get_process_group_ranks(group):
    rank_map = _pg_group_ranks.get(group)
    if rank_map is None:
        if group is _default_pg:
            return list(range(group.size()))
        raise ValueError("The given group is not registered")
    return sorted(rank_map.keys())


def get_node_local_rank(fallback_rank=None):
    for var in ("LOCAL_RANK", "MPI_LOCALRANKID", "OMPI_COMM_WORLD_LOCAL_RANK",
                "MV2_COMM_WORLD_LOCAL_RANK"):
        val = os.environ.get(var)
        if val is not None:
            return int(val)
    if fallback_rank is not None:
        return fallback_rank
    raise RuntimeError("Could not determine node-local rank")


def get_pg_count():
    return _group_count


class _GroupModule:
    """Module-like object so that dist.group.WORLD works."""
    @property
    def WORLD(self):
        return GroupMember.WORLD

group = _GroupModule()


# ---------------------------------------------------------------------------
# Collective operations
# ---------------------------------------------------------------------------

def _split_flat_to_list(flat_tensor, tensor_list, numel, dtype):
    """Copy shards from a flat buffer into individual tensors in tensor_list."""
    device_type = flat_tensor.storage().device.type
    itemsize = dtype.itemsize
    shard_bytes = numel * itemsize
    src_base = flat_tensor.storage().data_ptr()

    if device_type == "npu":
        from .._backends.npu import runtime as npu_runtime
        ACL_MEMCPY_D2D = 3
        dev_id = flat_tensor.storage().device.index or 0
        runtime = npu_runtime.get_runtime(dev_id)
        runtime.activate()
        for i, t in enumerate(tensor_list):
            src_ptr = src_base + i * shard_bytes
            dst_ptr = t.storage().data_ptr()
            ret = npu_runtime.acl.rt.memcpy(
                dst_ptr, shard_bytes, src_ptr, shard_bytes,
                ACL_MEMCPY_D2D,
            )
            if ret != 0:
                raise RuntimeError(f"acl.rt.memcpy D2D failed: {ret}")
    else:
        import ctypes
        for i, t in enumerate(tensor_list):
            src_ptr = src_base + i * shard_bytes
            dst_ptr = t.storage().data_ptr()
            ctypes.memmove(dst_ptr, src_ptr, shard_bytes)

def barrier(group=None, async_op=False, device_ids=None):
    pg = group or _default_pg
    work = pg.barrier()
    if not async_op:
        work.wait()
        return None
    return work


def all_reduce(tensor, op=ReduceOp.SUM, group=None, async_op=False):
    pg = group or _default_pg
    work = pg.allreduce(tensor, op)
    if not async_op:
        work.wait()
        return None
    return work


def all_reduce_coalesced(tensors, op=ReduceOp.SUM, group=None, async_op=False):
    pg = group or _default_pg
    works = []
    for t in tensors:
        works.append(pg.allreduce(t, op))
    if not async_op:
        for w in works:
            w.wait()
        return None
    return works[-1] if works else None


def broadcast(tensor, src=None, group=None, async_op=False, group_src=None):
    pg = group or _default_pg
    if src is None and group_src is None:
        src = 0
    elif src is None:
        src = group_src
    work = pg.broadcast(tensor, root=src)
    if not async_op:
        work.wait()
        return None
    return work


def reduce(tensor, dst=None, op=ReduceOp.SUM, group=None, async_op=False,
           group_dst=None):
    pg = group or _default_pg
    if dst is None and group_dst is None:
        dst = 0
    elif dst is None:
        dst = group_dst
    work = pg.reduce(tensor, dst=dst, op=op)
    if not async_op:
        work.wait()
        return None
    return work


def all_gather(tensor_list, tensor, group=None, async_op=False):
    pg = group or _default_pg
    import mindtorch_v2 as torch

    # PyTorch signature: tensor_list is a list of pre-allocated tensors,
    # one per rank. HCCL writes into a single contiguous buffer.
    # We allocate a flat output buffer, call HcclAllGather, then split.
    world_size = pg.size()
    numel = tensor.numel()
    flat_output = torch.zeros(world_size * numel, dtype=tensor.dtype,
                              device=tensor.device)
    work = pg.allgather(flat_output, tensor)
    if not async_op:
        work.wait()
        _split_flat_to_list(flat_output, tensor_list, numel, tensor.dtype)
        return None
    return work


def all_gather_into_tensor(output_tensor, input_tensor, group=None,
                           async_op=False):
    pg = group or _default_pg
    work = pg.allgather(output_tensor, input_tensor)
    if not async_op:
        work.wait()
        return None
    return work


# Deprecated alias
def _all_gather_base(output_tensor, input_tensor, group=None, async_op=False):
    return all_gather_into_tensor(output_tensor, input_tensor, group, async_op)


def all_gather_coalesced(output_tensor_lists, input_tensor_list, group=None,
                         async_op=False):
    pg = group or _default_pg
    works = []
    for out_list, inp in zip(output_tensor_lists, input_tensor_list):
        w = all_gather(out_list, inp, group=pg, async_op=True)
        works.append(w)
    if not async_op:
        for w in works:
            if w is not None:
                w.wait()
        return None
    return works[-1] if works else None


def gather(tensor, gather_list=None, dst=None, group=None, async_op=False,
           group_dst=None):
    pg = group or _default_pg
    if dst is None and group_dst is None:
        dst = 0
    elif dst is None:
        dst = group_dst

    # Implement gather via all_gather + discard
    import mindtorch_v2 as torch
    world_size = pg.size()
    rank = pg.rank()
    temp_list = [torch.zeros_like(tensor) for _ in range(world_size)]
    all_gather(temp_list, tensor, group=pg)
    if rank == dst and gather_list is not None:
        for i in range(world_size):
            gather_list[i].storage().copy_(temp_list[i].storage())
    if not async_op:
        return None
    # For async, return a completed Work since we already waited in all_gather
    w = Work()
    w._completed = True
    return w


def scatter(tensor, scatter_list=None, src=None, group=None, async_op=False,
            group_src=None):
    pg = group or _default_pg
    if src is None and group_src is None:
        src = 0
    elif src is None:
        src = group_src

    if scatter_list is not None and isinstance(scatter_list, (list, tuple)):
        # Concatenate scatter_list into a single buffer for HCCL
        import mindtorch_v2 as torch
        flat = torch.cat(scatter_list, dim=0)
        work = pg.scatter(tensor, flat, src=src)
    else:
        input_tensor = scatter_list or tensor
        work = pg.scatter(tensor, input_tensor, src=src)

    if not async_op:
        work.wait()
        return None
    return work


def reduce_scatter(output, input_list, op=ReduceOp.SUM, group=None,
                   async_op=False):
    pg = group or _default_pg
    if isinstance(input_list, (list, tuple)):
        import mindtorch_v2 as torch
        flat = torch.cat(input_list, dim=0)
    else:
        flat = input_list
    work = pg.reduce_scatter(output, flat, op)
    if not async_op:
        work.wait()
        return None
    return work


def reduce_scatter_tensor(output, input, op=ReduceOp.SUM, group=None,
                          async_op=False):
    pg = group or _default_pg
    work = pg.reduce_scatter(output, input, op)
    if not async_op:
        work.wait()
        return None
    return work


# Deprecated alias
def _reduce_scatter_base(output, input, op=ReduceOp.SUM, group=None,
                         async_op=False):
    return reduce_scatter_tensor(output, input, op, group, async_op)


def all_to_all(output_tensor_list, input_tensor_list, group=None,
               async_op=False):
    raise NotImplementedError("all_to_all is not supported by HCCL")


def all_to_all_single(output, input, output_split_sizes=None,
                      input_split_sizes=None, group=None, async_op=False):
    raise NotImplementedError("all_to_all_single is not supported by HCCL")


# ---------------------------------------------------------------------------
# Point-to-point operations
# ---------------------------------------------------------------------------

def send(tensor, dst=None, group=None, tag=0, group_dst=None):
    pg = group or _default_pg
    if dst is None and group_dst is None:
        raise ValueError("send requires dst or group_dst")
    if dst is None:
        dst = group_dst
    work = pg.send(tensor, dst)
    work.wait()


def recv(tensor, src=None, group=None, tag=0, group_src=None):
    pg = group or _default_pg
    if src is None and group_src is None:
        raise ValueError("HCCL recv requires explicit src rank")
    if src is None:
        src = group_src
    work = pg.recv(tensor, src)
    work.wait()
    return src


def isend(tensor, dst=None, group=None, tag=0, group_dst=None):
    pg = group or _default_pg
    if dst is None and group_dst is None:
        raise ValueError("isend requires dst or group_dst")
    if dst is None:
        dst = group_dst
    return pg.send(tensor, dst)


def irecv(tensor, src=None, group=None, tag=0, group_src=None):
    pg = group or _default_pg
    if src is None and group_src is None:
        raise ValueError("HCCL irecv requires explicit src rank")
    if src is None:
        src = group_src
    return pg.recv(tensor, src)


# ---------------------------------------------------------------------------
# Sub-groups
# ---------------------------------------------------------------------------

def new_group(ranks=None, timeout=None, backend=None, pg_options=None,
              use_local_synchronization=False, group_desc=None,
              device_id=None):
    global _group_count

    if _default_pg is None:
        raise RuntimeError("Default process group not initialized")

    timeout = _parse_timeout(timeout)
    timeout_sec = int(timeout.total_seconds())

    world_rank = _default_pg.rank()
    world_size = _default_pg.size()

    if ranks is None:
        ranks = list(range(world_size))

    if world_rank not in ranks:
        return GroupMember.NON_GROUP_MEMBER

    group_rank = ranks.index(world_rank)
    group_size = len(ranks)

    # Reuse the default store with a prefix to avoid key collisions
    _, default_store = _pg_map[_default_pg]
    prefix = f"pg{_group_count}"
    prefixed_store = PrefixStore(prefix, default_store)

    # Inherit backend from parent PG if not specified
    if backend is None:
        parent_backend = str(_pg_map[_default_pg][0])
        backend = parent_backend

    backend = str(backend).lower()

    if backend == "gloo":
        pg = ProcessGroupGloo(
            prefixed_store, group_rank, group_size,
            group_name=group_desc or prefix,
            group_ranks=ranks,
        )
    elif backend in ("hccl", "nccl"):
        dev_id = None
        if device_id is not None:
            dev_id = int(getattr(device_id, "index", device_id))
        else:
            dev_id = world_rank % 8
        pg = ProcessGroupHCCL(
            prefixed_store, group_rank, group_size,
            device_id=dev_id,
            group_name=group_desc or prefix,
            group_ranks=ranks,
        )
    else:
        raise ValueError(f"Unsupported backend: {backend}")

    _pg_map[pg] = (Backend(backend), prefixed_store)
    _pg_names[pg] = group_desc or prefix
    _pg_group_ranks[pg] = {ranks[i]: i for i in range(group_size)}
    _group_count += 1
    return pg


def new_subgroups(group_size=None, group=None, timeout=None, backend=None,
                  pg_options=None):
    if _default_pg is None:
        raise RuntimeError("Default process group not initialized")
    pg = group or _default_pg
    world_size = pg.size()
    if group_size is None:
        raise ValueError("group_size must be specified")
    if world_size % group_size != 0:
        raise ValueError(
            f"world_size {world_size} not divisible by group_size {group_size}"
        )
    num_groups = world_size // group_size
    subgroups = []
    for i in range(num_groups):
        ranks = list(range(i * group_size, (i + 1) * group_size))
        subgroups.append(new_group(ranks, timeout=timeout, backend=backend,
                                   pg_options=pg_options))
    rank = pg.rank()
    my_group_idx = rank // group_size
    return subgroups[my_group_idx], subgroups


def new_subgroups_by_enumeration(ranks_per_subgroup_list, timeout=None,
                                 backend=None, pg_options=None):
    if _default_pg is None:
        raise RuntimeError("Default process group not initialized")
    subgroups = []
    for ranks in ranks_per_subgroup_list:
        subgroups.append(new_group(ranks, timeout=timeout, backend=backend,
                                   pg_options=pg_options))
    rank = _default_pg.rank()
    my_group = GroupMember.NON_GROUP_MEMBER
    for i, ranks in enumerate(ranks_per_subgroup_list):
        if rank in ranks:
            my_group = subgroups[i]
            break
    return my_group, subgroups


def split_group(parent_pg, color, key=None):
    raise NotImplementedError("split_group is not yet supported")


def monitored_barrier(group=None, timeout=None, wait_all_ranks=False):
    # HCCL does not support monitored barrier. Fall back to regular barrier.
    barrier(group=group)


def supports_complex(op):
    return op in (ReduceOp.SUM, ReduceOp.AVG)


# ---------------------------------------------------------------------------
# Broadcast coalesced (used by DDP for initial param sync)
# ---------------------------------------------------------------------------

def _broadcast_coalesced(tensors, src=0, group=None):
    """Broadcast a list of tensors from src rank, one by one."""
    pg = group or _default_pg
    for t in tensors:
        pg.broadcast(t, root=src).wait()


# ---------------------------------------------------------------------------
# Exports
# ---------------------------------------------------------------------------

__all__ = [
    # Classes
    "Backend", "GroupMember", "P2POp",
    "ProcessGroup", "ReduceOp", "Work",
    "Store", "PrefixStore", "TCPStore",
    # Initialization
    "init_process_group", "destroy_process_group",
    "is_available", "is_initialized",
    "is_nccl_available", "is_gloo_available", "is_mpi_available",
    "is_ucc_available", "is_hccl_available", "is_backend_available",
    "is_torchelastic_launched",
    # Utility
    "get_rank", "get_world_size", "get_backend", "get_backend_config",
    "get_group_rank", "get_global_rank", "get_process_group_ranks",
    "get_node_local_rank", "get_pg_count",
    "get_default_backend_for_device",
    "group",
    # Collectives
    "all_reduce", "all_reduce_coalesced",
    "broadcast",
    "reduce",
    "all_gather", "all_gather_into_tensor", "all_gather_coalesced",
    "gather",
    "scatter",
    "reduce_scatter", "reduce_scatter_tensor",
    "all_to_all", "all_to_all_single",
    "barrier", "monitored_barrier",
    # P2P
    "send", "recv", "isend", "irecv",
    "batch_isend_irecv",
    # Object collectives
    "broadcast_object_list", "all_gather_object",
    "gather_object", "scatter_object_list",
    # Sub-groups
    "new_group", "new_subgroups", "new_subgroups_by_enumeration",
    "split_group",
    "init_device_mesh",
    # Misc
    "supports_complex",
    "default_pg_timeout", "get_debug_level",
    "DebugLevel",
    # Deprecated
    "reduce_op",
    "_all_gather_base", "_reduce_scatter_base",
    # Options (compatibility stubs)
    "AllreduceOptions", "AllreduceCoalescedOptions", "AllToAllOptions",
    "BarrierOptions", "BroadcastOptions", "GatherOptions",
    "ReduceOptions", "ReduceScatterOptions", "ScatterOptions",
    "RedOpType",
]
