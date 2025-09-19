# mypy: allow-untyped-decorators
import socket
import uuid
from contextlib import contextmanager
from datetime import timedelta
from enum import Enum
from functools import partial
from typing import Any, Callable, Dict, Generator, List, Optional, Tuple

import mindtorch
import mindtorch.distributed._functional_collectives as funcol
import mindtorch.distributed.distributed_c10d as c10d
from mindtorch._C._distributed_c10d import _SymmetricMemory, Work as _Work


_group_name_to_store: Dict[str, c10d.Store] = {}


def enable_symm_mem_for_group(group_name: str) -> None:
    """
    Enables symmetric memory for a process group.

    Args:
        group_name (str): the name of the process group.
    """
    if group_name in _group_name_to_store:
        return

    group = c10d._resolve_process_group(group_name)
    global_ranks = sorted(c10d._world.pg_group_ranks[group].keys())
    # Different subgroups with the same name should use different stores
    global_ranks_str = "_".join(map(str, global_ranks))
    store = c10d.PrefixStore(
        f"symmetric_memory-{global_ranks_str}",
        c10d._get_process_group_store(group),
    )
    # Use one store-based broadcast to bootstrap a file store from the process
    # and simultaneously verify that all ranks are on the same host.
    hostname = socket.gethostname()
    if group.rank() == 0:
        uid = str(uuid.uuid4())
        msg = f"{hostname}/{uid}"
        store.set("init", msg)
    else:
        msg = store.get("init").decode("utf-8")
        tokens = msg.split("/")
        assert len(tokens) == 2, tokens
        rank_0_hostname, uid = tokens
        if hostname != rank_0_hostname:
            raise RuntimeError(
                "init_symmetric_memory_for_process_group() failed for "
                f'group "{group_name}". Rank 0 and rank {group.rank()} '
                f"are on different hosts ({rank_0_hostname} and {hostname})"
            )
    store = mindtorch._C._distributed_c10d.FileStore(f"/tmp/{uid}", group.size())
    # TODO: check device connectiivity
    _group_name_to_store[group_name] = store
    _SymmetricMemory.set_group_info(
        group_name,
        group.rank(),
        group.size(),
        store,
    )


_is_test_mode: bool = False


@contextmanager
def _test_mode() -> Generator[None, None, None]:
    """
    Forces ``is_symm_mem_enabled_for_group()`` to return ``True`` and the ops
    defined in the ``symm_mem`` namespace to use fallback implementations.

    The context manager is not thread safe.
    """
    global _is_test_mode
    prev = _is_test_mode
    try:
        _is_test_mode = True
        yield
    finally:
        _is_test_mode = prev


def is_symm_mem_enabled_for_group(group_name: str) -> bool:
    """
    Check if symmetric memory is enabled for a process group.

    Args:
        group_name (str): the name of the process group.
    """
    return _is_test_mode or group_name in _group_name_to_store


_group_name_to_workspace_tensor: Dict[str, Optional[mindtorch.Tensor]] = {}


def get_symm_mem_workspace(group_name: str, min_size: int) -> _SymmetricMemory:
    """
    Get the symmetric memory workspace associated with the process group. If
    ``min_size`` is greater than the workspace associated with ``group_name``,
    the workspace will be re-allocated and re-rendezvous'd.

    Args:
        group_name (str): the name of the process group.
        min_size (int): the size requirement for the workspace in bytes.

    Returns:
        _SymmetricMemory: the symmetric memory workspace associated with the
        group.
    """
    enable_symm_mem_for_group(group_name)

    tensor = _group_name_to_workspace_tensor.get(group_name)
    size = tensor.numel() * tensor.element_size() if tensor is not None else 0
    if tensor is None or size < min_size:
        if mindtorch.cuda.is_current_stream_capturing():
            curr_size = 0 if tensor is None else tensor.numel() * tensor.element_size()
            raise RuntimeError(
                f"get_symm_mem_workspace(): the requested size ({min_size} bytes) "
                "is greater than the size of the currently allocated workspace "
                f"({curr_size} bytes). It's currently not possible to expand the "
                "workspace size during graph capture. Please invoke "
                f'`get_symm_mem_workspace(group_name="{group_name}", '
                f'min_size="{min_size}")` before initiating the graph capture '
                "and try again."
            )
        tensor = _SymmetricMemory.empty_strided_p2p(
            (max(size, min_size),),
            [1],
            mindtorch.uint8,
            mindtorch.device(f"cuda:{mindtorch.cuda.current_device()}"),
            group_name,
        )
        _group_name_to_workspace_tensor[group_name] = tensor
    return _SymmetricMemory.rendezvous(tensor)


_backend_streams: Dict[int, mindtorch.cuda.Stream] = {}


def _get_backend_stream(priority: int = 0) -> mindtorch.cuda.Stream:
    if priority not in _backend_streams:
        _backend_streams[priority] = mindtorch.cuda.Stream(priority=priority)
    return _backend_streams[priority]


def _pipelined_multi_all_gather_and_consume(
    shard: List[mindtorch.Tensor],
    shard_consumer: Callable[[List[mindtorch.Tensor], int], None],
    ag_out: List[mindtorch.Tensor],
    group_name: str,
) -> None:
    """
    Perform the following logic with micro-pipelined computation and
    communication:

        gathered = [
            all_gather_tensor(x, gather_dim=0, group=group)
            for x in shard
        ]

        shards = [[] for _ in range(group_size)]
        for x in ag_out:
            for i, y in enumerate(x.chunk(group_size)):
                shards[i].append(y)

        for src_rank, shard in enumerate(shards):
            shard_consumer(shard, src_rank)
    """
    p2p_workspace_size_req = 0
    for x in shard:
        p2p_workspace_size_req += x.numel() * x.element_size()
    symm_mem = get_symm_mem_workspace(group_name, min_size=p2p_workspace_size_req)
    group_size = symm_mem.world_size
    rank = symm_mem.rank

    symm_mem.barrier(channel=0)
    backend_stream = _get_backend_stream()
    backend_stream.wait_stream(mindtorch.cuda.current_stream())

    for x, y in zip(shard, ag_out):
        assert x.is_contiguous(), (
            "_pipelined_all_gather_and_consume: all tensors "
            "in `shard` must be contiguous"
        )
        assert y.is_contiguous(), (
            "_pipelined_all_gather_and_consume: all tensors "
            "in `ag_out` must be contiguous"
        )
        assert x.shape[0] * group_size == y.shape[0]
        assert x.shape[1:] == y.shape[1:]

    def copy_shard(dst: List[mindtorch.Tensor], src: List[mindtorch.Tensor]) -> None:
        for d, s in zip(dst, src):
            d.copy_(s)

    def get_p2p_bufs(remote_rank: int) -> List[mindtorch.Tensor]:
        offset_bytes = 0
        bufs = []
        for x in shard:
            buf = symm_mem.get_buffer(
                remote_rank,
                x.shape,
                x.dtype,
                storage_offset=offset_bytes // x.element_size(),
            )
            bufs.append(buf)
            offset_bytes += buf.numel() * buf.element_size()
        return bufs

    local_p2p_bufs = get_p2p_bufs(rank)

    # shards[i] => shard from rank i
    shards: List[List[mindtorch.Tensor]] = [[] for _ in range(group_size)]
    for x in ag_out:
        for i, y in enumerate(x.chunk(group_size)):
            shards[i].append(y)

    # Parallelization strategy: after each rank copies its shard into its local
    # p2p buffer, every rank issues independent p2p copy -> shard_consumer
    # sequences to two streams. In addition to computation/communication
    # overlapping, the strategy allows for computation/computation overlapping,
    # greatly reducing quantization inefficiency.
    #
    # Notation:
    # - "mv" for the copy to local buffer
    # - "cp" for p2p copies
    # - "b" for barriers
    #
    # Constraints:
    # - The GPU scheduler may or may not overlap "mv" with the first shard_consumer.
    # - "cp" from different streams cannot overlap.
    #
    # Ideal scenario 0 - "mv" overlaps with the first shard_consumer:
    #
    # stream 0: [ shard_consumer ][ cp ][ shard_consumer ]
    # stream 1: [ mv ][b][ cp ][ shard_consumer ]
    #
    # Ideal scenario 1 - "mv" is scheduled before the first shard_consumer:
    #
    # stream 0:       [ shard_consumer ][ cp ][ shard_consumer ]
    # stream 1: [ mv ][b][ cp ][ shard_consumer ]
    #
    # Suboptimal scenario 0 - "mv" is scheduled after the first shard_consumer:
    #
    # stream 0: [ shard_consumer ]               [ cp ][ shard_consumer ]
    # stream 1:                   [ mv ][b][ cp ][ shard_consumer ]
    #
    # Suboptimal scenario 0 - "b" is scheduled after the first shard_consumer:
    #
    # stream 0:       [ shard_consumer ]         [ cp ][ shard_consumer ]
    # stream 1: [ mv ]                  [b][ cp ][ shard_consumer ]
    #
    # We haven't yet figured out a way to ensure "mv" and "b" are either
    # overlapped with or scheduled before the first shard_consumer. Thus, to
    # prevent suboptimal scenarios, we are giving up the chance to overlap "mv"
    # and "b" with the first shard_consumer for now.
    copy_shard(dst=local_p2p_bufs, src=shard)
    symm_mem.barrier(channel=1)
    backend_stream.wait_stream(mindtorch.cuda.current_stream())

    # At this point, all ranks have copied their local shard to
    # their local p2p buffer. Each rank can now copy and consume
    # remote shards.
    shard_consumer(shard, rank)

    for step in range(1, group_size):
        if step % 2 == 0:
            stream = mindtorch.cuda.current_stream()
        else:
            stream = backend_stream
        remote_rank = (step + rank) % group_size
        remote_p2p_bufs = get_p2p_bufs(remote_rank)
        with mindtorch.cuda.stream(stream):
            copy_shard(dst=shards[remote_rank], src=remote_p2p_bufs)
            shard_consumer(shards[remote_rank], remote_rank)

    # Copy from input to the all-gather output. Opportunistically overlap it
    # with the last shard_consumer.
    if group_size % 2 == 0:
        stream = mindtorch.cuda.current_stream()
    else:
        stream = backend_stream
    with mindtorch.cuda.stream(stream):
        copy_shard(dst=shards[rank], src=shard)

    mindtorch.cuda.current_stream().wait_stream(backend_stream)
    symm_mem.barrier(channel=0)


def _pipelined_all_gather_and_consume(
    shard: mindtorch.Tensor,
    shard_consumer: Callable[[mindtorch.Tensor, int], None],
    ag_out: mindtorch.Tensor,
    group_name: str,
) -> None:
    """
    Perform the following logic with micro-pipelined computation and
    communication:

        ag_out = all_gather_tensor(shard, gather_dim=0, group=group)
        shards = ag_out.chunk(group.size())
        for src_rank, shard in enumerate(shards):
            shard_consumer(shard, src_rank)
    """

    def adapter(shard: List[mindtorch.Tensor], rank: int) -> None:
        shard_consumer(shard[0], rank)

    _pipelined_multi_all_gather_and_consume(
        [shard],
        adapter,
        [ag_out],
        group_name,
    )


def _pipelined_produce_and_all2all(
    chunk_producer: Callable[[int, mindtorch.Tensor], None],
    output: mindtorch.Tensor,
    group_name: str,
) -> None:
    """
    Perform the following logic with micro-pipelined computation and
    communication:

        chunks = [
            chunk_producer(dst_rank, chunks[dst_rank])
            for dst_rank in range(group_size):
        ]
        dist.all_to_all_single(output=output, input=mindtorch.cat(chunks))
    """
    out_chunks = output.chunk(c10d._get_group_size_by_name(group_name))
    p2p_workspace_size_req = out_chunks[0].numel() * out_chunks[0].element_size() * 2
    symm_mem = get_symm_mem_workspace(group_name, min_size=p2p_workspace_size_req)
    group_size = symm_mem.world_size
    rank = symm_mem.rank

    symm_mem.barrier(channel=0)
    backend_stream = _get_backend_stream()
    backend_stream.wait_stream(mindtorch.cuda.current_stream())

    def get_p2p_buf(rank: int, idx: int) -> mindtorch.Tensor:
        assert idx in (0, 1)
        offset = 0 if idx == 0 else out_chunks[0].numel()
        return symm_mem.get_buffer(
            rank, out_chunks[0].shape, out_chunks[0].dtype, offset
        )

    # Prepare two local p2p buffers, so that a remote rank can pull the result
    # of step [i] in one p2p buffer while the local rank can compute the
    # result of step [i+1] and write it directly the other p2p buffer.
    local_p2p_buf_0 = get_p2p_buf(rank, 0)
    local_p2p_buf_1 = get_p2p_buf(rank, 1)

    for step in range(1, group_size):
        remote_rank = (rank - step) % group_size
        if step % 2 == 0:
            stream = mindtorch.cuda.current_stream()
            p2p_buf = local_p2p_buf_1
            remote_p2p_buf = get_p2p_buf(remote_rank, 1)
        else:
            stream = backend_stream
            p2p_buf = local_p2p_buf_0
            remote_p2p_buf = get_p2p_buf(remote_rank, 0)
        with mindtorch.cuda.stream(stream):
            # Parallelization strategy: every rank issues independent compute
            # -> barrier -> p2p copy sequences on two streams. In addition to
            # computation/communication overlapping, the strategy allows for
            # computation/computation overlapping, greatly reducing
            # quantization inefficiency.
            #
            # Ideally, stream activities would look like this ("b" for
            # barriers, "cp" for p2p copies):
            #
            # [rank 0]
            # stream 0:         [  chunk_producer  ][b][ cp ][  chunk_producer ][b][ cp ]
            # stream 1: [  chunk_producer  ][b][ cp ][  chunk_producer  ][b][ cp ]
            #
            # [rank 1]
            # stream 0:         [  chunk_producer  ][b][ cp ][  chunk_producer ][b][ cp ]
            # stream 1: [  chunk_producer  ][b][ cp ][  chunk_producer  ][b][ cp ]
            #
            # Note that the barriers synchronize streams with the same ID
            # across ranks. They don't synchronize streams on the same rank.
            #
            # Since the work on both streams is independent, there's no
            # guarantee that the chunk_producer from stream 0 or stream 1 will
            # be scheduled first. If there is a scheduling mismatch across
            # ranks, the barrier forces all ranks to wait for the slowest.
            #
            # When scheduling mismatches occur among ranks, the stream
            # activities might look like this (note that p2p copies from
            # different streams cannot overlap with each other):
            #
            # [rank 0]
            # stream 0: [  chunk_producer  ][b        ][ cp ][  chunk_producer ][b       ][ cp ]
            # stream 1:         [  chunk_producer  ][b]      [ cp ][  chunk_producer  ][b]      [ cp ]
            #
            # [rank 1]
            # stream 0:         [  chunk_producer  ][b]      [ cp ][  chunk_producer  ][b]      [ cp ]
            # stream 1: [  chunk_producer  ][b        ][ cp ][  chunk_producer  ][b      ][ cp ]
            #
            # To prevent this, we need to ensure that the chunk_producer on
            # stream 1 gets scheduled first on every rank. Without access to
            # the underlying kernels, CUDA offers no API to control the
            # scheduling order of two independent, overlapping kernels. Our
            # solution is to issue a small sleep kernel in stream 0. The sleep
            # duration is insignificant, but having an extra task in stream 0
            # will almost guarantee that the chunk_producer on stream 1 gets
            # scheduled first. Once the first chunk_producer is scheduled in
            # the correct order, there's very little room for the scheduling
            # order of subsequent kernels to be inconsistent across ranks.
            if step == 2:
                mindtorch.cuda._sleep(100)
            chunk_producer((rank + step) % group_size, p2p_buf)
            symm_mem.barrier(channel=step % 2)
            out_chunks[remote_rank].copy_(remote_p2p_buf)
            # The local P2P buffer can only be overwritten by the next
            # chunk_producer after all peers have finished reading from it.
            symm_mem.barrier(channel=step % 2)

    chunk_producer(rank, out_chunks[rank])
    mindtorch.cuda.current_stream().wait_stream(backend_stream)
    symm_mem.barrier(channel=0)


lib = mindtorch.library.Library("symm_mem", "DEF")  # noqa: TOR901
lib.define(
    "fused_all_gather_matmul(Tensor A, Tensor[] Bs, int gather_dim, str group_name) -> (Tensor, Tensor[])"
)
lib.define(
    "fused_all_gather_scaled_matmul("
    "Tensor A, Tensor[] Bs, Tensor A_scale, Tensor[] B_scales, "
    "int gather_dim, str group_name, "
    "Tensor?[] biases, "
    "Tensor?[] result_scales, "
    "ScalarType?[] out_dtypes, "
    "bool[] use_fast_accum) -> (Tensor, Tensor[])"
)
lib.define(
    "fused_matmul_reduce_scatter(Tensor A, Tensor B, str reduce_op, int scatter_dim, str group_name) -> Tensor"
)
lib.define(
    "fused_scaled_matmul_reduce_scatter("
    "Tensor A, Tensor B, Tensor A_scale, Tensor B_scale, "
    "str reduce_op, int scatter_dim, str group_name, "
    "Tensor? bias = None, "
    "Tensor? result_scale = None, "
    "ScalarType? out_dtype = None, "
    "bool use_fast_accum = False) -> Tensor"
)
lib.define("_low_contention_all_gather(Tensor tensor, str group_name) -> Tensor")
lib.define(
    "_low_contention_reduce_scatter(Tensor tensor, str reduce_op, str group_name) -> Tensor"
)


class _ScaleMode(Enum):
    UNSCALED = "unscaled"
    TENSOR_WISE = "tensor-wise"
    ROW_WISE_SHARDED = "row-wise-sharded"
    ROW_WISE_REPLICATED = "row-wise-replicated"


def _check_and_verify_fp8_all_gather_scale_mode(
    shard: mindtorch.Tensor, scale: Optional[mindtorch.Tensor], gather_dim: int, group_size: int
) -> _ScaleMode:
    full_shape = list(shard.shape)
    full_shape[gather_dim] *= group_size

    if scale is None:
        return _ScaleMode.UNSCALED
    elif scale.shape[:-1] == shard.shape[:-1] and scale.shape[-1] == 1:
        # Row-wise scaling
        #
        # NOTE: when the last dim of both A_shard and A_scale is one, we can't
        # tell if A_scale is replicated tensor-wise scale or sharded row-wise
        # scale. Treating it as row-wise scaling for safety.
        return _ScaleMode.ROW_WISE_SHARDED
    elif scale.numel() == 1:
        return _ScaleMode.TENSOR_WISE
    elif list(scale.shape[:-1]) == full_shape[:-1]:
        return _ScaleMode.ROW_WISE_REPLICATED
    else:
        raise ValueError(
            "Invalid scale shape for fp8 all-gather "
            f"(shard shape: {shard.shape}, scale shape: {scale.shape})"
        )


def _fused_all_gather_matmul_impl(
    mm_out_op: mindtorch._ops.OpOverload,
    A_shard: mindtorch.Tensor,
    Bs: List[mindtorch.Tensor],
    A_scale: Optional[mindtorch.Tensor],
    kwargs_list: List[Dict[str, Any]],
    out_dtypes: List[Optional[mindtorch.dtype]],
    gather_dim: int,
    group_name: str,
) -> Tuple[mindtorch.Tensor, List[mindtorch.Tensor]]:
    if A_shard.dim() < 2:
        raise ValueError("A_shard must be a matrix")
    for B in Bs:
        if B.dim() != 2:
            raise ValueError("B must be a matrix")
    if len(out_dtypes) != len(Bs):
        raise ValueError("len(out_types) must be the same as len(Bs)")
    if len(kwargs_list) != len(Bs):
        raise ValueError("len(kwargs_list) must be the same as len(Bs)")
    if gather_dim < 0 or gather_dim >= A_shard.dim():
        raise ValueError("Invalid gather_dim")

    group = c10d._resolve_process_group(group_name)

    # Move the gather_dim to the front and flatten the tensor into a 2D matrix.
    # The flattened tensor doesn't need to be contiguous (for computation
    # efficiency), as _pipelined_all_gather_and_consume guarantees that shards
    # passed to shard_consumer are contiguous.
    A_shard_flat = A_shard.movedim(gather_dim, 0)
    leading_dims = [group.size()] + list(A_shard_flat.shape[:-1])
    A_shard_flat = A_shard_flat.flatten(0, -2)

    # Helper function for reverting the above transformation
    def unflatten(t: mindtorch.Tensor) -> mindtorch.Tensor:
        return t.view(*leading_dims, -1).flatten(0, 1).movedim(0, gather_dim)

    A_flat = A_shard_flat.new_empty(
        A_shard_flat.shape[0] * group.size(),
        A_shard_flat.shape[1],
    )

    outputs = [
        A_flat.new_empty(A_flat.shape[0], B.shape[1], dtype=out_dtype or B.dtype)
        for B, out_dtype in zip(Bs, out_dtypes)
    ]
    output_shards = [output.chunk(group.size()) for output in outputs]

    scale_mode = _check_and_verify_fp8_all_gather_scale_mode(
        shard=A_shard, scale=A_scale, gather_dim=gather_dim, group_size=group.size()
    )

    # Computing block-wise matmul along the first dim of A
    if scale_mode == _ScaleMode.ROW_WISE_SHARDED:
        assert A_scale is not None
        A_scale_shard = A_scale.movedim(gather_dim, 0).flatten(0, -2)
        A_scale_flat = A_scale_shard.new_empty(
            A_scale_shard.shape[0] * group.size(),
            A_scale_shard.shape[1],
        )

        def row_wise_sharded_consumer(shard: List[mindtorch.Tensor], rank: int) -> None:
            for idx, (B, kwargs) in enumerate(zip(Bs, kwargs_list)):
                mm_out_op(
                    shard[0],
                    B,
                    scale_a=shard[1],
                    **kwargs,
                    out=output_shards[idx][rank],
                )

        _pipelined_multi_all_gather_and_consume(
            [A_shard_flat, A_scale_shard],
            row_wise_sharded_consumer,
            [A_flat, A_scale_flat],
            group_name,
        )
    elif scale_mode == _ScaleMode.ROW_WISE_REPLICATED:
        assert A_scale is not None
        A_scale_shards = (
            A_scale.movedim(gather_dim, 0).flatten(0, -2).chunk(group.size())
        )

        def row_wise_replicated_consumer(shard: mindtorch.Tensor, rank: int) -> None:
            for idx, (B, kwargs) in enumerate(zip(Bs, kwargs_list)):
                mm_out_op(
                    shard,
                    B,
                    scale_a=A_scale_shards[rank],
                    **kwargs,
                    out=output_shards[idx][rank],
                )

        _pipelined_all_gather_and_consume(
            A_shard_flat,
            row_wise_replicated_consumer,
            A_flat,
            group_name,
        )
    else:
        if scale_mode == _ScaleMode.TENSOR_WISE:
            assert A_scale is not None
            for kwargs in kwargs_list:
                kwargs["scale_a"] = A_scale
        else:
            assert scale_mode == _ScaleMode.UNSCALED

        def default_consumer(shard: mindtorch.Tensor, rank: int) -> None:
            for idx, (B, kwargs) in enumerate(zip(Bs, kwargs_list)):
                mm_out_op(shard, B, **kwargs, out=output_shards[idx][rank])

        _pipelined_all_gather_and_consume(
            A_shard_flat,
            default_consumer,
            A_flat,
            group_name,
        )

    return unflatten(A_flat), [unflatten(output) for output in outputs]


@mindtorch.library.impl(lib, "fused_all_gather_matmul", "Meta")
def _fused_all_gather_matmul_fallback(
    A_shard: mindtorch.Tensor,
    Bs: List[mindtorch.Tensor],
    gather_dim: int,
    group_name: str,
) -> Tuple[mindtorch.Tensor, List[mindtorch.Tensor]]:
    group_size = c10d._get_group_size_by_name(group_name)
    A = mindtorch.ops._c10d_functional.all_gather_into_tensor(
        A_shard.contiguous(), group_size, group_name
    )
    A = mindtorch.ops._c10d_functional.wait_tensor(A)
    A = A.view(group_size, *A_shard.shape).movedim(gather_dim + 1, 1).flatten(0, 1)
    return A.movedim(0, gather_dim), [
        mindtorch.matmul(A, B).movedim(0, gather_dim) for B in Bs
    ]


@mindtorch.library.impl(lib, "fused_all_gather_matmul", "CUDA")
def _fused_all_gather_matmul(
    A_shard: mindtorch.Tensor,
    Bs: List[mindtorch.Tensor],
    gather_dim: int,
    group_name: str,
) -> Tuple[mindtorch.Tensor, List[mindtorch.Tensor]]:
    """
    Perform the following logic with micro-pipelined computation and
    communication:

        all_gather_tensor(A_shard, gather_dim, group_name) @ B

    Optimal stride order for A_shard - if A_shard.movedim(gather_dim, 0) is
    contiguous, no extra copy is required for input layout transformation.
    Otherwise A_shard needs to be copied once.
    """
    if _is_test_mode:
        return _fused_all_gather_matmul_fallback(A_shard, Bs, gather_dim, group_name)

    with mindtorch.profiler.record_function("fused_all_gather_matmul"):
        return _fused_all_gather_matmul_impl(
            mindtorch.ops.aten.mm.out,
            A_shard,
            Bs,
            None,
            [{} for B in Bs],
            [B.dtype for B in Bs],
            gather_dim,
            group_name,
        )


def _fused_all_gather_matmul_native(
    A_shard: mindtorch.Tensor,
    B: mindtorch.Tensor,
    group_name: str,
) -> Tuple[mindtorch.Tensor, mindtorch.Tensor]:
    symm_mem = _SymmetricMemory.rendezvous(A_shard)
    if symm_mem is None:
        symm_mem = get_symm_mem_workspace(
            group_name, A_shard.numel() * A_shard.element_size()
        )
        symm_mem.barrier()
        buf = symm_mem.get_buffer(symm_mem.rank, A_shard.shape, A_shard.dtype)
        buf.copy_(A_shard)
        A_shard = buf

    rank = symm_mem.rank
    world_size = symm_mem.world_size

    current_stream = mindtorch.cuda.current_stream()
    backend_stream = _get_backend_stream(priority=-1)

    symm_mem.barrier()
    current_stream.wait_stream(backend_stream)
    backend_stream.wait_stream(current_stream)

    A = A_shard.new_empty(A_shard.shape[0] * world_size, A_shard.shape[1])
    A_signals = mindtorch.zeros(world_size, dtype=mindtorch.uint32, device=A_shard.device)
    A_shards = A.chunk(world_size)

    A_shards[rank].copy_(A_shard)
    _SymmetricMemory.stream_write_value32(A_signals, rank, 1)

    out = mindtorch.ops.symm_mem._async_input_mm(A, B, A_signals, rank)
    for step in range(1, world_size):
        src_rank = (rank + step) % world_size
        src_buf = symm_mem.get_buffer(src_rank, A_shard.shape, A_shard.dtype)
        with mindtorch.cuda.stream(backend_stream):
            A_shards[src_rank].copy_(src_buf)
            # cuStreamWriteValue32 issues a system level fence before the write
            _SymmetricMemory.stream_write_value32(A_signals, src_rank, 1)

    current_stream.wait_stream(backend_stream)
    backend_stream.wait_stream(current_stream)

    symm_mem.barrier()
    return A, out


@mindtorch.library.impl(lib, "fused_all_gather_scaled_matmul", "Meta")
def _fused_all_gather_scaled_matmul_fallback(
    A_shard: mindtorch.Tensor,
    Bs: List[mindtorch.Tensor],
    A_scale: mindtorch.Tensor,
    B_scales: List[mindtorch.Tensor],
    gather_dim: int,
    group_name: str,
    biases: List[Optional[mindtorch.Tensor]],
    result_scales: List[Optional[mindtorch.Tensor]],
    out_dtypes: List[Optional[mindtorch.dtype]],
    use_fast_accum: List[bool],
) -> Tuple[mindtorch.Tensor, List[mindtorch.Tensor]]:
    out_dtypes = _maybe_convert_scalar_types_to_dtypes(out_dtypes)

    group_size = c10d._get_group_size_by_name(group_name)
    A = mindtorch.ops._c10d_functional.all_gather_into_tensor(
        A_shard.contiguous(), group_size, group_name
    )
    A = mindtorch.ops._c10d_functional.wait_tensor(A)
    A = A.view(group_size, *A_shard.shape).movedim(gather_dim + 1, 1).flatten(0, 1)

    scale_mode = _check_and_verify_fp8_all_gather_scale_mode(
        shard=A_shard, scale=A_scale, gather_dim=gather_dim, group_size=group_size
    )
    if scale_mode == _ScaleMode.ROW_WISE_SHARDED:
        A_scale_shard = A_scale
        A_scale = mindtorch.ops._c10d_functional.all_gather_into_tensor(
            A_scale.contiguous(), group_size, group_name
        )
        A_scale = mindtorch.ops._c10d_functional.wait_tensor(A_scale)
        A_scale = (
            A_scale.view(group_size, *A_scale_shard.shape)
            .movedim(gather_dim + 1, 1)
            .flatten(0, -2)
        )
    elif scale_mode == _ScaleMode.ROW_WISE_REPLICATED:
        A_scale = A_scale.movedim(gather_dim, 0).flatten(0, -2)
    else:
        assert scale_mode == _ScaleMode.TENSOR_WISE

    def scaled_matmul(
        A: mindtorch.Tensor,
        B: mindtorch.Tensor,
        A_scale: mindtorch.Tensor,
        B_scale: mindtorch.Tensor,
        bias: Optional[mindtorch.Tensor],
        result_scale: Optional[mindtorch.Tensor],
        out_dtype: Optional[mindtorch.dtype],
        use_fast_accum: bool,
    ) -> mindtorch.Tensor:
        leading_dims = A.shape[:-1]
        res = mindtorch.ops.aten._scaled_mm(
            A.flatten(0, -2),
            B,
            A_scale,
            B_scale,
            bias,
            result_scale,
            out_dtype=out_dtype,
            use_fast_accum=use_fast_accum,
        )
        return res.unflatten(0, leading_dims)

    return A.movedim(0, gather_dim), [
        scaled_matmul(
            A, B, A_scale, B_scale, bias, result_scale, out_dtype, fast_accum
        ).movedim(0, gather_dim)
        for B, B_scale, bias, result_scale, out_dtype, fast_accum in zip(
            Bs, B_scales, biases, result_scales, out_dtypes, use_fast_accum
        )
    ]


@mindtorch.library.impl(lib, "fused_all_gather_scaled_matmul", "CUDA")
def _fused_all_gather_scaled_matmul(
    A_shard: mindtorch.Tensor,
    Bs: List[mindtorch.Tensor],
    A_scale: mindtorch.Tensor,
    B_scales: List[mindtorch.Tensor],
    gather_dim: int,
    group_name: str,
    biases: List[Optional[mindtorch.Tensor]],
    result_scales: List[Optional[mindtorch.Tensor]],
    out_dtypes: List[Optional[mindtorch.dtype]],
    use_fast_accum: List[bool],
) -> Tuple[mindtorch.Tensor, List[mindtorch.Tensor]]:
    """
    Perform the following logic with micro-pipelined computation and
    communication:

        A = all_gather_tensor(A_shard, gather_dim, group_name)
        leading_dims = A.shape[:-1]
        res = mindtorch.ops.aten._scaled_mm(A.flatten(0, -2), B, A_scale, B_scale)
        res = res.unflatten(0, leading_dims)

    The input `A_scale` can be tensor-wise, row-wise-sharded or
    row-wise-replicated.

    Optimal stride order for `A_shard` - if `A_shard.movedim(gather_dim, 0)` is
    contiguous, no extra copy is required for input layout transformation.
    Otherwise A_shard needs to be copied once.
    """
    out_dtypes = _maybe_convert_scalar_types_to_dtypes(out_dtypes)

    if len(biases) != len(Bs):
        raise ValueError("len(biases) must be the same as len(Bs)")
    if len(result_scales) != len(Bs):
        raise ValueError("len(result_scales) must be the same as len(Bs)")
    if len(out_dtypes) != len(Bs):
        raise ValueError("len(out_dtypes) must be the same as len(Bs)")
    if len(use_fast_accum) != len(Bs):
        raise ValueError("len(use_gast_accum_list) must be the same as len(Bs)")

    if _is_test_mode:
        return _fused_all_gather_scaled_matmul_fallback(
            A_shard,
            Bs,
            A_scale,
            B_scales,
            gather_dim,
            group_name,
            biases,
            result_scales,
            out_dtypes,
            use_fast_accum,
        )

    with mindtorch.profiler.record_function("fused_all_gather_scaled_matmul"):
        return _fused_all_gather_matmul_impl(
            mindtorch.ops.aten._scaled_mm.out,
            A_shard,
            Bs,
            A_scale,
            [
                {
                    "scale_b": B_scale,
                    "bias": bias,
                    "scale_result": result_scale,
                    "out_dtype": out_dtype,
                    "use_fast_accum": fast_accum,
                }
                for B_scale, bias, result_scale, out_dtype, fast_accum in zip(
                    B_scales, biases, result_scales, out_dtypes, use_fast_accum
                )
            ],
            out_dtypes,
            gather_dim,
            group_name,
        )


def make_contiguous_for_perm(
    t: mindtorch.Tensor,
    perm: List[int],
) -> mindtorch.Tensor:
    """
    Restride `t` such that `t.permute(perm)` is contiguous.
    """
    inv_perm = [0] * len(perm)
    for i, p in enumerate(perm):
        inv_perm[p] = i
    return t.permute(perm).contiguous().permute(inv_perm)


def restride_A_shard_for_fused_all_gather_matmul(
    t: mindtorch.Tensor,
    gather_dim: int,
) -> mindtorch.Tensor:
    """
    Restride the `A_shard` arg of `fused_all_gather_matmul` for optimal perf.
    See the doc for `fused_all_gather_matmul` for detail.
    """
    perm = list(range(len(t.shape)))
    perm.insert(0, perm.pop(gather_dim))
    return make_contiguous_for_perm(t, perm)


def _fused_matmul_reduce_scatter_impl(
    mm_out_op: mindtorch._ops.OpOverload,
    A: mindtorch.Tensor,
    B: mindtorch.Tensor,
    A_scale: Optional[mindtorch.Tensor],
    kwargs: Dict[str, Any],
    out_dtype: Optional[mindtorch.dtype],
    reduce_op: str,
    scatter_dim: int,
    group_name: str,
) -> mindtorch.Tensor:
    if A.dim() < 2:
        raise ValueError("A_shard must be a matrix")
    if scatter_dim < 0 or scatter_dim >= A.dim():
        raise ValueError("Invalid gather_dim")
    if B.dim() != 2:
        raise ValueError("B must be a matrix")
    if reduce_op == "sum":
        reduce_fn = partial(mindtorch.sum, dim=0)
    elif reduce_op == "avg":
        reduce_fn = partial(mindtorch.mean, dim=0)
    else:
        raise ValueError("reduce_op must be sum or avg")

    group = c10d._resolve_process_group(group_name)
    out_shape = [*A.shape[:-1], B.shape[1]]
    out_shape[scatter_dim] //= group.size()

    # Move the scatter_dim to the front and flatten the tensor into a 2D matrix
    x = A.movedim(scatter_dim, 0)
    leading_dims = [group.size()] + list(x.shape[:-1])
    leading_dims[1] //= group.size()
    x = x.flatten(0, -2)
    A_shards = x.chunk(group.size())

    A_scale_shards = None
    if A_scale is None:
        pass
    elif A_scale.numel() == 1:
        A_scale_shards = [A_scale] * group.size()
    else:
        if A_scale.shape[:-1] != A.shape[:-1]:
            raise ValueError(
                "For row-wise scaling, the leading dims of A_scale "
                "must match the leading dims of A "
                f"(A shape: {A.shape}, A_scale shape: {A_scale.shape})"
            )
        A_scale = A_scale.movedim(scatter_dim, 0).contiguous().flatten(0, -2)
        A_scale_shards = list(A_scale.chunk(group.size()))

    # Computing block-wise matmul along the first dim of A
    def chunk_producer(rank: int, out: mindtorch.Tensor) -> None:
        if A_scale_shards is not None:
            mm_out_op(
                A_shards[rank], B, scale_a=A_scale_shards[rank], **kwargs, out=out
            )
        else:
            mm_out_op(A_shards[rank], B, **kwargs, out=out)

    stacked_partials = x.new_empty(x.shape[0], B.shape[1], dtype=out_dtype or A.dtype)

    _pipelined_produce_and_all2all(
        chunk_producer,
        stacked_partials,
        group_name,
    )
    # Ensures that the transpose and reduction produce contiguous result
    # in a single reduction kernel.
    return reduce_fn(
        stacked_partials.view(*leading_dims, -1)
        .movedim(1, scatter_dim + 1)
        .movedim(0, scatter_dim),
        dim=scatter_dim,
    )


@mindtorch.library.impl(lib, "fused_matmul_reduce_scatter", "Meta")
def _fused_matmul_reduce_scatter_fallback(
    A: mindtorch.Tensor,
    B: mindtorch.Tensor,
    reduce_op: str,
    scatter_dim: int,
    group_name: str,
) -> mindtorch.Tensor:
    res = funcol.reduce_scatter_tensor(A @ B, reduce_op, scatter_dim, group_name)
    res = funcol.wait_tensor(res)
    return res


@mindtorch.library.impl(lib, "fused_matmul_reduce_scatter", "CUDA")
def _fused_matmul_reduce_scatter(
    A: mindtorch.Tensor,
    B: mindtorch.Tensor,
    reduce_op: str,
    scatter_dim: int,
    group_name: str,
) -> mindtorch.Tensor:
    """
    Perform the following logic with micro-pipelined computation and
    communication:

        reduce_scatter_tensor(A @ B, reduce_op, scatter_dim, group_name)

    Optimal stride order for A - if A.movedim(scatter_dim, 0) is contiguous, no
    extra copy is required for input layout transformation. Otherwise A needs
    to be copied once.
    """
    if _is_test_mode:
        return _fused_matmul_reduce_scatter_fallback(
            A, B, reduce_op, scatter_dim, group_name
        )

    with mindtorch.profiler.record_function("fused_matmul_reduce_scatter"):
        return _fused_matmul_reduce_scatter_impl(
            mm_out_op=mindtorch.ops.aten.mm.out,
            A=A,
            B=B,
            A_scale=None,
            kwargs={},
            out_dtype=A.dtype,
            reduce_op=reduce_op,
            scatter_dim=scatter_dim,
            group_name=group_name,
        )


@mindtorch.library.impl(lib, "fused_scaled_matmul_reduce_scatter", "Meta")
def _fused_scaled_matmul_reduce_scatter_fallback(
    A: mindtorch.Tensor,
    B: mindtorch.Tensor,
    A_scale: mindtorch.Tensor,
    B_scale: mindtorch.Tensor,
    reduce_op: str,
    scatter_dim: int,
    group_name: str,
    bias: Optional[mindtorch.Tensor] = None,
    result_scale: Optional[mindtorch.Tensor] = None,
    out_dtype: Optional[mindtorch.dtype] = None,
    use_fast_accum: bool = False,
) -> mindtorch.Tensor:
    if A_scale.numel() > 1:
        if A_scale.shape[:-1] != A.shape[:-1]:
            raise ValueError(
                "For row-wise scaling, the leading dims of A_scale "
                "must match the leading dims of A "
                f"(A shape: {A.shape}, A_scale shape: {A_scale.shape})"
            )
        A_scale = A_scale.flatten(0, -2).contiguous()
    elif A_scale.numel() != 1:
        raise ValueError(
            "Invalid A_scale shape "
            f"(A shape: {A.shape}, A_scale shape: {A_scale.shape})"
        )

    C = mindtorch._scaled_mm(
        A.flatten(0, -2).contiguous(),
        B,
        A_scale,
        B_scale,
        bias,
        result_scale,
        out_dtype,
        use_fast_accum,
    )
    C = C.view(*A.shape[:-1], B.shape[1])
    res = funcol.reduce_scatter_tensor(
        C,
        reduce_op,
        scatter_dim,
        group_name,
    )
    res = funcol.wait_tensor(res)
    return res


@mindtorch.library.impl(lib, "fused_scaled_matmul_reduce_scatter", "CUDA")
def _fused_scaled_matmul_reduce_scatter(
    A: mindtorch.Tensor,
    B: mindtorch.Tensor,
    A_scale: mindtorch.Tensor,
    B_scale: mindtorch.Tensor,
    reduce_op: str,
    scatter_dim: int,
    group_name: str,
    bias: Optional[mindtorch.Tensor] = None,
    result_scale: Optional[mindtorch.Tensor] = None,
    out_dtype: Optional[mindtorch.dtype] = None,
    use_fast_accum: bool = False,
) -> mindtorch.Tensor:
    if _is_test_mode:
        return _fused_scaled_matmul_reduce_scatter_fallback(
            A,
            B,
            A_scale,
            B_scale,
            reduce_op,
            scatter_dim,
            group_name,
            bias,
            result_scale,
            out_dtype,
            use_fast_accum,
        )
    with mindtorch.profiler.record_function("fused_matmul_reduce_scatter"):
        return _fused_matmul_reduce_scatter_impl(
            mm_out_op=mindtorch.ops.aten._scaled_mm.out,
            A=A,
            B=B,
            A_scale=A_scale,
            kwargs={
                "scale_b": B_scale,
                "bias": bias,
                "scale_result": result_scale,
                "out_dtype": out_dtype,
                "use_fast_accum": use_fast_accum,
            },
            out_dtype=out_dtype,
            reduce_op=reduce_op,
            scatter_dim=scatter_dim,
            group_name=group_name,
        )


def restride_A_for_fused_matmul_reduce_scatter(
    t: mindtorch.Tensor,
    scatter_dim: int,
) -> mindtorch.Tensor:
    """
    Restride the `A_shard` arg of `fused_matmul_reduce_scatter` for optimal
    perf. See the doc for `fused_matmul_reduce_scatter` for detail.
    """
    perm = list(range(len(t.shape)))
    perm.insert(0, perm.pop(scatter_dim))
    return make_contiguous_for_perm(t, perm)


def _maybe_convert_scalar_types_to_dtypes(
    scalar_types: List[Any],
) -> List[Optional[mindtorch.dtype]]:
    """
    When a list of `mindtorch.dtype`s is passed through the dispatcher as
    `ScalarType[]`, it is converted to a list of scalar type enum values. This
    function converts it back to a list of `mindtorch.dtype`s.
    """
    # Order defined in https://github.com/pytorch/pytorch/blob/344defc9733a45fee8d0c4d3f5530f631e823196/c10/core/ScalarType.h
    _SCALAR_TYPE_TO_DTYPE = {
        0: mindtorch.uint8,
        1: mindtorch.int8,
        2: mindtorch.short,
        3: mindtorch.int,
        4: mindtorch.int64,
        5: mindtorch.half,
        6: mindtorch.float,
        7: mindtorch.double,
        8: mindtorch.complex32,
        9: mindtorch.complex64,
        10: mindtorch.complex128,
        11: mindtorch.bool,
        12: mindtorch.qint8,
        13: mindtorch.quint8,
        14: mindtorch.qint32,
        15: mindtorch.bfloat16,
        16: mindtorch.float8_e5m2,
        17: mindtorch.float8_e4m3fn,
        18: mindtorch.float8_e5m2fnuz,
        19: mindtorch.float8_e4m3fnuz,
    }
    if any(not isinstance(x, (type(None), int)) for x in scalar_types):
        return scalar_types

    dtypes: List[Optional[mindtorch.dtype]] = []
    for scalar_type in scalar_types:
        if scalar_type is None:
            dtypes.append(scalar_type)
        elif scalar_type not in _SCALAR_TYPE_TO_DTYPE:
            raise ValueError("Unrecognized scalar type {scalar_type}")
        else:
            dtypes.append(_SCALAR_TYPE_TO_DTYPE[scalar_type])
    return dtypes


class Work(_Work):
    def __init__(self) -> None:
        super().__init__()
        self.event = mindtorch.cuda.Event()
        self.event.record()

    def wait(self, timeout: timedelta = timedelta(seconds=0)) -> bool:
        self.event.wait()
        return True


"""
NOTE [low-contention collectives]
When a collective is overlapped with abundant compute, it makes sense to
prioritize reducing the contention between the collective and the overlapped
compute, even at the cost of a slightly slower collective.

Common collective implementations (e.g., NCCL without user buffer
registration) optimize for throughput with no ambient compute. However, such
implementations may not be optimal when they are overlapped with compute:
- These implementations typically fuse the entire collective into a single
kernel and reserve SM resources based on the most demanding portion of the
collective, even when a large portion of the collective does not require this
much resource.
- These implementations often use SM-based P2P copy as opposed to copy
engine-based P2P copy. Copy engine-based P2P copy may not have a significant
advantage when there's no ambient compute. However, it may significantly
improve overall resource utilization in the presence of ambient compute.

When overlapped with intensive compute (e.g., persistent matmul kernels), the
SM-usage of a collective can lead to inefficient overlapping.

Low-contention collectives achieve their goals with the following strategies:
- Use copy engine-based copy whenever possible.
- Break down portions of a collective with different resource requirements
into multiple kernels. This improves the overlapping efficiency at the cost
of additional launching overhead.
"""


@mindtorch.library.impl(lib, "_low_contention_all_gather", "Meta")
def _low_contention_all_gather_meta(
    tensor: mindtorch.Tensor,
    group_name: str,
) -> mindtorch.Tensor:
    group_size = c10d._get_group_size_by_name(group_name)
    return tensor.new_empty(tensor.shape[0] * group_size, *tensor.shape[1:])


@mindtorch.library.impl(lib, "_low_contention_all_gather", "CUDA")
def _low_contention_all_gather(
    tensor: mindtorch.Tensor,
    group_name: str,
) -> mindtorch.Tensor:
    """
    Performs all-gather with symmetric memory in a low-contention fashion.

    When `tensor` is already in symmetric memory:
        - The collective is carried out without using SMs.
        - No symmetric memory workspace is required.

    When `tensor` is not in symmetric memory:
        - An extra SM-based copy is performed to copy the input data into the
          symmetric memory workspace.
        - Symmetric memory workspace size requirement: the size of `tensor`.
    """
    symm_mem = _SymmetricMemory.rendezvous(tensor)
    if symm_mem is not None:
        input_is_symm_mem = True
    else:
        symm_mem = get_symm_mem_workspace(
            group_name, tensor.numel() * tensor.element_size()
        )
        input_is_symm_mem = False

    rank = symm_mem.rank
    world_size = symm_mem.world_size

    output = tensor.new_empty(tensor.shape[0] * world_size, *tensor.shape[1:])
    chunks = output.chunk(world_size)

    _get_backend_stream().wait_stream(mindtorch.cuda.current_stream())
    with mindtorch.cuda.stream(_get_backend_stream()):
        if not input_is_symm_mem:
            local_buf = symm_mem.get_buffer(rank, tensor.shape, tensor.dtype)
            local_buf.copy_(tensor)
        # pull
        symm_mem.barrier()
        for step in range(0, world_size):
            remote_rank = (rank - step) % world_size
            src_buf = symm_mem.get_buffer(remote_rank, tensor.shape, tensor.dtype)
            chunks[remote_rank].copy_(src_buf)
        symm_mem.barrier()
        mindtorch._C._distributed_c10d._register_work(output, Work())
        return output


@mindtorch.library.impl(lib, "_low_contention_reduce_scatter", "Meta")
def _low_contention_reduce_scatter_meta(
    tensor: mindtorch.Tensor,
    reduce_op: str,
    group_name: str,
) -> mindtorch.Tensor:
    group_size = c10d._get_group_size_by_name(group_name)
    return tensor.unflatten(0, (group_size, -1)).mean(dim=0)


def _low_contention_reduce_scatter_with_symm_mem_input(
    tensor: mindtorch.Tensor,
    reduce_op: str,
    symm_mem: _SymmetricMemory,
) -> mindtorch.Tensor:
    rank = symm_mem.rank
    world_size = symm_mem.world_size

    assert tensor.shape[0] % world_size == 0
    a2a_res = mindtorch.empty_like(tensor)
    chunks = a2a_res.chunk(world_size)

    _get_backend_stream().wait_stream(mindtorch.cuda.current_stream())
    with mindtorch.cuda.stream(_get_backend_stream()):
        # pull + offline reduction
        symm_mem.barrier()
        for step in range(0, world_size):
            remote_rank = (rank - step) % world_size
            src_buf = symm_mem.get_buffer(
                remote_rank,
                chunks[0].shape,
                chunks[0].dtype,
                chunks[0].numel() * rank,
            )
            chunks[remote_rank].copy_(src_buf)
        symm_mem.barrier()

        ret = a2a_res.unflatten(0, (world_size, -1))
        if reduce_op == "sum":
            ret = ret.sum(dim=0)
        elif reduce_op == "avg":
            ret = ret.mean(dim=0)
        else:
            raise ValueError(f"reduce_op ({reduce_op}) is not supported")
        mindtorch._C._distributed_c10d._register_work(ret, Work())
        return ret


def _low_contention_reduce_scatter_with_workspace(
    tensor: mindtorch.Tensor,
    reduce_op: str,
    workspace: _SymmetricMemory,
) -> mindtorch.Tensor:
    rank = workspace.rank
    world_size = workspace.world_size

    assert tensor.shape[0] % world_size == 0
    chunks = tensor.chunk(world_size)

    _get_backend_stream().wait_stream(mindtorch.cuda.current_stream())
    with mindtorch.cuda.stream(_get_backend_stream()):
        # push + offline reduction
        workspace.barrier()
        for step in range(0, world_size):
            remote_rank = (rank - step) % world_size
            dst_buf = workspace.get_buffer(
                remote_rank, chunks[0].shape, chunks[0].dtype, chunks[0].numel() * rank
            )
            dst_buf.copy_(chunks[remote_rank])
        workspace.barrier()

        buf = workspace.get_buffer(rank, tensor.shape, tensor.dtype)
        ret = buf.unflatten(0, (world_size, -1))
        if reduce_op == "sum":
            ret = ret.sum(dim=0)
        elif reduce_op == "avg":
            ret = ret.mean(dim=0)
        else:
            raise ValueError(f"reduce_op ({reduce_op}) is not supported")
        mindtorch._C._distributed_c10d._register_work(ret, Work())
        return ret


@mindtorch.library.impl(lib, "_low_contention_reduce_scatter", "CUDA")
def _low_contention_reduce_scatter(
    tensor: mindtorch.Tensor,
    reduce_op: str,
    group_name: str,
) -> mindtorch.Tensor:
    """
    Performs reduce-scatter with symmetric memory in a low-contention fashion.

    This implementation performs a P2P-based all-to-all followed by an offline
    reduction.

    When `tensor` is already in symmetric memory:
        - Pull-based all-to-all is used.
        - No symmetric memory workspace is required.

    When `tensor` is not in symmetric memory:
        - Push-based all-to-all is used.
        - Symmetric memory workspace size requirement: the size of `tensor`.

    SM-usage:
        - SM-based copy of the rank's own chunk for the all-to-all.
        - Reduction on the all-to-all result.

    TODO(yifu): the SM-based copy can be avoided with a list-based reduction
    kernel.
    """
    symm_mem = _SymmetricMemory.rendezvous(tensor)
    if symm_mem is not None:
        return _low_contention_reduce_scatter_with_symm_mem_input(
            tensor, reduce_op, symm_mem
        )
    else:
        workspace = get_symm_mem_workspace(
            group_name, tensor.numel() * tensor.element_size()
        )
        return _low_contention_reduce_scatter_with_workspace(
            tensor, reduce_op, workspace
        )


# =============================================================================
# User-facing APIs
# =============================================================================


from typing import Any, overload, Sequence, TYPE_CHECKING, Union

from mindtorch.types import _device, _dtype, _int


if TYPE_CHECKING:
    from ..c10d import ProcessGroup


@overload
def empty(
    *size: _int, dtype: Optional[_dtype] = None, device: Optional[_device] = None
) -> mindtorch.Tensor:
    ...


@overload
def empty(
    size: Sequence[_int],
    *,
    dtype: Optional[_dtype] = None,
    device: Optional[_device] = None,
) -> mindtorch.Tensor:
    ...


def empty(  # type: ignore[misc]
    *size: Any,
    dtype: Optional[_dtype] = None,
    device: Optional[_device] = None,
) -> mindtorch.Tensor:
    r"""
    empty(*size, *, dtype=None, device=None) -> Tensor

    Similar to :func:`mindtorch.empty()`. The returned tensor can be used by
    :func:`mindtorch._distributed._symmetric_memory.rendezvous()` to establish a
    symmetric memory tensor among participating processes.

    Args:
        size (int...): a sequence of integers defining the shape of the output tensor.
            Can be a variable number of arguments or a collection like a list or tuple.

    Keyword args:
        dtype (:class:`mindtorch.dtype`, optional): the desired data type of returned tensor.
            Default: if ``None``, uses a global default (see :func:`mindtorch.set_default_dtype`).
        device (:class:`mindtorch.device`, optional): the desired device of returned tensor.
            Default: if ``None``, uses the current device for the default tensor type
            (see :func:`mindtorch.set_default_device`). :attr:`device` will be the CPU
            for CPU tensor types and the current CUDA device for CUDA tensor types.
    """
    if len(size) == 1 and isinstance(size[0], Sequence):
        size = tuple(size[0])
    else:
        size = tuple(size)

    if dtype is None:
        dtype = mindtorch.get_default_dtype()

    if device is None:
        device = mindtorch.get_default_device()

    return _SymmetricMemory.empty_strided_p2p(
        size=size,
        stride=mindtorch._prims_common.make_contiguous_strides_for(size),
        dtype=dtype,
        device=mindtorch.device(device),
    )


def rendezvous(
    tensor: mindtorch.Tensor, group: Union[str, "ProcessGroup"]
) -> _SymmetricMemory:
    r"""
    rendezvous(tensor, group) -> _SymmetricMemory

    Establish a symmetric memory tensor among participating processes. This is
    a collective operation.

    Args:
        tensor (:class:`mindtorch.Tensor`): the local tensor used to establish the symmetric memory tensor.
            It must be allocated via :func:`mindtorch._distributed._symmetric_memory.empty()`. The shape,
            dtype, and device type must be identical across all participating processes.
        group (Union[str, :class:`mindtorch.distributed.ProcessGroup`]): The group identifying the
            participating processes. This can be either a group name or a process group object.
    """
    from ..c10d import ProcessGroup

    if isinstance(group, str):
        group_name = group
    elif isinstance(group, ProcessGroup):
        group_name = group.group_name
    else:
        raise TypeError(f"rendezvous: unsupported group type: {type(group)}")

    enable_symm_mem_for_group(group_name)
    return _SymmetricMemory.rendezvous(tensor, group_name)


__all__ = ["empty", "rendezvous"]
