import socket

import pytest

import mindtorch_v2 as torch
import mindtorch_v2.distributed as dist


def _free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def _cleanup_pg() -> None:
    if dist.is_initialized():
        dist.destroy_process_group()


def test_init_process_group_error_includes_runtime_context() -> None:
    _cleanup_pg()
    port = _free_port()
    # Explicitly provide unsupported backend to force a stable, local failure.
    with pytest.raises(ValueError, match=r"stage=init_process_group") as ei:
        dist.init_process_group(
            backend="unknown_backend",
            rank=0,
            world_size=1,
            init_method=f"tcp://127.0.0.1:{port}",
        )

    msg = str(ei.value)
    assert "backend=unknown_backend" in msg
    assert "rank=0" in msg
    assert "world_size=1" in msg


def test_collective_error_includes_runtime_context() -> None:
    _cleanup_pg()
    t = torch.tensor([1.0])
    # Uninitialized call path should fail and expose runtime context.
    with pytest.raises(Exception, match=r"stage=all_reduce") as ei:
        dist.all_reduce(t)

    msg = str(ei.value)
    assert "backend=uninitialized" in msg
    assert "rank=0" in msg
    assert "world_size=1" in msg
    assert "op=all_reduce" in msg


def test_barrier_error_includes_runtime_context() -> None:
    _cleanup_pg()
    with pytest.raises(Exception, match=r"stage=barrier") as ei:
        dist.barrier()

    msg = str(ei.value)
    assert "backend=uninitialized" in msg
    assert "rank=0" in msg
    assert "world_size=1" in msg
    assert "op=barrier" in msg


def test_broadcast_error_includes_runtime_context() -> None:
    _cleanup_pg()
    t = torch.tensor([1.0])
    with pytest.raises(Exception, match=r"stage=broadcast") as ei:
        dist.broadcast(t, src=0)

    msg = str(ei.value)
    assert "backend=uninitialized" in msg
    assert "rank=0" in msg
    assert "world_size=1" in msg
    assert "op=broadcast" in msg
