import socket
import os

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


def test_hccl_init_preflight_rejects_negative_device_id_with_context() -> None:
    _cleanup_pg()
    port = _free_port()
    with pytest.raises(ValueError, match=r"stage=init_process_group") as ei:
        dist.init_process_group(
            backend="hccl",
            rank=0,
            world_size=1,
            init_method=f"tcp://127.0.0.1:{port}",
            device_id=-1,
        )

    msg = str(ei.value)
    assert "backend=hccl" in msg
    assert "device_id=-1" in msg


def test_hccl_init_preflight_rejects_rank_out_of_range_with_context() -> None:
    _cleanup_pg()
    old_rank = os.environ.get("RANK")
    old_world = os.environ.get("WORLD_SIZE")
    old_addr = os.environ.get("MASTER_ADDR")
    old_port = os.environ.get("MASTER_PORT")
    try:
        os.environ["RANK"] = "3"
        os.environ["WORLD_SIZE"] = "2"
        os.environ["MASTER_ADDR"] = "127.0.0.1"
        os.environ["MASTER_PORT"] = str(_free_port())
        with pytest.raises(ValueError, match=r"stage=init_process_group") as ei:
            dist.init_process_group(backend="hccl")

        msg = str(ei.value)
        assert "backend=hccl" in msg
        assert "rank=3" in msg
        assert "world_size=2" in msg
    finally:
        if old_rank is None:
            os.environ.pop("RANK", None)
        else:
            os.environ["RANK"] = old_rank
        if old_world is None:
            os.environ.pop("WORLD_SIZE", None)
        else:
            os.environ["WORLD_SIZE"] = old_world
        if old_addr is None:
            os.environ.pop("MASTER_ADDR", None)
        else:
            os.environ["MASTER_ADDR"] = old_addr
        if old_port is None:
            os.environ.pop("MASTER_PORT", None)
        else:
            os.environ["MASTER_PORT"] = old_port


def test_hccl_init_preflight_requires_rank_env_with_context() -> None:
    _cleanup_pg()
    old_rank = os.environ.get("RANK")
    old_world = os.environ.get("WORLD_SIZE")
    old_addr = os.environ.get("MASTER_ADDR")
    old_port = os.environ.get("MASTER_PORT")
    try:
        os.environ.pop("RANK", None)
        os.environ["WORLD_SIZE"] = "2"
        os.environ["MASTER_ADDR"] = "127.0.0.1"
        os.environ["MASTER_PORT"] = str(_free_port())
        with pytest.raises(ValueError, match=r"stage=init_process_group") as ei:
            dist.init_process_group(backend="hccl", rank=-1, world_size=-1)

        msg = str(ei.value)
        assert "backend=hccl" in msg
        assert "rank=0" in msg
        assert "world_size=2" in msg
        assert "requires RANK env" in msg
    finally:
        if old_rank is None:
            os.environ.pop("RANK", None)
        else:
            os.environ["RANK"] = old_rank
        if old_world is None:
            os.environ.pop("WORLD_SIZE", None)
        else:
            os.environ["WORLD_SIZE"] = old_world
        if old_addr is None:
            os.environ.pop("MASTER_ADDR", None)
        else:
            os.environ["MASTER_ADDR"] = old_addr
        if old_port is None:
            os.environ.pop("MASTER_PORT", None)
        else:
            os.environ["MASTER_PORT"] = old_port


def test_hccl_init_preflight_requires_world_size_env_with_context() -> None:
    _cleanup_pg()
    old_rank = os.environ.get("RANK")
    old_world = os.environ.get("WORLD_SIZE")
    old_addr = os.environ.get("MASTER_ADDR")
    old_port = os.environ.get("MASTER_PORT")
    try:
        os.environ["RANK"] = "0"
        os.environ.pop("WORLD_SIZE", None)
        os.environ["MASTER_ADDR"] = "127.0.0.1"
        os.environ["MASTER_PORT"] = str(_free_port())
        with pytest.raises(ValueError, match=r"stage=init_process_group") as ei:
            dist.init_process_group(backend="hccl", rank=-1, world_size=-1)

        msg = str(ei.value)
        assert "backend=hccl" in msg
        assert "rank=0" in msg
        assert "world_size=1" in msg
        assert "requires WORLD_SIZE env" in msg
    finally:
        if old_rank is None:
            os.environ.pop("RANK", None)
        else:
            os.environ["RANK"] = old_rank
        if old_world is None:
            os.environ.pop("WORLD_SIZE", None)
        else:
            os.environ["WORLD_SIZE"] = old_world
        if old_addr is None:
            os.environ.pop("MASTER_ADDR", None)
        else:
            os.environ["MASTER_ADDR"] = old_addr
        if old_port is None:
            os.environ.pop("MASTER_PORT", None)
        else:
            os.environ["MASTER_PORT"] = old_port


def test_hccl_preflight_error_includes_master_endpoint_context() -> None:
    _cleanup_pg()
    old_addr = os.environ.get("MASTER_ADDR")
    old_port = os.environ.get("MASTER_PORT")
    try:
        os.environ["MASTER_ADDR"] = "127.0.0.9"
        os.environ["MASTER_PORT"] = "29677"
        with pytest.raises(ValueError, match=r"stage=init_process_group") as ei:
            dist.init_process_group(
                backend="hccl",
                rank=3,
                world_size=2,
            )

        msg = str(ei.value)
        assert "master_addr=127.0.0.9" in msg
        assert "master_port=29677" in msg
    finally:
        if old_addr is None:
            os.environ.pop("MASTER_ADDR", None)
        else:
            os.environ["MASTER_ADDR"] = old_addr
        if old_port is None:
            os.environ.pop("MASTER_PORT", None)
        else:
            os.environ["MASTER_PORT"] = old_port


def test_hccl_preflight_error_includes_rank_world_source_context() -> None:
    _cleanup_pg()
    old_rank = os.environ.get("RANK")
    old_world = os.environ.get("WORLD_SIZE")
    old_addr = os.environ.get("MASTER_ADDR")
    old_port = os.environ.get("MASTER_PORT")
    try:
        os.environ["RANK"] = "7"
        os.environ["WORLD_SIZE"] = "4"
        os.environ["MASTER_ADDR"] = "127.0.0.1"
        os.environ["MASTER_PORT"] = str(_free_port())
        with pytest.raises(ValueError, match=r"stage=init_process_group") as ei:
            dist.init_process_group(backend="hccl", rank=-1, world_size=-1)

        msg = str(ei.value)
        assert "rank_source=env" in msg
        assert "world_size_source=env" in msg
    finally:
        if old_rank is None:
            os.environ.pop("RANK", None)
        else:
            os.environ["RANK"] = old_rank
        if old_world is None:
            os.environ.pop("WORLD_SIZE", None)
        else:
            os.environ["WORLD_SIZE"] = old_world
        if old_addr is None:
            os.environ.pop("MASTER_ADDR", None)
        else:
            os.environ["MASTER_ADDR"] = old_addr
        if old_port is None:
            os.environ.pop("MASTER_PORT", None)
        else:
            os.environ["MASTER_PORT"] = old_port
