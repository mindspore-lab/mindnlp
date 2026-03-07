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


def test_hccl_new_group_inherits_parent_device_id_by_default(monkeypatch) -> None:
    _cleanup_pg()

    class _FakeStore:
        def __init__(self):
            self.data = {}

        def set(self, key, value):
            self.data[key] = value

        def get(self, key):
            return self.data.get(key, b"")

        def wait(self, keys, timeout=None):
            return None

    class _FakeHCCLPG:
        def __init__(self, rank, size, device_id):
            self._rank = rank
            self._size = size
            self._device_id = device_id

        def rank(self):
            return self._rank

        def size(self):
            return self._size

        def destroy(self):
            return None

    created = {}

    def _fake_hccl_ctor(store, rank, size, device_id=None, group_name="", group_ranks=None):
        created["device_id"] = device_id
        return _FakeHCCLPG(rank, size, device_id)

    fake_world = _FakeHCCLPG(rank=3, size=8, device_id=5)
    fake_store = _FakeStore()

    monkeypatch.setattr(dist, "_default_pg", fake_world)
    monkeypatch.setattr(dist, "_group_count", 1)
    monkeypatch.setitem(dist._pg_map, fake_world, (dist.Backend("hccl"), fake_store))
    monkeypatch.setitem(dist._pg_names, fake_world, "default_pg")
    monkeypatch.setitem(dist._pg_group_ranks, fake_world, {i: i for i in range(8)})
    monkeypatch.setattr(dist, "GroupMember", type("_GM", (), {"WORLD": fake_world, "NON_GROUP_MEMBER": object()}))
    monkeypatch.setattr("mindtorch_v2.distributed.ProcessGroupHCCL", _fake_hccl_ctor)

    subgroup = dist.new_group(ranks=[3, 5], backend="hccl")
    assert subgroup is not None
    assert created["device_id"] == 5
