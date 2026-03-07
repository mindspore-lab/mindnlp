import os
import socket

import mindtorch_v2 as torch
import mindtorch_v2.nn as nn
import mindtorch_v2.distributed as dist


def _free_port():
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def _set_env(port):
    os.environ["MASTER_ADDR"] = "127.0.0.1"
    os.environ["MASTER_PORT"] = str(port)


def _cleanup_pg():
    if dist.is_initialized():
        dist.destroy_process_group()


def test_process_group_lifecycle_smoke_cpu():
    _cleanup_pg()
    port = _free_port()
    _set_env(port)

    assert dist.is_initialized() is False
    dist.init_process_group("gloo", rank=0, world_size=1, init_method=f"tcp://127.0.0.1:{port}")

    assert dist.is_initialized() is True
    assert dist.get_rank() == 0
    assert dist.get_world_size() == 1
    dist.barrier()

    dist.destroy_process_group()
    assert dist.is_initialized() is False


def test_all_reduce_sum_world1_smoke_cpu():
    _cleanup_pg()
    port = _free_port()
    _set_env(port)

    dist.init_process_group("gloo", rank=0, world_size=1, init_method=f"tcp://127.0.0.1:{port}")
    try:
        t = torch.tensor([1.0, 3.0])
        dist.all_reduce(t, op=dist.ReduceOp.SUM)
        assert t.numpy().tolist() == [1.0, 3.0]
    finally:
        _cleanup_pg()


def test_ddp_single_process_train_step_cpu():
    _cleanup_pg()
    port = _free_port()
    _set_env(port)

    dist.init_process_group("gloo", rank=0, world_size=1, init_method=f"tcp://127.0.0.1:{port}")
    try:
        model = nn.Linear(4, 2)
        ddp_model = nn.parallel.DistributedDataParallel(model)
        x = torch.randn((3, 4))

        loss = ddp_model(x).sum()
        loss.backward()

        assert model.weight.grad is not None
        assert model.bias.grad is not None
    finally:
        _cleanup_pg()


def test_all_gather_async_wait_populates_tensor_list_world1_cpu():
    _cleanup_pg()
    port = _free_port()
    _set_env(port)

    dist.init_process_group("gloo", rank=0, world_size=1, init_method=f"tcp://127.0.0.1:{port}")
    try:
        source = torch.tensor([2.0, 4.0])
        gathered = [torch.zeros_like(source)]

        work = dist.all_gather(gathered, source, async_op=True)
        assert work is not None
        work.wait()

        assert gathered[0].numpy().tolist() == [2.0, 4.0]
    finally:
        _cleanup_pg()
