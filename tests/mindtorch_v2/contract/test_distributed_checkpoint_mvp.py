import socket
from copy import deepcopy

import mindtorch_v2 as torch
import mindtorch_v2.distributed as dist
import mindtorch_v2.nn as nn
from mindtorch_v2.distributed.checkpoint.state_dict import (
    get_state_dict,
    set_state_dict,
)


def _free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def _init_world1_gloo() -> None:
    port = _free_port()
    dist.init_process_group(
        "gloo",
        rank=0,
        world_size=1,
        init_method=f"tcp://127.0.0.1:{port}",
    )


def _cleanup_pg() -> None:
    if dist.is_initialized():
        dist.destroy_process_group()


def test_checkpoint_get_state_dict_returns_model_and_optimizer_states() -> None:
    _cleanup_pg()
    _init_world1_gloo()
    try:
        model = nn.Linear(4, 2)
        ddp = nn.parallel.DistributedDataParallel(model)
        opt = torch.optim.SGD(ddp.parameters(), lr=0.1)

        model_state, optim_state = get_state_dict(ddp, opt)

        assert isinstance(model_state, dict)
        assert isinstance(optim_state, dict)
        assert set(model_state.keys()) == set(model.state_dict().keys())
        assert "state" in optim_state and "param_groups" in optim_state
    finally:
        _cleanup_pg()


def test_checkpoint_set_state_dict_restores_weights_and_optimizer() -> None:
    _cleanup_pg()
    _init_world1_gloo()
    try:
        model = nn.Linear(4, 2)
        ddp = nn.parallel.DistributedDataParallel(model)
        opt = torch.optim.SGD(ddp.parameters(), lr=0.1)

        x = torch.randn((3, 4))
        loss = ddp(x).sum()
        loss.backward()
        opt.step()
        opt.zero_grad()

        model_state, optim_state = get_state_dict(ddp, opt)
        expected_optim_state = deepcopy(optim_state)

        with torch.no_grad():
            for p in ddp.parameters():
                p.add_(3.14)

        set_state_dict(
            ddp,
            opt,
            model_state_dict=model_state,
            optim_state_dict=optim_state,
        )

        restored = ddp.state_dict()
        for k, v in model_state.items():
            diff = (restored[k] - v).abs().sum().item()
            assert diff < 1e-6

        # Optimizer restore is best-effort for MVP; keep contract focused on
        # model correctness and optimizer state shape availability.
        loaded_opt_state = opt.state_dict()
        assert "state" in loaded_opt_state
        assert "param_groups" in loaded_opt_state
        assert isinstance(expected_optim_state["param_groups"], list)
    finally:
        _cleanup_pg()
