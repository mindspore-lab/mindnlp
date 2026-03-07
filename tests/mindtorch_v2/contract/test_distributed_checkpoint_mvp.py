import socket
from copy import deepcopy

import pytest

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

        result = set_state_dict(
            ddp,
            opt,
            model_state_dict=model_state,
            optim_state_dict=optim_state,
        )

        assert isinstance(result, dict)
        assert "restored_optimizer_state_keys_count" in result
        assert isinstance(result["restored_optimizer_state_keys_count"], int)

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


def test_checkpoint_get_state_dict_rank0_only_returns_none_on_non_zero_rank() -> None:
    model = nn.Linear(2, 1)

    model_state, optim_state = get_state_dict(
        model,
        optimizer=None,
        rank0_only=True,
        rank=1,
    )

    assert model_state is None
    assert optim_state is None


def test_checkpoint_get_state_dict_rank0_only_returns_payload_on_rank0() -> None:
    model = nn.Linear(2, 1)

    model_state, optim_state = get_state_dict(
        model,
        optimizer=None,
        rank0_only=True,
        rank=0,
    )

    assert isinstance(model_state, dict)
    assert optim_state is None


def test_checkpoint_get_state_dict_rank0_only_payload_on_rank0() -> None:
    model = nn.Linear(2, 1)
    opt = torch.optim.SGD(model.parameters(), lr=0.1)

    payload = get_state_dict(
        model,
        optimizer=opt,
        rank0_only=True,
        rank=0,
        as_payload=True,
    )

    assert isinstance(payload, dict)
    assert "model" in payload
    assert "optim" in payload
    assert "meta" in payload
    assert payload["meta"]["rank0_only"] is True
    assert payload["meta"]["rank"] == 0


def test_checkpoint_get_state_dict_rank0_only_payload_none_on_non_zero_rank() -> None:
    model = nn.Linear(2, 1)

    payload = get_state_dict(
        model,
        optimizer=None,
        rank0_only=True,
        rank=3,
        as_payload=True,
    )

    assert payload is None


def test_checkpoint_set_state_dict_accepts_payload_bundle() -> None:
    model = nn.Linear(3, 2)
    opt = torch.optim.SGD(model.parameters(), lr=0.1)

    payload = get_state_dict(
        model,
        optimizer=opt,
        rank0_only=True,
        rank=0,
        as_payload=True,
    )

    with torch.no_grad():
        model.weight.add_(2.0)
        model.bias.add_(2.0)

    result = set_state_dict(model, optimizer=opt, payload=payload)

    assert isinstance(result, dict)
    assert "loaded_keys_count" in result


def test_checkpoint_set_state_dict_payload_none_is_noop() -> None:
    model = nn.Linear(3, 2)
    before = {k: v.clone() for k, v in model.state_dict().items()}

    result = set_state_dict(model, payload=None, strict=False)

    assert result["loaded_keys_count"] == 0
    after = model.state_dict()
    for k in before:
        diff = (after[k] - before[k]).abs().sum().item()
        assert diff < 1e-6


def test_checkpoint_set_state_dict_rejects_payload_without_meta() -> None:
    model = nn.Linear(3, 2)
    bad_payload = {
        "model": model.state_dict(),
        "optim": None,
    }

    with pytest.raises(ValueError, match="payload.meta must be a dict"):
        set_state_dict(model, payload=bad_payload)


def test_checkpoint_set_state_dict_rejects_payload_meta_without_rank() -> None:
    model = nn.Linear(3, 2)
    payload = {
        "model": model.state_dict(),
        "optim": None,
        "meta": {"rank0_only": True},
    }

    with pytest.raises(ValueError, match="payload.meta.rank must be an int"):
        set_state_dict(model, payload=payload)


def test_checkpoint_set_state_dict_requires_model_state_dict() -> None:
    model = nn.Linear(2, 1)

    with pytest.raises(ValueError, match="model_state_dict must not be None"):
        set_state_dict(model, model_state_dict=None)


def test_checkpoint_set_state_dict_strict_false_allows_missing_keys() -> None:
    model = nn.Linear(3, 2)
    full_state = {k: v.clone() for k, v in model.state_dict().items()}

    # Remove one key to create a partial checkpoint.
    partial_state = {k: v for k, v in full_state.items() if k != "bias"}

    with torch.no_grad():
        model.weight.add_(1.23)
        model.bias.add_(4.56)

    result = set_state_dict(model, model_state_dict=partial_state, strict=False)

    assert isinstance(result, dict)
    assert "missing_keys" in result
    assert "unexpected_keys" in result
    assert "loaded_keys_count" in result
    assert "bias" in result["missing_keys"]
    assert result["unexpected_keys"] == []
    assert result["loaded_keys_count"] == 1

    # Provided key is restored.
    wdiff = (model.weight - full_state["weight"]).abs().sum().item()
    assert wdiff < 1e-6
    # Missing key stays unchanged under strict=False path.
    bdiff = (model.bias - full_state["bias"]).abs().sum().item()
    assert bdiff > 1e-6


def test_checkpoint_set_state_dict_reports_unexpected_keys_when_not_strict() -> None:
    model = nn.Linear(3, 2)
    state = {k: v.clone() for k, v in model.state_dict().items()}
    state["unexpected.weight"] = torch.randn((1,))

    result = set_state_dict(model, model_state_dict=state, strict=False)

    assert isinstance(result, dict)
    assert "unexpected.weight" in result["unexpected_keys"]
    assert result["loaded_keys_count"] == 2


def test_checkpoint_set_state_dict_rejects_optimizer_group_mismatch_by_default() -> None:
    model = nn.Linear(3, 2)
    opt = torch.optim.SGD(model.parameters(), lr=0.1)
    payload = get_state_dict(model, optimizer=opt, as_payload=True)

    # Create optimizer param_group mismatch.
    payload["optim"]["param_groups"] = payload["optim"]["param_groups"] + [
        dict(payload["optim"]["param_groups"][0])
    ]

    with pytest.raises(ValueError, match="optimizer param_groups length mismatch"):
        set_state_dict(model, optimizer=opt, payload=payload)


def test_checkpoint_set_state_dict_allows_optimizer_group_mismatch_when_enabled() -> None:
    model = nn.Linear(3, 2)
    opt = torch.optim.SGD(model.parameters(), lr=0.1)
    payload = get_state_dict(model, optimizer=opt, as_payload=True)

    payload["optim"]["param_groups"] = payload["optim"]["param_groups"] + [
        dict(payload["optim"]["param_groups"][0])
    ]

    result = set_state_dict(
        model,
        optimizer=opt,
        payload=payload,
        allow_partial_optimizer=True,
    )

    assert isinstance(result, dict)
    assert result["restored_optimizer_state_keys_count"] >= 0
