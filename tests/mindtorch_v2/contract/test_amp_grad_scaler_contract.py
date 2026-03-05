import pytest
import mindtorch_v2 as torch


class _DummyOptimizer:
    def __init__(self):
        self.p = torch.tensor([1.0], dtype=torch.float32)
        self.p.grad = torch.tensor([2.0], dtype=torch.float32)
        self.param_groups = [{"params": [self.p]}]
        self.step_calls = 0

    def step(self, *args, **kwargs):
        self.step_calls += 1
        return self.step_calls


def test_grad_scaler_double_unscale_raises():
    optim = _DummyOptimizer()
    scaler = torch.amp.GradScaler("cpu")
    scaler.scale(torch.tensor(1.0))
    scaler.unscale_(optim)

    with pytest.raises(RuntimeError):
        scaler.unscale_(optim)


def test_grad_scaler_step_twice_raises():
    optim = _DummyOptimizer()
    scaler = torch.amp.GradScaler("cpu")
    scaler.scale(torch.tensor(1.0))
    scaler.step(optim)

    with pytest.raises(RuntimeError):
        scaler.step(optim)


def test_grad_scaler_state_dict_keys_match_torch_contract():
    scaler = torch.amp.GradScaler("cpu")
    state = scaler.state_dict()
    expected = {"scale", "growth_factor", "backoff_factor", "growth_interval", "_growth_tracker"}
    assert set(state.keys()) == expected


def test_grad_scaler_unscale_after_step_raises():
    optim = _DummyOptimizer()
    scaler = torch.amp.GradScaler("cpu")
    scaler.scale(torch.tensor(1.0))
    scaler.step(optim)

    with pytest.raises(RuntimeError):
        scaler.unscale_(optim)
