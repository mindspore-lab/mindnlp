import numpy as np
import torch
import io
import builtins
from collections import OrderedDict

import mindtorch_v2 as mt
import mindtorch_v2.nn as nn


def test_serialization_does_not_import_torch_runtime(monkeypatch):
    real_import = builtins.__import__

    def guarded_import(name, *args, **kwargs):
        if name == "torch" or name.startswith("torch."):
            raise AssertionError("torch import is not allowed in mindtorch_v2 serialization")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", guarded_import)

    buf = io.BytesIO()
    mt.save({"x": mt.tensor([1.0, 2.0])}, buf)
    buf.seek(0)
    out = mt.load(buf)
    assert out["x"].tolist() == [1.0, 2.0]


def _assert_state_dict_close(lhs, rhs):
    assert set(lhs.keys()) == set(rhs.keys())
    for k in lhs.keys():
        lv = lhs[k]
        rv = rhs[k]
        assert tuple(lv.shape) == tuple(rv.shape)
        assert str(lv.dtype).split(".")[-1] == str(rv.dtype).split(".")[-1]
        lcpu = lv.detach().to("cpu").numpy()
        rcpu = rv.detach().to("cpu").numpy()
        np.testing.assert_allclose(lcpu, rcpu)


def test_torch_save_then_mindtorch_load_state_dict(tmp_path):
    model = torch.nn.Linear(4, 3)
    path = tmp_path / "torch_state.pth"
    torch.save(model.state_dict(), path)

    loaded = mt.load(path)

    assert isinstance(loaded, OrderedDict)
    assert set(loaded.keys()) == {"weight", "bias"}
    for v in loaded.values():
        assert isinstance(v, mt.Tensor)

    ref = {k: v.detach().cpu() for k, v in model.state_dict().items()}
    _assert_state_dict_close(ref, loaded)


def test_mindtorch_save_then_torch_load_state_dict(tmp_path):
    model = nn.Linear(4, 3)
    state = model.state_dict()
    path = tmp_path / "mindtorch_state.pth"
    mt.save(state, path)

    loaded = torch.load(path, map_location="cpu")

    assert isinstance(loaded, OrderedDict)
    assert set(loaded.keys()) == {"weight", "bias"}
    for v in loaded.values():
        assert isinstance(v, torch.Tensor)

    expected = {k: torch.tensor(v.detach().to("cpu").numpy()) for k, v in state.items()}
    assert set(expected.keys()) == set(loaded.keys())
    for k in expected.keys():
        assert expected[k].shape == loaded[k].shape
        assert expected[k].dtype == loaded[k].dtype
        assert torch.allclose(expected[k], loaded[k])


def test_mindtorch_save_load_nested_common_checkpoint(tmp_path):
    model = nn.Linear(4, 3)
    ckpt = {
        "model": model.state_dict(),
        "epoch": 7,
        "meta": {"tag": "baseline", "lr": 1e-3},
    }
    path = tmp_path / "nested_ckpt.pth"
    mt.save(ckpt, path)

    loaded = mt.load(path)

    assert loaded["epoch"] == 7
    assert loaded["meta"]["tag"] == "baseline"
    assert abs(loaded["meta"]["lr"] - 1e-3) < 1e-12
    assert isinstance(loaded["model"], dict)
    assert set(loaded["model"].keys()) == {"weight", "bias"}


def test_mindtorch_save_load_bytesio_roundtrip():
    model = nn.Linear(4, 3)
    ckpt = {"model": model.state_dict(), "step": 11}

    buf = io.BytesIO()
    mt.save(ckpt, buf)
    buf.seek(0)

    loaded = mt.load(buf)

    assert loaded["step"] == 11
    assert set(loaded["model"].keys()) == {"weight", "bias"}


def test_torch_save_optimizer_state_then_mindtorch_load(tmp_path):
    model = torch.nn.Linear(4, 3)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)

    x = torch.randn(2, 4)
    y = model(x).sum()
    y.backward()
    opt.step()

    path = tmp_path / "torch_opt_state.pth"
    torch.save(opt.state_dict(), path)

    loaded = mt.load(path, map_location="cpu")

    assert isinstance(loaded, dict)
    assert "state" in loaded
    assert "param_groups" in loaded
    assert isinstance(loaded["state"], dict)
    assert isinstance(loaded["param_groups"], list)
