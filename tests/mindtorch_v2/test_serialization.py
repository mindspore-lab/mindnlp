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


def test_torch_load_with_map_location_dict(tmp_path):
    model = torch.nn.Linear(4, 3)
    path = tmp_path / "torch_state_dict_map_loc_dict.pth"
    torch.save(model.state_dict(), path)

    loaded = mt.load(path, map_location={"cpu": "cpu"})

    assert isinstance(loaded, OrderedDict)
    assert set(loaded.keys()) == {"weight", "bias"}


def test_torch_load_with_map_location_callable(tmp_path):
    model = torch.nn.Linear(4, 3)
    path = tmp_path / "torch_state_dict_map_loc_callable.pth"
    torch.save(model.state_dict(), path)

    calls = []

    def mapper(storage, loc):
        calls.append(loc)
        return storage

    loaded = mt.load(path, map_location=mapper)

    assert isinstance(loaded, OrderedDict)
    assert set(loaded.keys()) == {"weight", "bias"}
    assert calls and set(calls) == {"cpu"}


def test_mindtorch_roundtrip_preserves_storage_aliasing():
    base = mt.tensor([0.0, 1.0, 2.0, 3.0])
    view1 = base[:3]
    view2 = base[1:]

    buf = io.BytesIO()
    mt.save({"v1": view1, "v2": view2}, buf)
    buf.seek(0)
    loaded = mt.load(buf)

    lv1 = loaded["v1"]
    lv2 = loaded["v2"]
    assert lv1.storage().data_ptr() == lv2.storage().data_ptr()
    assert lv1.offset == 0
    assert lv2.offset == 1
    assert lv1.tolist() == [0.0, 1.0, 2.0]
    assert lv2.tolist() == [1.0, 2.0, 3.0]


def test_torch_to_mindtorch_preserves_storage_aliasing(tmp_path):
    t = torch.arange(6.0)
    obj = {"a": t[:4], "b": t[2:]}
    path = tmp_path / "torch_alias.pth"
    torch.save(obj, path)

    loaded = mt.load(path)

    la = loaded["a"]
    lb = loaded["b"]
    assert la.storage().data_ptr() == lb.storage().data_ptr()
    assert la.offset == 0
    assert lb.offset == 2
    assert la.tolist() == [0.0, 1.0, 2.0, 3.0]
    assert lb.tolist() == [2.0, 3.0, 4.0, 5.0]


def test_torch_to_mindtorch_preserves_non_contiguous_stride(tmp_path):
    t = torch.arange(12.0).reshape(3, 4).t()
    path = tmp_path / "torch_noncontig.pth"
    torch.save({"x": t}, path)

    loaded = mt.load(path)
    x = loaded["x"]

    assert tuple(x.shape) == (4, 3)
    assert tuple(int(s) for s in x.stride) == (1, 4)
    assert x.tolist() == t.tolist()


def test_mindtorch_to_torch_preserves_non_contiguous_stride(tmp_path):
    t = mt.arange(0, 12).reshape((3, 4)).transpose(0, 1)
    path = tmp_path / "mindtorch_noncontig.pth"
    mt.save({"x": t}, path)

    loaded = torch.load(path, map_location="cpu")
    x = loaded["x"]

    assert tuple(x.shape) == (4, 3)
    assert tuple(x.stride()) == (1, 4)
    assert x.tolist() == t.tolist()


import pytest


@pytest.mark.parametrize(
    "dtype",
    [torch.float16, torch.float32, torch.float64, torch.int64, torch.bool],
)
def test_torch_to_mindtorch_dtype_matrix(tmp_path, dtype):
    if dtype is torch.bool:
        t = torch.tensor([[True, False], [False, True]], dtype=dtype)
    else:
        t = torch.arange(4, dtype=torch.float32).reshape(2, 2).to(dtype)
    path = tmp_path / f"torch_dtype_{str(dtype).split('.')[-1]}.pth"
    torch.save({"x": t}, path)

    loaded = mt.load(path)
    x = loaded["x"]

    assert tuple(x.shape) == tuple(t.shape)
    assert str(x.dtype).split(".")[-1] == str(t.dtype).split(".")[-1]
    assert x.tolist() == t.tolist()


@pytest.mark.parametrize(
    "dtype",
    [mt.float16, mt.float32, mt.float64, mt.int64, mt.bool],
)
def test_mindtorch_to_torch_dtype_matrix(tmp_path, dtype):
    if dtype == mt.bool:
        t = mt.tensor([[True, False], [False, True]], dtype=dtype)
    else:
        t = mt.arange(0, 4, dtype=mt.float32).reshape((2, 2)).to(dtype=dtype)
    path = tmp_path / f"mindtorch_dtype_{dtype.name}.pth"
    mt.save({"x": t}, path)

    loaded = torch.load(path, map_location="cpu")
    x = loaded["x"]

    assert tuple(x.shape) == tuple(t.shape)
    assert str(x.dtype).split(".")[-1] == dtype.name
    assert x.tolist() == t.tolist()
