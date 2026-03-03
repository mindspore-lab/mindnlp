import numpy as np
import torch
import io
import builtins
import pickle
import zipfile
import collections
from collections import OrderedDict

import mindtorch_v2 as mt
import mindtorch_v2.serialization as ser
import mindtorch_v2.nn as nn


class _CustomPayload:
    def __init__(self, value):
        self.value = value


class _TorchUnsafeGlobal:
    def __init__(self, value):
        self.value = value


class _CountingPickleModule:
    load_calls = 0
    unpickler_inits = 0

    @staticmethod
    def load(*args, **kwargs):
        _CountingPickleModule.load_calls += 1
        return pickle.load(*args, **kwargs)

    class Unpickler(pickle.Unpickler):
        def __init__(self, *args, **kwargs):
            _CountingPickleModule.unpickler_inits += 1
            super().__init__(*args, **kwargs)


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


def test_torch_legacy_save_then_mindtorch_load_state_dict(tmp_path):
    model = torch.nn.Linear(4, 3)
    path = tmp_path / "torch_state_legacy.pth"
    torch.save(model.state_dict(), path, _use_new_zipfile_serialization=False)

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


def test_weights_only_allows_torch_zip_state_dict(tmp_path):
    model = torch.nn.Linear(4, 3)
    path = tmp_path / "torch_weights_only_zip.pth"
    torch.save(model.state_dict(), path)

    loaded = mt.load(path, weights_only=True)

    assert isinstance(loaded, OrderedDict)
    assert set(loaded.keys()) == {"weight", "bias"}


def test_weights_only_allows_torch_legacy_state_dict(tmp_path):
    model = torch.nn.Linear(4, 3)
    path = tmp_path / "torch_weights_only_legacy.pth"
    torch.save(model.state_dict(), path, _use_new_zipfile_serialization=False)

    loaded = mt.load(path, weights_only=True)

    assert isinstance(loaded, OrderedDict)
    assert set(loaded.keys()) == {"weight", "bias"}


def test_weights_only_rejects_custom_global(tmp_path):
    obj = {
        "weight": mt.tensor([1.0]),
        "meta": _CustomPayload(7),
    }
    path = tmp_path / "weights_only_reject_custom.pth"
    mt.save(obj, path)

    with pytest.raises(pickle.UnpicklingError, match="weights_only"):
        mt.load(path, weights_only=True)


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


def test_torch_load_with_map_location_torch_device_cpu(tmp_path):
    model = torch.nn.Linear(4, 3)
    path = tmp_path / "torch_state_dict_map_loc_device_cpu.pth"
    torch.save(model.state_dict(), path)

    loaded = mt.load(path, map_location=torch.device("cpu"))

    assert isinstance(loaded, OrderedDict)
    assert set(loaded.keys()) == {"weight", "bias"}


def _patch_checkpoint_location_tag(src_path, dst_path, old, new):
    assert len(old) == len(new)
    with zipfile.ZipFile(src_path, "r") as zin, zipfile.ZipFile(dst_path, "w", compression=zipfile.ZIP_STORED) as zout:
        for name in zin.namelist():
            data = zin.read(name)
            if name.endswith("/data.pkl"):
                data = data.replace(old, new)
            zout.writestr(name, data)


def test_load_zip_non_cpu_location_with_dict_remap_to_cpu(tmp_path):
    src = tmp_path / "src_non_cpu_remap.pth"
    patched = tmp_path / "patched_non_cpu_remap.pth"
    torch.save({"x": torch.tensor([1.0, 2.0])}, src)
    _patch_checkpoint_location_tag(src, patched, old=b"cpu", new=b"npu")

    loaded = mt.load(patched, map_location={"npu": "cpu"})

    assert loaded["x"].tolist() == [1.0, 2.0]


def test_load_zip_non_cpu_location_without_remap_raises(tmp_path):
    src = tmp_path / "src_non_cpu_no_remap.pth"
    patched = tmp_path / "patched_non_cpu_no_remap.pth"
    torch.save({"x": torch.tensor([3.0])}, src)
    _patch_checkpoint_location_tag(src, patched, old=b"cpu", new=b"npu")

    with pytest.raises(RuntimeError):
        mt.load(patched)


def test_strict_map_location_callable_invoked_for_non_cpu_source(tmp_path):
    src = tmp_path / "src_strict_callable.pth"
    patched = tmp_path / "patched_strict_callable.pth"
    torch.save({"x": torch.tensor([5.0])}, src)
    _patch_checkpoint_location_tag(src, patched, old=b"cpu", new=b"npu")

    calls = []

    def mapper(storage, loc):
        calls.append(loc)
        return None

    with pytest.raises(RuntimeError):
        mt.load(patched, map_location=mapper)

    assert calls == ["npu"]


def test_strict_map_location_string_non_cpu_does_not_fallback_to_cpu(tmp_path):
    path = tmp_path / "strict_map_loc_string.pth"
    torch.save({"x": torch.tensor([7.0])}, path)

    with pytest.raises(RuntimeError):
        mt.load(path, map_location="npu:0")


def test_strict_map_location_cuda_string_does_not_fallback_to_cpu(tmp_path):
    path = tmp_path / "strict_map_loc_cuda.pth"
    torch.save({"x": torch.tensor([9.0])}, path)

    with pytest.raises(RuntimeError):
        mt.load(path, map_location="cuda:0")


def test_save_legacy_flag_false_is_explicitly_rejected(tmp_path):
    path = tmp_path / "legacy_flag_false.pth"
    with pytest.raises(NotImplementedError):
        mt.save({"x": mt.tensor([1.0])}, path, _use_new_zipfile_serialization=False)


def test_save_legacy_flag_true_still_writes_zip(tmp_path):
    path = tmp_path / "legacy_flag_true.pth"
    mt.save({"x": mt.tensor([1.0])}, path, _use_new_zipfile_serialization=True)
    assert zipfile.is_zipfile(path)


def test_strict_save_preserves_source_location_in_storage_ref():
    base = mt.tensor([1.0, 2.0])

    class _FakeTensor:
        device = mt.device("npu:0")
        requires_grad = False

        def detach(self):
            return base

    proxy = ser._tensor_to_proxy(_FakeTensor(), {})
    assert proxy.storage_ref.location == "npu:0"


def test_strict_save_cpu_source_location_stays_cpu():
    t = mt.tensor([3.0])
    proxy = ser._tensor_to_proxy(t, {})
    assert proxy.storage_ref.location == "cpu"


def test_torch_load_with_map_location_callable_returning_none(tmp_path):
    model = torch.nn.Linear(4, 3)
    path = tmp_path / "torch_state_dict_map_loc_callable_none.pth"
    torch.save(model.state_dict(), path)

    calls = []

    def mapper(storage, loc):
        calls.append((storage, loc))
        return None

    loaded = mt.load(path, map_location=mapper)

    assert isinstance(loaded, OrderedDict)
    assert set(loaded.keys()) == {"weight", "bias"}
    assert calls and {loc for _, loc in calls} == {"cpu"}


def test_torch_load_rejects_unsupported_map_location_target(tmp_path):
    model = torch.nn.Linear(4, 3)
    path = tmp_path / "torch_state_dict_map_loc_bad_target.pth"
    torch.save(model.state_dict(), path)

    with pytest.raises(RuntimeError, match="unsupported checkpoint storage location"):
        mt.load(path, map_location={"cpu": "cuda:0"})


def test_save_unsupported_dtype_uint16_raises(tmp_path):
    x = mt.tensor([1, 2, 3], dtype=mt.uint16)
    path = tmp_path / "mindtorch_uint16.pth"

    with pytest.raises(TypeError, match="unsupported dtype for serialization"):
        mt.save({"x": x}, path)


def test_load_zip_checkpoint_missing_data_pkl_raises(tmp_path):
    path = tmp_path / "bad_missing_data_pkl.pth"
    with zipfile.ZipFile(path, mode="w", compression=zipfile.ZIP_STORED) as zf:
        zf.writestr("archive/version", b"3")
        zf.writestr("archive/byteorder", b"little")

    with pytest.raises(RuntimeError, match="checkpoint missing data.pkl record"):
        mt.load(path)


def test_load_zip_checkpoint_unknown_typename_raises(tmp_path):
    seed_path = tmp_path / "good_unknown_typename_seed.pth"
    mt.save({"x": mt.tensor([1.0])}, seed_path)

    path = tmp_path / "bad_unknown_typename.pth"
    with zipfile.ZipFile(seed_path, mode="r") as src, zipfile.ZipFile(
        path, mode="w", compression=zipfile.ZIP_STORED
    ) as dst:
        for name in src.namelist():
            payload = src.read(name)
            if name.endswith("data.pkl"):
                assert b"storage" in payload
                payload = payload.replace(b"storage", b"notstor", 1)
            dst.writestr(name, payload)

    with pytest.raises(RuntimeError, match="Unknown typename for persistent_load"):
        mt.load(path)


def test_load_legacy_truncated_storage_payload_raises(tmp_path):
    tensor = torch.tensor([1.0, 2.0, 3.0], dtype=torch.float32)
    src_path = tmp_path / "legacy_ok.pth"
    torch.save({"x": tensor}, src_path, _use_new_zipfile_serialization=False)

    with open(src_path, "rb") as fh:
        original = fh.read()

    bad = original[:-1]

    bad_path = tmp_path / "legacy_truncated.pth"
    with open(bad_path, "wb") as fh:
        fh.write(bad)

    with pytest.raises(RuntimeError, match="truncated storage payload"):
        mt.load(bad_path)


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
    [
        torch.float16,
        torch.float32,
        torch.float64,
        torch.int8,
        torch.int16,
        torch.int32,
        torch.int64,
        torch.uint8,
        torch.bool,
        torch.complex64,
        torch.complex128,
    ],
)
def test_torch_to_mindtorch_dtype_matrix(tmp_path, dtype):
    if dtype is torch.bool:
        t = torch.tensor([[True, False], [False, True]], dtype=dtype)
    elif dtype in (torch.complex64, torch.complex128):
        base = torch.arange(4, dtype=torch.float32).reshape(2, 2)
        t = torch.complex(base, base + 1).to(dtype)
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
    [
        mt.float16,
        mt.float32,
        mt.float64,
        mt.int8,
        mt.int16,
        mt.int32,
        mt.int64,
        mt.uint8,
        mt.bool,
        mt.bfloat16,
        mt.complex64,
        mt.complex128,
    ],
)
def test_mindtorch_to_torch_dtype_matrix(tmp_path, dtype):
    if dtype == mt.bool:
        t = mt.tensor([[True, False], [False, True]], dtype=dtype)
    elif dtype in (mt.complex64, mt.complex128):
        real = mt.arange(0, 4, dtype=mt.float32).reshape((2, 2)).to(dtype=mt.float32)
        imag = mt.arange(10, 14, dtype=mt.float32).reshape((2, 2)).to(dtype=mt.float32)
        t = mt.tensor(real.numpy() + 1j * imag.numpy(), dtype=dtype)
    else:
        t = mt.arange(0, 4, dtype=mt.float32).reshape((2, 2)).to(dtype=dtype)
    path = tmp_path / f"mindtorch_dtype_{dtype.name}.pth"
    mt.save({"x": t}, path)

    loaded = torch.load(path, map_location="cpu")
    x = loaded["x"]

    assert tuple(x.shape) == tuple(t.shape)
    assert str(x.dtype).split(".")[-1] == dtype.name
    if dtype == mt.bfloat16:
        assert x.to(torch.float32).tolist() == t.to(dtype=mt.float32).tolist()
    else:
        assert x.tolist() == t.tolist()


@pytest.mark.parametrize("dtype", [mt.uint16, mt.uint32, mt.uint64])
def test_mindtorch_save_unsupported_unsigned_dtypes_raise(tmp_path, dtype):
    t = mt.tensor([1, 2, 3], dtype=dtype)
    path = tmp_path / f"mindtorch_{dtype.name}_unsupported.pth"

    with pytest.raises(TypeError, match="unsupported dtype for serialization"):
        mt.save({"x": t}, path)


def test_load_with_mmap_pathlike_zip_checkpoint(tmp_path):
    path = tmp_path / "mmap_zip_roundtrip.pth"
    mt.save({"x": mt.tensor([1.0, 2.0, 3.0])}, path)

    loaded = mt.load(path, mmap=True)

    assert loaded["x"].tolist() == [1.0, 2.0, 3.0]


def test_load_with_mmap_filelike_raises_value_error(tmp_path):
    path = tmp_path / "mmap_filelike.pth"
    mt.save({"x": mt.tensor([1.0])}, path)

    with open(path, "rb") as fh:
        with pytest.raises(ValueError, match="f must be a string filename"):
            mt.load(fh, mmap=True)


def test_load_with_mmap_pathlike_uses_memmap_backing(tmp_path):
    path = tmp_path / "mmap_backing_check.pth"
    mt.save({"x": mt.tensor([10.0, 20.0, 30.0])}, path)

    loaded = mt.load(path, mmap=True)
    arr = loaded["x"].storage().data

    assert isinstance(arr.base, np.memmap)
    assert loaded["x"].tolist() == [10.0, 20.0, 30.0]


def test_weights_only_rejects_torch_custom_global(tmp_path):
    path = tmp_path / "weights_only_reject_torch_custom.pth"
    torch.save({"x": torch.tensor([1.0]), "unsafe": _TorchUnsafeGlobal(1)}, path)

    with pytest.raises(pickle.UnpicklingError, match="weights_only"):
        mt.load(path, weights_only=True)


def test_weights_only_rejects_defaultdict_global(tmp_path):
    path = tmp_path / "weights_only_reject_defaultdict.pth"
    payload = {"x": torch.tensor([1.0]), "d": collections.defaultdict(int, a=1)}
    torch.save(payload, path)

    with pytest.raises(pickle.UnpicklingError, match="weights_only"):
        mt.load(path, weights_only=True)


def test_weights_only_false_allows_defaultdict_global(tmp_path):
    path = tmp_path / "weights_only_false_defaultdict.pth"
    payload = {"x": torch.tensor([1.0]), "d": collections.defaultdict(int, a=1)}
    torch.save(payload, path)

    loaded = mt.load(path, weights_only=False)

    assert loaded["x"].tolist() == [1.0]
    assert isinstance(loaded["d"], collections.defaultdict)
    assert loaded["d"]["a"] == 1


def test_weights_only_none_allows_defaultdict_global(tmp_path):
    path = tmp_path / "weights_only_none_defaultdict.pth"
    payload = {"x": torch.tensor([1.0]), "d": collections.defaultdict(int, a=2)}
    torch.save(payload, path)

    loaded = mt.load(path, weights_only=None)

    assert loaded["x"].tolist() == [1.0]
    assert isinstance(loaded["d"], collections.defaultdict)
    assert loaded["d"]["a"] == 2


def test_load_honors_custom_pickle_module_on_zip_path(tmp_path):
    _CountingPickleModule.load_calls = 0
    _CountingPickleModule.unpickler_inits = 0

    path = tmp_path / "custom_pickle_zip.pth"
    mt.save({"x": mt.tensor([1.0])}, path)

    loaded = mt.load(path, pickle_module=_CountingPickleModule)

    assert loaded["x"].tolist() == [1.0]
    assert _CountingPickleModule.unpickler_inits >= 1


def test_load_honors_custom_pickle_module_on_legacy_path(tmp_path):
    _CountingPickleModule.load_calls = 0
    _CountingPickleModule.unpickler_inits = 0

    path = tmp_path / "custom_pickle_legacy.pth"
    torch.save(
        {"x": torch.tensor([1.0])},
        path,
        _use_new_zipfile_serialization=False,
    )

    loaded = mt.load(path, pickle_module=_CountingPickleModule)

    assert loaded["x"].tolist() == [1.0]
    assert _CountingPickleModule.load_calls >= 1
    assert _CountingPickleModule.unpickler_inits >= 1
