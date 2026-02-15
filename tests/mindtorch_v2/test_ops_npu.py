import numpy as np
import pytest
import mindtorch_v2 as torch


def test_npu_add():
    if not torch.npu.is_available():
        pytest.skip("NPU not available")
    x = torch.tensor([1.0, 2.0]).to("npu")
    y = torch.tensor([3.0, 4.0]).to("npu")
    z = torch.add(x, y).to("cpu")
    assert z.storage().data.tolist() == [4.0, 6.0]


def test_npu_matmul_2d():
    if not torch.npu.is_available():
        pytest.skip("NPU not available")
    a = torch.tensor([[1.0, 2.0], [3.0, 4.0]], device="npu", dtype=torch.float16)
    b = torch.tensor([[1.0, 0.0], [0.0, 1.0]], device="npu", dtype=torch.float16)
    out = torch.matmul(a, b)
    assert out.device.type == "npu"
    assert np.allclose(out.to("cpu").numpy(), np.matmul(a.to("cpu").numpy(), b.to("cpu").numpy()))


def test_npu_matmul_1d_2d():
    if not torch.npu.is_available():
        pytest.skip("NPU not available")
    a = torch.tensor([1.0, 2.0, 3.0], device="npu", dtype=torch.float16)
    b = torch.tensor([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]], device="npu", dtype=torch.float16)
    out = torch.matmul(a, b)
    assert out.shape == (2,)
    assert np.allclose(out.to("cpu").numpy(), np.matmul(a.to("cpu").numpy(), b.to("cpu").numpy()))


def test_npu_matmul_batched_broadcast():
    if not torch.npu.is_available():
        pytest.skip("NPU not available")
    a = torch.tensor(
        np.arange(2 * 1 * 2 * 3, dtype=np.float16).reshape(2, 1, 2, 3),
        device="npu",
        dtype=torch.float16,
    )
    b = torch.tensor(
        np.arange(1 * 4 * 3 * 5, dtype=np.float16).reshape(1, 4, 3, 5),
        device="npu",
        dtype=torch.float16,
    )
    out = torch.matmul(a, b)
    assert out.shape == (2, 4, 2, 5)
    assert np.allclose(out.to("cpu").numpy(), np.matmul(a.to("cpu").numpy(), b.to("cpu").numpy()))


def test_npu_model_dir_probe():
    if not torch.npu.is_available():
        pytest.skip("NPU not available")
    ok = torch._C._npu_probe_model_dirs()
    assert ok is True


def test_npu_model_dir_selected():
    if not torch.npu.is_available():
        pytest.skip("NPU not available")
    path = torch._C._npu_model_dir()
    assert path in {
        "/usr/local/Ascend/ascend-toolkit/latest/opp",
        "/home/lvyufeng/lvyufeng/acl_engine",
    }


def test_npu_aclnn_available():
    if not torch.npu.is_available():
        pytest.skip("NPU not available")
    assert torch._C._npu_aclnn_available() is True


def test_aclnn_symbols_present():
    if not torch.npu.is_available():
        pytest.skip("NPU not available")
    assert torch._C._npu_aclnn_symbols_ok() is True


def test_aclnn_ones_zero_symbols_present():
    if not torch.npu.is_available():
        pytest.skip("NPU not available")
    assert torch._C._npu_aclnn_ones_zero_ok() is True


def test_npu_add_execute():
    if not torch.npu.is_available():
        pytest.skip("NPU not available")
    a = torch.tensor([1.0, 2.0], device="npu")
    b = torch.tensor([3.0, 4.0], device="npu")
    out = a + b
    assert out.device.type == "npu"
    assert out.to("cpu").numpy().tolist() == [4.0, 6.0]


def test_npu_mul_relu():
    if not torch.npu.is_available():
        pytest.skip("NPU not available")
    a = torch.tensor([-1.0, 2.0], device="npu")
    b = torch.tensor([3.0, 4.0], device="npu")
    prod = a * b
    relu = a.relu()
    assert prod.device.type == "npu"
    assert relu.device.type == "npu"
    assert prod.to("cpu").numpy().tolist() == [-3.0, 8.0]
    assert relu.to("cpu").numpy().tolist() == [0.0, 2.0]


def test_npu_sum():
    if not torch.npu.is_available():
        pytest.skip("NPU not available")
    a = torch.tensor([[1.0, 2.0]], device="npu")
    total = a.sum()
    kept = a.sum(dim=1, keepdim=True)
    assert total.device.type == "npu"
    assert kept.device.type == "npu"
    assert total.to("cpu").numpy().tolist() == 3.0
    assert kept.to("cpu").numpy().tolist() == [[3.0]]

def test_npu_device_index_preserved():
    if not torch.npu.is_available():
        pytest.skip("NPU not available")
    out = torch.ones((1,), device="npu:0")
    assert out.device.type == "npu"
    assert out.device.index == 0


def test_npu_cross_device_copy():
    if not torch.npu.is_available():
        pytest.skip("NPU not available")
    if torch._C._npu_device_count() < 2:
        pytest.skip("Need 2 NPUs")
    src = torch.ones((2,), device="npu:0")
    dst = src.to("npu:1")
    assert dst.device.index == 1
    assert dst.to("cpu").numpy().tolist() == [1.0, 1.0]


def test_npu_ones():
    if not torch.npu.is_available():
        pytest.skip("NPU not available")
    out = torch.ones((1,), device="npu")
    assert out.device.type == "npu"
    assert out.to("cpu").numpy().tolist() == [1.0]


def test_npu_zeros():
    if not torch.npu.is_available():
        pytest.skip("NPU not available")
    out = torch.zeros((2,), device="npu")
    assert out.device.type == "npu"
    assert out.to("cpu").numpy().tolist() == [0.0, 0.0]


def test_npu_to_cpu_synchronizes(monkeypatch):
    if not torch.npu.is_available():
        pytest.skip("NPU not available")
    calls = []

    from mindtorch_v2._backends.npu import runtime as npu_runtime
    runtime = npu_runtime.get_runtime(0)

    def fake_sync():
        calls.append("sync")

    monkeypatch.setattr(runtime, "synchronize", fake_sync)

    t = torch.ones((1,), device="npu")
    _ = t.to("cpu")
    assert "sync" in calls
