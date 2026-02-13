import pytest
import mindtorch_v2 as torch


def test_npu_add():
    if not torch.npu.is_available():
        pytest.skip("NPU not available")
    x = torch.tensor([1.0, 2.0]).to("npu")
    y = torch.tensor([3.0, 4.0]).to("npu")
    z = torch.add(x, y).to("cpu")
    assert z.storage().data.tolist() == [4.0, 6.0]


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
