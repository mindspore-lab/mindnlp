import mindtorch_v2._backends.ascend as ascend
import mindtorch_v2._backends.npu as npu


def test_backend_shim_imports():
    assert ascend.is_available is npu.is_available
