import mindtorch_v2 as torch


def test_dispatch_prefers_meta_when_input_meta():
    a = torch.ones((2,), device="meta")
    b = torch.ones((2,), device="meta")
    c = torch.add(a, b)
    assert c.device.type == "meta"


def test_dispatch_prefers_npu_over_cpu():
    a = torch.ones((2,), device="npu")
    b = torch.ones((2,), device="npu")
    c = torch.add(a, b)
    assert c.device.type == "npu"
