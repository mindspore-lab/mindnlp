import pytest
import mindtorch_v2 as torch


def test_autocast_fp32_policy_softmax():
    if not hasattr(torch, "softmax"):
        pytest.skip("softmax not available")
    x = torch.randn((4, 4), dtype=torch.float32)
    with torch.amp.autocast("cpu", dtype=torch.bfloat16):
        out = torch.softmax(x, dim=0)
    assert out.dtype == torch.float32


def test_autocast_fp16_policy_matmul():
    x = torch.randn((4, 4), dtype=torch.float32)
    with torch.amp.autocast("cpu", dtype=torch.bfloat16):
        out = torch.matmul(x, x)
    assert out.dtype == torch.bfloat16
