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


def test_autocast_dot_mixed_dtype_matches_torch_error_semantics():
    a = torch.randn((8,), dtype=torch.float32)
    b = torch.randn((8,), dtype=torch.float16)
    with torch.amp.autocast("cpu", dtype=torch.bfloat16):
        with pytest.raises(RuntimeError, match="same dtype"):
            torch.dot(a, b)


def test_autocast_tensordot_mixed_dtype_matches_torch_error_semantics():
    a = torch.randn((4, 4), dtype=torch.float32)
    b = torch.randn((4, 4), dtype=torch.float16)
    with torch.amp.autocast("cpu", dtype=torch.bfloat16):
        with pytest.raises(RuntimeError, match="same dtype"):
            torch.tensordot(a, b, ([1], [0]))


def test_autocast_cross_mixed_dtype_matches_torch_error_semantics():
    a = torch.randn((4, 3), dtype=torch.float32)
    b = torch.randn((4, 3), dtype=torch.float16)
    with torch.amp.autocast("cpu", dtype=torch.bfloat16):
        with pytest.raises(RuntimeError, match="same dtype"):
            torch.cross(a, b, dim=1)
