import mindtorch_v2 as torch


def test_tensor_repr_cpu_default_dtype():
    t = torch.ones((2, 2))
    rep = repr(t)
    assert rep.startswith("tensor(")
    assert "dtype=" not in rep
    assert "device=" not in rep
    assert rep == str(t)


def test_tensor_repr_cpu_non_default_dtype():
    t = torch.ones((2, 2), dtype=torch.float16)
    rep = repr(t)
    assert "dtype=torch.float16" in rep


def test_tensor_repr_meta_includes_device():
    t = torch.ones((2, 2), device="meta")
    rep = repr(t)
    assert rep.startswith("tensor(")
    assert "..." in rep
    assert "device='meta'" in rep
    assert "dtype=torch.float32" in rep


def test_tensor_repr_respects_precision():
    prev = torch.get_printoptions()
    try:
        torch.set_printoptions(precision=2)
        t = torch.tensor([1.23456])
        rep = repr(t)
        assert "1.23" in rep
    finally:
        torch.set_printoptions(**prev)


def test_tensor_repr_npu_includes_device():
    if not torch.npu.is_available():
        return
    t = torch.ones((1,))
    t = t.to("npu")
    rep = repr(t)
    assert "device='npu'" in rep
    assert "1" in rep


def test_tensor_repr_npu_index():
    if not torch.npu.is_available():
        return
    if torch._C._npu_device_count() < 2:
        return
    t = torch.ones((1,), device="npu:1")
    rep = repr(t)
    assert "device='npu:1'" in rep
