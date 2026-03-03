import mindtorch_v2 as torch


def test_pipeline_context_records_ops():
    a = torch.tensor([1.0, 2.0])
    b = torch.tensor([3.0, 4.0])
    with torch.pipeline():
        c = a + b
        d = c * b
        assert c._pending is True
        assert d._pending is True
        assert c.device.type == "cpu"
        assert d.device.type == "cpu"


def test_pipeline_dispatch_marks_pending():
    a = torch.tensor([1.0, 2.0])
    b = torch.tensor([3.0, 4.0])
    with torch.pipeline():
        c = a + b
        assert c._pending is True
    assert c._pending is False


def test_pipeline_flush_on_to_cpu():
    a = torch.tensor([1.0, 2.0])
    b = torch.tensor([3.0, 4.0])
    with torch.pipeline():
        c = a + b
        out = c.to("cpu")
        assert c._pending is False
    assert out.storage().data.tolist() == [4.0, 6.0]


def test_pipeline_meta_shapes():
    a = torch.tensor([[1.0, 2.0]])
    b = torch.tensor([[3.0, 4.0]])
    with torch.pipeline():
        c = a + b
        d = c.relu()
        e = d.sum()
        assert c._pending is True
        assert d._pending is True
        assert e._pending is True
        assert c.shape == a.shape
        assert d.shape == a.shape
        assert e.shape == ()
        assert c.device.type == "cpu"
    assert c._pending is False
    assert d._pending is False
    assert e._pending is False
