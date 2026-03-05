import mindtorch_v2 as torch
import mindtorch_v2.nn.functional as F


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


def test_pipeline_context_max_ops_auto_flush():
    a = torch.tensor([1.0, 2.0])
    b = torch.tensor([3.0, 4.0])
    with torch.pipeline(max_ops=1):
        c = a + b
        # max_ops=1 should flush immediately after recording this op.
        assert c._pending is False
        out = c.to("cpu")
    assert out.storage().data.tolist() == [4.0, 6.0]


def test_pipeline_min_defer_ops_gates_deferral():
    a = torch.tensor([1.0, 2.0])
    b = torch.tensor([3.0, 4.0])
    with torch.pipeline(min_defer_ops=2):
        c = a + b
        assert c._pending is False
        d = c + b
        assert d._pending is True
    assert d._pending is False


def test_pipeline_adaptive_defer_uses_prev_window_prediction():
    a = torch.tensor([1.0, 2.0])
    b = torch.tensor([3.0, 4.0])
    old_cfg = torch.get_pipeline_config()
    try:
        torch.pipeline_config(adaptive_defer=True, adaptive_small_window_ops=4, adaptive_min_defer_ops=3, min_defer_ops=None)

        # First window has no prediction yet, so first op is deferred.
        with torch.pipeline():
            x = a + b
            assert x._pending is True
            y = x + b
            assert y._pending is True

        # Next window uses previous window size (2 <= 4), so first two ops go eager.
        with torch.pipeline():
            c = a + b
            assert c._pending is False
            d = c + b
            assert d._pending is False
            e = d + b
            assert e._pending is True
        assert e._pending is False
    finally:
        torch.pipeline_config(**old_cfg)


def test_pipeline_global_config_applies_max_ops():
    old_cfg = torch.get_pipeline_config()
    try:
        torch.pipeline_config(max_ops=1)
        a = torch.tensor([1.0, 2.0])
        b = torch.tensor([3.0, 4.0])
        with torch.pipeline():
            c = a + b
            assert c._pending is False
    finally:
        torch.pipeline_config(**old_cfg)


def test_pipeline_softmax_meta_accepts_dim():
    x = torch.randn((2, 4))
    with torch.pipeline():
        y = F.softmax(x, dim=-1)
        assert y._pending is True
        assert y.shape == x.shape
    assert y._pending is False


def test_pipeline_layer_norm_meta_accepts_args():
    x = torch.randn((2, 4))
    with torch.pipeline():
        y = F.layer_norm(x, (4,))
        assert y._pending is True
        assert y.shape == x.shape
    assert y._pending is False
