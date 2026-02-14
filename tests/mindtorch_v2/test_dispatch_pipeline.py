import mindtorch_v2 as torch


def test_pipeline_defers_execution():
    with torch.pipeline() as pipe:
        a = torch.ones((2,))
        b = torch.ones((2,))
        c = torch.add(a, b)
        assert getattr(c, "_pending", False) is True
        pipe.flush()
        assert getattr(c, "_pending", False) is False


def test_pipeline_flushes_on_backward():
    with torch.pipeline():
        a = torch.ones((2,))
        a.requires_grad = True
        b = torch.sum(a)
        assert getattr(b, "_pending", False) is True
        b.backward()
        assert getattr(b, "_pending", False) is False
        assert a.grad is not None
