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


def test_pipeline_handles_multi_output():
    with torch.pipeline() as pipe:
        x = torch.tensor([1.0, 2.0])
        values, indices = torch.cummax(x, dim=0)
        assert getattr(values, "_pending", False) is True
        assert getattr(indices, "_pending", False) is True
        pipe.flush()
        assert getattr(values, "_pending", False) is False
        assert getattr(indices, "_pending", False) is False
