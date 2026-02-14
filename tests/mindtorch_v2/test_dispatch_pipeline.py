import mindtorch_v2 as torch


def test_pipeline_defers_execution():
    with torch.pipeline() as pipe:
        a = torch.ones((2,))
        b = torch.ones((2,))
        c = torch.add(a, b)
        assert getattr(c, "_pending", False) is True
        pipe.flush()
        assert getattr(c, "_pending", False) is False
