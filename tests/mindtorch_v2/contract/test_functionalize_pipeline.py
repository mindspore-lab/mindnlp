import mindtorch_v2 as torch


def test_pipeline_records_functionalized_inplace():
    a = torch.tensor([1.0, 2.0])
    b = torch.tensor([3.0, 4.0])
    with torch.functionalize():
        with torch.pipeline():
            out = a.add_(b)
            assert out._pending is True
    assert out._pending is False
