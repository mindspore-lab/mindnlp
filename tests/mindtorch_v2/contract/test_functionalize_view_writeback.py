import mindtorch_v2 as torch


def test_functionalize_writeback_respects_view():
    base = torch.tensor([1.0, 2.0, 3.0, 4.0])
    view = base.view((2, 2))
    with torch.functionalize():
        view.add_(torch.ones((2, 2)))
    assert base.storage().data.tolist() == [2.0, 3.0, 4.0, 5.0]
