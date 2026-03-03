import mindtorch_v2 as torch
from mindtorch_v2 import nn, optim


def test_sgd_step():
    layer = nn.Linear(2, 1)
    opt = optim.SGD(layer.parameters(), lr=0.1)
    x = torch.tensor([[1.0, 2.0]])
    y = layer(x)
    y.sum().backward()
    opt.step()
    assert layer.weight.grad is not None
