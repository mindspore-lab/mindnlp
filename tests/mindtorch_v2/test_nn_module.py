# tests/mindtorch_v2/test_nn_module.py
import numpy as np
import mindtorch_v2 as torch
from mindtorch_v2.nn import Module, Parameter


class SimpleModule(Module):
    def __init__(self):
        super().__init__()
        self.weight = Parameter(torch.randn(3, 4))
        self.bias = Parameter(torch.randn(4))

    def forward(self, x):
        return x @ self.weight + self.bias


def test_module_parameters():
    """Module tracks parameters."""
    m = SimpleModule()
    params = list(m.parameters())
    assert len(params) == 2


def test_module_named_parameters():
    """Module provides named parameters."""
    m = SimpleModule()
    named = dict(m.named_parameters())
    assert 'weight' in named
    assert 'bias' in named


def test_module_call():
    """Module is callable and runs forward."""
    m = SimpleModule()
    x = torch.randn(2, 3)
    y = m(x)
    assert y.shape == (2, 4)


def test_module_train_eval():
    """Module has train/eval modes."""
    m = SimpleModule()
    assert m.training == True
    m.eval()
    assert m.training == False
    m.train()
    assert m.training == True


def test_module_to_device():
    """Module.to() returns self."""
    m = SimpleModule()
    result = m.to('cpu')
    assert result is m


def test_module_children():
    """Module tracks child modules."""
    class Parent(Module):
        def __init__(self):
            super().__init__()
            self.child = SimpleModule()
        def forward(self, x):
            return self.child(x)

    p = Parent()
    children = list(p.children())
    assert len(children) == 1
    assert isinstance(children[0], SimpleModule)


def test_module_modules():
    """Module.modules() returns all modules recursively."""
    class Parent(Module):
        def __init__(self):
            super().__init__()
            self.child = SimpleModule()
        def forward(self, x):
            return self.child(x)

    p = Parent()
    modules = list(p.modules())
    assert len(modules) == 2  # Parent + SimpleModule
