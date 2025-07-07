from unittest import TestCase
import mindnlp
from mindnlp import core as torch
from mindnlp.core import tensor


class TestSplit(TestCase):
    def test_simple_split(self):
        x = torch.randn(3, 2)
        y1, y2 = x.tensor_split(2, -1)
        assert y1.shape == (3, 1)
        assert y2.shape == (3, 1)

    def test_split_backward(self):
        # scalar add
        x = torch.randn(3, 2, requires_grad=True)
        y1, y2 = x.tensor_split(2, -1)
        assert y1.shape == (3, 1)
        assert y2.shape == (3, 1)
        z = y1 + y2
        z.sum().backward()
        print(x.grad)
