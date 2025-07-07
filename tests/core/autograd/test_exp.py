from unittest import TestCase

import numpy as np
import mindnlp
from mindnlp.core import tensor


class TestExp(TestCase):

    def test_exp(self):
        # scalar exp
        t1 = tensor(2.0)
        t2 = t1.exp()
        np.testing.assert_allclose(t2.array, np.exp(2))

        t1 = tensor(2.0, requires_grad=True)
        t2 = t1.exp()
        t2.backward()
        np.testing.assert_allclose(t1.grad.array, np.exp(2))

        # vector exp
        t1 = tensor([1.0, 2.0])
        t2 = t1.exp()
        np.testing.assert_allclose(t2.array, np.exp([1, 2]))

        t1 = tensor([1.0, 2.0], requires_grad=True)
        t2 = t1.exp()
        t2.backward(tensor([1.0, 1.0]))
        np.testing.assert_allclose(t1.grad.array, np.exp([1, 2]))