from unittest import TestCase
import mindnlp
from mindnlp.core import tensor


class TestAdd(TestCase):

    def test_simple_add(self):
        # scalar add
        t1 = tensor(1.0)
        t2 = tensor(2.0)
        t3 = t1 + t2
        self.assertEqual(t3.data.tolist(), 3.0)

        t1 = tensor(2.0, requires_grad=True)
        t2 = tensor(3.0)
        t3 = t1 + t2
        print(t1.grad_fn, t2.grad_fn)
        t3.backward()
        print(t3.grad_fn)
        self.assertEqual(t1.grad.data.tolist(), 1.0)

        t1 = tensor(2.0)
        t2 = tensor(3.0, requires_grad=True)
        t3 = t1 + t2
        t3.backward()
        self.assertEqual(t2.grad.data.tolist(), 1.0)

        t1 = tensor(2.0, requires_grad=True)
        t2 = tensor(3.0, requires_grad=True)
        t3 = t1 + t2
        t3.backward()
        self.assertEqual(t1.grad.data.tolist(), 1.0)
        self.assertEqual(t2.grad.data.tolist(), 1.0)

        # vector add
        t1 = tensor([1.0, 2.0])
        t2 = tensor([2.0, 3.0])
        t3 = t1 + t2
        self.assertEqual(t3.data.tolist(), [3.0, 5.0])

        t1 = tensor([1.0, 2.0], requires_grad=True)
        t2 = tensor([2.0, 3.0])
        t3 = t1 + t2
        t3.backward(tensor([1.0, 1.0]))
        self.assertEqual(t1.grad.data.tolist(), [1.0, 1.0])

        t1 = tensor([1.0, 2.0])
        t2 = tensor([2.0, 3.0], requires_grad=True)
        t3 = t1 + t2
        t3.backward(tensor([1.0, 1.0]))
        self.assertEqual(t2.grad.data.tolist(), [1.0, 1.0])

        t1 = tensor([1.0, 2.0], requires_grad=True)
        t2 = tensor([2.0, 3.0], requires_grad=True)
        t3 = t1 + t2
        t3.backward(tensor([1.0, 1.0]))
        self.assertEqual(t1.grad.data.tolist(), [1.0, 1.0])
        self.assertEqual(t2.grad.data.tolist(), [1.0, 1.0])

    def test_broadcast_add(self):
        # (2,) + ()
        t1 = tensor([1.0, 2.0], requires_grad=True)
        t2 = tensor(2.0, requires_grad=True)
        t3 = t1 + t2
        t3.backward(tensor([1.0, 1.0]))
        self.assertEqual(t1.grad.data.tolist(), [1.0, 1.0])
        self.assertEqual(t2.grad.data.tolist(), 2.0)

        # (2,) + (1,)
        t1 = tensor([1.0, 2.0], requires_grad=True)
        t2 = tensor([2.0], requires_grad=True)
        t3 = t1 + t2
        t3.backward(tensor([1.0, 1.0]))
        self.assertEqual(t1.grad.data.tolist(), [1.0, 1.0])
        self.assertEqual(t2.grad.data.tolist(), [2.0])

        # (2, 2) + ()
        t1 = tensor([[1.0, 2.0], [3.0, 4.0]], requires_grad=True)
        t2 = tensor(2.0, requires_grad=True)
        t3 = t1 + t2
        t3.backward(tensor([[1.0, 1.0], [1.0, 1.0]]))
        self.assertEqual(t1.grad.data.tolist(), [[1.0, 1.0], [1.0, 1.0]])
        self.assertEqual(t2.grad.data.tolist(), 4.0)

        # (2, 2) + (1,)
        t1 = tensor([[1.0, 2.0], [3.0, 4.0]], requires_grad=True)
        t2 = tensor([2.0], requires_grad=True)
        t3 = t1 + t2
        t3.backward(tensor([[1.0, 1.0], [1.0, 1.0]]))
        self.assertEqual(t1.grad.data.tolist(), [[1.0, 1.0], [1.0, 1.0]])
        self.assertEqual(t2.grad.data.tolist(), [4.0])

        # (2, 2) + (2, )
        t1 = tensor([[1.0, 2.0], [3.0, 4.0]], requires_grad=True)
        t2 = tensor([2.0, 3.0], requires_grad=True)
        t3 = t1 + t2
        t3.backward(tensor([[1.0, 1.0], [1.0, 1.0]]))
        self.assertEqual(t1.grad.data.tolist(), [[1.0, 1.0], [1.0, 1.0]])
        self.assertEqual(t2.grad.data.tolist(), [2.0, 2.0])