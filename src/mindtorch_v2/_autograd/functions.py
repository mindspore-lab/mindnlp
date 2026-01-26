"""Backward functions for autograd."""

from .node import Node


class AddBackward(Node):
    """Backward for element-wise addition."""

    def __init__(self):
        super().__init__()
        self._name = "AddBackward"

    def backward(self, grad_outputs):
        grad_output = grad_outputs[0]
        return (grad_output, grad_output)


class SubBackward(Node):
    """Backward for element-wise subtraction."""

    def __init__(self):
        super().__init__()
        self._name = "SubBackward"

    def backward(self, grad_outputs):
        grad_output = grad_outputs[0]
        from .. import neg
        return (grad_output, neg(grad_output))


class MulBackward(Node):
    """Backward for element-wise multiplication."""

    def __init__(self):
        super().__init__()
        self._name = "MulBackward"

    def backward(self, grad_outputs):
        grad_output = grad_outputs[0]
        a, b = self.saved_tensors
        from .. import mul
        return (mul(grad_output, b), mul(grad_output, a))


class DivBackward(Node):
    """Backward for element-wise division."""

    def __init__(self):
        super().__init__()
        self._name = "DivBackward"

    def backward(self, grad_outputs):
        grad_output = grad_outputs[0]
        a, b = self.saved_tensors
        from .. import div, mul, neg, pow
        grad_a = div(grad_output, b)
        grad_b = neg(div(mul(grad_output, a), pow(b, 2)))
        return (grad_a, grad_b)


class NegBackward(Node):
    """Backward for negation."""

    def __init__(self):
        super().__init__()
        self._name = "NegBackward"

    def backward(self, grad_outputs):
        grad_output = grad_outputs[0]
        from .. import neg
        return (neg(grad_output),)


class PowBackward(Node):
    """Backward for power."""

    def __init__(self):
        super().__init__()
        self._name = "PowBackward"

    def backward(self, grad_outputs):
        grad_output = grad_outputs[0]
        base, exp = self.saved_tensors
        from .. import mul, pow, sub
        from .._tensor import Tensor

        exp_minus_1 = sub(exp, Tensor(1.0))
        grad_base = mul(mul(grad_output, exp), pow(base, exp_minus_1))
        return (grad_base, None)


class SumBackward(Node):
    """Backward for sum reduction."""

    def __init__(self):
        super().__init__()
        self._name = "SumBackward"

    def backward(self, grad_outputs):
        grad_output = grad_outputs[0]
        input_shape = self._input_shape
        from .._creation import ones
        from .. import mul
        grad = mul(ones(input_shape), grad_output)
        return (grad,)


class MeanBackward(Node):
    """Backward for mean reduction."""

    def __init__(self):
        super().__init__()
        self._name = "MeanBackward"

    def backward(self, grad_outputs):
        grad_output = grad_outputs[0]
        input_shape = self._input_shape
        numel = 1
        for s in input_shape:
            numel *= s
        from .._creation import ones
        from .. import mul, div
        from .._tensor import Tensor
        grad = div(mul(ones(input_shape), grad_output), Tensor(float(numel)))
        return (grad,)


class MatmulBackward(Node):
    """Backward for matrix multiplication."""

    def __init__(self):
        super().__init__()
        self._name = "MatmulBackward"

    def backward(self, grad_outputs):
        grad_output = grad_outputs[0]
        a, b = self.saved_tensors
        from .. import matmul
        grad_a = matmul(grad_output, b.t())
        grad_b = matmul(a.t(), grad_output)
        return (grad_a, grad_b)


class ExpBackward(Node):
    """Backward for exp."""

    def __init__(self):
        super().__init__()
        self._name = "ExpBackward"

    def backward(self, grad_outputs):
        grad_output = grad_outputs[0]
        result = self.saved_tensors[0]
        from .. import mul
        return (mul(grad_output, result),)


class LogBackward(Node):
    """Backward for log."""

    def __init__(self):
        super().__init__()
        self._name = "LogBackward"

    def backward(self, grad_outputs):
        grad_output = grad_outputs[0]
        x = self.saved_tensors[0]
        from .. import div
        return (div(grad_output, x),)


class SqrtBackward(Node):
    """Backward for sqrt."""

    def __init__(self):
        super().__init__()
        self._name = "SqrtBackward"

    def backward(self, grad_outputs):
        grad_output = grad_outputs[0]
        result = self.saved_tensors[0]
        from .. import div, mul
        from .._tensor import Tensor
        return (div(grad_output, mul(Tensor(2.0), result)),)
