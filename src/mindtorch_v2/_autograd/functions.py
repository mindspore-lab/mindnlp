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


class TransposeBackward(Node):
    """Backward for transpose."""

    def __init__(self):
        super().__init__()
        self._name = "TransposeBackward"

    def backward(self, grad_outputs):
        grad_output = grad_outputs[0]
        dim0, dim1 = self._dims
        # Transpose gradient back
        return (grad_output.transpose(dim0, dim1),)


class EmbeddingBackward(Node):
    """Backward for embedding lookup."""

    def __init__(self):
        super().__init__()
        self._name = "EmbeddingBackward"

    def backward(self, grad_outputs):
        grad_output = grad_outputs[0]
        indices, weight = self.saved_tensors
        num_embeddings = weight.shape[0]
        embedding_dim = weight.shape[1]

        # Create zero gradient for weight
        import numpy as np
        from .._tensor import Tensor
        grad_weight_np = np.zeros((num_embeddings, embedding_dim), dtype=np.float32)

        # Scatter add: grad_weight[indices] += grad_output
        indices_np = indices.numpy().astype(np.int64).flatten()
        grad_output_np = grad_output.numpy().reshape(-1, embedding_dim)

        for i, idx in enumerate(indices_np):
            grad_weight_np[idx] += grad_output_np[i]

        grad_weight = Tensor(grad_weight_np)
        # Indices don't have gradients
        return (None, grad_weight)


class LayerNormBackward(Node):
    """Backward for layer normalization."""

    def __init__(self):
        super().__init__()
        self._name = "LayerNormBackward"

    def backward(self, grad_outputs):
        grad_output = grad_outputs[0]
        input_tensor, normalized_shape, weight, bias, eps = self._saved_info

        import numpy as np
        from .._tensor import Tensor

        x = input_tensor.numpy()
        grad_out = grad_output.numpy()

        # Determine axes to normalize over
        ndim = len(normalized_shape)
        axes = tuple(range(-ndim, 0))
        normalized_size = 1
        for s in normalized_shape:
            normalized_size *= s

        # Recompute forward values
        mean = np.mean(x, axis=axes, keepdims=True)
        var = np.var(x, axis=axes, keepdims=True)
        std = np.sqrt(var + eps)
        x_norm = (x - mean) / std

        # Gradient w.r.t. weight and bias
        grad_weight = None
        grad_bias = None
        if weight is not None:
            # Sum over all axes except the normalized ones
            sum_axes = tuple(range(x.ndim - ndim))
            grad_weight = Tensor(np.sum(grad_out * x_norm, axis=sum_axes))
        if bias is not None:
            sum_axes = tuple(range(x.ndim - ndim))
            grad_bias = Tensor(np.sum(grad_out, axis=sum_axes))

        # Gradient w.r.t. input
        if weight is not None:
            grad_out = grad_out * weight.numpy()

        # d(norm)/dx = (1/std) * (I - (1/N) - (1/N) * x_norm * x_norm)
        grad_x_norm = grad_out
        grad_var = np.sum(grad_x_norm * (x - mean) * -0.5 * (var + eps) ** (-1.5), axis=axes, keepdims=True)
        grad_mean = np.sum(grad_x_norm * -1 / std, axis=axes, keepdims=True) + grad_var * np.mean(-2 * (x - mean), axis=axes, keepdims=True)
        grad_x = grad_x_norm / std + grad_var * 2 * (x - mean) / normalized_size + grad_mean / normalized_size

        return (Tensor(grad_x.astype(np.float32)), None, grad_weight, grad_bias, None)


class ReluBackward(Node):
    """Backward for ReLU."""

    def __init__(self):
        super().__init__()
        self._name = "ReluBackward"

    def backward(self, grad_outputs):
        grad_output = grad_outputs[0]
        input_tensor = self.saved_tensors[0]

        import numpy as np
        from .._tensor import Tensor

        x = input_tensor.numpy()
        grad_out = grad_output.numpy()

        # ReLU derivative: 1 if x > 0, else 0
        grad_x = grad_out * (x > 0).astype(np.float32)
        return (Tensor(grad_x),)


class GeluBackward(Node):
    """Backward for GELU."""

    def __init__(self):
        super().__init__()
        self._name = "GeluBackward"

    def backward(self, grad_outputs):
        grad_output = grad_outputs[0]
        input_tensor = self.saved_tensors[0]
        approximate = self._approximate

        import numpy as np
        import math
        from .._tensor import Tensor

        x = input_tensor.numpy()
        grad_out = grad_output.numpy()

        if approximate == 'tanh':
            # Approximate GELU derivative
            coef = math.sqrt(2.0 / math.pi)
            inner = coef * (x + 0.044715 * x ** 3)
            tanh_inner = np.tanh(inner)
            sech2 = 1 - tanh_inner ** 2
            grad_x = 0.5 * (1 + tanh_inner) + 0.5 * x * sech2 * coef * (1 + 3 * 0.044715 * x ** 2)
        else:
            # Exact GELU derivative
            from scipy.special import erf
            cdf = 0.5 * (1 + erf(x / math.sqrt(2)))
            pdf = np.exp(-0.5 * x ** 2) / math.sqrt(2 * math.pi)
            grad_x = cdf + x * pdf

        grad_x = grad_out * grad_x
        return (Tensor(grad_x.astype(np.float32)),)


class SiluBackward(Node):
    """Backward for SiLU/Swish."""

    def __init__(self):
        super().__init__()
        self._name = "SiluBackward"

    def backward(self, grad_outputs):
        grad_output = grad_outputs[0]
        input_tensor = self.saved_tensors[0]

        import numpy as np
        from .._tensor import Tensor

        x = input_tensor.numpy()
        grad_out = grad_output.numpy()

        # SiLU = x * sigmoid(x)
        # d/dx SiLU = sigmoid(x) + x * sigmoid(x) * (1 - sigmoid(x))
        #           = sigmoid(x) * (1 + x * (1 - sigmoid(x)))
        sigmoid_x = 1.0 / (1.0 + np.exp(-x))
        grad_x = grad_out * (sigmoid_x * (1 + x * (1 - sigmoid_x)))
        return (Tensor(grad_x.astype(np.float32)),)
