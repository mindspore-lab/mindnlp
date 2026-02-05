"""Backward functions for autograd."""

from .node import Node


class AddBackward(Node):
    """Backward for element-wise addition."""

    def __init__(self):
        super().__init__()
        self._name = "AddBackward"

    def backward(self, grad_outputs):
        grad_output = grad_outputs[0]
        import numpy as np
        from .._tensor import Tensor

        grad_np = grad_output.numpy()
        a_shape = self._a_shape
        b_shape = self._b_shape

        # Handle broadcasting: reduce gradients to match input shapes
        grad_a = _reduce_gradient(grad_np, a_shape)
        grad_b = _reduce_gradient(grad_np, b_shape)

        return (Tensor(grad_a.astype(np.float32)), Tensor(grad_b.astype(np.float32)))


def _reduce_gradient(grad, target_shape):
    """Reduce gradient to match target shape by summing over broadcasted dimensions."""
    import numpy as np

    # First, sum over extra leading dimensions
    while grad.ndim > len(target_shape):
        grad = grad.sum(axis=0)

    # Then sum over dimensions that were broadcast (size 1 in target)
    for i in range(len(target_shape)):
        if target_shape[i] == 1 and grad.shape[i] != 1:
            grad = grad.sum(axis=i, keepdims=True)
        elif i < grad.ndim and grad.shape[i] != target_shape[i]:
            # This handles the case where grad has larger shape
            if target_shape[i] == 1:
                grad = grad.sum(axis=i, keepdims=True)

    return grad


class SubBackward(Node):
    """Backward for element-wise subtraction."""

    def __init__(self):
        super().__init__()
        self._name = "SubBackward"

    def backward(self, grad_outputs):
        grad_output = grad_outputs[0]
        import numpy as np
        from .._tensor import Tensor

        grad_np = grad_output.numpy()
        a_shape = self._a_shape
        b_shape = self._b_shape

        # Handle broadcasting: reduce gradients to match input shapes
        grad_a = _reduce_gradient(grad_np, a_shape)
        grad_b = _reduce_gradient(-grad_np, b_shape)  # Negate for subtraction

        return (Tensor(grad_a.astype(np.float32)), Tensor(grad_b.astype(np.float32)))


class MulBackward(Node):
    """Backward for element-wise multiplication."""

    def __init__(self):
        super().__init__()
        self._name = "MulBackward"

    def backward(self, grad_outputs):
        grad_output = grad_outputs[0]
        import numpy as np
        from .._tensor import Tensor

        grad_np = grad_output.numpy()

        # Handle scalar multiplication case
        if len(self.saved_tensors) == 1:
            # Scalar case: y = x * scalar
            a = self.saved_tensors[0]
            scalar = self._scalar_multiplier
            a_np = a.numpy()

            # Gradient for tensor: grad * scalar
            grad_a = grad_np * scalar
            grad_a = _reduce_gradient(grad_a, a_np.shape)
            return (Tensor(grad_a.astype(np.float32)),)
        else:
            # Two-tensor case: y = a * b
            a, b = self.saved_tensors
            a_np = a.numpy()
            b_np = b.numpy()

            # Compute gradients
            grad_a_raw = grad_np * b_np
            grad_b_raw = grad_np * a_np

            # Handle broadcasting: reduce gradients to match input shapes
            grad_a = _reduce_gradient(grad_a_raw, a_np.shape)
            grad_b = _reduce_gradient(grad_b_raw, b_np.shape)

            return (Tensor(grad_a.astype(np.float32)), Tensor(grad_b.astype(np.float32)))


class DivBackward(Node):
    """Backward for element-wise division."""

    def __init__(self):
        super().__init__()
        self._name = "DivBackward"

    def backward(self, grad_outputs):
        grad_output = grad_outputs[0]
        saved = self.saved_tensors

        if len(saved) == 1:
            # Division by scalar - only need grad w.r.t. first arg
            a = saved[0]
            b_val = self._scalar_divisor
            from .. import div
            grad_a = div(grad_output, b_val)
            return (grad_a,)
        else:
            a, b = saved
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
        import numpy as np
        from .._tensor import Tensor

        # Handle scalar exponent case
        if len(self.saved_tensors) == 1:
            base = self.saved_tensors[0]
            exp_val = self._scalar_exponent
            base_np = base.numpy()
            grad_out_np = grad_output.numpy()

            # d/dx (x^n) = n * x^(n-1)
            grad_base = grad_out_np * exp_val * np.power(base_np, exp_val - 1)
            return (Tensor(grad_base.astype(np.float32)),)
        else:
            base, exp = self.saved_tensors
            from .. import mul, pow, sub

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

        import numpy as np
        from .._tensor import Tensor

        grad_out = grad_output.numpy()
        a_np = a.numpy()
        b_np = b.numpy()

        # Handle n-dimensional matmul with broadcasting
        # C = A @ B
        # dA = dC @ B.T
        # dB = A.T @ dC
        grad_a = np.matmul(grad_out, np.swapaxes(b_np, -2, -1))
        grad_b = np.matmul(np.swapaxes(a_np, -2, -1), grad_out)

        # Handle broadcasting: sum gradients over broadcasted dimensions
        # grad_a should match a_np shape
        while grad_a.ndim > a_np.ndim:
            grad_a = grad_a.sum(axis=0)
        for i in range(a_np.ndim):
            if i < len(a_np.shape) and a_np.shape[i] == 1 and grad_a.shape[i] != 1:
                grad_a = grad_a.sum(axis=i, keepdims=True)

        # grad_b should match b_np shape
        while grad_b.ndim > b_np.ndim:
            grad_b = grad_b.sum(axis=0)
        for i in range(b_np.ndim):
            if i < len(b_np.shape) and b_np.shape[i] == 1 and grad_b.shape[i] != 1:
                grad_b = grad_b.sum(axis=i, keepdims=True)

        return (Tensor(grad_a.astype(np.float32)), Tensor(grad_b.astype(np.float32)))


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


class TanhBackward(Node):
    """Backward for tanh."""

    def __init__(self):
        super().__init__()
        self._name = "TanhBackward"

    def backward(self, grad_outputs):
        grad_output = grad_outputs[0]
        result = self.saved_tensors[0]  # tanh output

        import numpy as np
        from .._tensor import Tensor

        tanh_out = result.numpy()
        grad_out = grad_output.numpy()

        # d/dx tanh(x) = 1 - tanh(x)^2
        grad_x = grad_out * (1 - tanh_out ** 2)
        return (Tensor(grad_x.astype(np.float32)),)


class SoftmaxBackward(Node):
    """Backward for softmax."""

    def __init__(self):
        super().__init__()
        self._name = "SoftmaxBackward"

    def backward(self, grad_outputs):
        grad_output = grad_outputs[0]
        result = self.saved_tensors[0]  # softmax output

        import numpy as np
        from .._tensor import Tensor

        s = result.numpy()
        grad_out = grad_output.numpy()
        dim = self._dim

        # Jacobian: diag(s) - s @ s.T (element-wise for each batch)
        # Simplified: grad_input = s * (grad_out - sum(grad_out * s, dim))
        sum_grad_s = np.sum(grad_out * s, axis=dim, keepdims=True)
        grad_x = s * (grad_out - sum_grad_s)

        return (Tensor(grad_x.astype(np.float32)),)


class BmmBackward(Node):
    """Backward for batched matrix multiplication."""

    def __init__(self):
        super().__init__()
        self._name = "BmmBackward"

    def backward(self, grad_outputs):
        grad_output = grad_outputs[0]
        a, b = self.saved_tensors

        import numpy as np
        from .._tensor import Tensor

        grad_out = grad_output.numpy()
        a_np = a.numpy()
        b_np = b.numpy()

        # C = A @ B
        # dA = dC @ B.T
        # dB = A.T @ dC
        grad_a = np.matmul(grad_out, np.swapaxes(b_np, -2, -1))
        grad_b = np.matmul(np.swapaxes(a_np, -2, -1), grad_out)

        return (Tensor(grad_a.astype(np.float32)), Tensor(grad_b.astype(np.float32)))


class CloneBackward(Node):
    """Backward for clone."""

    def __init__(self):
        super().__init__()
        self._name = "CloneBackward"

    def backward(self, grad_outputs):
        # Clone just passes gradients through
        return (grad_outputs[0],)


class ContiguousBackward(Node):
    """Backward for contiguous (identity)."""

    def __init__(self):
        super().__init__()
        self._name = "ContiguousBackward"

    def backward(self, grad_outputs):
        return (grad_outputs[0],)


class ViewBackward(Node):
    """Backward for view/reshape."""

    def __init__(self):
        super().__init__()
        self._name = "ViewBackward"

    def backward(self, grad_outputs):
        grad_output = grad_outputs[0]
        input_shape = self._input_shape

        # Reshape gradient back to input shape
        return (grad_output.reshape(*input_shape),)


class SelectBackward(Node):
    """Backward for indexing/slicing operations."""

    def __init__(self):
        super().__init__()
        self._name = "SelectBackward"

    def backward(self, grad_outputs):
        grad_output = grad_outputs[0]
        input_shape = self._input_shape
        key = self._key

        import numpy as np
        from .._tensor import Tensor

        # Create zero gradient with input shape
        grad_input = np.zeros(input_shape, dtype=np.float32)

        # Scatter gradient to the indexed positions
        grad_out_np = grad_output.numpy()
        grad_input[key] = grad_out_np

        return (Tensor(grad_input),)


class DropoutBackward(Node):
    """Backward for dropout."""

    def __init__(self):
        super().__init__()
        self._name = "DropoutBackward"

    def backward(self, grad_outputs):
        grad_output = grad_outputs[0]
        mask = self._mask  # Saved during forward
        p = self._p

        import numpy as np
        from .._tensor import Tensor

        grad_out_np = grad_output.numpy()
        # Apply same mask to gradient (scaled by 1/(1-p))
        grad_input = grad_out_np * mask

        return (Tensor(grad_input.astype(np.float32)),)


class PermuteBackward(Node):
    """Backward for permute."""

    def __init__(self):
        super().__init__()
        self._name = "PermuteBackward"

    def backward(self, grad_outputs):
        grad_output = grad_outputs[0]
        dims = self._dims

        # Inverse permutation
        inv_dims = [0] * len(dims)
        for i, d in enumerate(dims):
            inv_dims[d] = i

        return (grad_output.permute(*inv_dims),)


class ContiguousBackward(Node):
    """Backward for contiguous (identity pass-through)."""

    def __init__(self):
        super().__init__()
        self._name = "ContiguousBackward"

    def backward(self, grad_outputs):
        return (grad_outputs[0],)
