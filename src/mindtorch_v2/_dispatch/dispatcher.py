"""Dispatcher routes ops to correct implementations based on dispatch keys."""

from typing import Any, Tuple
from .keys import DispatchKey
from .registry import get_op_impl


def _get_backend_key(args: Tuple) -> DispatchKey:
    """Determine backend dispatch key from arguments.

    Looks at tensor arguments to determine which backend to use.
    """
    from .._tensor import Tensor

    for arg in args:
        if isinstance(arg, Tensor):
            device_type = arg.device.type
            if device_type == "cpu":
                return DispatchKey.Backend_CPU
            elif device_type == "cuda":
                return DispatchKey.Backend_CUDA
            elif device_type in ("npu", "ascend"):
                return DispatchKey.Backend_Ascend

    # Default to CPU
    return DispatchKey.Backend_CPU


def _requires_grad(args) -> bool:
    """Check if any input tensor requires grad."""
    from .._tensor import Tensor
    for arg in args:
        if isinstance(arg, Tensor) and arg.requires_grad:
            return True
    return False


def _make_grad_fn(op_name: str, args, result, kwargs=None):
    """Create gradient function node for an op."""
    from .._tensor import Tensor
    from .._autograd.node import AccumulateGrad
    from .._autograd import functions as F

    if kwargs is None:
        kwargs = {}

    # Map op names to backward classes
    backward_classes = {
        'add': F.AddBackward,
        'sub': F.SubBackward,
        'mul': F.MulBackward,
        'div': F.DivBackward,
        'neg': F.NegBackward,
        'pow': F.PowBackward,
        'sum': F.SumBackward,
        'mean': F.MeanBackward,
        'matmul': F.MatmulBackward,
        'exp': F.ExpBackward,
        'log': F.LogBackward,
        'sqrt': F.SqrtBackward,
        'transpose': F.TransposeBackward,
        'embedding': F.EmbeddingBackward,
        'layer_norm': F.LayerNormBackward,
        'relu': F.ReluBackward,
        'gelu': F.GeluBackward,
        'silu': F.SiluBackward,
    }

    BackwardClass = backward_classes.get(op_name)
    if BackwardClass is None:
        return None  # No backward for this op

    grad_fn = BackwardClass()

    # Build next_functions
    next_fns = []
    tensors_to_save = []

    for arg in args:
        if isinstance(arg, Tensor):
            if arg.requires_grad:
                if arg.grad_fn is not None:
                    next_fns.append((arg.grad_fn, 0))
                else:
                    # Leaf tensor - create AccumulateGrad
                    acc_grad = AccumulateGrad(arg)
                    next_fns.append((acc_grad, 0))
            else:
                next_fns.append((None, 0))
            tensors_to_save.append(arg)
        else:
            # Scalar - wrap in tensor for saving
            tensors_to_save.append(Tensor(arg))

    grad_fn._next_functions = tuple(next_fns)

    # Save tensors for backward
    if op_name in ('mul', 'div', 'pow', 'matmul', 'embedding'):
        grad_fn.save_for_backward(*tensors_to_save)
    elif op_name == 'exp':
        grad_fn.save_for_backward(result)  # Save result for exp backward
    elif op_name == 'log':
        grad_fn.save_for_backward(tensors_to_save[0])  # Save input
    elif op_name == 'sqrt':
        grad_fn.save_for_backward(result)  # Save result
    elif op_name in ('sum', 'mean'):
        grad_fn._input_shape = tensors_to_save[0].shape
    elif op_name == 'transpose':
        # For transpose, save the dims from kwargs or args
        if 'dim0' in kwargs and 'dim1' in kwargs:
            grad_fn._dims = (kwargs['dim0'], kwargs['dim1'])
        elif len(args) >= 3:
            grad_fn._dims = (args[1], args[2])
    elif op_name == 'layer_norm':
        # Save input, normalized_shape, weight, bias, eps for layer_norm backward
        # args = (input, normalized_shape, weight, bias, eps)
        grad_fn._saved_info = (args[0], args[1], args[2] if len(args) > 2 else None,
                               args[3] if len(args) > 3 else None,
                               args[4] if len(args) > 4 else 1e-5)
    elif op_name in ('relu', 'silu'):
        grad_fn.save_for_backward(tensors_to_save[0])  # Save input
    elif op_name == 'gelu':
        grad_fn.save_for_backward(tensors_to_save[0])  # Save input
        grad_fn._approximate = kwargs.get('approximate', 'none')

    return grad_fn


def dispatch(op_name: str, *args, **kwargs) -> Any:
    """Dispatch an operation to the correct implementation.

    Args:
        op_name: Name of the operation (e.g., "add", "matmul")
        *args: Positional arguments to pass to the op
        **kwargs: Keyword arguments to pass to the op

    Returns:
        Result of the operation

    Raises:
        NotImplementedError: If no implementation found for the op
    """
    from .._tensor import Tensor
    from .._autograd import is_grad_enabled

    # Determine which backend to use
    backend_key = _get_backend_key(args)

    # Try to get implementation for this backend
    impl = get_op_impl(op_name, backend_key)

    if impl is None:
        # Try composite fallback
        impl = get_op_impl(op_name, DispatchKey.CompositeExplicit)

    if impl is None:
        raise NotImplementedError(
            f"No implementation found for op '{op_name}' with dispatch key {backend_key}"
        )

    # Execute forward
    result = impl(*args, **kwargs)

    # Record autograd if needed
    if is_grad_enabled() and _requires_grad(args) and isinstance(result, Tensor):
        grad_fn = _make_grad_fn(op_name, args, result, kwargs)
        if grad_fn is not None:
            result._grad_fn = grad_fn
            result._requires_grad = True

    return result
