"""Dispatcher routes ops to correct implementations based on dispatch keys.

The dispatch system supports two modes:
1. New standardized ops (from _ops module) - use op's own backward() method
2. Legacy dispatch (from registry) - use _autograd.functions backward classes

New ops are tried first for better maintainability and performance.
"""

from typing import Any, Tuple
from .keys import DispatchKey
from .registry import get_op_impl


def _get_backend_key(args: Tuple) -> DispatchKey:
    """Determine backend dispatch key from arguments.

    Looks at tensor arguments to determine which backend to use.
    Also checks tensors inside lists/tuples (e.g., for cat/stack).
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
        elif isinstance(arg, (list, tuple)):
            # Check tensors inside lists/tuples (for cat, stack, etc.)
            for item in arg:
                if isinstance(item, Tensor):
                    device_type = item.device.type
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
        elif isinstance(arg, (list, tuple)):
            for item in arg:
                if isinstance(item, Tensor) and item.requires_grad:
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
        'bmm': F.BmmBackward,
        'exp': F.ExpBackward,
        'log': F.LogBackward,
        'sqrt': F.SqrtBackward,
        'tanh': F.TanhBackward,
        'transpose': F.TransposeBackward,
        'embedding': F.EmbeddingBackward,
        'layer_norm': F.LayerNormBackward,
        'relu': F.ReluBackward,
        'gelu': F.GeluBackward,
        'silu': F.SiluBackward,
        'softmax': F.SoftmaxBackward,
        'clone': F.CloneBackward,
        'dropout': F.DropoutBackward,
        'cat': F.CatBackward,
        'stack': F.StackBackward,
    }

    BackwardClass = backward_classes.get(op_name)
    if BackwardClass is None:
        return None  # No backward for this op

    grad_fn = BackwardClass()

    # Build next_functions - special handling for ops with complex signatures
    next_fns = []
    tensors_to_save = []

    if op_name == 'layer_norm':
        # layer_norm args: (input, normalized_shape, weight, bias, eps)
        # We need to track: input, weight, bias
        input_tensor = args[0]
        weight = args[2] if len(args) > 2 else kwargs.get('weight')
        bias = args[3] if len(args) > 3 else kwargs.get('bias')

        # Input tensor
        if isinstance(input_tensor, Tensor) and input_tensor.requires_grad:
            if input_tensor.grad_fn is not None:
                next_fns.append((input_tensor.grad_fn, 0))
            else:
                acc_grad = AccumulateGrad(input_tensor)
                next_fns.append((acc_grad, 0))
        else:
            next_fns.append((None, 0))

        # normalized_shape - not a tensor, no gradient
        next_fns.append((None, 0))

        # Weight tensor
        if weight is not None and isinstance(weight, Tensor) and weight.requires_grad:
            if weight.grad_fn is not None:
                next_fns.append((weight.grad_fn, 0))
            else:
                acc_grad = AccumulateGrad(weight)
                next_fns.append((acc_grad, 0))
        else:
            next_fns.append((None, 0))

        # Bias tensor
        if bias is not None and isinstance(bias, Tensor) and bias.requires_grad:
            if bias.grad_fn is not None:
                next_fns.append((bias.grad_fn, 0))
            else:
                acc_grad = AccumulateGrad(bias)
                next_fns.append((acc_grad, 0))
        else:
            next_fns.append((None, 0))

        # eps - not a tensor, no gradient
        next_fns.append((None, 0))

    elif op_name == 'embedding':
        # embedding args: (indices, weight)
        # Only weight needs gradient, indices don't
        indices = args[0]
        weight = args[1]

        # Indices - no gradient
        next_fns.append((None, 0))

        # Weight tensor
        if isinstance(weight, Tensor) and weight.requires_grad:
            if weight.grad_fn is not None:
                next_fns.append((weight.grad_fn, 0))
            else:
                acc_grad = AccumulateGrad(weight)
                next_fns.append((acc_grad, 0))
        else:
            next_fns.append((None, 0))

        tensors_to_save = [indices, weight]

    elif op_name in ('cat', 'stack'):
        # cat/stack args: (tensors_list, dim)
        # First arg is a list/tuple of tensors
        tensor_list = args[0]
        dim = kwargs.get('dim', 0)
        if len(args) > 1:
            dim = args[1]

        for t in tensor_list:
            if isinstance(t, Tensor) and t.requires_grad:
                if t.grad_fn is not None:
                    next_fns.append((t.grad_fn, 0))
                else:
                    acc_grad = AccumulateGrad(t)
                    next_fns.append((acc_grad, 0))
            else:
                next_fns.append((None, 0))

    else:
        # Default handling for other ops
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
                # Non-tensor args don't get gradients
                pass

    grad_fn._next_functions = tuple(next_fns)

    # Save tensors for backward
    if op_name == 'mul':
        # Handle scalar multiplier
        if len(tensors_to_save) == 1:
            grad_fn.save_for_backward(tensors_to_save[0])
            # Find the scalar (non-tensor) argument
            for arg in args:
                if not isinstance(arg, Tensor):
                    grad_fn._scalar_multiplier = arg
                    break
        else:
            grad_fn.save_for_backward(*tensors_to_save)
    elif op_name == 'pow':
        # Handle scalar exponent
        if len(tensors_to_save) == 1:
            grad_fn.save_for_backward(tensors_to_save[0])
            # Find the scalar (non-tensor) argument - the exponent
            for arg in args[1:]:  # Skip first arg (base)
                if not isinstance(arg, Tensor):
                    grad_fn._scalar_exponent = arg
                    break
        else:
            grad_fn.save_for_backward(*tensors_to_save)
    elif op_name in ('matmul', 'embedding', 'bmm'):
        grad_fn.save_for_backward(*tensors_to_save)
    elif op_name == 'div':
        # Handle scalar divisor
        if len(tensors_to_save) == 1:
            grad_fn.save_for_backward(tensors_to_save[0])
            grad_fn._scalar_divisor = args[1]  # Save the scalar
        else:
            grad_fn.save_for_backward(*tensors_to_save)
    elif op_name == 'exp':
        grad_fn.save_for_backward(result)  # Save result for exp backward
    elif op_name == 'log':
        grad_fn.save_for_backward(tensors_to_save[0])  # Save input
    elif op_name == 'sqrt':
        grad_fn.save_for_backward(result)  # Save result
    elif op_name == 'tanh':
        grad_fn.save_for_backward(result)  # Save result (tanh output)
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
    elif op_name == 'softmax':
        grad_fn.save_for_backward(result)  # Save softmax output
        grad_fn._dim = kwargs.get('dim', -1)
    elif op_name == 'clone':
        pass  # No tensors to save for clone
    elif op_name == 'cat':
        # Save shapes and dim for cat backward
        tensor_list = args[0]
        grad_fn._input_shapes = [t.shape for t in tensor_list]
        grad_fn._dim = kwargs.get('dim', 0)
        if len(args) > 1:
            grad_fn._dim = args[1]
    elif op_name == 'stack':
        # Save num inputs and dim for stack backward
        tensor_list = args[0]
        grad_fn._num_inputs = len(tensor_list)
        grad_fn._dim = kwargs.get('dim', 0)
        if len(args) > 1:
            grad_fn._dim = args[1]
    elif op_name in ('add', 'sub'):
        # Save shapes for broadcasting gradient reduction
        if len(tensors_to_save) >= 2:
            grad_fn._a_shape = tensors_to_save[0].shape
            grad_fn._b_shape = tensors_to_save[1].shape
        elif len(tensors_to_save) == 1:
            # Scalar case
            grad_fn._a_shape = tensors_to_save[0].shape
            grad_fn._b_shape = ()

    return grad_fn


def _dispatch_new_op(op_name: str, op, args, kwargs):
    """Dispatch using new standardized op.

    Args:
        op_name: Name of the operation
        op: Op instance from _ops module
        args: Positional arguments
        kwargs: Keyword arguments

    Returns:
        Result Tensor with autograd set up if needed
    """
    from .._tensor import Tensor
    from .._autograd import is_grad_enabled
    from .._autograd.node import AccumulateGrad
    from ..configs import DEVICE_TARGET
    import numpy as np

    # Determine device from input tensors
    device_str = "cpu"
    for arg in args:
        if isinstance(arg, Tensor):
            device_str = str(arg.device)
            break

    # Get correct backend based on device/context
    if DEVICE_TARGET == 'Ascend':
        from .._backends.pyboost_ascend import _get_ms_data, _wrap_result
    else:
        from .._backends.pyboost_cpu import _get_ms_data, _wrap_result

    # Convert Tensor args to MindSpore tensors
    ms_args = []
    tensor_args = []
    for arg in args:
        if isinstance(arg, Tensor):
            ms_args.append(_get_ms_data(arg))
            tensor_args.append(arg)
        elif isinstance(arg, (np.ndarray, np.floating, np.integer, np.bool_)):
            # Convert numpy types to MindSpore tensors
            ms_args.append(_get_ms_data(arg))
        else:
            ms_args.append(arg)

    # Execute forward
    ms_result = op.forward(*ms_args, **kwargs)

    # Wrap result with correct device
    result = _wrap_result(ms_result, device=device_str)

    # Setup autograd if needed
    requires_grad = any(t.requires_grad for t in tensor_args)
    if is_grad_enabled() and requires_grad:
        result._requires_grad = True
        result._grad_fn = _make_new_grad_fn(op_name, op, tensor_args, ms_args, ms_result, device_str)

    return result


def _make_new_grad_fn(op_name, op, tensor_args, ms_args, ms_result, device_str="cpu"):
    """Create grad_fn using new op's backward method.

    Args:
        op_name: Name of the operation
        op: Op instance with backward() method
        tensor_args: Original Tensor arguments
        ms_args: MindSpore tensor arguments
        ms_result: MindSpore tensor result
        device_str: Device string for output tensors

    Returns:
        Node representing the backward function
    """
    from .._autograd.node import Node, AccumulateGrad
    from .._tensor import Tensor
    from ..configs import DEVICE_TARGET

    # Get correct backend based on device/context
    if DEVICE_TARGET == 'Ascend':
        from .._backends.pyboost_ascend import _get_ms_data, _wrap_result
    else:
        from .._backends.pyboost_cpu import _get_ms_data, _wrap_result

    class OpBackward(Node):
        """Backward node using standardized op's backward method."""

        def __init__(self, op, saved_tensors, saved_ms, ms_result, device):
            super().__init__()
            self._op = op
            self._saved_tensors = saved_tensors
            self._saved_ms = saved_ms
            self._ms_result = ms_result
            self._device = device
            self._name = f"{op.name}Backward"

        def backward(self, grad_outputs):
            grad_out = grad_outputs[0]
            ms_grad_out = _get_ms_data(grad_out)

            # Check if op needs forward result for backward (exp, sqrt, etc.)
            if getattr(self._op, 'needs_forward_result', False):
                ms_grads = self._op.backward(ms_grad_out, self._ms_result)
            else:
                # Pass input args for backward
                ms_grads = self._op.backward(ms_grad_out, *self._saved_ms)

            # Wrap results with correct device
            grads = []
            for g in ms_grads:
                if g is not None:
                    grads.append(_wrap_result(g, device=self._device))
                else:
                    grads.append(None)

            return tuple(grads)

    grad_fn = OpBackward(op, tensor_args, ms_args, ms_result, device_str)

    # Build next_functions
    next_fns = []
    for t in tensor_args:
        if t.requires_grad:
            if t.grad_fn is not None:
                next_fns.append((t.grad_fn, 0))
            else:
                next_fns.append((AccumulateGrad(t), 0))
        else:
            next_fns.append((None, 0))

    grad_fn._next_functions = tuple(next_fns)

    return grad_fn


def _has_meta_tensor(args) -> bool:
    """Check if any argument is a meta tensor.

    Meta tensors have device.type == "meta" and are used for shape inference
    without actual computation.
    """
    from .._tensor import Tensor

    for arg in args:
        if isinstance(arg, Tensor) and arg.device.type == "meta":
            return True
    return False


def _dispatch_meta(op_name: str, args, kwargs) -> Any:
    """Dispatch operation to meta backend for shape inference.

    Meta operations compute output shape without actual computation.
    This is used for model initialization and shape inference.

    Args:
        op_name: Name of the operation
        args: Positional arguments
        kwargs: Keyword arguments

    Returns:
        Meta tensor with correct output shape
    """
    from .._backends.meta import dispatch_meta

    return dispatch_meta(op_name, *args, **kwargs)


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

    # Check for meta tensors FIRST - route to meta backend for shape inference
    if _has_meta_tensor(args):
        return _dispatch_meta(op_name, args, kwargs)

    # Use legacy dispatch - backends handle device-specific implementations
    return _dispatch_legacy(op_name, args, kwargs)


def _dispatch_legacy(op_name: str, args, kwargs) -> Any:
    """Legacy dispatch using registry and _autograd.functions.

    Args:
        op_name: Name of the operation
        args: Positional arguments
        kwargs: Keyword arguments

    Returns:
        Result of the operation
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

    # Special handling for dropout - need mask for backward
    mask = None
    if op_name == 'dropout' and is_grad_enabled() and _requires_grad(args):
        result, mask = impl(*args, **kwargs, return_mask=True)
    else:
        # Execute forward
        result = impl(*args, **kwargs)

    # Record autograd if needed
    if is_grad_enabled() and _requires_grad(args) and isinstance(result, Tensor):
        grad_fn = _make_grad_fn(op_name, args, result, kwargs)
        if grad_fn is not None:
            # Special handling for dropout - save mask
            if op_name == 'dropout' and mask is not None:
                grad_fn._mask = mask
                grad_fn._p = kwargs.get('p', 0.5)
            result._grad_fn = grad_fn
            result._requires_grad = True

    return result
