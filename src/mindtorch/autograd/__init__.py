"""autograd"""
import warnings
from typing import Any, Callable, cast, List, Optional, Sequence, Tuple, Union
import mindtorch
from mindtorch.types import _size, _TensorOrTensors
from .node import Node
from .function import Function, value_and_grad
from .grad_mode import no_grad, enable_grad, inference_mode
from .graph import _engine_run_backward


_OptionalTensor = Optional[mindtorch.Tensor]
_ShapeorNestedShape = Union[_size, Sequence[_size], mindtorch.Tensor]


def _calculate_shape(
    output: mindtorch.Tensor, grad: mindtorch.Tensor, is_grads_batched: bool
) -> Tuple[_ShapeorNestedShape, _ShapeorNestedShape]:
    if output.is_nested:
        if is_grads_batched:
            raise RuntimeError("Batched grads are not supported with Nested Tensor.")
        out_shape = output._nested_tensor_size()
        grad_shape = grad._nested_tensor_size()
        return out_shape, grad_shape
    reg_out_shape = output.shape
    reg_grad_shape = grad.shape if not is_grads_batched else grad.shape[1:]
    return reg_out_shape, reg_grad_shape


def _tensor_or_tensors_to_tuple(
    tensors: Optional[_TensorOrTensors], length: int
) -> Tuple[_OptionalTensor, ...]:
    if tensors is None:
        return (None,) * length
    if isinstance(tensors, mindtorch.Tensor):
        return (tensors,)
    return tuple(tensors)

def _make_grads(
    outputs: Sequence[mindtorch.Tensor],
    grads: Sequence[_OptionalTensor],
    is_grads_batched: bool,
) -> Tuple[_OptionalTensor, ...]:
    new_grads: List[_OptionalTensor] = []
    for out, grad in zip(outputs, grads):
        if isinstance(grad, mindtorch.Tensor):
            first_grad = grad if not is_grads_batched else grad[0]
            if not mindtorch.is_same_size(out, first_grad):
                out_shape, grad_shape = _calculate_shape(out, first_grad, is_grads_batched)
                if is_grads_batched:
                    raise RuntimeError(
                        "If `is_grads_batched=True`, we interpret the first "
                        "dimension of each grad_output as the batch dimension. "
                        "The sizes of the remaining dimensions are expected to match "
                        "the shape of corresponding output, but a mismatch "
                        "was detected: grad_output["
                        + str(grads.index(grad))
                        + "] has a shape of "
                        + str(grad_shape)
                        + " and output["
                        + str(outputs.index(out))
                        + "] has a shape of "
                        + str(out_shape)
                        + ". "
                        "If you only want some tensors in `grad_output` to be considered "
                        "batched, consider using vmap."
                    )
                else:
                    raise RuntimeError(
                        "Mismatch in shape: grad_output["
                        + str(grads.index(grad))
                        + "] has a shape of "
                        + str(grad_shape)
                        + " and output["
                        + str(outputs.index(out))
                        + "] has a shape of "
                        + str(out_shape)
                        + "."
                    )
            if out.dtype.is_complex != grad.dtype.is_complex:
                raise RuntimeError(
                    "For complex Tensors, both grad_output and output"
                    " are required to have the same dtype."
                    " Mismatch in dtype: grad_output["
                    + str(grads.index(grad))
                    + "] has a dtype of "
                    + str(grad.dtype)
                    + " and output["
                    + str(outputs.index(out))
                    + "] has a dtype of "
                    + str(out.dtype)
                    + "."
                )
            new_grads.append(grad)
        elif grad is None:
            if out.numel() != 1:
                raise RuntimeError("grad can be implicitly created only for scalar outputs")
            if not out.dtype.is_floating_point:
                raise RuntimeError("grad can be implicitly created only for real scalar outputs but got {out.dtype}")
            new_grads.append(mindtorch.ones_like(out, memory_format=mindtorch.preserve_format))
        else:
            raise TypeError("gradients can be either Tensors or None, but got "+ type(grad).__name__)
    return tuple(new_grads)

def backward(
    tensors: _TensorOrTensors,
    grad_tensors: Optional[_TensorOrTensors] = None,
    retain_graph: Optional[bool] = None,
    create_graph: bool = False,
    grad_variables: Optional[_TensorOrTensors] = None,
    inputs: Optional[_TensorOrTensors] = None,
) -> None:
    if grad_variables is not None:
        warnings.warn("'grad_variables' is deprecated. Use 'grad_tensors' instead.")
        if grad_tensors is None:
            grad_tensors = grad_variables
        else:
            raise RuntimeError("'grad_tensors' and 'grad_variables' (deprecated) arguments both passed to backward(). Please only use 'grad_tensors'.")
    if inputs is not None and len(inputs) == 0:
        raise RuntimeError("'inputs' argument to backward() cannot be empty.")
    tensors = (tensors,) if isinstance(tensors, mindtorch.Tensor) else tuple(tensors)
    inputs = ((inputs,) if isinstance(inputs, mindtorch.Tensor) else tuple(inputs) if inputs is not None else tuple())
    grad_tensors_ = _tensor_or_tensors_to_tuple(grad_tensors, len(tensors))
    grad_tensors_ = _make_grads(tensors, grad_tensors_, is_grads_batched=False)
    if retain_graph is None:
        retain_graph = create_graph
    _engine_run_backward(tensors, grad_tensors_, retain_graph, create_graph, inputs, allow_unreachable=True, accumulate_grad=True)  # Calls into the C++ engine to run the backward pass
