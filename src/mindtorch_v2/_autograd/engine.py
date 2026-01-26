"""Autograd backward engine."""

from typing import Dict, List
from collections import defaultdict


def _topological_sort(root_nodes):
    """Topological sort of nodes for backward pass.

    Returns nodes in order from outputs to inputs.
    """
    visited = set()
    order = []

    def dfs(node):
        if node is None or id(node) in visited:
            return
        visited.add(id(node))

        for next_node, _ in node.next_functions:
            dfs(next_node)

        order.append(node)

    for node in root_nodes:
        dfs(node)

    return list(reversed(order))


def backward(
    tensors,
    grad_tensors=None,
    retain_graph: bool = False,
    create_graph: bool = False,
):
    """Compute gradients of tensors w.r.t. graph leaves."""
    from .node import AccumulateGrad
    from .._tensor import Tensor

    if not isinstance(tensors, (tuple, list)):
        tensors = (tensors,)

    # Initialize grad_tensors
    if grad_tensors is None:
        grad_tensors = []
        for t in tensors:
            if t.numel() != 1:
                raise RuntimeError(
                    "grad can be implicitly created only for scalar outputs"
                )
            grad_tensors.append(Tensor([1.0], dtype=t.dtype, device=str(t.device)))
        grad_tensors = tuple(grad_tensors)
    elif not isinstance(grad_tensors, (tuple, list)):
        grad_tensors = (grad_tensors,)

    # Collect root nodes
    root_nodes = []
    node_to_grad: Dict[int, List] = defaultdict(list)

    for tensor, grad in zip(tensors, grad_tensors):
        if tensor.grad_fn is not None:
            root_nodes.append(tensor.grad_fn)
            node_to_grad[id(tensor.grad_fn)].append(grad)

    if not root_nodes:
        return

    # Topological sort
    sorted_nodes = _topological_sort(root_nodes)

    # Compute gradients
    for node in sorted_nodes:
        grads = node_to_grad[id(node)]
        if not grads:
            continue

        # Sum gradients if multiple
        if len(grads) == 1:
            grad_output = grads[0]
        else:
            from .. import add
            grad_output = grads[0]
            for g in grads[1:]:
                grad_output = add(grad_output, g)

        # Handle AccumulateGrad specially
        if isinstance(node, AccumulateGrad):
            variable = node.variable

            # Call hooks
            if hasattr(variable, '_hooks') and variable._hooks:
                grad_output = variable._call_hooks(grad_output)

            if variable.grad is None:
                variable.grad = grad_output
            else:
                from .. import add
                variable.grad = add(variable.grad, grad_output)
            continue

        # Compute input gradients
        try:
            input_grads = node.backward((grad_output,))
        except Exception as e:
            raise RuntimeError(f"Error in backward for {node.name}: {e}") from e

        if not isinstance(input_grads, tuple):
            input_grads = (input_grads,)

        # Propagate to next functions
        for (next_node, idx), grad in zip(node.next_functions, input_grads):
            if next_node is not None and grad is not None:
                node_to_grad[id(next_node)].append(grad)
