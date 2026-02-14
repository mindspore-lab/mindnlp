from .grad_mode import no_grad
from .._functional import add


def backward(tensor, grad=None, retain_graph=False, create_graph=False):
    if grad is None:
        if tensor.numel() != 1:
            raise RuntimeError("grad can be implicitly created only for scalar outputs")
        grad = tensor._ones_like()
    if tensor._backward_hooks:
        for hook in tensor._backward_hooks.values():
            result = hook(grad)
            if result is not None:
                grad = result
    tensor.grad = grad
    if tensor.grad_fn is None:
        return
    grads = tensor.grad_fn.backward(grad)
    for t, g in zip(tensor.grad_fn.inputs, grads):
        if g is None:
            continue
        if t._backward_hooks:
            for hook in t._backward_hooks.values():
                result = hook(g)
                if result is not None:
                    g = result
        if t.grad is None:
            if t.grad_fn is None or getattr(t, "_retain_grad", False):
                t.grad = g
        else:
            with no_grad():
                t.grad = add(t.grad, g)
        if t.grad_fn is not None:
            backward(t, t.grad, retain_graph=retain_graph, create_graph=create_graph)
