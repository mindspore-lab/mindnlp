from .grad_mode import no_grad
from .._functional import add


def backward(tensor, grad=None, retain_graph=False, create_graph=False):
    if grad is None:
        if tensor.numel() != 1:
            raise RuntimeError("grad can be implicitly created only for scalar outputs")
        grad = tensor._ones_like()
    tensor.grad = grad
    if tensor.grad_fn is None:
        return
    grads = tensor.grad_fn.backward(grad)
    for t, g in zip(tensor.grad_fn.inputs, grads):
        if g is None:
            continue
        if t.grad is None:
            if t.grad_fn is None or getattr(t, "_retain_grad", False):
                t.grad = g
        else:
            with no_grad():
                t.grad = add(t.grad, g)
        if t.grad_fn is not None:
            backward(t, t.grad, retain_graph=retain_graph, create_graph=create_graph)
