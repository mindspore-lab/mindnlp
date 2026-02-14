from .grad_mode import no_grad
from .._functional import add


def backward(tensor, grad=None):
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
            t.grad = g
        else:
            with no_grad():
                t.grad = add(t.grad, g)
        if t.grad_fn is not None:
            backward(t, t.grad)
