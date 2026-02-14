from .grad_mode import no_grad


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
            from .._functional import add
            with no_grad():
                t.grad = add(t.grad, g)
        if t.grad_fn is not None:
            backward(t, t.grad, retain_graph=retain_graph, create_graph=create_graph)


def grad(outputs, inputs, grad_outputs=None, retain_graph=None, create_graph=False, allow_unused=False):
    if retain_graph is None:
        retain_graph = create_graph
    outs = outputs if isinstance(outputs, (tuple, list)) else (outputs,)
    ins = inputs if isinstance(inputs, (tuple, list)) else (inputs,)
    if all(out.grad_fn is None and not out.requires_grad for out in outs):
        if allow_unused:
            return tuple(None for _ in ins)
        raise RuntimeError(
            "element 0 of tensors does not require grad and does not have a grad_fn"
        )
    if grad_outputs is None:
        grad_outputs = []
        for out in outs:
            if out.numel() != 1:
                raise RuntimeError("grad can be implicitly created only for scalar outputs")
            grad_outputs.append(out._ones_like())
        grad_outputs = tuple(grad_outputs)
    else:
        grad_outputs = grad_outputs if isinstance(grad_outputs, (tuple, list)) else (grad_outputs,)
        if len(grad_outputs) != len(outs):
            raise RuntimeError("grad_outputs must be the same length as outputs")
    grads_map = {}
    for out, grad_value in zip(outs, grad_outputs):
        if grad_value is None:
            continue
        _accumulate_grad(out, grad_value, grads_map, retain_graph=retain_graph, create_graph=create_graph)
    results = []
    for inp in ins:
        grad_val = grads_map.get(inp)
        if grad_val is None and not allow_unused:
            raise RuntimeError(
                "One of the differentiated Tensors appears to not have been used in the graph."
            )
        results.append(grad_val)
    return tuple(results)


def _apply_hooks(tensor, grad):
    if tensor._backward_hooks:
        for hook in tensor._backward_hooks.values():
            result = hook(grad)
            if result is not None:
                grad = result
    return grad


def _accumulate_grad(tensor, grad, grads_map, retain_graph=False, create_graph=False):
    grad = _apply_hooks(tensor, grad)
    if tensor.grad_fn is None:
        prev = grads_map.get(tensor)
        if prev is None:
            grads_map[tensor] = grad
        else:
            from .._functional import add
            with no_grad():
                grads_map[tensor] = add(prev, grad)
        return
    grads = tensor.grad_fn.backward(grad)
    for t, g in zip(tensor.grad_fn.inputs, grads):
        if g is None:
            continue
        _accumulate_grad(t, g, grads_map, retain_graph=retain_graph, create_graph=create_graph)
