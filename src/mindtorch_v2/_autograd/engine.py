from collections import deque
from contextlib import nullcontext
import threading

from .grad_mode import no_grad


_GRAPH_STATE = threading.local()


def _task_stack():
    stack = getattr(_GRAPH_STATE, "stack", None)
    if stack is None:
        stack = []
        _GRAPH_STATE.stack = stack
    return stack


def is_create_graph_enabled():
    stack = _task_stack()
    if not stack:
        return False
    return bool(stack[-1].create_graph)


class _GraphTask:
    def _grad_accum_context(self):
        if self.create_graph:
            return nullcontext()
        return no_grad()

    def __init__(self, dependencies, *, retain_graph, create_graph, accumulate_grad, grads_map=None):
        self.dependencies = dependencies
        self.received = {}
        self.node_grads = {}
        self.ready = deque()
        self.retain_graph = retain_graph
        self.create_graph = create_graph
        self.accumulate_grad = accumulate_grad
        self.grads_map = grads_map if grads_map is not None else {}

    def _accumulate_tensor_grad(self, tensor, grad):
        grad = _apply_hooks(tensor, grad)
        if self.create_graph and grad is not None:
            grad.requires_grad = True
        if self.accumulate_grad:
            if tensor.grad_fn is None or getattr(tensor, "_retain_grad", False):
                if tensor.grad is None:
                    tensor.grad = grad
                else:
                    from .._functional import add
                    with self._grad_accum_context():
                        tensor.grad = add(tensor.grad, grad)
        else:
            prev = self.grads_map.get(tensor)
            if prev is None:
                self.grads_map[tensor] = grad
            else:
                from .._functional import add
                with self._grad_accum_context():
                    self.grads_map[tensor] = add(prev, grad)
        return grad

    def _accumulate_node_grad(self, node, grad):
        existing = self.node_grads.get(node)
        if existing is None:
            self.node_grads[node] = grad
        else:
            from .._functional import add
            with self._grad_accum_context():
                self.node_grads[node] = add(existing, grad)
        self.received[node] = self.received.get(node, 0) + 1
        if self.received[node] >= self.dependencies.get(node, 0):
            self.ready.append(node)

    def run(self):
        while self.ready:
            node = self.ready.popleft()
            grad = self.node_grads.pop(node, None)
            if grad is None:
                continue
            grads = node.backward(grad)
            if grads is None:
                grads = ()
            for t, g in zip(node.inputs, grads):
                if g is None:
                    continue
                g = self._accumulate_tensor_grad(t, g)
                if t.grad_fn is not None:
                    self._accumulate_node_grad(t.grad_fn, g)
            if not self.retain_graph:
                node.release_saved_tensors()


def _apply_hooks(tensor, grad):
    if tensor._backward_hooks:
        for hook in tensor._backward_hooks.values():
            result = hook(grad)
            if result is not None:
                grad = result
    return grad


def _build_dependencies(outputs):
    nodes = set()
    stack = [out.grad_fn for out in outputs if out.grad_fn is not None]
    while stack:
        node = stack.pop()
        if node in nodes:
            continue
        nodes.add(node)
        for inp in node.inputs:
            fn = getattr(inp, "grad_fn", None)
            if fn is not None:
                stack.append(fn)
    deps = {node: 0 for node in nodes}
    for node in nodes:
        seen = set()
        for inp in node.inputs:
            fn = getattr(inp, "grad_fn", None)
            if fn is None:
                continue
            if fn in seen:
                continue
            seen.add(fn)
            if fn in deps:
                deps[fn] += 1
    return deps


def _run_backward(outputs, grad_outputs, *, retain_graph, create_graph, accumulate_grad, inputs=None, allow_unused=False):
    task = _GraphTask(
        _build_dependencies(outputs),
        retain_graph=retain_graph,
        create_graph=create_graph,
        accumulate_grad=accumulate_grad,
    )
    _task_stack().append(task)
    try:
        for out, grad in zip(outputs, grad_outputs):
            if grad is None:
                continue
            grad = _apply_hooks(out, grad)
            if out.grad_fn is None:
                task._accumulate_tensor_grad(out, grad)
            else:
                task._accumulate_node_grad(out.grad_fn, grad)
        task.run()
    finally:
        _task_stack().pop()

    if inputs is None:
        return None
    results = []
    for inp in inputs:
        grad_val = task.grads_map.get(inp)
        if grad_val is None and not allow_unused:
            raise RuntimeError(
                "One of the differentiated Tensors appears to not have been used in the graph."
            )
        results.append(grad_val)
    return tuple(results)


def backward(tensor, grad=None, retain_graph=False, create_graph=False):
    if grad is None:
        if tensor.numel() != 1:
            raise RuntimeError("grad can be implicitly created only for scalar outputs")
        grad = tensor._ones_like()
    if create_graph and not retain_graph:
        retain_graph = True
    _run_backward((tensor,), (grad,), retain_graph=retain_graph, create_graph=create_graph, accumulate_grad=True)


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
    return _run_backward(
        outs,
        grad_outputs,
        retain_graph=retain_graph,
        create_graph=create_graph,
        accumulate_grad=False,
        inputs=ins,
        allow_unused=allow_unused,
    )
