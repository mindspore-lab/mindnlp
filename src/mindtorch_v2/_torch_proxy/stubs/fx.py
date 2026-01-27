"""Stub for torch.fx module."""


class Proxy:
    """Proxy class stub for torch.fx."""
    pass


class Node:
    """Node class stub for torch.fx."""
    pass


class Graph:
    """Graph class stub for torch.fx."""
    pass


class GraphModule:
    """GraphModule class stub for torch.fx."""
    pass


class Tracer:
    """Tracer class stub for torch.fx."""
    pass


class Interpreter:
    """Interpreter class stub for torch.fx."""
    pass


def symbolic_trace(root, concrete_args=None):
    """Symbolic trace stub - raises NotImplementedError."""
    raise NotImplementedError("torch.fx.symbolic_trace not available in mindtorch_v2")


def wrap(fn):
    """Wrap function - returns identity decorator."""
    return fn


__all__ = [
    'Proxy',
    'Node',
    'Graph',
    'GraphModule',
    'Tracer',
    'Interpreter',
    'symbolic_trace',
    'wrap',
]
