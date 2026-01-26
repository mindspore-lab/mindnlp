"""PyTorch-style dispatch keys for operation routing."""

from enum import IntEnum, auto


class DispatchKey(IntEnum):
    """Dispatch keys ordered by priority (lower = higher priority)."""

    # Autograd wrapper - records ops for backward
    Autograd = auto()

    # Automatic mixed precision
    AutocastCPU = auto()
    AutocastGPU = auto()

    # Batching (vmap)
    Batched = auto()

    # Functionalization (mutations → copies)
    Functionalize = auto()

    # JIT tracing
    Tracing = auto()

    # Backend execution
    Backend_CPU = auto()
    Backend_CUDA = auto()
    Backend_Ascend = auto()

    # Composite ops (decompose to primitives)
    CompositeExplicit = auto()


# Default active keys for normal execution
DEFAULT_DISPATCH_KEYS = frozenset({
    DispatchKey.Backend_CPU,
})
