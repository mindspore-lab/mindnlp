"""Base class for all ops."""
from typing import Any, Tuple, Optional


class Op:
    """Base class for tensor operations.

    Each op implements:
    - forward(*args, **kwargs) -> result (MindSpore tensors)
    - backward(grad_output, *saved) -> tuple of gradients (MindSpore tensors)

    All forward/backward use pyboost primitives only.
    NEVER use mindspore.ops or mindspore.mint directly.

    Set needs_forward_result = True for ops where backward needs the forward
    result (e.g., exp, sqrt, tanh) rather than the input.
    """

    # Override in subclass if backward needs the forward result
    needs_forward_result: bool = False

    @property
    def name(self) -> str:
        return self.__class__.__name__

    def forward(self, *args, **kwargs) -> Any:
        raise NotImplementedError(f"{self.name}.forward not implemented")

    def backward(self, grad_output: Any, *saved: Any) -> Tuple[Optional[Any], ...]:
        raise NotImplementedError(f"{self.name}.backward not implemented")

    def __call__(self, *args, **kwargs):
        return self.forward(*args, **kwargs)
