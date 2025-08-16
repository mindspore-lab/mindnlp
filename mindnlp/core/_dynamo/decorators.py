from typing import Any, Callable, Optional, overload, TYPE_CHECKING, TypeVar, Union

def mark_static_address(t: Any, guard: bool = True) -> None:
    """
    Marks an input tensor whose data_ptr will not change across multiple calls
    to a dynamo-compiled function. This indicates to cudagraphs that an extra allocation
    is not needed for this input. The data_ptr will be guarded if guard=True. Note:
    Tensors marked in this way will be kept alive until `torch._dynamo.reset()` is called.
    """
    # if not isinstance(t, torch.Tensor):
    #     raise TypeError(f"mark_static_address expects a tensor but received {type(t)}")

    # if guard:
    #     t._dynamo_static_input_type = "guarded"  # type: ignore[attr-defined]
    # else:
    #     t._dynamo_static_input_type = "unguarded"  # type: ignore[attr-defined]
    pass

def allow_in_graph(fn):
    """
    Customize which functions TorchDynamo will include in the generated
    graph. Similar to `torch.fx.wrap()`.
    ::

        torch._dynamo.allow_in_graph(my_custom_function)

        @torch._dynamo.optimize(...)
        def fn(a):
            x = torch.add(x, 1)
            x = my_custom_function(x)
            x = torch.add(x, 1)
            return x

        fn(...)

    Will capture a single graph containing `my_custom_function()`.
    """
    if isinstance(fn, (list, tuple)):
        return [allow_in_graph(x) for x in fn]
    assert callable(fn), "allow_in_graph expects a callable"
    return fn

def disable(fn=None, recursive=True, *, reason=None, wrapping=True):  # type: ignore[no-untyped-def]
    """
    Decorator to disable TorchDynamo

    If recursive=True, Dynamo is completely skipped on the decorated function
    frame as well as the recursively invoked functions.

    If recursive=False, Dynamo skips frames associated with the function code,
    but still process recursively invoked frames.

    If reason is provided, it will be printed when Dynamo attempts to trace the disabled function.
    """
    return fn