from typing import *  # noqa: F403
import mindtorch

def detach_variable(inputs: Tuple[Any, ...]) -> Tuple[mindtorch.Tensor, ...]:
    if isinstance(inputs, tuple):
        out = []
        for inp in inputs:
            if not isinstance(inp, mindtorch.Tensor):
                out.append(inp)
                continue

            x = inp.detach()
            x.requires_grad = inp.requires_grad
            out.append(x)
        return tuple(out)
    else:
        raise RuntimeError(
            "Only tuple of tensors is supported. Got Unsupported input type: ",
            type(inputs).__name__,
        )

def checkpoint(
    function,
    *args,
    use_reentrant = None,
    context_fn = None,
    determinism_check = None,
    debug = None,
    **kwargs
):
    return function(*args, **kwargs)