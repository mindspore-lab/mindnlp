from .._autograd.grad_mode import no_grad, enable_grad
from .._autograd.engine import _run_backward
from .._autograd.node import Node


def checkpoint(function, *args, use_reentrant=True, preserve_rng_state=True, **kwargs):
    """Checkpoint a function to trade compute for memory.

    Runs function(*args) without saving intermediates in forward,
    then recomputes them during backward.

    Args:
        function: The function to checkpoint
        *args: Arguments to pass to function
        use_reentrant: Whether to use reentrant checkpoint (for API compatibility)
        preserve_rng_state: Whether to preserve RNG state (not implemented, for API compatibility)
        **kwargs: Keyword arguments to pass to function
    """
    # Separate tensor and non-tensor args
    tensor_inputs = []
    tensor_indices = []
    for i, arg in enumerate(args):
        if hasattr(arg, 'requires_grad') and hasattr(arg, 'grad_fn'):
            tensor_inputs.append(arg)
            tensor_indices.append(i)

    # Forward: run without saving intermediates
    with no_grad():
        outputs = function(*args, **kwargs)

    # If no tensor input requires grad, just return
    if not any(t.requires_grad for t in tensor_inputs):
        return outputs

    is_tuple = isinstance(outputs, tuple)
    if not is_tuple:
        outputs = (outputs,)

    # Detach inputs for recomputation
    def make_recompute_inputs():
        new_args = list(args)
        detached = []
        for i, idx in enumerate(tensor_indices):
            d = tensor_inputs[i].detach()
            d.requires_grad_(tensor_inputs[i].requires_grad)
            new_args[idx] = d
            detached.append(d)
        return new_args, detached

    def _checkpoint_backward(grad):
        new_args, detached = make_recompute_inputs()
        with enable_grad():
            recomputed = function(*new_args, **kwargs)
        if not isinstance(recomputed, tuple):
            recomputed = (recomputed,)

        # Only backward through outputs that got gradients
        out_with_grad = []
        grad_outputs = []
        for r, o in zip(recomputed, outputs):
            out_with_grad.append(r)
            grad_outputs.append(grad if len(recomputed) == 1 else None)

        result = _run_backward(
            tuple(out_with_grad), tuple(grad_outputs),
            retain_graph=False, create_graph=False,
            accumulate_grad=False, inputs=detached,
            allow_unused=True,
        )
        # Map back to original tensor_inputs order
        all_grads = [None] * len(tensor_inputs)
        for i, g in enumerate(result):
            all_grads[i] = g
        return tuple(all_grads)

    node = Node(_checkpoint_backward, tuple(tensor_inputs))
    # Attach grad_fn to outputs and mark as requiring grad
    for out in outputs:
        if hasattr(out, 'grad_fn'):
            out.grad_fn = node
            out.requires_grad = True

    if is_tuple:
        return outputs
    return outputs[0]


def checkpoint_sequential(functions, segments, input, **kwargs):
    """Checkpoint a sequential model by splitting into segments."""
    funcs = list(functions)
    segment_size = (len(funcs) + segments - 1) // segments

    def run_segment(start, end, inp):
        def segment_fn(x):
            for f in funcs[start:end]:
                x = f(x)
            return x
        return checkpoint(segment_fn, inp, **kwargs)

    x = input
    for start in range(0, len(funcs), segment_size):
        end = min(start + segment_size, len(funcs))
        x = run_segment(start, end, x)
    return x
