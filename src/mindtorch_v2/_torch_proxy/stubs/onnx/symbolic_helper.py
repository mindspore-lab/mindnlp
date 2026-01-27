"""Stub for torch.onnx.symbolic_helper module."""


def parse_args(*args, **kwargs):
    """Parse args decorator - returns identity decorator."""
    def decorator(fn):
        return fn
    return decorator


def _get_tensor_sizes(x, allow_nonstatic=True):
    return None


def _get_tensor_dim_size(x, dim):
    return None


def _unimplemented(op, msg):
    def wrapper(*args, **kwargs):
        raise NotImplementedError(f"ONNX op {op}: {msg}")
    return wrapper


def _onnx_unsupported(op_name):
    def wrapper(*args, **kwargs):
        raise NotImplementedError(f"ONNX op {op_name} is not supported")
    return wrapper


def _onnx_opset_unsupported(op_name, current_opset, required_opset):
    def wrapper(*args, **kwargs):
        raise NotImplementedError(
            f"ONNX op {op_name} requires opset {required_opset}, but current opset is {current_opset}"
        )
    return wrapper


# Quantization helpers
def quantized_args(*arg_names):
    """Decorator for quantized args - returns identity decorator."""
    def decorator(fn):
        return fn
    return decorator


__all__ = [
    'parse_args',
    '_get_tensor_sizes',
    '_get_tensor_dim_size',
    '_unimplemented',
    '_onnx_unsupported',
    '_onnx_opset_unsupported',
    'quantized_args',
]
