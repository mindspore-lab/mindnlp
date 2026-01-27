"""Stub for torch.onnx module."""


def export(model, args, f, export_params=True, verbose=False, training=None,
           input_names=None, output_names=None, operator_export_type=None,
           opset_version=None, do_constant_folding=True, dynamic_axes=None,
           keep_initializers_as_inputs=None, custom_opsets=None,
           export_modules_as_functions=False):
    """Export model to ONNX - stub, raises NotImplementedError."""
    raise NotImplementedError("ONNX export not available in mindtorch_v2")


def is_in_onnx_export():
    """Check if in ONNX export context."""
    return False


def select_model_mode_for_export(model, mode):
    """Select model mode for export - no-op stub."""
    return model


# Symbolic helper stub
class symbolic_helper:
    """Stub for torch.onnx.symbolic_helper."""

    @staticmethod
    def parse_args(*args, **kwargs):
        """Parse args decorator - returns identity decorator."""
        def decorator(fn):
            return fn
        return decorator

    @staticmethod
    def _get_tensor_sizes(x, allow_nonstatic=True):
        return None

    @staticmethod
    def _get_tensor_dim_size(x, dim):
        return None

    @staticmethod
    def _unimplemented(op, msg):
        def wrapper(*args, **kwargs):
            raise NotImplementedError(f"ONNX op {op}: {msg}")
        return wrapper


# Symbolic opset stubs
class _SymbolicOpset:
    """Stub for symbolic opset."""

    def __getattr__(self, name):
        def op_stub(g, *args, **kwargs):
            return None
        return op_stub


symbolic_opset9 = _SymbolicOpset()
symbolic_opset10 = _SymbolicOpset()
symbolic_opset11 = _SymbolicOpset()
symbolic_opset12 = _SymbolicOpset()
symbolic_opset13 = _SymbolicOpset()
symbolic_opset14 = _SymbolicOpset()
symbolic_opset15 = _SymbolicOpset()
symbolic_opset16 = _SymbolicOpset()
symbolic_opset17 = _SymbolicOpset()
symbolic_opset18 = _SymbolicOpset()


# Register custom op stub
def register_custom_op_symbolic(symbolic_name, symbolic_fn, opset_version):
    """Register custom op symbolic - no-op stub."""
    pass


def unregister_custom_op_symbolic(symbolic_name, opset_version):
    """Unregister custom op symbolic - no-op stub."""
    pass


# Verification stub
class verification:
    @staticmethod
    def verify(*args, **kwargs):
        pass


# Utils stub
class utils:
    @staticmethod
    def model_info(*args, **kwargs):
        return {}


# Export types
class OperatorExportTypes:
    ONNX = 0
    ONNX_ATEN = 1
    ONNX_ATEN_FALLBACK = 2
    RAW = 3


class TrainingMode:
    EVAL = 0
    PRESERVE = 1
    TRAINING = 2


__all__ = [
    'export',
    'is_in_onnx_export',
    'select_model_mode_for_export',
    'symbolic_helper',
    'symbolic_opset9',
    'symbolic_opset10',
    'symbolic_opset11',
    'symbolic_opset12',
    'symbolic_opset13',
    'symbolic_opset14',
    'symbolic_opset15',
    'symbolic_opset16',
    'symbolic_opset17',
    'symbolic_opset18',
    'register_custom_op_symbolic',
    'unregister_custom_op_symbolic',
    'verification',
    'utils',
    'OperatorExportTypes',
    'TrainingMode',
]
