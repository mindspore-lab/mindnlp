"""Stub for torch._C - PyTorch's C++ extension module.

This is a Tier 3 stub - imported by transformers but not actually
used for BERT forward/backward passes.
"""

from types import ModuleType


class _CStub(ModuleType):
    """Stub module that returns None for any attribute access."""

    def __getattr__(self, name):
        # Return None for most attributes
        return None

    def __call__(self, *args, **kwargs):
        return None


# Create module-level attributes that transformers might check
_C = _CStub('torch._C')

# Common attributes that might be accessed
_C._get_tracing_state = lambda: None
_C._is_tracing = lambda: False
_C.ScriptModule = type('ScriptModule', (), {})
_C.ScriptFunction = type('ScriptFunction', (), {})
_C.Graph = type('Graph', (), {})
_C.Node = type('Node', (), {})
_C.Value = type('Value', (), {})
_C.Type = type('Type', (), {})
_C.TensorType = type('TensorType', (), {})
_C.ListType = type('ListType', (), {})
_C.DictType = type('DictType', (), {})
_C.OptionalType = type('OptionalType', (), {})
_C.TupleType = type('TupleType', (), {})
_C.ClassType = type('ClassType', (), {})
_C.InterfaceType = type('InterfaceType', (), {})
_C.AnyType = type('AnyType', (), {})
_C.NoneType = type('NoneType', (), {})
_C.BoolType = type('BoolType', (), {})
_C.IntType = type('IntType', (), {})
_C.FloatType = type('FloatType', (), {})
_C.ComplexType = type('ComplexType', (), {})
_C.StringType = type('StringType', (), {})
_C.DeviceObjType = type('DeviceObjType', (), {})
_C.StreamObjType = type('StreamObjType', (), {})
_C.FunctionType = type('FunctionType', (), {})
_C.PyObjectType = type('PyObjectType', (), {})

# Export all attributes
__all__ = ['_C']

# Make this module behave like _C itself
import sys
sys.modules[__name__].__class__ = _CStub
