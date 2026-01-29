"""Stub for torch._C - PyTorch's C++ extension module.

This is a Tier 3 stub - imported by transformers but not actually
used for BERT forward/backward passes.
"""

from types import ModuleType


class _CStub(ModuleType):
    """Stub module that returns None for unknown attribute access."""

    def __getattr__(self, name):
        # __getattr__ is only called if attribute not found normally
        # Return None for unknown attributes (fallback)
        return None

    def __call__(self, *args, **kwargs):
        return None


# Create module-level attributes that transformers might check
_C = _CStub('torch._C')

# Common attributes that might be accessed - set directly on __dict__ to bypass __getattr__
_C.__dict__['_get_tracing_state'] = lambda: None
_C.__dict__['_is_tracing'] = lambda: False
_C.__dict__['_jit_clear_class_registry'] = lambda: None  # Used by transformers tests for cleanup
_C.__dict__['ScriptModule'] = type('ScriptModule', (), {})
_C.__dict__['ScriptFunction'] = type('ScriptFunction', (), {})
_C.__dict__['Graph'] = type('Graph', (), {})
_C.__dict__['Node'] = type('Node', (), {})
_C.__dict__['Value'] = type('Value', (), {})
_C.__dict__['Type'] = type('Type', (), {})
_C.__dict__['TensorType'] = type('TensorType', (), {})
_C.__dict__['ListType'] = type('ListType', (), {})
_C.__dict__['DictType'] = type('DictType', (), {})
_C.__dict__['OptionalType'] = type('OptionalType', (), {})
_C.__dict__['TupleType'] = type('TupleType', (), {})
_C.__dict__['ClassType'] = type('ClassType', (), {})
_C.__dict__['InterfaceType'] = type('InterfaceType', (), {})
_C.__dict__['AnyType'] = type('AnyType', (), {})
_C.__dict__['NoneType'] = type('NoneType', (), {})
_C.__dict__['BoolType'] = type('BoolType', (), {})
_C.__dict__['IntType'] = type('IntType', (), {})
_C.__dict__['FloatType'] = type('FloatType', (), {})
_C.__dict__['ComplexType'] = type('ComplexType', (), {})
_C.__dict__['StringType'] = type('StringType', (), {})
_C.__dict__['DeviceObjType'] = type('DeviceObjType', (), {})
_C.__dict__['StreamObjType'] = type('StreamObjType', (), {})
_C.__dict__['FunctionType'] = type('FunctionType', (), {})
_C.__dict__['PyObjectType'] = type('PyObjectType', (), {})

# Export all attributes
__all__ = ['_C']

# Make this module behave like _C itself
import sys
sys.modules[__name__].__class__ = _CStub
