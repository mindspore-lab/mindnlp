#!/usr/bin/env python
# -*- coding: utf-8 -*-

from .._jit_internal import (
    # _Await,
    # _drop,
    # _IgnoreContextManager,
    # _isinstance,
    # _overload,
    # _overload_method,
    # export,
    Final,
    # Future,
    # ignore,
    # is_scripting,
    unused,
)

from ._trace import (
    # _flatten,
    # _get_trace_graph,
    _script_if_tracing,
    # _unique_state_dict,
    # is_tracing,
    # ONNXTracedModule,
    # TopLevelTracedModule,
    # trace,
    # trace_module,
    # TracedModule,
    # TracerWarning,
    # TracingCheckError,
)

def is_tracing():
    return False

def is_scripting():
    return False

def script(obj, optimize=None, _frames_up=0, _rcb=None, example_inputs=None):
    return obj

def ignore(drop=False, **kwargs):

    if callable(drop):
        return drop

    def decorator(fn):
        return fn

    return decorator

def _overload_method(func):
    pass

def interface(obj):
    pass

def script_if_tracing(fn):
    """
    Compiles ``fn`` when it is first called during tracing.

    ``torch.jit.script`` has a non-negligible start up time when it is first called due to
    lazy-initializations of many compiler builtins. Therefore you should not use
    it in library code. However, you may want to have parts of your library work
    in tracing even if they use control flow. In these cases, you should use
    ``@torch.jit.script_if_tracing`` to substitute for
    ``torch.jit.script``.

    Args:
        fn: A function to compile.

    Returns:
        If called during tracing, a :class:`ScriptFunction` created by `torch.jit.script` is returned.
        Otherwise, the original function `fn` is returned.
    """
    return _script_if_tracing(fn)