import inspect
import typing

from ._dispatch.registry import registry, dispatch_key_from_string
from ._dispatch.keys import DispatchKey


class Library:
    """Registration handle for a namespace of custom operators.

    Mirrors torch.library.Library for defining schemas and registering
    dispatch-key-specific kernel implementations.

    Args:
        ns: Namespace string (e.g. "myops").
        kind: One of "DEF", "IMPL", "FRAGMENT".  Currently only "DEF"
              is meaningful; others are accepted for compatibility.
    """

    def __init__(self, ns, kind="DEF"):
        self._ns = ns
        self._kind = kind

    def _qualname(self, name):
        if "::" in name:
            return name
        return f"{self._ns}::{name}"

    def define(self, schema):
        """Register an operator schema.

        Args:
            schema: Schema string, e.g. "my_add(Tensor x, Tensor y) -> Tensor".
                    The op name is extracted from the schema and qualified with
                    the library namespace.
        """
        # Extract op name from schema (everything before the first "(")
        paren = schema.index("(")
        op_name = schema[:paren].strip()
        qualname = self._qualname(op_name)
        # Rewrite schema with qualified name for the registry
        qualified_schema = qualname + schema[paren:]
        registry.register_schema(qualname, qualified_schema)

    def impl(self, name, fn=None, dispatch_key="CompositeImplicitAutograd"):
        """Register a kernel implementation for a dispatch key.

        Can be used as a direct call or as a decorator::

            lib.impl("my_add", my_add_cpu, "CPU")

            @lib.impl("my_add", dispatch_key="CPU")
            def my_add_cpu(x, y): ...

        Args:
            name: Operator name (qualified or bare).
            fn: Kernel function.  If None, returns a decorator.
            dispatch_key: Dispatch key string (e.g. "CPU", "NPU", "Meta").
        """
        qualname = self._qualname(name)
        key = dispatch_key_from_string(dispatch_key)

        if fn is not None:
            registry.register_kernel(qualname, key, fn)
            return fn

        # Decorator form
        def decorator(func):
            registry.register_kernel(qualname, key, func)
            return func
        return decorator


def impl(qualname, dispatch_key="CompositeImplicitAutograd"):
    """Standalone decorator to register a kernel for an existing op.

    Usage::

        @torch.library.impl("myops::my_add", "CPU")
        def my_add_cpu(x, y):
            return x + y

    The op schema must already be registered (via Library.define or
    registry.register_schema).
    """
    key = dispatch_key_from_string(dispatch_key)

    def decorator(fn):
        registry.register_kernel(qualname, key, fn)
        return fn
    return decorator


def register_fake(qualname):
    """Standalone decorator to register a fake (meta) kernel for an op.

    Usage::

        @torch.library.register_fake("myops::my_add")
        def my_add_fake(x, y):
            return torch.empty_like(x)

    Equivalent to ``impl(qualname, "Meta")``.
    """
    def decorator(fn):
        registry.register_kernel(qualname, DispatchKey.Meta, fn)
        return fn
    return decorator


# ---------------------------------------------------------------------------
# Schema inference from type hints
# ---------------------------------------------------------------------------

# Map Python type annotations to schema type strings.
_TYPE_MAP = {
    "Tensor": "Tensor",
    "int": "int",
    "float": "float",
    "bool": "bool",
    "str": "str",
}


def _annotation_to_schema_type(annotation, mutates=False):
    """Convert a Python type annotation to a schema type string."""
    if annotation is inspect.Parameter.empty:
        return "Tensor"  # default assumption

    # Handle string annotations
    if isinstance(annotation, str):
        base = _TYPE_MAP.get(annotation)
        if base:
            return f"{base}!" if mutates else base
        return annotation

    # Get the origin for generic types (e.g. Optional, List)
    origin = getattr(annotation, "__origin__", None)

    # Optional[X] -> "X?"
    if origin is typing.Union:
        args = annotation.__args__
        # Optional[X] is Union[X, None]
        non_none = [a for a in args if a is not type(None)]
        if len(non_none) == 1:
            inner = _annotation_to_schema_type(non_none[0], mutates=mutates)
            return f"{inner}?"

    # List[X] -> "X[]"
    if origin is list:
        args = getattr(annotation, "__args__", ())
        if args:
            inner = _annotation_to_schema_type(args[0])
            return f"{inner}[]"
        return "int[]"

    # Plain types
    name = getattr(annotation, "__name__", None) or str(annotation)
    base = _TYPE_MAP.get(name)
    if base:
        return f"{base}!" if mutates else base

    # Fallback
    return "Tensor" if mutates else name


def _return_annotation_to_schema(annotation):
    """Convert a return type annotation to a schema return string."""
    if annotation is inspect.Parameter.empty or annotation is None:
        return "Tensor"

    origin = getattr(annotation, "__origin__", None)

    # tuple[X, Y, ...] -> "(X, Y, ...)"
    if origin is tuple:
        args = getattr(annotation, "__args__", ())
        parts = [_annotation_to_schema_type(a) for a in args]
        return f"({', '.join(parts)})"

    return _annotation_to_schema_type(annotation)


def _infer_schema(fn, mutates_args=()):
    """Infer an operator schema string from a function's type hints.

    Args:
        fn: The function to infer the schema from.
        mutates_args: Tuple of parameter names that are mutated in-place.

    Returns:
        Schema string, e.g. "op_name(Tensor x, Tensor y, float scale=1.0) -> Tensor"
    """
    sig = inspect.signature(fn)
    name = fn.__name__

    params = []
    for pname, param in sig.parameters.items():
        mutates = pname in mutates_args
        schema_type = _annotation_to_schema_type(param.annotation, mutates=mutates)
        if param.default is not inspect.Parameter.empty:
            default = param.default
            if default is None:
                default_str = "None"
            elif isinstance(default, bool):
                default_str = "True" if default else "False"
            elif isinstance(default, (int, float)):
                default_str = str(default)
            elif isinstance(default, str):
                default_str = f'"{default}"'
            else:
                default_str = str(default)
            params.append(f"{schema_type} {pname}={default_str}")
        else:
            params.append(f"{schema_type} {pname}")

    ret = _return_annotation_to_schema(sig.return_annotation)
    return f"{name}({', '.join(params)}) -> {ret}"


# ---------------------------------------------------------------------------
# @custom_op decorator and CustomOpHandle
# ---------------------------------------------------------------------------

# Autograd keys to strip when redispatching inside the autograd wrapper.
_AUTOGRAD_KEYS = (
    DispatchKey.Autograd,
    DispatchKey.AutogradCPU,
    DispatchKey.AutogradNPU,
    DispatchKey.AutogradMeta,
    DispatchKey.AutogradOther,
    DispatchKey.AutogradXPU,
    DispatchKey.ADInplaceOrView,
)


class CustomOpHandle:
    """Handle returned by @custom_op.  Callable (dispatches the op) and
    provides registration methods for fake, per-device kernels, and autograd.
    """

    def __init__(self, qualname, fn):
        self._qualname = qualname
        self._fn = fn
        # Copy function metadata for nice repr
        self.__name__ = fn.__name__
        self.__qualname__ = fn.__qualname__
        self.__module__ = fn.__module__
        self.__doc__ = fn.__doc__

    def __call__(self, *args, **kwargs):
        from ._dispatch.dispatcher import dispatch
        from ._tensor import Tensor
        # Infer device from tensor args
        device = "cpu"
        for a in args:
            if isinstance(a, Tensor):
                device = a.device
                break
        return dispatch(self._qualname, device, *args, **kwargs)

    def register_fake(self, fn):
        """Register a fake (meta) kernel for shape/dtype inference."""
        registry.register_kernel(self._qualname, DispatchKey.Meta, fn)
        return fn

    def register_kernel(self, device):
        """Decorator to register a device-specific kernel.

        Usage::

            @op.register_kernel("npu")
            def my_op_npu(x, y):
                ...
        """
        key = dispatch_key_from_string(device.upper() if device in ("cpu", "npu", "meta") else device)

        def decorator(fn):
            registry.register_kernel(self._qualname, key, fn)
            return fn
        return decorator

    def register_autograd(self, backward_fn, *, setup_context=None):
        """Register autograd support for this custom op.

        Creates an autograd wrapper that:
        1. Strips autograd keys and redispatches to get the forward result
        2. Builds a Node with the backward closure
        3. Attaches grad_fn to the output

        Args:
            backward_fn: Function(ctx, grad_output) -> tuple of gradients.
            setup_context: Optional function(ctx, inputs, output) to save
                tensors for backward.
        """
        qualname = self._qualname

        def autograd_wrapper(*args, **kwargs):
            from ._autograd.function import FunctionCtx
            from ._autograd.node import Node
            from ._autograd.grad_mode import is_grad_enabled
            from ._dispatch.dispatcher import redispatch, current_dispatch_keyset
            from ._tensor import Tensor

            # Check if any input requires grad
            needs_grad = any(
                isinstance(a, Tensor) and a.requires_grad
                for a in args
            ) and is_grad_enabled()

            # Strip autograd keys and redispatch
            keyset = current_dispatch_keyset()
            inner_keyset = keyset.without(_AUTOGRAD_KEYS)
            output = redispatch(qualname, inner_keyset, *args, **kwargs)

            if not needs_grad:
                return output

            # Build autograd graph
            ctx = FunctionCtx()
            input_tensors = [a for a in args if isinstance(a, Tensor) and a.requires_grad]
            ctx._needs_input_grad = tuple(
                isinstance(a, Tensor) and a.requires_grad for a in args
            )

            if setup_context is not None:
                setup_context(ctx, args, output)

            materialize = ctx._materialize_grads

            def _backward(grad):
                if materialize and grad is None:
                    from ._functional import zeros_like
                    grad = zeros_like(output)
                return backward_fn(ctx, grad)

            node = Node(_backward, input_tensors)

            if ctx._to_save is not None:
                node.save_for_backward(*ctx._to_save)
                ctx._saved_tensors = node._saved_tensors

            if isinstance(output, Tensor):
                output.grad_fn = node
                output.requires_grad = True
            elif isinstance(output, tuple):
                for o in output:
                    if isinstance(o, Tensor):
                        o.grad_fn = node
                        o.requires_grad = True

            return output

        # Register the autograd wrapper for all autograd dispatch keys
        for key in (DispatchKey.Autograd, DispatchKey.AutogradCPU,
                    DispatchKey.AutogradNPU, DispatchKey.AutogradMeta):
            registry.register_kernel(self._qualname, key, autograd_wrapper)


def custom_op(qualname, *, mutates_args=(), device_types=None, schema=None):
    """Decorator to define a custom operator with dispatch integration.

    Usage::

        @custom_op("myops::scaled_add", mutates_args=())
        def scaled_add(x: Tensor, y: Tensor, scale: float = 1.0) -> Tensor:
            return x + scale * y

    Args:
        qualname: Qualified name, e.g. "myops::scaled_add".
        mutates_args: Tuple of argument names that are mutated in-place.
        device_types: Optional device type(s) to register ("cpu", "npu", or both).
            If None, registers as CompositeImplicitAutograd (all backends).
        schema: Optional explicit schema string. If None, inferred from type hints.

    Returns:
        CustomOpHandle wrapping the function.
    """
    def decorator(fn):
        # Infer or use provided schema
        if schema is not None:
            schema_str = schema
        else:
            schema_str = _infer_schema(fn, mutates_args)

        # Register the schema (replace inferred name with qualname)
        paren = schema_str.index("(")
        full_schema = qualname + schema_str[paren:]
        registry.register_schema(qualname, full_schema)

        # Register the implementation
        if device_types is None:
            # Register for all backends so dispatch finds it
            registry.register_kernel(qualname, DispatchKey.CPU, fn)
            registry.register_kernel(qualname, DispatchKey.NPU, fn)
        else:
            types = (device_types,) if isinstance(device_types, str) else device_types
            for dt in types:
                key = dispatch_key_from_string(dt.upper() if dt in ("cpu", "npu", "meta") else dt)
                registry.register_kernel(qualname, key, fn)

        return CustomOpHandle(qualname, fn)

    return decorator
