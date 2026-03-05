# Custom Operator Support for mindtorch_v2

**Date:** 2026-02-27
**Status:** Approved

## Goal

Add custom operator support to mindtorch_v2 with three objectives:

1. **PyTorch API compatibility** — `torch.autograd.Function` and `torch.library.custom_op`
2. **Simplified AscendC integration** — External NPU users register compiled AscendC kernels via Python
3. **Deep integration with existing dispatch** — Reuse DispatchKey, OpRegistry, autograd engine

Primary users: external NPU users who have their own AscendC kernels and want to register them into mindtorch_v2 via Python without any C++ compilation of the framework itself.

## Implementation Strategy

Bottom-up, in four phases:

1. `torch.autograd.Function` (foundation)
2. `torch.library.Library` + `impl` + `register_fake` (registration layer)
3. `torch.library.custom_op` (recommended user API)
4. AscendC convenience layer: `KernelLauncher`, `ascendc_op`, helpers

## Phase 1: torch.autograd.Function

### Files

| File | Action |
|------|--------|
| `_autograd/function.py` | New |
| `_autograd/__init__.py` | Export `Function` |

### Components

#### FunctionCtx

Context object passed between forward and backward. Independent of `Node` — it is a lightweight container that `apply()` wires into the autograd graph.

```python
class FunctionCtx:
    def save_for_backward(self, *tensors): ...
    saved_tensors: property  # -> tuple[Tensor, ...]
    needs_input_grad: tuple[bool, ...]
    mark_dirty(self, *tensors): ...
    mark_non_differentiable(self, *tensors): ...
    set_materialize_grads(self, value: bool): ...
```

Internal state:
- `_to_save: list[Tensor]` — tensors queued by `save_for_backward()`
- `_saved_tensors: list[SavedTensor]` — materialized after `apply()` wires them into a `Node`
- `_non_differentiable: set[int]` — ids of outputs that should not get `grad_fn`
- `_dirty: set[int]` — ids of in-place modified tensors
- `_materialize_grads: bool` — whether None grads become zeros (default True)

#### FunctionMeta (metaclass)

Inspects `forward()` signature at class definition time to detect old-style vs new-style:

- **Old-style**: `forward(ctx, input, ...)` — first param is ctx
- **New-style**: `forward(input, ...)` — no ctx, must have `setup_context(ctx, inputs, output)`

Sets `cls._new_style: bool` for `apply()` to branch on.

#### Function base class

```python
class Function(metaclass=FunctionMeta):
    @classmethod
    def apply(cls, *args, **kwargs): ...
```

#### apply() execution flow

1. Flatten args, identify which are tensors with `requires_grad`
2. Build `needs_input_grad` tuple
3. Create `FunctionCtx`
4. If old-style: call `cls.forward(ctx, *args, **kwargs)`
   If new-style: call `cls.forward(*args, **kwargs)`, then `cls.setup_context(ctx, args, output)`
5. If any input requires grad and `is_grad_enabled()`:
   a. Create `Node` with a backward closure that calls `cls.backward(ctx, grad_output)`
   b. Transfer `ctx._to_save` into `Node` via `node.save_for_backward()`
   c. Wire `ctx._saved_tensors` to read from `node.saved_tensors()`
   d. Set `output.grad_fn = node` and `output.requires_grad = True`
   e. Skip for outputs in `ctx._non_differentiable`
6. Return output

#### Out of scope

- `jvp()` (forward-mode AD) — no forward-mode engine in mindtorch_v2
- `vmap()` / `generate_vmap_rule` — no vmap support

### Usage

```python
from torch.autograd import Function

# Old-style
class MyReLU(Function):
    @staticmethod
    def forward(ctx, input):
        ctx.save_for_backward(input)
        return input.clamp(min=0)

    @staticmethod
    def backward(ctx, grad_output):
        input, = ctx.saved_tensors
        grad = grad_output.clone()
        grad[input < 0] = 0
        return (grad,)

output = MyReLU.apply(x)

# New-style
class MyMul(Function):
    @staticmethod
    def forward(a, b):
        return a * b

    @staticmethod
    def setup_context(ctx, inputs, output):
        a, b = inputs
        ctx.save_for_backward(a, b)

    @staticmethod
    def backward(ctx, grad_output):
        a, b = ctx.saved_tensors
        return grad_output * b, grad_output * a

output = MyMul.apply(x, y)
```

## Phase 2: torch.library API

### Files

| File | Action |
|------|--------|
| `library.py` | Expand (currently empty) |

### Components

#### Library class

```python
lib = Library("myops", "DEF")
lib.define("my_add(Tensor x, Tensor y) -> Tensor")
lib.impl("my_add", my_add_cpu, "CPU")
lib.impl("my_add", my_add_npu, "NPU")
```

Implementation:
- `define(schema)` — parse schema, call `registry.register_schema(qualname, schema)`
- `impl(name, fn, dispatch_key)` — map dispatch_key string to DispatchKey enum, call `registry.register_kernel(qualname, key, fn)`
- `kind` parameter ("DEF"/"IMPL"/"FRAGMENT") — simplified to just "DEF" initially

#### Standalone decorators

```python
@torch.library.impl("myops::my_add", "CPU")
def my_add_cpu(x, y): ...

@torch.library.register_fake("myops::my_add")
def my_add_fake(x, y): ...
```

These are sugar over `registry.register_kernel()`.

#### Dispatch key string resolution

Add `_dispatch_key_from_string(name) -> DispatchKey` to map:

| String | DispatchKey |
|--------|-------------|
| `"CPU"` | `DispatchKey.CPU` |
| `"NPU"` / `"PrivateUse1"` | `DispatchKey.NPU` |
| `"Meta"` | `DispatchKey.Meta` |
| `"Autograd"` | `DispatchKey.Autograd` |
| `"AutogradCPU"` | `DispatchKey.AutogradCPU` |
| `"AutogradNPU"` / `"AutogradPrivateUse1"` | `DispatchKey.AutogradNPU` |
| `"CompositeImplicitAutograd"` | `DispatchKey.CompositeImplicitAutograd` |

## Phase 3: @custom_op decorator

### Files

| File | Action |
|------|--------|
| `library.py` | Add `custom_op`, `CustomOpHandle` |

### CustomOpHandle

Returned by `@custom_op`. Is both a callable (dispatches the op) and a registration hub.

```python
handle = CustomOpHandle(qualname, schema, impl_fn, mutates_args)
handle.register_fake(fn)           # -> DispatchKey.Meta
handle.register_kernel("npu")(fn)  # -> DispatchKey.NPU
handle.register_autograd(backward_fn, setup_context=setup_fn)
handle(*args, **kwargs)            # -> dispatch(qualname, ...)
```

#### @custom_op decorator

```python
@custom_op("myops::scaled_add", mutates_args=())
def scaled_add(x: Tensor, y: Tensor, scale: float = 1.0) -> Tensor:
    return x + scale * y
```

Implementation:
1. Infer schema from type hints if not provided
2. `registry.register_schema(qualname, schema)`
3. Register implementation as default kernel (CPU + NPU unless `device_types` restricts)
4. Return wrapper function with handle methods attached

#### Schema inference from type hints

```python
def my_op(x: Tensor, y: Tensor, scale: float = 1.0) -> Tensor: ...
# -> "my_op(Tensor x, Tensor y, float scale=1.0) -> Tensor"
```

Type mapping:
- `Tensor` -> `Tensor`
- `int` -> `int`
- `float` -> `float`
- `bool` -> `bool`
- `Optional[Tensor]` -> `Tensor?`
- `List[int]` -> `int[]`

#### register_autograd() implementation

Creates an autograd wrapper kernel using the same `FunctionCtx` + `Node` pattern from Phase 1:

1. Strip autograd keys from keyset
2. `redispatch()` to get forward result
3. Create `FunctionCtx`, call `setup_context(ctx, inputs, output)`
4. Create `Node` with backward closure
5. Set `output.grad_fn = node`
6. Register this wrapper for `AutogradCPU`, `AutogradNPU`, `Autograd` keys

This does NOT go through `Function.apply()` — it reuses the same `FunctionCtx` and `Node` building blocks directly, avoiding unnecessary overhead.

### Usage

```python
@custom_op("myops::scaled_add", mutates_args=())
def scaled_add(x: Tensor, y: Tensor, scale: float = 1.0) -> Tensor:
    return x + scale * y

@scaled_add.register_fake
def _(x, y, scale=1.0):
    return torch.empty_like(x)

@scaled_add.register_kernel("npu")
def _(x, y, scale=1.0):
    # NPU-specific fast path
    ...

def setup(ctx, inputs, output):
    x, y, scale = inputs
    ctx.save_for_backward(y)
    ctx.scale = scale

def backward(ctx, grad):
    y, = ctx.saved_tensors
    return grad, grad * ctx.scale, None

scaled_add.register_autograd(backward, setup_context=setup)

# Call
z = scaled_add(x, y, scale=2.0)
```

## Phase 4: AscendC Convenience Layer

### Files

| File | Action |
|------|--------|
| `_backends/npu/custom_kernel.py` | New |

### KernelLauncher

Loads a compiled AscendC .so and provides a `launch()` method.

```python
class KernelLauncher:
    def __init__(self, library_path: str):
        self._lib = ctypes.CDLL(library_path)
        self._cache = {}  # kernel_name -> ctypes function

    def launch(self, kernel_name: str, block_dim: int,
               args: list, stream=None):
        """
        Launch an AscendC kernel.

        Args:
            kernel_name: Entry point name (e.g. "add_custom")
            block_dim: Number of AI cores
            args: List of kernel arguments (int pointers and scalars)
            stream: ACL stream (default: current stream)
        """
```

Implementation:
- Look up `aclrtlaunch_<kernel_name>` symbol via `getattr(self._lib, ...)`
- Cache function pointers
- Convert Python args to ctypes (`int` -> `c_uint64`, `float` -> `c_double`)
- If stream is None, use `npu_state.current_stream().stream`
- Call the launch function

### Helper functions

```python
def tensor_ptr(t: Tensor) -> int:
    """Extract device memory pointer from tensor."""
    return t.storage().data_ptr()

def alloc_like(t: Tensor) -> Tensor:
    """Allocate output tensor with same shape/dtype/device."""
    runtime = get_runtime(t.device.index or 0)
    ptr = _alloc_device(t.numel() * t.element_size(), runtime=runtime)
    storage = npu_typed_storage_from_ptr(ptr, t.numel(), t.dtype, device=t.device)
    return _wrap_tensor(storage, t.shape, _contiguous_stride(t.shape))
```

### @ascendc_op decorator

Sugar over `@custom_op(..., device_types="npu")` with one addition: auto-generates a default `register_fake` based on return type hint (`torch.empty_like` for single Tensor return). User can override with explicit `@op.register_fake`.

```python
@ascendc_op("mylib::vector_add")
def vector_add(x: Tensor, y: Tensor) -> Tensor:
    out = alloc_like(x)
    launcher.launch("add_custom", block_dim=8,
                    args=[tensor_ptr(x), tensor_ptr(y), tensor_ptr(out), x.numel()])
    return out
```

### Phase 4b (deferred): ascendc_load — JIT compilation

```python
my_ext = ascendc_load(name="my_kernels", sources=["add_custom.cpp"])
```

Deferred because:
- Requires CANN compiler path detection, SoC version detection
- Compilation caching logic
- Users can compile with CANN tools directly for now

### Out of scope

- No wrapping of aclnn two-phase API (GetWorkspaceSize + Execute) — that is for built-in ops
- No automatic tiling generation — tiling is the kernel developer's responsibility
- No `torch.ops.namespace.opname` access — P2, add later

## Complete File List

| File | Action | Phase |
|------|--------|-------|
| `_autograd/function.py` | New — `Function`, `FunctionCtx`, `FunctionMeta` | 1 |
| `_autograd/__init__.py` | Export `Function` | 1 |
| `library.py` | Expand — `Library`, `impl`, `register_fake`, `custom_op`, `CustomOpHandle` | 2-3 |
| `_backends/npu/custom_kernel.py` | New — `KernelLauncher`, `ascendc_op`, `tensor_ptr`, `alloc_like` | 4 |
| `_dispatch/registry.py` | Add `_dispatch_key_from_string()` helper | 2 |

## Dependencies on Existing Code

| Component | Depends on | Already exists? |
|-----------|-----------|-----------------|
| `Function.apply()` | `Node`, `SavedTensor`, `is_grad_enabled()` | Yes |
| `Library.define()` | `registry.register_schema()`, schema parser | Yes |
| `Library.impl()` | `registry.register_kernel()` | Yes |
| `custom_op` autograd wrapper | `Node`, `FunctionCtx`, `redispatch()` | Yes (Node, redispatch); New (FunctionCtx) |
| `KernelLauncher` | `ctypes`, `npu_state.current_stream()` | Yes |
| `alloc_like()` | `_alloc_device`, `npu_typed_storage_from_ptr`, `_wrap_tensor` | Yes |
