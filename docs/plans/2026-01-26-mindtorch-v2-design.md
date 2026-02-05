# mindtorch v2 Design Document

**Date**: 2026-01-26
**Status**: Approved
**Goal**: Deep PyTorch compatibility with MindSpore as kernel backend

## Overview

mindtorch v2 is a ground-up reimplementation providing deep PyTorch compatibility while using MindSpore purely as a kernel execution backend.

### Problems Solved

1. **Reduce maintenance burden** - No more patching MindSpore or transformers
2. **Improve compatibility** - Deep PyTorch API compatibility including internals
3. **Performance** - Real storage model, efficient dispatch, direct primitive calls

### Core Principle

PyTorch semantics are authoritative. MindSpore is invisible to users except through explicit interop APIs.

## Architecture

```
┌─────────────────────────────────────────┐
│  PyTorch-compatible Public API          │  ← torch.*, torch.nn.*, etc.
├─────────────────────────────────────────┤
│  Dispatcher (dispatch keys)             │  ← autograd, batching, tracing
├─────────────────────────────────────────┤
│  Operator Registry                      │  ← native ops, composite ops
├─────────────────────────────────────────┤
│  Tensor + Storage                       │  ← stride/offset logic
├─────────────────────────────────────────┤
│  Autograd Engine                        │  ← tape-based, custom backward
├─────────────────────────────────────────┤
│  Backend Abstraction                    │  ← cpu, cuda, ascend
├─────────────────────────────────────────┤
│  _op_prim (pyboost/primitives)          │  ← actual compute
└─────────────────────────────────────────┘
```

### Key Decisions

| Decision | Choice | Rationale |
|----------|--------|-----------|
| Architecture | Independent tensor/autograd | Full control over PyTorch semantics |
| MindSpore role | Kernel backend + interop | Clean API, escape hatch for power users |
| Hardware targets | CPU, GPU, Ascend | All three, develop on CPU/macOS first |
| Migration | Parallel rewrite | mindtorch_v2 alongside existing mindtorch |
| Validation target | Full BERT forward/backward | Proves architecture works for real models |
| Compatibility level | Deep (including internals) | Support transformers, diffusers, etc. |

## Component Designs

### 1. Tensor & Storage

**Storage Class** - Wraps contiguous MindSpore tensor:
```
Storage
├── _ms_tensor: mindspore.Tensor  (contiguous data)
├── _size: int                     (byte count)
├── _dtype: torch.dtype
├── _device: torch.device
└── _weak_refs: list[Tensor]       (for memory tracking)
```

**Tensor Class** - View metadata over Storage:
```
Tensor
├── _storage: Storage
├── _storage_offset: int
├── _shape: tuple[int, ...]
├── _stride: tuple[int, ...]
├── _requires_grad: bool
├── _grad_fn: Optional[Node]       (autograd link)
├── _grad: Optional[Tensor]
└── _version: int                  (in-place mutation tracking)
```

**View Semantics**: Reshape/transpose/slice create new Tensor with same Storage, different offset/stride. Mutations propagate. `_version` counter enables autograd safety checks.

### 2. Autograd Engine

**Graph Structure**:
```
Node (base class for all grad_fns)
├── next_functions: tuple[(Node, int), ...]   (inputs to this op)
├── _saved_tensors: tuple[Tensor, ...]        (saved for backward)
├── _needs_input_grad: tuple[bool, ...]       (which inputs need grad)
└── backward(grad_outputs) → grad_inputs       (override per op)
```

**Recording**: When `requires_grad=True` tensor enters an op:
1. Op executes forward, produces output
2. Creates Node with saved tensors and links to input nodes
3. Output tensor's `_grad_fn` points to this Node

**Backward Pass** (`torch.autograd.backward`):
1. Topologically sort nodes from output to inputs
2. For each node, call `node.backward(grad)`
3. Accumulate gradients into leaf tensors' `_grad`
4. Respect `retain_graph`, `create_graph` flags

**Required Features**:
- `register_hook` on Tensors (grad hooks)
- `register_full_backward_hook` on Nodes
- `torch.no_grad()` context manager
- `torch.enable_grad()` / `torch.set_grad_enabled()`
- `grad_fn.next_functions` introspection
- Gradient accumulation (not replacement)

### 3. Dispatcher System

**Dispatch Keys** (ordered by priority):
```python
class DispatchKey(Enum):
    Autograd = auto()          # Record ops for backward
    AutocastCPU = auto()       # AMP (CPU)
    AutocastGPU = auto()       # AMP (GPU)
    Batched = auto()           # vmap support
    Functionalize = auto()     # Mutations → copies
    Tracing = auto()           # torch.jit tracing
    Backend_CPU = auto()       # CPU kernel execution
    Backend_CUDA = auto()      # GPU kernel execution
    Backend_Ascend = auto()    # NPU kernel execution
    CompositeExplicit = auto() # Decompose to primitives
```

**Dispatch Flow**:
1. Op called (e.g., `torch.add(a, b)`)
2. Dispatcher checks active keys (tensor properties + context)
3. Walks keys in priority order until one handles
4. Autograd key: wraps op, records Node, delegates to next key
5. Backend key: executes actual kernel

**Operator Registration**:
```python
@register_op("add", DispatchKey.Backend_CPU)
def add_cpu(a, b):
    from mindtorch._op_prim.cpu import add as prim_add
    return prim_add(a._storage._ms_tensor, b._storage._ms_tensor)
```

**Context Managers**:
- `torch.no_grad()` → disables Autograd key
- `torch.autocast()` → enables Autocast key
- `torch._C._set_tracing_state()` → enables Tracing key

### 4. Composite Ops

Problematic kernels defined as compositions of working primitives:

```python
@register_composite("softmax")
def softmax_composite(x, dim):
    x_max = x.max(dim=dim, keepdim=True).values
    exp_x = (x - x_max).exp()
    return exp_x / exp_x.sum(dim=dim, keepdim=True)

@register_composite("layer_norm")
def layer_norm_composite(x, normalized_shape, weight, bias, eps):
    mean = x.mean(dim=-1, keepdim=True)
    var = x.var(dim=-1, keepdim=True, unbiased=False)
    out = (x - mean) / (var + eps).sqrt()
    return out * weight + bias
```

**Op Status Tracking**:
```python
# ops/_status.py
OP_STATUS = {
    "softmax": {"cpu": "composite", "cuda": "native", "ascend": "native"},
    "layer_norm": {"cpu": "composite", "cuda": "composite", "ascend": "native"},
    "gelu": {"cpu": "native", "cuda": "native", "ascend": "native"},
}
```

### 5. MindSpore Interop

**API**:
```python
# mindtorch Tensor → MindSpore tensor
ms_tensor = torch_tensor.to_mindspore()

# MindSpore tensor → mindtorch Tensor
torch_tensor = mindtorch.from_mindspore(ms_tensor)

# Zero-copy when possible
torch_tensor = mindtorch.from_mindspore(ms_tensor, copy=False)
```

**Rules**:
1. `to_mindspore()` on non-contiguous → forces contiguous copy
2. `from_mindspore(copy=False)` wraps MS tensor as Storage directly
3. Autograd does not cross boundary - detaches gradient history
4. Device must match

**Warning System**:
```python
mindtorch.set_interop_warning(True)  # Help find migration gaps
```

## Project Structure

```
src/
├── mindnlp/              # Unchanged
├── mindtorch/            # Current implementation (v1)
└── mindtorch_v2/         # New implementation
    ├── __init__.py       # Public torch.* API
    ├── _tensor.py        # Tensor class
    ├── _storage.py       # Storage class
    ├── _autograd/
    │   ├── __init__.py
    │   ├── engine.py     # Backward pass execution
    │   ├── node.py       # Node base class
    │   ├── grad_mode.py  # no_grad, enable_grad
    │   └── function.py   # Function class (user custom autograd)
    ├── _dispatch/
    │   ├── __init__.py
    │   ├── keys.py       # DispatchKey enum
    │   ├── registry.py   # Op registration
    │   └── dispatcher.py # Dispatch logic
    ├── _ops/
    │   ├── __init__.py
    │   ├── _native/      # Direct MindSpore kernel calls
    │   ├── _composite/   # Composite op definitions
    │   └── _status.py    # Op status tracking
    ├── _backends/
    │   ├── cpu.py        # CPU backend (calls _op_prim/cpu)
    │   ├── cuda.py       # GPU backend (calls _op_prim/gpu)
    │   └── ascend.py     # NPU backend (calls _op_prim/npu)
    ├── nn/               # torch.nn modules
    ├── optim/            # torch.optim optimizers
    ├── utils/            # torch.utils.*
    └── _interop.py       # MindSpore conversion
```

## Testing Strategy

### Phase 1: Development Testing

**Snapshot Tests**:
```python
# tests/snapshots/test_ops.py
def test_add():
    a = torch.randn(3, 4)
    b = torch.randn(3, 4)
    result = torch.add(a, b)
    assert_matches_snapshot(result, "add_3x4")
```
- Snapshots stored as `.npz` files
- Generated once from PyTorch, compared against mindtorch

**PyTorch Test Suite Port**:
- Port tests from `pytorch/test/` for core ops
- Focus: `test_torch.py`, `test_autograd.py`, `test_nn.py`
- Track coverage: which tests pass, which need work

### Phase 2: Integration Testing

Use test-runner agent workflow:
1. Activate conda: `source ~/miniconda3/bin/activate mindnlp`
2. Run: `python tests/run_test.py -vs {test_file}`
3. Analyze failures, fix bugs in `src/mindtorch_v2/` only
4. Never modify files in `tests/transformers/`
5. Re-run until passing

**Milestone Validation**:
```bash
python tests/run_test.py -vs tests/transformers/tests/models/bert/test_modeling_bert.py
```

### Test Organization

```
tests/
├── snapshots/          # Op-level snapshot tests
│   ├── data/           # .npz snapshot files
│   └── test_*.py
├── pytorch_port/       # Ported PyTorch tests
│   ├── test_torch.py
│   └── test_autograd.py
└── transformers/       # Existing transformers tests (read-only)
```

## Implementation Roadmap

### Phase 1: Foundation
- [ ] Storage class with MindSpore tensor wrapper
- [ ] Tensor class with shape, stride, offset, dtype
- [ ] Basic tensor creation: `empty`, `zeros`, `ones`, `randn`, `tensor`
- [ ] View operations: `view`, `reshape`, `transpose`, `permute`, `contiguous`
- [ ] Indexing: `__getitem__`, `__setitem__` with full slice/index support

### Phase 2: Dispatch + Core Ops
- [ ] DispatchKey enum and dispatcher skeleton
- [ ] Op registration decorators
- [ ] Backend abstraction layer (cpu first)
- [ ] Math ops: `add`, `sub`, `mul`, `div`, `matmul`, `pow`
- [ ] Reduction ops: `sum`, `mean`, `max`, `min`, `softmax`
- [ ] Comparison ops: `eq`, `ne`, `gt`, `lt`, `ge`, `le`

### Phase 3: Autograd
- [ ] Node base class and graph structure
- [ ] Backward engine with topological sort
- [ ] Gradient recording in dispatcher
- [ ] `no_grad`, `enable_grad` context managers
- [ ] Backward functions for Phase 2 ops

### Phase 4: nn.Module + Layers
- [ ] Module, Parameter, ParameterList, ModuleList
- [ ] Linear, Embedding, LayerNorm, Dropout
- [ ] Conv1d/2d, BatchNorm, Pooling
- [ ] Activation functions: ReLU, GELU, SiLU, Softmax

### Phase 5: BERT Validation
- [ ] Remaining ops needed for BERT (identified via test failures)
- [ ] Optimizers: AdamW
- [ ] Run BERT forward + backward
- [ ] Fix gaps until tests pass

## Backend Migration Notes

Development happens on macOS/CPU. Architecture supports easy migration:

1. **Backend abstraction**: All kernel calls go through `_backends/{cpu,cuda,ascend}.py`
2. **Op status tracking**: `_ops/_status.py` tracks which ops need composite fallbacks per backend
3. **Primitive calls**: Use `_op_prim/{cpu,gpu,npu}` directly, not `mindspore.ops` or `mindspore.mint`

To add GPU/Ascend support:
1. Implement `_backends/cuda.py` or `_backends/ascend.py`
2. Update `OP_STATUS` for ops that work natively on that backend
3. Add dispatch key routing for new device type
