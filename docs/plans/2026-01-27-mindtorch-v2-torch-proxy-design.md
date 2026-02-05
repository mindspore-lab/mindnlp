# mindtorch_v2 Torch Proxy Design

**Date**: 2026-01-27
**Status**: Ready for Implementation
**Goal**: Make mindtorch_v2 a drop-in replacement for torch so HuggingFace `transformers.BertModel` works directly

## Success Criteria

`BertModelTest::test_model` from transformers test suite passes with mindtorch_v2 as the torch backend.

## Scope

- **Target**: BERT-first minimal compatibility, expand incrementally
- **Operations**: Forward + backward pass (inference + gradient computation)
- **Implementation**: HuggingFace's actual `transformers.BertModel`
- **Approach**: Implement real equivalents of PyTorch internals (no fragile stubs)

## High-Level Architecture

```
┌─────────────────────────────────────────────────────────┐
│                    Python Import System                  │
├─────────────────────────────────────────────────────────┤
│  sys.meta_path.insert(0, MindTorchV2Finder)             │
│         │                                                │
│         ▼                                                │
│  ┌─────────────────┐    ┌─────────────────────────────┐ │
│  │ MindTorchV2Finder│───▶│ MindTorchV2Loader          │ │
│  │ (finds torch.*)  │    │ (returns shim modules)     │ │
│  └─────────────────┘    └─────────────────────────────┘ │
│                                   │                      │
│                                   ▼                      │
│                    ┌─────────────────────────────────┐  │
│                    │      mindtorch_v2 + shims       │  │
│                    │  ┌───────────┬───────────────┐  │  │
│                    │  │ Real impl │ Stub modules  │  │  │
│                    │  │ (Tensor,  │ (_C, _dynamo, │  │  │
│                    │  │  nn, etc) │  jit, etc)    │  │  │
│                    │  └───────────┴───────────────┘  │  │
│                    └─────────────────────────────────┘  │
└─────────────────────────────────────────────────────────┘
```

## Module Classification

### Tier 1: Real Implementation (must work correctly)

Actually used by BERT's forward/backward:

| torch module | mindtorch_v2 equivalent |
|--------------|------------------------|
| `torch` | `mindtorch_v2` |
| `torch.nn` | `mindtorch_v2.nn` |
| `torch.nn.functional` | `mindtorch_v2.nn.functional` |
| `torch.nn.Module` | `mindtorch_v2.nn.Module` |
| `torch.Tensor` | `mindtorch_v2.Tensor` |
| `torch.autograd` | `mindtorch_v2._autograd` |

### Tier 2: Functional Stubs (called but can be no-ops)

Called but don't affect BERT correctness:

| Function | Stub behavior |
|----------|---------------|
| `torch.cuda.is_available()` | `return False` |
| `torch.jit.script()` | `return function unchanged` |
| `torch.compile()` | `return function unchanged` |
| `torch.backends.cudnn.enabled` | `False` |
| `torch.distributed.is_initialized()` | `return False` |

### Tier 3: Import-Only Stubs (must exist but never called)

Imported by transformers' init but BERT never uses:

| Module | Stub approach |
|--------|---------------|
| `torch._C` | Empty module with `__getattr__` returning None |
| `torch._dynamo` | Empty module |
| `torch._inductor` | Empty module |
| `torch.utils._pytree` | Minimal `tree_flatten`/`tree_unflatten` |

## File Structure

```
src/mindtorch_v2/
├── __init__.py              # Main module (already exists)
├── _torch_proxy/            # NEW: Proxy loader system
│   ├── __init__.py          # Initializes proxy, exports install()
│   ├── finder.py            # MindTorchV2Finder class
│   ├── loader.py            # MindTorchV2Loader class
│   └── stubs/               # Stub modules for Tier 2 & 3
│       ├── __init__.py
│       ├── _C.py            # torch._C stub
│       ├── cuda.py          # torch.cuda stub
│       ├── jit.py           # torch.jit stub
│       ├── backends.py      # torch.backends stub
│       ├── distributed.py   # torch.distributed stub
│       └── _pytree.py       # torch.utils._pytree stub
├── _tensor.py               # (exists)
├── _autograd/               # (exists)
├── nn/                      # (exists)
└── optim/                   # (exists)
```

## Implementation Details

### MindTorchV2Finder

```python
class MindTorchV2Finder:
    """Intercept all torch.* imports."""

    def find_module(self, fullname, path=None):
        if fullname == 'torch' or fullname.startswith('torch.'):
            return MindTorchV2Loader()
        return None  # Let normal import handle non-torch
```

### MindTorchV2Loader

```python
class MindTorchV2Loader:
    """Return mindtorch_v2 or stubs for torch.* imports."""

    REAL_MODULES = {
        'torch': 'mindtorch_v2',
        'torch.nn': 'mindtorch_v2.nn',
        'torch.nn.functional': 'mindtorch_v2.nn.functional',
        'torch.optim': 'mindtorch_v2.optim',
        'torch.autograd': 'mindtorch_v2._autograd',
    }

    def load_module(self, fullname):
        if fullname in self.REAL_MODULES:
            real_name = self.REAL_MODULES[fullname]
            module = importlib.import_module(real_name)
        else:
            module = self._create_stub(fullname)

        sys.modules[fullname] = module
        return module
```

### Critical Stubs

**torch._C:**
```python
class _C:
    def __getattr__(self, name):
        return None
```

**torch.utils._pytree:**
```python
def tree_flatten(obj):
    if isinstance(obj, (list, tuple)):
        return list(obj), type(obj)
    return [obj], None

def tree_unflatten(values, spec):
    if spec is None:
        return values[0]
    return spec(values)
```

**torch.cuda:**
```python
def is_available(): return False
def device_count(): return 0
def current_device(): return 0
def synchronize(): pass
```

**torch.jit:**
```python
def script(fn): return fn
def trace(fn, *args): return fn
def is_scripting(): return False
def is_tracing(): return False
```

### Missing Tier 1 Implementations

**Add to mindtorch_v2/__init__.py:**
```python
FloatTensor = lambda *args: Tensor(*args, dtype=float32)
LongTensor = lambda *args: Tensor(*args, dtype=int64)
from_numpy = lambda arr: Tensor(arr)
einsum = lambda eq, *ops: Tensor(np.einsum(eq, *[o.numpy() for o in ops]))
compile = lambda fn: fn  # No-op for now
```

**Add to mindtorch_v2/nn/functional.py:**
```python
def scaled_dot_product_attention(query, key, value, attn_mask=None, ...):
    scores = matmul(query, key.transpose(-2, -1)) / math.sqrt(query.shape[-1])
    if attn_mask is not None:
        scores = scores + attn_mask
    attn = softmax(scores, dim=-1)
    return matmul(attn, value)
```

## Verification Strategy

```
Step 1: Import succeeds
────────────────────────
from mindtorch_v2._torch_proxy import install
install()
from transformers import BertModel, BertConfig  # No error

Step 2: Model creation works
────────────────────────────
config = BertConfig(hidden_size=64, num_hidden_layers=2, ...)
model = BertModel(config)  # No error

Step 3: Forward pass works
──────────────────────────
input_ids = torch.randint(0, 1000, (2, 8))
output = model(input_ids)
assert output.last_hidden_state.shape == (2, 8, 64)

Step 4: Backward pass works
───────────────────────────
loss = output.last_hidden_state.sum()
loss.backward()
assert all(p.grad is not None for p in model.parameters())

Step 5: Test suite passes
─────────────────────────
pytest tests/transformers/tests/models/bert/test_modeling_bert.py::BertModelTest::test_model
```

## Implementation Tasks

1. Create `_torch_proxy/` directory structure
2. Implement `MindTorchV2Finder` and `MindTorchV2Loader`
3. Create Tier 3 stubs (`_C`, `_pytree`, `_dynamo`, etc.)
4. Create Tier 2 stubs (`cuda`, `jit`, `backends`, `distributed`)
5. Add missing Tier 1 functions to mindtorch_v2
6. Add `scaled_dot_product_attention` to nn.functional
7. Verify Step 1-4 progressively
8. Run `BertModelTest::test_model` and fix failures
