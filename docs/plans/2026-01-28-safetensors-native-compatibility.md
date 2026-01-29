# Safetensors Native Compatibility for mindtorch_v2

## Problem Statement

When using mindtorch_v2's torch_proxy to intercept `import torch`, safetensors' Rust-based `safe_open(..., framework='pt').get_tensor()` returns `bool` instead of a tensor.

**Root Cause**: Safetensors' Rust code (compiled with PyO3) links directly to PyTorch's **C extension library**. When mindtorch_v2 intercepts `import torch`, only the Python-level module is replaced - the Rust code still uses PyTorch's C library for tensor creation, which fails silently.

**Goal**: Use safetensors' native APIs that work with mindtorch_v2 - no patches, no wrappers.

## Working Native APIs (Verified)

### API 1: `safetensors.torch.load(bytes)` ✅

```python
from safetensors.torch import load
with open(filename, 'rb') as f:
    data = f.read()
tensors = load(data)  # Uses deserialize() + _view2torch()
```

This works because `_view2torch()` creates tensors using Python-level torch APIs:
- `torch.frombuffer(data, dtype)`
- `torch.empty(shape, dtype)`
- `tensor.reshape(shape)`

### API 2: `safe_open(..., framework='numpy')` + `torch.from_numpy()` ✅

```python
from safetensors import safe_open
with safe_open(filename, framework='numpy') as f:
    for k in f.keys():
        np_array = f.get_tensor(k)  # Returns numpy.ndarray
        tensor = torch.from_numpy(np_array)  # Convert to mindtorch_v2 tensor
```

This works because:
- `framework='numpy'` tells Rust to return numpy arrays (no torch C API needed)
- `torch.from_numpy()` is a Python-level API that mindtorch_v2 implements

### API 3: `safe_open(..., framework='pt').get_tensor()` ❌

```python
from safetensors import safe_open
with safe_open(filename, framework='pt') as f:
    tensor = f.get_tensor(k)  # Returns bool - BROKEN
```

This fails because Rust calls PyTorch's C library directly.

## The Challenge: HuggingFace Transformers

HuggingFace transformers uses `safe_open(..., framework='pt')` directly in `modeling_utils.py`:

```python
# From transformers/modeling_utils.py line 484
with safe_open(checkpoint_file, framework="pt") as f:
    metadata = f.metadata()
    for k in f.keys():
        state_dict[k] = f.get_tensor(k)  # Calls broken Rust API
```

We need transformers to use either:
1. `safetensors.torch.load(bytes)` - read file → pass bytes
2. `safe_open(..., framework='numpy')` + `torch.from_numpy()` - numpy intermediate

## Implementation Strategy

### Option A: Modify HuggingFace transformers (In MindNLP fork)

Since MindNLP has its own fork/copy of transformers at `tests/transformers/src/transformers/`, we can modify `modeling_utils.py` to detect mindtorch_v2 and use a compatible loading path.

**Pros**: Clean, no monkey-patching
**Cons**: Need to maintain the modification across transformers updates

### Option B: Upstream contribution to safetensors

Contribute a feature to safetensors that provides a pure-Python fallback when the Rust tensor creation fails or when a non-PyTorch torch is detected.

**Pros**: Benefits entire ecosystem
**Cons**: Requires upstream acceptance, longer timeline

### Option C: Upstream contribution to transformers

Modify HuggingFace transformers to support alternative tensor backends or use `safetensors.torch.load()` instead of `safe_open`.

**Pros**: Benefits entire ecosystem
**Cons**: Requires upstream acceptance, longer timeline

**Recommended**: Start with **Option A** for immediate progress, then pursue **Option B/C** for long-term solution.

## Implementation Plan

### Phase 1: Verify mindtorch_v2 API Completeness

Verify these APIs work correctly with `safetensors.torch.load()`:

| API | Purpose | Status |
|-----|---------|--------|
| `torch.frombuffer(data, dtype)` | Create tensor from bytes | ✅ Verify |
| `torch.empty(shape, dtype)` | Create empty tensor | ✅ Verify |
| `torch.from_numpy(array)` | Convert numpy to tensor | ✅ Works |
| `tensor.reshape(shape)` | Reshape tensor | ✅ Verify |
| `torch.__version__` | Version check | ⚠️ Add |

### Phase 2: Modify transformers loading (Option A)

Modify `tests/transformers/src/transformers/modeling_utils.py`:

```python
def load_state_dict_with_safetensors(checkpoint_file, map_location=None):
    """Load safetensors checkpoint - compatible with mindtorch_v2."""
    # Check if using mindtorch_v2
    import torch
    use_numpy_fallback = not hasattr(torch, '_C')  # mindtorch_v2 has no _C module

    if use_numpy_fallback:
        # Use numpy framework + from_numpy (works with mindtorch_v2)
        from safetensors import safe_open
        state_dict = {}
        with safe_open(checkpoint_file, framework='numpy') as f:
            for k in f.keys():
                np_array = f.get_tensor(k)
                state_dict[k] = torch.from_numpy(np_array)
        return state_dict
    else:
        # Original PyTorch path
        with safe_open(checkpoint_file, framework="pt") as f:
            ...
```

### Phase 3: Test Full Model Loading

1. Run Albert model tests with modified transformers
2. Verify `from_pretrained()` works end-to-end
3. Validate model outputs match expected values

### Phase 4: Upstream Contributions (Future)

1. **safetensors PR**: Add `framework='auto'` or pure-Python fallback
2. **transformers PR**: Support alternative torch backends

## Required mindtorch_v2 APIs (Complete List)

### For `safetensors.torch.load()` / `_view2torch()`:

| API | Used For |
|-----|----------|
| `torch.frombuffer(data, dtype)` | Create tensor from raw bytes |
| `torch.empty(shape, dtype)` | Create empty tensor for zero-size case |
| `tensor.reshape(shape)` | Reshape loaded tensor |
| `torch.from_numpy(array)` | Big-endian byteswap fallback |
| `torch.float32, float16, bfloat16, int64, int32, int16, int8, uint8, bool, float64, complex64` | All supported dtypes |

### For `safetensors.torch.save()` / `_tobytes()`:

| API | Used For |
|-----|----------|
| `tensor.layout` | Check tensor is strided |
| `torch.strided` | Layout constant |
| `tensor.is_contiguous()` | Check contiguity |
| `tensor.contiguous()` | Make contiguous copy |
| `tensor.device.type` | Check device is CPU |
| `tensor.to("cpu")` | Move to CPU if needed |
| `tensor.data_ptr()` | Get memory pointer for ctypes |
| `tensor.shape` | Get shape for metadata |
| `tensor.dtype` | Get dtype for metadata |
| `str(dtype).split(".")[-1]` | Dtype string format |

### For `_find_shared_tensors()` (shared tensor detection):

| API | Used For |
|-----|----------|
| `tensor.untyped_storage()` | Get underlying storage |
| `storage.data_ptr()` | Storage memory address |
| `storage.nbytes()` | Storage size |
| `tensor.view(-1)[-1].data_ptr()` | End pointer calculation |
| `tensor.nelement()` | Element count |
| `torch.device("meta")` | Meta device comparison |

### For version checking:

| API | Used For |
|-----|----------|
| `torch.__version__` | Version comparison (e.g., "2.3.0") |

## Testing Strategy

1. **Unit tests for each API**
   - `test_frombuffer_all_dtypes()`
   - `test_dtype_string_format()`
   - `test_storage_data_ptr()`
   - `test_storage_nbytes()`

2. **Integration tests**
   - `test_safetensors_save_load_roundtrip()`
   - `test_load_with_numpy_framework()`

3. **Model loading tests**
   - Full Albert model `from_pretrained()` test
   - Verify model outputs match expected values

## Success Criteria

1. `safetensors.torch.save_file()` works (already verified ✅)
2. `safetensors.torch.load(bytes)` works (already verified ✅)
3. `safe_open(framework='numpy')` + `from_numpy()` works (already verified ✅)
4. Modified transformers can load Albert model via `from_pretrained()`
5. All Albert unit tests pass

## Files to Modify

1. `src/mindtorch_v2/__init__.py` - Add `__version__` (if missing)
2. `tests/transformers/src/transformers/modeling_utils.py` - Add numpy fallback for safetensors loading
3. `tests/run_test_v2.py` - Remove safetensors patch (use native APIs)
4. `tests/mindtorch_v2/test_safetensors_compat.py` - New test file for API verification

## Summary

**Key Insight**: Safetensors' Rust `safe_open(..., framework='pt')` cannot work with mindtorch_v2 because it's linked to PyTorch's C library at compile time.

**Solution**: Use native safetensors APIs that don't require PyTorch's C library:
1. `safetensors.torch.load(bytes)` - pure Python tensor creation
2. `safe_open(..., framework='numpy')` - returns numpy arrays, convert with `torch.from_numpy()`

**No wrappers, no patches** - just use the right native APIs and modify transformers to detect mindtorch_v2 and use the compatible loading path.
