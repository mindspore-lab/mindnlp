# Random Seed and Dropout Implementation Design

## Overview

Implement PyTorch-compatible random number generation (RNG) system with manual seed control and dropout support for both CPU and NPU backends.

## Current State

**Working:**
- CPU randn using NumPy (no seed control)
- CPU dropout using NumPy random (no seed control)

**Missing:**
- `torch.manual_seed()` - global seed control
- `torch.random` module - RNG state management
- `torch.Generator` class - per-device RNG state
- NPU randn - random number generation on NPU
- NPU dropout - dropout operation on NPU
- Reproducible random operations

## PyTorch API Reference

```python
# Global seed control
torch.manual_seed(seed: int) -> Generator
torch.initial_seed() -> int

# RNG state management
torch.get_rng_state() -> Tensor
torch.set_rng_state(new_state: Tensor)

# Device-specific (torch.cuda, torch.npu)
torch.npu.manual_seed(seed: int)
torch.npu.manual_seed_all(seed: int)
torch.npu.get_rng_state(device: int = -1) -> Tensor
torch.npu.set_rng_state(new_state: Tensor, device: int = -1)

# Generator class
gen = torch.Generator(device='cpu')
gen.manual_seed(seed)
gen.initial_seed()
gen.get_state() -> Tensor
gen.set_state(new_state: Tensor)

# Usage in operations
torch.randn(5, 5, generator=gen)
F.dropout(x, p=0.5, training=True)  # Uses global RNG
```

## Architecture Design

### 1. Global RNG State Management

**File:** `src/mindtorch_v2/_random.py`

```python
# Global RNG state per device type
_cpu_rng_state = None  # NumPy RandomState
_npu_rng_states = {}   # Dict[int, NPU RNG state]
_initial_seed = None

def manual_seed(seed: int):
    """Set global seed for all devices."""
    global _initial_seed, _cpu_rng_state
    _initial_seed = seed
    _cpu_rng_state = np.random.RandomState(seed)
    # Set NPU seeds for all available devices
    if npu.is_available():
        for device_id in range(npu.device_count()):
            _set_npu_seed(device_id, seed)
    return default_generator

def initial_seed() -> int:
    """Get the initial seed."""
    return _initial_seed

def get_rng_state():
    """Get CPU RNG state as tensor."""
    global _cpu_rng_state
    if _cpu_rng_state is None:
        _cpu_rng_state = np.random.RandomState()
    state = _cpu_rng_state.get_state()
    # Convert to tensor (state is tuple: (str, ndarray, int, int, float))
    return tensor(state[1], dtype=uint8)

def set_rng_state(new_state):
    """Set CPU RNG state from tensor."""
    global _cpu_rng_state
    state_array = new_state.numpy()
    # Reconstruct state tuple
    _cpu_rng_state = np.random.RandomState()
    _cpu_rng_state.set_state(('MT19937', state_array, 624, 0, 0.0))
```

### 2. Generator Class

**File:** `src/mindtorch_v2/_random.py`

```python
class Generator:
    """RNG generator for a specific device."""

    def __init__(self, device='cpu'):
        self.device = Device(device) if not isinstance(device, Device) else device
        self._seed = None
        if self.device.type == 'cpu':
            self._rng = np.random.RandomState()
        elif self.device.type == 'npu':
            # NPU uses global state, just track seed
            self._rng = None

    def manual_seed(self, seed: int):
        """Set seed for this generator."""
        self._seed = seed
        if self.device.type == 'cpu':
            self._rng.seed(seed)
        elif self.device.type == 'npu':
            _set_npu_seed(self.device.index or 0, seed)
        return self

    def initial_seed(self) -> int:
        """Get initial seed."""
        return self._seed

    def get_state(self):
        """Get RNG state."""
        if self.device.type == 'cpu':
            state = self._rng.get_state()
            return tensor(state[1], dtype=uint8)
        else:
            raise NotImplementedError(f"get_state not supported for {self.device.type}")

    def set_state(self, new_state):
        """Set RNG state."""
        if self.device.type == 'cpu':
            state_array = new_state.numpy()
            self._rng.set_state(('MT19937', state_array, 624, 0, 0.0))
        else:
            raise NotImplementedError(f"set_state not supported for {self.device.type}")

# Default generator
default_generator = Generator('cpu')
```

### 3. CPU Backend Updates

**File:** `src/mindtorch_v2/_backends/cpu/ops.py`

```python
def dropout(a, p=0.5, training=True):
    """Dropout using global RNG state."""
    if not training or p == 0:
        return a

    # Use global RNG state for reproducibility
    from ..._random import _get_cpu_rng
    rng = _get_cpu_rng()

    arr = _to_numpy(a)
    mask = (rng.random(arr.shape) >= p).astype(arr.dtype)
    result = arr * mask / (1.0 - p)
    return _from_numpy(result, a.dtype, a.device)
```

**File:** `src/mindtorch_v2/_backends/cpu/creation.py`

```python
def randn_create(shape, dtype, device):
    """Create random normal tensor using global RNG state."""
    from ..._random import _get_cpu_rng
    rng = _get_cpu_rng()

    arr = rng.randn(*shape).astype(_dtype_to_numpy(dtype))
    storage = cpu_typed_storage_from_numpy(arr, dtype, device)
    stride = _contiguous_stride(shape)
    return _wrap_tensor(storage, shape, stride)
```

### 4. NPU Backend Implementation

**Strategy:** Use MindSpore's random ops with device context

**File:** `src/mindtorch_v2/_backends/npu/ops.py`

```python
def dropout(a, p=0.5, training=True):
    """Dropout using MindSpore Dropout op."""
    if not training or p == 0:
        return a

    # Use MindSpore Dropout primitive
    from mindspore.ops.auto_generate import gen_ops_prim
    dropout_op = gen_ops_prim.Dropout(keep_prob=1.0 - p).set_device('Ascend')

    runtime = npu_runtime.get_runtime((a.device.index or 0))
    stream = npu_state.current_stream((a.device.index or 0))

    # Convert to MindSpore tensor
    ms_tensor = _to_mindspore_tensor(a)

    # Apply dropout (returns tuple: output, mask)
    output, _ = dropout_op(ms_tensor)

    # Convert back
    return _from_mindspore_tensor(output, a.dtype, a.device)
```

**File:** `src/mindtorch_v2/_backends/npu/creation.py`

```python
def randn_create(shape, dtype, device):
    """Create random normal tensor using MindSpore StandardNormal."""
    from mindspore.ops.auto_generate import gen_ops_prim

    # Use StandardNormal primitive
    std_normal_op = gen_ops_prim.StandardNormal(seed=0, seed2=0).set_device('Ascend')

    # Create shape tensor
    import mindspore
    shape_tensor = mindspore.Tensor(shape, dtype=mindspore.int64)

    # Generate random tensor
    ms_output = std_normal_op(shape_tensor)

    # Convert to target dtype and wrap
    if dtype != float32:
        ms_output = ms_output.astype(_dtype_to_mindspore(dtype))

    # Wrap as mindtorch tensor
    return _from_mindspore_tensor(ms_output, dtype, device)
```

### 5. NPU Seed Management

**File:** `src/mindtorch_v2/npu.py`

```python
def manual_seed(seed: int):
    """Set seed for current NPU device."""
    import mindspore
    mindspore.set_seed(seed)

def manual_seed_all(seed: int):
    """Set seed for all NPU devices."""
    import mindspore
    mindspore.set_seed(seed)

def get_rng_state(device: int = -1):
    """Get NPU RNG state (placeholder)."""
    # MindSpore doesn't expose RNG state directly
    raise NotImplementedError("NPU RNG state management not supported by MindSpore")

def set_rng_state(new_state, device: int = -1):
    """Set NPU RNG state (placeholder)."""
    raise NotImplementedError("NPU RNG state management not supported by MindSpore")
```

## Implementation Plan

### Phase 1: Core RNG Infrastructure (CPU)

1. Create `_random.py` with:
   - Global RNG state management
   - `manual_seed()`, `initial_seed()`
   - `get_rng_state()`, `set_rng_state()`
   - `Generator` class

2. Update CPU backend:
   - Modify `cpu/ops.py` dropout to use global RNG
   - Modify `cpu/creation.py` randn to use global RNG

3. Export in `__init__.py`:
   - `manual_seed`, `initial_seed`
   - `get_rng_state`, `set_rng_state`

### Phase 2: NPU RNG Support

4. Implement NPU randn:
   - Add `randn_create()` in `npu/creation.py`
   - Use MindSpore StandardNormal op
   - Register in `npu/__init__.py`

5. Implement NPU dropout:
   - Update `npu/ops.py` dropout
   - Use MindSpore Dropout op
   - Register in `npu/__init__.py`

6. Add NPU seed control:
   - Add `manual_seed()` to `npu.py`
   - Use `mindspore.set_seed()`

### Phase 3: Testing

7. Write CPU tests:
   - Test manual_seed reproducibility
   - Test dropout with seed control
   - Test randn with seed control
   - Test get/set_rng_state

8. Write NPU tests:
   - Test NPU randn
   - Test NPU dropout
   - Test NPU manual_seed

## Testing Strategy

```python
# Test 1: CPU manual_seed reproducibility
torch.manual_seed(42)
x1 = torch.randn(5, 5)
torch.manual_seed(42)
x2 = torch.randn(5, 5)
assert torch.allclose(x1, x2)

# Test 2: CPU dropout reproducibility
torch.manual_seed(42)
out1 = F.dropout(x, p=0.5, training=True)
torch.manual_seed(42)
out2 = F.dropout(x, p=0.5, training=True)
assert torch.allclose(out1, out2)

# Test 3: NPU randn
x_npu = torch.randn(5, 5, device='npu')
assert x_npu.device.type == 'npu'

# Test 4: NPU dropout
x_npu = torch.randn(5, 5, device='npu')
out_npu = F.dropout(x_npu, p=0.5, training=True)
assert out_npu.device.type == 'npu'

# Test 5: RNG state save/restore
state = torch.get_rng_state()
x1 = torch.randn(5, 5)
torch.set_rng_state(state)
x2 = torch.randn(5, 5)
assert torch.allclose(x1, x2)
```

## Files to Modify

| File | Changes |
|------|---------|
| `src/mindtorch_v2/_random.py` | NEW: RNG infrastructure |
| `src/mindtorch_v2/__init__.py` | Export manual_seed, get/set_rng_state |
| `src/mindtorch_v2/_backends/cpu/ops.py` | Update dropout to use global RNG |
| `src/mindtorch_v2/_backends/cpu/creation.py` | Update randn to use global RNG |
| `src/mindtorch_v2/_backends/npu/creation.py` | Add randn_create |
| `src/mindtorch_v2/_backends/npu/ops.py` | Implement dropout |
| `src/mindtorch_v2/_backends/npu/__init__.py` | Register randn, dropout |
| `src/mindtorch_v2/npu.py` | Add manual_seed, manual_seed_all |
| `tests/mindtorch_v2/test_random_seed.py` | NEW: RNG tests |
| `tests/mindtorch_v2/test_dropout.py` | NEW: Dropout tests |

## Success Criteria

1. `torch.manual_seed(42)` makes randn and dropout reproducible on CPU
2. NPU randn works and creates tensors on NPU device
3. NPU dropout works and applies dropout on NPU device
4. `torch.npu.manual_seed()` controls NPU random operations
5. All existing tests still pass
6. New tests verify reproducibility and correctness
