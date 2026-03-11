# Random Seed Reproducibility Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Make mindtorch_v2 random number generation fully reproducible — `manual_seed(42)` must produce identical results on both CPU and NPU across runs.

**Architecture:** Replace ad-hoc global RNG state with a proper Generator-based architecture matching PyTorch's design. CPU uses numpy MT19937 via Generator objects. NPU uses seed+offset pairs passed to ACLNN kernels via Generator's `philox_engine_inputs()`. All random ops (creation, in-place, bernoulli, multinomial, dropout) route through Generator objects, respecting both default and user-provided generators.

**Tech Stack:** Python, numpy (CPU RNG), ACLNN ctypes bindings (NPU RNG)

---

## Bugs Found During Analysis

| # | Bug | Location | Impact |
|---|-----|----------|--------|
| 1 | `randint()` uses `np.random.randint()` directly, ignoring `manual_seed` and device dispatch | `_creation.py:66-79` | **Critical**: randint never reproducible, never runs on NPU |
| 2 | `randperm()` uses `np.random.permutation()` directly, same issues | `_creation.py:82-88` | **Critical**: randperm never reproducible on CPU, NPU dispatch broken |
| 3 | NPU `randn_create`/`rand_create` generate on CPU then copy to NPU | `_backends/npu/creation.py:79-118` | NPU random tensors use CPU RNG, not ACLNN |
| 4 | NPU `aclnn.randperm()` uses `random.randint()` for seed | `_backends/npu/aclnn.py:11800` | NPU randperm ignores `manual_seed` |
| 5 | `bernoulli()`/`multinomial()` ignore `generator` parameter | `_random.py:210-276` | Generator parameter is dead code |
| 6 | NPU offset advances by `numel`, torch_npu uses fixed increment | `npu.py:416-421`, `_backends/npu/ops.py` | Offset grows too fast, inconsistent with torch_npu |
| 7 | `Generator` class NPU support incomplete — no offset, no state serialization | `_random.py:113-203` | Cannot use independent NPU generators |
| 8 | No `npu.get_rng_state()`/`npu.set_rng_state()` | `npu.py` | Cannot save/restore NPU RNG state |

---

### Task 1: Fix `_creation.py` — `randint` and `randperm` must use dispatch

**Files:**
- Modify: `src/mindtorch_v2/_creation.py:66-88`

**Step 1: Fix `randint()` to use dispatch**

Replace the broken `randint` that uses `np.random.randint()` with one that calls through dispatch:

```python
# _creation.py lines 66-79 — REPLACE with:
def randint(low, high=None, size=None, *, dtype=None, device=None, generator=None):
    return randint_dispatch(low, high=high, size=size, dtype=dtype, device=device)
```

Note: `randint_dispatch` is already imported from `_functional.py` (line 14) but was never used.

**Step 2: Fix `randperm()` to use dispatch**

Replace the broken `randperm` that uses `np.random.permutation()`:

```python
# _creation.py lines 82-88 — REPLACE with:
def randperm(n, *, dtype=None, device=None, generator=None):
    return randperm_dispatch(n, dtype=dtype, device=device)
```

Note: `randperm_dispatch` is already imported from `_functional.py` (line 15) but was never used.

**Step 3: Add `generator` parameter to `randn` and `rand`**

```python
# _creation.py lines 58-63 — REPLACE with:
def randn(*shape, dtype=float32, device=None, memory_format=None, generator=None):
    return randn_dispatch(*shape, dtype=dtype, device=device, memory_format=memory_format)

def rand(*shape, dtype=float32, device=None, memory_format=None, generator=None):
    return rand_dispatch(*shape, dtype=dtype, device=device, memory_format=memory_format)
```

Generator parameter threading will be added in Task 6.

**Step 4: Fix CPU `randint_create` to use `_get_cpu_rng()`**

Verify that `src/mindtorch_v2/_backends/cpu/creation.py:127-143` already uses `_get_cpu_rng()`. It does — this path was just never reached because `_creation.py` bypassed dispatch.

**Step 5: Run test to verify CPU random ops are now deterministic**

```bash
source ~/miniconda3/bin/activate mindspore
cd /home/lvyufeng/lvyufeng/mindnlp
python -c "
import sys; sys.path.insert(0, 'src')
from mindtorch_v2 import manual_seed, randn, rand, randint, randperm

manual_seed(42)
a1 = randn(3, 3)
b1 = rand(3, 3)
c1 = randint(0, 10, size=(3, 3))
d1 = randperm(10)

manual_seed(42)
a2 = randn(3, 3)
b2 = rand(3, 3)
c2 = randint(0, 10, size=(3, 3))
d2 = randperm(10)

import numpy as np
assert np.array_equal(a1.numpy(), a2.numpy()), 'randn not reproducible'
assert np.array_equal(b1.numpy(), b2.numpy()), 'rand not reproducible'
assert np.array_equal(c1.numpy(), c2.numpy()), 'randint not reproducible'
assert np.array_equal(d1.numpy(), d2.numpy()), 'randperm not reproducible'
print('PASS: All CPU random ops are reproducible with manual_seed')
"
```

**Step 6: Commit**

```bash
git add src/mindtorch_v2/_creation.py
git commit -m "fix(mindtorch_v2): route randint/randperm through dispatch for reproducibility"
```

---

### Task 2: Redesign Generator class with NPU support

**Files:**
- Modify: `src/mindtorch_v2/_random.py`

**Step 1: Rewrite Generator class**

Replace the existing Generator class (lines 113-203) with a complete implementation:

```python
class Generator:
    """Random number generator for a specific device.

    Args:
        device (str or Device): The device for this generator. Default: 'cpu'
    """

    # PyTorch's default seed constant
    _DEFAULT_SEED = 67280421310721

    def __init__(self, device='cpu'):
        self.device = Device(device) if not isinstance(device, Device) else device
        self._seed = self._DEFAULT_SEED

        if self.device.type == 'cpu':
            self._rng = np.random.RandomState(self._seed & 0xffffffff)
        elif self.device.type == 'npu':
            self._rng = None
            self._offset = 0
        else:
            raise ValueError(f"Unsupported device type: {self.device.type}")

    def manual_seed(self, seed: int):
        """Set the seed for this generator."""
        if seed < -0x8000_0000_0000_0000 or seed > 0xffff_ffff_ffff_ffff:
            raise RuntimeError(
                f"Seed must be within range [-2^63, 2^64-1], got {seed}"
            )
        if seed < 0:
            seed = 0xffff_ffff_ffff_ffff + seed

        self._seed = seed

        if self.device.type == 'cpu':
            self._rng = np.random.RandomState(seed & 0xffffffff)
        elif self.device.type == 'npu':
            self._offset = 0
        return self

    def seed(self):
        """Set the seed to a random value and return it."""
        import time
        s = int(time.time() * 1000000) & 0xffff_ffff_ffff_ffff
        self.manual_seed(s)
        return s

    def initial_seed(self) -> int:
        """Get the current seed for this generator."""
        return self._seed

    def get_state(self):
        """Get the RNG state as a ByteTensor."""
        from ._creation import tensor

        if self.device.type == 'cpu':
            state = self._rng.get_state()
            state_bytes = state[1].view(np.uint8)
            pos = np.array([state[2]], dtype=np.int32).view(np.uint8)
            return tensor(np.concatenate([state_bytes, pos]), dtype=uint8)
        elif self.device.type == 'npu':
            # NPU state: seed (uint64) + offset (int64) = 16 bytes
            buf = np.zeros(16, dtype=np.uint8)
            buf[:8] = np.array([self._seed], dtype=np.uint64).view(np.uint8)
            buf[8:] = np.array([self._offset], dtype=np.int64).view(np.uint8)
            return tensor(buf, dtype=uint8)

    def set_state(self, new_state):
        """Set the RNG state from a ByteTensor."""
        if hasattr(new_state, 'device') and new_state.device.type != 'cpu':
            new_state = new_state.to('cpu')
        raw = new_state.numpy()

        if self.device.type == 'cpu':
            state_array = raw[:624 * 4].view(np.uint32)
            pos = raw[624 * 4:624 * 4 + 4].view(np.int32)[0]
            self._rng = np.random.RandomState()
            self._rng.set_state(('MT19937', state_array, int(pos), 0, 0.0))
        elif self.device.type == 'npu':
            self._seed = int(raw[:8].view(np.uint64)[0])
            self._offset = int(raw[8:16].view(np.int64)[0])

    def philox_engine_inputs(self, increment=10):
        """Get (seed, offset) for NPU ACLNN kernels and advance offset.

        This matches torch_npu's NPUGeneratorImpl::philox_engine_inputs().
        The increment represents Philox rounds of separation between ops.

        Args:
            increment: Number of Philox rounds to advance. Default: 10.

        Returns:
            tuple: (seed, offset) as integers.
        """
        if self.device.type != 'npu':
            raise RuntimeError("philox_engine_inputs only for NPU generators")
        seed = self._seed
        offset = self._offset
        self._offset += increment
        return seed, offset
```

**Step 2: Update `default_generator` initialization**

```python
# Use PyTorch's default seed constant
default_generator = Generator('cpu')
```

**Step 3: Run existing tests**

```bash
python tests/run_test_v2.py -vs tests/mindtorch_v2/test_random_seed.py
```

**Step 4: Commit**

```bash
git add src/mindtorch_v2/_random.py
git commit -m "feat(mindtorch_v2): redesign Generator class with NPU philox support"
```

---

### Task 3: Redesign `npu.py` seed management — use Generator objects

**Files:**
- Modify: `src/mindtorch_v2/npu.py`

**Step 1: Replace global seed variables with default generator**

Remove the old globals `_NPU_SEED` and `_NPU_SEED_OFFSET`. Replace with a lazily-created default generator per device.

```python
# Remove these globals:
# _NPU_SEED = None
# _NPU_SEED_OFFSET = 0

# Add:
_default_generators = {}  # device_index -> Generator

def _get_default_generator(device_index=0):
    """Get or create the default NPU generator for a device."""
    if device_index not in _default_generators:
        from ._random import Generator
        gen = Generator(f'npu:{device_index}')
        _default_generators[device_index] = gen
    return _default_generators[device_index]

# For backward compat, expose as a property-like accessor
class _DefaultGeneratorsAccessor:
    """Lazy accessor for default NPU generators, mimicking torch.cuda.default_generators."""
    def __getitem__(self, index):
        return _get_default_generator(index)

default_generators = _DefaultGeneratorsAccessor()
```

**Step 2: Rewrite `manual_seed`, `_get_seed`, `_get_and_advance_offset`**

```python
def manual_seed(seed: int):
    """Set the seed for generating random numbers for the current NPU device."""
    dev_idx = current_device()
    gen = _get_default_generator(dev_idx)
    gen.manual_seed(seed)

def manual_seed_all(seed: int):
    """Set the seed for generating random numbers on all NPU devices."""
    from ._backends.npu.runtime import device_count
    try:
        n = device_count()
    except Exception:
        n = 1
    for i in range(n):
        gen = _get_default_generator(i)
        gen.manual_seed(seed)

def _get_seed(device_index=None):
    """Get current NPU seed for the given device."""
    if device_index is None:
        device_index = current_device()
    return _get_default_generator(device_index)._seed

def _get_and_advance_offset(device_index=None, increment=10):
    """Get (seed, offset) and advance offset for the given device."""
    if device_index is None:
        device_index = current_device()
    gen = _get_default_generator(device_index)
    return gen.philox_engine_inputs(increment)
```

**Step 3: Add `get_rng_state` / `set_rng_state` / `get_rng_state_all` / `set_rng_state_all`**

```python
def get_rng_state(device=None):
    """Get NPU RNG state as a ByteTensor."""
    dev = _normalize_npu_device(device)
    gen = _get_default_generator(dev.index or 0)
    return gen.get_state()

def set_rng_state(new_state, device=None):
    """Set NPU RNG state from a ByteTensor."""
    dev = _normalize_npu_device(device)
    gen = _get_default_generator(dev.index or 0)
    gen.set_state(new_state)

def get_rng_state_all():
    """Get RNG state for all NPU devices."""
    from ._backends.npu.runtime import device_count
    try:
        n = device_count()
    except Exception:
        n = 1
    return [get_rng_state(i) for i in range(n)]

def set_rng_state_all(states):
    """Set RNG state for all NPU devices."""
    for i, state in enumerate(states):
        set_rng_state(state, device=i)
```

**Step 4: Update `__all__` to export new functions**

Add `manual_seed`, `manual_seed_all`, `get_rng_state`, `set_rng_state`, `get_rng_state_all`, `set_rng_state_all`, `default_generators` to `__all__`.

**Step 5: Commit**

```bash
git add src/mindtorch_v2/npu.py
git commit -m "feat(mindtorch_v2): replace NPU global seed with Generator-based RNG management"
```

---

### Task 4: Update `_random.py` top-level functions to use Generator

**Files:**
- Modify: `src/mindtorch_v2/_random.py`

**Step 1: Rewrite `manual_seed()` to use `default_generator`**

```python
def manual_seed(seed: int):
    """Set the seed for generating random numbers on all devices."""
    global _cpu_rng_state

    # Seed the CPU default generator
    default_generator.manual_seed(seed)

    # Keep _cpu_rng_state in sync (used by backend creation ops)
    _cpu_rng_state = default_generator._rng

    # Propagate to all NPU devices
    try:
        from . import npu
        if npu.is_available():
            npu.manual_seed_all(seed)
    except Exception:
        pass

    return default_generator
```

**Step 2: Update `_get_cpu_rng()` to use `default_generator`**

```python
def _get_cpu_rng():
    """Get the CPU RNG state from the default generator."""
    return default_generator._rng
```

This ensures all CPU random ops (creation ops, in-place ops) use the same RNG state as the default generator.

**Step 3: Rewrite `get_rng_state()` / `set_rng_state()`**

```python
def get_rng_state():
    """Get the CPU RNG state as a ByteTensor."""
    return default_generator.get_state()

def set_rng_state(new_state):
    """Set the CPU RNG state from a ByteTensor."""
    global _cpu_rng_state
    default_generator.set_state(new_state)
    _cpu_rng_state = default_generator._rng
```

**Step 4: Rewrite `seed()` and `initial_seed()`**

```python
def seed():
    """Set the seed to a random number and return it."""
    s = default_generator.seed()
    # Propagate to NPU
    try:
        from . import npu
        if npu.is_available():
            npu.manual_seed_all(s)
    except Exception:
        pass
    return s

def initial_seed() -> int:
    """Get the initial seed for the default CPU generator."""
    return default_generator.initial_seed()
```

**Step 5: Update `bernoulli()` to respect generator parameter**

```python
def bernoulli(input, *, generator=None):
    """Sample Bernoulli distribution given probabilities tensor."""
    from ._creation import tensor
    from ._dtype import float32

    rng = generator._rng if (generator is not None and generator.device.type == 'cpu') else _get_cpu_rng()
    if hasattr(input, '_numpy_view'):
        probs = input._numpy_view().copy()
    else:
        probs = np.array(input, dtype=np.float32)
    uniform = rng.uniform(0.0, 1.0, size=probs.shape)
    out = (uniform < probs).astype(probs.dtype)
    return tensor(out, dtype=input.dtype if hasattr(input, 'dtype') else float32)
```

**Step 6: Update `multinomial()` to respect generator parameter**

```python
def multinomial(input, num_samples, replacement=False, *, generator=None):
    """Sample indices from a multinomial distribution."""
    from ._creation import tensor
    from ._dtype import int64

    rng = generator._rng if (generator is not None and generator.device.type == 'cpu') else _get_cpu_rng()
    # ... rest of implementation unchanged, just uses `rng` variable ...
```

**Step 7: Commit**

```bash
git add src/mindtorch_v2/_random.py
git commit -m "feat(mindtorch_v2): unify RNG state through Generator, support generator param"
```

---

### Task 5: Fix NPU random ops — use ACLNN kernels and fix offset

**Files:**
- Modify: `src/mindtorch_v2/_backends/npu/creation.py:79-118`
- Modify: `src/mindtorch_v2/_backends/npu/ops.py` (uniform_, normal_, dropout)
- Modify: `src/mindtorch_v2/_backends/npu/aclnn.py` (randperm)

**Step 1: Rewrite NPU `randn_create` to use ACLNN**

```python
# creation.py — replace randn_create (lines 79-97)
def randn_create(shape, dtype=None, device=None, requires_grad=False, memory_format=None):
    """Create a tensor filled with random numbers from N(0,1) on NPU."""
    if isinstance(shape, int):
        shape = (shape,)
    shape = tuple(shape)
    # Create empty NPU tensor, then fill with normal_ (uses ACLNN kernel)
    t = empty_create(shape, dtype=dtype, device=device, requires_grad=requires_grad,
                     memory_format=memory_format)
    from .ops import normal_
    normal_(t, mean=0.0, std=1.0)
    return t
```

**Step 2: Rewrite NPU `rand_create` to use ACLNN**

```python
# creation.py — replace rand_create (lines 100-118)
def rand_create(shape, dtype=None, device=None, requires_grad=False, memory_format=None):
    """Create a tensor filled with random numbers from U(0,1) on NPU."""
    if isinstance(shape, int):
        shape = (shape,)
    shape = tuple(shape)
    t = empty_create(shape, dtype=dtype, device=device, requires_grad=requires_grad,
                     memory_format=memory_format)
    from .ops import uniform_
    uniform_(t, low=0.0, high=1.0)
    return t
```

**Step 3: Fix NPU `uniform_` and `normal_` to use fixed offset increment**

In `ops.py`, change `_get_and_advance_offset(advance=_numel(a.shape))` to use the new `_get_and_advance_offset()` which returns `(seed, offset)`:

```python
# ops.py uniform_ — REPLACE seed/offset retrieval:
# OLD:
#   seed = npu_mod._get_seed()
#   offset = npu_mod._get_and_advance_offset(advance=_numel(a.shape))
# NEW:
    seed, offset = npu_mod._get_and_advance_offset(
        device_index=(a.device.index or 0), increment=10
    )
```

Same change for `normal_` and `dropout`.

**Step 4: Fix `aclnn.randperm()` to accept seed parameter**

```python
# aclnn.py — change randperm function signature and implementation:
# OLD (line 11800):
#   seed = random.randint(0, 2**31 - 1)
# NEW: accept seed as parameter
def randperm(n, out_ptr, dtype, runtime, stream=None, seed=None, offset=0):
    # ... existing setup code ...
    # Replace random.randint with:
    if seed is None:
        import random
        seed = random.randint(0, 2**31 - 1)
    # ... rest of function uses seed variable ...
```

**Step 5: Fix NPU `randperm` in ops.py to pass seed**

```python
# ops.py randperm — add seed from npu module:
    from ... import npu as npu_mod
    seed, offset = npu_mod._get_and_advance_offset(
        device_index=(device.index or 0), increment=10
    )
    aclnn.randperm(n, out_ptr, dtype, runtime, stream=stream.stream,
                   seed=seed, offset=offset)
```

**Step 6: Fix dropout seed retrieval**

```python
# ops.py dropout — same pattern:
# OLD:
#   seed = npu_mod._get_seed()
#   offset = npu_mod._get_and_advance_offset(advance=out_numel)
# NEW:
    seed, offset = npu_mod._get_and_advance_offset(
        device_index=(a.device.index or 0), increment=10
    )
```

**Step 7: Commit**

```bash
git add src/mindtorch_v2/_backends/npu/creation.py \
        src/mindtorch_v2/_backends/npu/ops.py \
        src/mindtorch_v2/_backends/npu/aclnn.py
git commit -m "fix(mindtorch_v2): NPU random ops use ACLNN kernels with proper seed management"
```

---

### Task 6: Write comprehensive reproducibility tests

**Files:**
- Modify: `tests/mindtorch_v2/test_random_seed.py`

**Step 1: Add CPU reproducibility tests**

```python
class TestCPUReproducibility(unittest.TestCase):
    """Test that all CPU random ops are reproducible with manual_seed."""

    def test_randn_reproducible(self):
        torch.manual_seed(42)
        a = torch.randn(5, 5)
        torch.manual_seed(42)
        b = torch.randn(5, 5)
        self.assertTrue(np.array_equal(a.numpy(), b.numpy()))

    def test_rand_reproducible(self):
        torch.manual_seed(42)
        a = torch.rand(5, 5)
        torch.manual_seed(42)
        b = torch.rand(5, 5)
        self.assertTrue(np.array_equal(a.numpy(), b.numpy()))

    def test_randint_reproducible(self):
        torch.manual_seed(42)
        a = torch.randint(0, 100, size=(5, 5))
        torch.manual_seed(42)
        b = torch.randint(0, 100, size=(5, 5))
        self.assertTrue(np.array_equal(a.numpy(), b.numpy()))

    def test_randperm_reproducible(self):
        torch.manual_seed(42)
        a = torch.randperm(100)
        torch.manual_seed(42)
        b = torch.randperm(100)
        self.assertTrue(np.array_equal(a.numpy(), b.numpy()))

    def test_uniform_reproducible(self):
        torch.manual_seed(42)
        a = torch.empty(5, 5).uniform_()
        torch.manual_seed(42)
        b = torch.empty(5, 5).uniform_()
        self.assertTrue(np.array_equal(a.numpy(), b.numpy()))

    def test_normal_reproducible(self):
        torch.manual_seed(42)
        a = torch.empty(5, 5).normal_()
        torch.manual_seed(42)
        b = torch.empty(5, 5).normal_()
        self.assertTrue(np.array_equal(a.numpy(), b.numpy()))

    def test_bernoulli_reproducible(self):
        probs = torch.full((5, 5), 0.5)
        torch.manual_seed(42)
        a = torch.bernoulli(probs)
        torch.manual_seed(42)
        b = torch.bernoulli(probs)
        self.assertTrue(np.array_equal(a.numpy(), b.numpy()))

    def test_multinomial_reproducible(self):
        weights = torch.tensor([1.0, 2.0, 3.0, 4.0])
        torch.manual_seed(42)
        a = torch.multinomial(weights, 10, replacement=True)
        torch.manual_seed(42)
        b = torch.multinomial(weights, 10, replacement=True)
        self.assertTrue(np.array_equal(a.numpy(), b.numpy()))

    def test_different_seeds_different_results(self):
        torch.manual_seed(42)
        a = torch.randn(100)
        torch.manual_seed(123)
        b = torch.randn(100)
        self.assertFalse(np.array_equal(a.numpy(), b.numpy()))

    def test_sequence_of_ops_reproducible(self):
        """Multiple ops in sequence must produce identical results."""
        torch.manual_seed(42)
        a1 = torch.randn(3, 3)
        a2 = torch.rand(3, 3)
        a3 = torch.randint(0, 10, size=(3, 3))
        a4 = torch.randperm(10)

        torch.manual_seed(42)
        b1 = torch.randn(3, 3)
        b2 = torch.rand(3, 3)
        b3 = torch.randint(0, 10, size=(3, 3))
        b4 = torch.randperm(10)

        self.assertTrue(np.array_equal(a1.numpy(), b1.numpy()))
        self.assertTrue(np.array_equal(a2.numpy(), b2.numpy()))
        self.assertTrue(np.array_equal(a3.numpy(), b3.numpy()))
        self.assertTrue(np.array_equal(a4.numpy(), b4.numpy()))
```

**Step 2: Add Generator tests**

```python
class TestGeneratorFull(unittest.TestCase):
    """Test Generator class functionality."""

    def test_cpu_generator_independent(self):
        """User generator should be independent from default generator."""
        g = torch.Generator('cpu')
        g.manual_seed(42)
        torch.manual_seed(999)  # set default to different seed
        # Operations using generator should not be affected by default seed
        a = torch.bernoulli(torch.full((100,), 0.5), generator=g)

        g2 = torch.Generator('cpu')
        g2.manual_seed(42)
        b = torch.bernoulli(torch.full((100,), 0.5), generator=g2)
        self.assertTrue(np.array_equal(a.numpy(), b.numpy()))

    def test_generator_state_save_restore(self):
        g = torch.Generator('cpu')
        g.manual_seed(42)
        _ = torch.bernoulli(torch.full((10,), 0.5), generator=g)  # advance state
        state = g.get_state()
        a = torch.bernoulli(torch.full((10,), 0.5), generator=g)
        g.set_state(state)
        b = torch.bernoulli(torch.full((10,), 0.5), generator=g)
        self.assertTrue(np.array_equal(a.numpy(), b.numpy()))

    def test_initial_seed_default(self):
        g = torch.Generator('cpu')
        self.assertEqual(g.initial_seed(), 67280421310721)

    def test_manual_seed_returns_self(self):
        g = torch.Generator('cpu')
        result = g.manual_seed(42)
        self.assertIs(result, g)
```

**Step 3: Add NPU reproducibility tests (skip if NPU unavailable)**

```python
@unittest.skipUnless(HAS_NPU, "NPU not available")
class TestNPUReproducibility(unittest.TestCase):
    """Test that NPU random ops are reproducible with manual_seed."""

    def test_randn_npu_reproducible(self):
        torch.manual_seed(42)
        a = torch.randn(5, 5, device='npu')
        torch.manual_seed(42)
        b = torch.randn(5, 5, device='npu')
        self.assertTrue(np.array_equal(a.cpu().numpy(), b.cpu().numpy()))

    def test_rand_npu_reproducible(self):
        torch.manual_seed(42)
        a = torch.rand(5, 5, device='npu')
        torch.manual_seed(42)
        b = torch.rand(5, 5, device='npu')
        self.assertTrue(np.array_equal(a.cpu().numpy(), b.cpu().numpy()))

    def test_uniform_npu_reproducible(self):
        torch.manual_seed(42)
        a = torch.empty(5, 5, device='npu').uniform_()
        torch.manual_seed(42)
        b = torch.empty(5, 5, device='npu').uniform_()
        self.assertTrue(np.array_equal(a.cpu().numpy(), b.cpu().numpy()))

    def test_normal_npu_reproducible(self):
        torch.manual_seed(42)
        a = torch.empty(5, 5, device='npu').normal_()
        torch.manual_seed(42)
        b = torch.empty(5, 5, device='npu').normal_()
        self.assertTrue(np.array_equal(a.cpu().numpy(), b.cpu().numpy()))

    def test_randperm_npu_reproducible(self):
        torch.manual_seed(42)
        a = torch.randperm(100, device='npu')
        torch.manual_seed(42)
        b = torch.randperm(100, device='npu')
        self.assertTrue(np.array_equal(a.cpu().numpy(), b.cpu().numpy()))

    def test_dropout_npu_reproducible(self):
        torch.manual_seed(42)
        x = torch.ones(10, 10, device='npu')
        a = torch.nn.functional.dropout(x, p=0.5, training=True)
        torch.manual_seed(42)
        b = torch.nn.functional.dropout(x, p=0.5, training=True)
        self.assertTrue(np.array_equal(a.cpu().numpy(), b.cpu().numpy()))

    def test_npu_rng_state_save_restore(self):
        torch.manual_seed(42)
        from mindtorch_v2 import npu
        state = npu.get_rng_state()
        a = torch.randn(5, 5, device='npu')
        npu.set_rng_state(state)
        b = torch.randn(5, 5, device='npu')
        self.assertTrue(np.array_equal(a.cpu().numpy(), b.cpu().numpy()))

    def test_npu_sequence_reproducible(self):
        """Multiple NPU ops in sequence must produce identical results."""
        torch.manual_seed(42)
        a1 = torch.randn(3, 3, device='npu')
        a2 = torch.rand(3, 3, device='npu')
        a3 = torch.empty(3, 3, device='npu').uniform_(-1, 1)
        a4 = torch.empty(3, 3, device='npu').normal_(0, 2)

        torch.manual_seed(42)
        b1 = torch.randn(3, 3, device='npu')
        b2 = torch.rand(3, 3, device='npu')
        b3 = torch.empty(3, 3, device='npu').uniform_(-1, 1)
        b4 = torch.empty(3, 3, device='npu').normal_(0, 2)

        self.assertTrue(np.array_equal(a1.cpu().numpy(), b1.cpu().numpy()))
        self.assertTrue(np.array_equal(a2.cpu().numpy(), b2.cpu().numpy()))
        self.assertTrue(np.array_equal(a3.cpu().numpy(), b3.cpu().numpy()))
        self.assertTrue(np.array_equal(a4.cpu().numpy(), b4.cpu().numpy()))
```

**Step 4: Run all tests**

```bash
python tests/run_test_v2.py -vs tests/mindtorch_v2/test_random_seed.py
```

**Step 5: Commit**

```bash
git add tests/mindtorch_v2/test_random_seed.py
git commit -m "test(mindtorch_v2): add comprehensive reproducibility tests for CPU and NPU"
```

---

### Task 7: Integration verification — run model tests

**Step 1: Run BERT model tests to verify nothing regresses**

```bash
python tests/run_test_v2.py -vs tests/transformers/tests/models/bert/test_modeling_bert.py::BertModelTest::test_model
```

**Step 2: Run Albert model tests**

```bash
python tests/run_test_v2.py -vs tests/transformers/tests/models/albert/test_modeling_albert.py::AlbertModelTest::test_model
```

**Step 3: Commit any additional fixes needed**

```bash
# Only if fixes were required
git add -u
git commit -m "fix(mindtorch_v2): address model test regressions from RNG refactor"
```

---

## Summary of Changes

| File | Change |
|------|--------|
| `src/mindtorch_v2/_creation.py` | Route randint/randperm through dispatch; add generator param |
| `src/mindtorch_v2/_random.py` | Redesign Generator with NPU support; unify all state through Generator; fix bernoulli/multinomial |
| `src/mindtorch_v2/npu.py` | Replace globals with Generator objects; add get/set_rng_state |
| `src/mindtorch_v2/_backends/npu/creation.py` | Use ACLNN kernels for randn/rand instead of CPU generation |
| `src/mindtorch_v2/_backends/npu/ops.py` | Fix offset increment to 10; update seed retrieval API |
| `src/mindtorch_v2/_backends/npu/aclnn.py` | Fix randperm to accept seed parameter |
| `tests/mindtorch_v2/test_random_seed.py` | Add comprehensive CPU + NPU reproducibility tests |
