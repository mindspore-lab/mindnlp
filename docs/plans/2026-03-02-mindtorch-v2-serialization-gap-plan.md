# MindTorch v2 Serialization Gap Closure Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Close the highest-impact serialization compatibility gaps in `mindtorch_v2` so `mt.save/mt.load` behaves closer to torch for common training/inference checkpoints while preserving torch-runtime-independent loading.

**Architecture:** Keep the current pure-Python zip/legacy loader/writer architecture in `src/mindtorch_v2/serialization.py` and expand behavior in small, test-driven increments. Prioritize API/semantic alignment first (`weights_only`, `map_location`, dtype/storage coverage, error semantics), then performance/ergonomics (`mmap`). Maintain strict backwards compatibility for currently passing round-trip/state_dict paths.

**Tech Stack:** Python, `pickle`, `zipfile`, `numpy`, `pytest`, `mindtorch_v2` tensor/storage internals.

---

## Current Capability Snapshot (Evidence)

- Baseline status: `PYTHONPATH=src pytest -q tests/mindtorch_v2/test_serialization.py` => `23 passed in 5.96s`.
- Public surface is currently `mt.save` / `mt.load` only (exported from `src/mindtorch_v2/__init__.py`).
- Implemented paths in `src/mindtorch_v2/serialization.py`:
  - Torch zip checkpoint load + mindtorch_v2 zip save.
  - Legacy checkpoint load.
  - Aliasing + non-contiguous stride roundtrip.
  - `map_location` supports `None`, `'cpu'`, `dict`, callable for zip path, but CPU-only remap target.
- Existing tests (`tests/mindtorch_v2/test_serialization.py`) cover state_dict, optimizer-state basic shape, alias/stride, bytesio, and a small dtype matrix.

## Capability Gap Inventory (Prioritized)

1. **`weights_only` argument is ignored**
- Evidence: `load(..., weights_only=False, **kwargs)` immediately discards it (`_ = pickle_module, weights_only, kwargs`).
- Risk: Security and behavior mismatch with torch semantics on untrusted checkpoints.

2. **`pickle_module` and load kwargs are effectively ignored for non-encoding behavior**
- Evidence: `load` hardcodes `encoding='utf-8'` and ignores caller-provided `pickle_module`/kwargs semantics.
- Risk: Compatibility gaps for callers relying on custom unpicklers or exact torch behavior.

3. **Serialization dtype coverage lags behind `mindtorch_v2` dtype surface**
- Evidence: `_DTYPE_NAME_TO_STORAGE` includes `float16/32/64`, `bfloat16`, `int8/16/32/64`, `uint8`, `bool`, `complex64/128`; excludes `uint16/32/64`, float8 family.
- Risk: `mt.save` raises `TypeError` on valid `mindtorch_v2` tensors for those dtypes.

4. **`map_location` is CPU-only for storage-location remap**
- Evidence: zip/legacy paths reject non-cpu locations.
- Risk: limited portability for checkpoints with non-cpu location tags.

5. **No `mmap` support in `mindtorch_v2.load`**
- Evidence: unlike v1 `mindtorch.serialization.load(..., mmap=...)`, v2 has no mmap argument behavior.
- Risk: performance/memory penalty for large checkpoints.

6. **Negative-path coverage is thin**
- Missing tests for corrupted zip records, unsupported map_location remaps, unsupported dtype failure contracts, and `weights_only` behavior.

---

### Task 1: Lock API Behavior Contracts Before Refactor

**Files:**
- Modify: `tests/mindtorch_v2/test_serialization.py`
- Test: `tests/mindtorch_v2/test_serialization.py`

**Step 1: Write failing tests for currently missing contracts**

```python
def test_load_rejects_unsupported_map_location_target(tmp_path):
    path = tmp_path / "x.pth"
    torch.save({"x": torch.tensor([1.0])}, path)
    with pytest.raises(NotImplementedError):
        mt.load(path, map_location={"cpu": "cuda:0"})


def test_save_unsupported_dtype_uint16_raises(tmp_path):
    x = mt.arange(0, 4, dtype=mt.uint16)
    with pytest.raises(TypeError):
        mt.save({"x": x}, tmp_path / "bad_dtype.pth")
```

**Step 2: Run tests to verify failures/behavior visibility**

Run: `PYTHONPATH=src pytest -q tests/mindtorch_v2/test_serialization.py -k "unsupported_map_location_target or unsupported_dtype_uint16"`

Expected: one pass (existing behavior), one explicit failure/contract visibility depending on implementation status.

**Step 3: Normalize assertions to document intended target behavior**

- For each gap test, add a short assertion on error type/message fragment.

**Step 4: Run full serialization suite**

Run: `PYTHONPATH=src pytest -q tests/mindtorch_v2/test_serialization.py`

Expected: known baseline + newly added contract checks.

**Step 5: Commit**

```bash
git add tests/mindtorch_v2/test_serialization.py
git commit -m "test(mindtorch_v2): add serialization gap contract coverage"
```

### Task 2: Implement Real `weights_only` Semantics (Security/Parity First)

**Files:**
- Modify: `src/mindtorch_v2/serialization.py`
- Modify: `tests/mindtorch_v2/test_serialization.py`
- Test: `tests/mindtorch_v2/test_serialization.py`

**Step 1: Write failing tests for `weights_only=True` and `weights_only=False` paths**

```python
def test_load_weights_only_blocks_non_tensor_globals(tmp_path):
    class Evil:
        pass
    path = tmp_path / "evil.pth"
    torch.save({"x": torch.tensor([1.0]), "evil": Evil()}, path)
    with pytest.raises(Exception):
        mt.load(path, weights_only=True)
```

**Step 2: Run targeted test to confirm failure first**

Run: `PYTHONPATH=src pytest -q tests/mindtorch_v2/test_serialization.py -k "weights_only"`

Expected: FAIL because current implementation ignores `weights_only`.

**Step 3: Implement minimal safe unpickler gate**

- In `serialization.py`, route `weights_only=True` through restricted `find_class` allowlist:
  - allow tensor rebuild helpers, storage markers, `OrderedDict`, basic containers.
  - reject arbitrary globals/modules with `UnpicklingError`/`RuntimeError`.

**Step 4: Re-run targeted + full suite**

Run:
- `PYTHONPATH=src pytest -q tests/mindtorch_v2/test_serialization.py -k "weights_only"`
- `PYTHONPATH=src pytest -q tests/mindtorch_v2/test_serialization.py`

Expected: targeted tests pass, no regressions in 23 baseline tests.

**Step 5: Commit**

```bash
git add src/mindtorch_v2/serialization.py tests/mindtorch_v2/test_serialization.py
git commit -m "feat(mindtorch_v2): honor weights_only in serialization load"
```

### Task 3: Expand Supported DType Matrix (v2 Surface Alignment)

**Files:**
- Modify: `src/mindtorch_v2/serialization.py`
- Modify: `tests/mindtorch_v2/test_serialization.py`
- Test: `tests/mindtorch_v2/test_serialization.py`

**Step 1: Add failing tests for additional dtypes**

- Add parameterized cases for `mt.int32`, `mt.int16`, `mt.int8`, `mt.uint8`, `mt.bfloat16`, `mt.complex64`, `mt.complex128` roundtrip.
- Add explicit expected-failure tests for unsupported `mt.uint16`, `mt.uint32`, `mt.uint64`, float8 family (until backend format decision is made).

**Step 2: Run targeted dtype tests**

Run: `PYTHONPATH=src pytest -q tests/mindtorch_v2/test_serialization.py -k "dtype_matrix or unsupported_dtype"`

Expected: failures that highlight missing support and/or mismatch.

**Step 3: Implement minimal dtype mapping extension where safe**

- Extend `_DTYPE_NAME_TO_STORAGE` and `_STORAGE_NAME_TO_DTYPE` only for torch-compatible storages that exist and are verified.
- Keep explicit `TypeError` for dtypes without stable torch storage counterpart.

**Step 4: Re-run dtype + full suite**

Run:
- `PYTHONPATH=src pytest -q tests/mindtorch_v2/test_serialization.py -k "dtype"`
- `PYTHONPATH=src pytest -q tests/mindtorch_v2/test_serialization.py`

Expected: expanded pass matrix, explicit failure contract for unsupported dtypes.

**Step 5: Commit**

```bash
git add src/mindtorch_v2/serialization.py tests/mindtorch_v2/test_serialization.py
git commit -m "feat(mindtorch_v2): expand serialization dtype coverage and contracts"
```

### Task 4: Improve `map_location` Behavior Parity

**Files:**
- Modify: `src/mindtorch_v2/serialization.py`
- Modify: `tests/mindtorch_v2/test_serialization.py`
- Test: `tests/mindtorch_v2/test_serialization.py`

**Step 1: Add failing tests for map-location edge behavior**

- callable returns `None` (keep original storage).
- dict missing key should preserve current behavior.
- explicit non-cpu remap should raise stable error class/message fragment.

**Step 2: Run targeted tests to verify failures or undefined behavior**

Run: `PYTHONPATH=src pytest -q tests/mindtorch_v2/test_serialization.py -k "map_location"`

**Step 3: Refactor `_validate_map_location` + `_apply_map_location` for clear contract**

- Keep CPU-only implementation for now, but make errors deterministic and consistent across zip/legacy paths.

**Step 4: Re-run targeted + full suite**

Run:
- `PYTHONPATH=src pytest -q tests/mindtorch_v2/test_serialization.py -k "map_location"`
- `PYTHONPATH=src pytest -q tests/mindtorch_v2/test_serialization.py`

**Step 5: Commit**

```bash
git add src/mindtorch_v2/serialization.py tests/mindtorch_v2/test_serialization.py
git commit -m "refactor(mindtorch_v2): tighten map_location contract for serialization"
```

### Task 5: Add Corruption and Negative-Path Robustness Tests

**Files:**
- Modify: `tests/mindtorch_v2/test_serialization.py`
- Test: `tests/mindtorch_v2/test_serialization.py`

**Step 1: Add failing tests for malformed checkpoint inputs**

- missing `data.pkl` in zip.
- truncated legacy storage payload.
- unknown storage typename.

**Step 2: Run targeted negative-path tests**

Run: `PYTHONPATH=src pytest -q tests/mindtorch_v2/test_serialization.py -k "corrupt or malformed or truncated"`

**Step 3: Adjust error assertions to lock expected exceptions/messages**

- Assert deterministic `RuntimeError`/`TypeError` with key message fragments.

**Step 4: Re-run full suite**

Run: `PYTHONPATH=src pytest -q tests/mindtorch_v2/test_serialization.py`

**Step 5: Commit**

```bash
git add tests/mindtorch_v2/test_serialization.py
git commit -m "test(mindtorch_v2): add serialization corruption-path coverage"
```

### Task 6: Optional Phase-2 `mmap` Support Design and Implementation

**Files:**
- Modify: `src/mindtorch_v2/serialization.py`
- Modify: `tests/mindtorch_v2/test_serialization.py`
- Create: `docs/plans/2026-03-02-mindtorch-v2-serialization-mmap-design.md` (if design split needed)
- Test: `tests/mindtorch_v2/test_serialization.py`

**Step 1: Write failing API parity tests for `mmap` argument**

- `mt.load(path, mmap=True)` allowed for path input.
- file-like + `mmap=True` raises `ValueError`.

**Step 2: Run targeted test to validate fail-first**

Run: `PYTHONPATH=src pytest -q tests/mindtorch_v2/test_serialization.py -k "mmap"`

**Step 3: Implement minimal zip-only mmap path**

- Use `numpy.memmap` for storage payload reads when `mmap=True` and `f` is path-like.
- Keep legacy path unsupported at first with explicit `RuntimeError`.

**Step 4: Re-run targeted + full serialization suite**

Run:
- `PYTHONPATH=src pytest -q tests/mindtorch_v2/test_serialization.py -k "mmap"`
- `PYTHONPATH=src pytest -q tests/mindtorch_v2/test_serialization.py`

**Step 5: Commit**

```bash
git add src/mindtorch_v2/serialization.py tests/mindtorch_v2/test_serialization.py
git commit -m "feat(mindtorch_v2): add optional mmap loading for zip checkpoints"
```

## Verification Gate Before Merge

- Required:
  - `PYTHONPATH=src pytest -q tests/mindtorch_v2/test_serialization.py`
- Recommended full v2 checkpoint-adjacent gate:
  - `PYTHONPATH=src pytest -q tests/mindtorch_v2`

## Out of Scope

- Distributed checkpoint API implementation (`mindtorch_v2.distributed.checkpoint` stubs).
- TorchScript archive loading support.
- Full non-CPU device restore semantics in v2 runtime.
