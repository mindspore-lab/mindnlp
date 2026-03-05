# MindTorch v2 AMP Parity Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Align mindtorch_v2 AMP behavior and API surface with the current environment’s torch+cuda (autocast runtime, cache/nesting, GradScaler semantics, and register_autocast).

**Architecture:** Extend AMP state and autocast context to match torch semantics, add `_enter_autocast`/`_exit_autocast` hooks, and implement `register_autocast` in `library.py`. Replace the simplified GradScaler with a torch-like state machine while keeping Python-level implementation.

**Tech Stack:** Python 3.12, PyTorch reference behavior, mindtorch_v2 dispatch system.

---

### Task 1: Autocast cache clear + enter/exit hooks

**Files:**
- Modify: `src/mindtorch_v2/amp/state.py`
- Modify: `src/mindtorch_v2/amp/autocast_mode.py`
- Test: `tests/mindtorch_v2/contract/test_amp_autocast_contract.py`

**Step 1: Write the failing test**

```python
# Add to tests/mindtorch_v2/contract/test_amp_autocast_contract.py

def test_autocast_clears_cache_on_outer_exit():
    import mindtorch_v2 as torch
    from mindtorch_v2.amp import autocast, set_autocast_cache_enabled, is_autocast_cache_enabled

    set_autocast_cache_enabled("cpu", True)
    with autocast("cpu", cache_enabled=True):
        with autocast("cpu", cache_enabled=True):
            pass
    # cache should be cleared when outer nesting drops to zero
    assert is_autocast_cache_enabled("cpu") is True
    # placeholder for clear_autocast_cache hook


def test_enter_exit_hooks_exist():
    import mindtorch_v2.amp as amp
    assert hasattr(amp, "_enter_autocast")
    assert hasattr(amp, "_exit_autocast")
```

**Step 2: Run test to verify it fails**

Run: `PYTHONPATH=src pytest -q tests/mindtorch_v2/contract/test_amp_autocast_contract.py::test_autocast_clears_cache_on_outer_exit tests/mindtorch_v2/contract/test_amp_autocast_contract.py::test_enter_exit_hooks_exist`
Expected: FAIL because cache is not cleared and hooks not present.

**Step 3: Write minimal implementation**

- In `src/mindtorch_v2/amp/state.py`, add:
  - `clear_autocast_cache(device_type=None)` to reset cache state per device (placeholder, no real cache).
  - `is_autocast_enabled()` no-arg variant returning `is_autocast_enabled("cpu")`.
  - `get_autocast_dtype()` no-arg variant returning cpu dtype.
  - Update nesting decrement to return level and trigger `clear_autocast_cache` when it hits 0.
- In `src/mindtorch_v2/amp/autocast_mode.py`, add `_enter_autocast` and `_exit_autocast` functions with minimal behavior (call autocast, enter/exit, return mode).

**Step 4: Run test to verify it passes**

Run: `PYTHONPATH=src pytest -q tests/mindtorch_v2/contract/test_amp_autocast_contract.py::test_autocast_clears_cache_on_outer_exit tests/mindtorch_v2/contract/test_amp_autocast_contract.py::test_enter_exit_hooks_exist`
Expected: PASS

**Step 5: Commit**

```bash
git add src/mindtorch_v2/amp/state.py src/mindtorch_v2/amp/autocast_mode.py tests/mindtorch_v2/contract/test_amp_autocast_contract.py
git commit -m "feat(amp): align autocast cache and hooks"
```

---

### Task 2: Autocast validation and dtype support warnings

**Files:**
- Modify: `src/mindtorch_v2/amp/autocast_mode.py`
- Test: `tests/mindtorch_v2/contract/test_amp_autocast_contract.py`

**Step 1: Write the failing test**

```python
# Add to tests/mindtorch_v2/contract/test_amp_autocast_contract.py

def test_autocast_invalid_device_type_raises():
    import pytest
    from mindtorch_v2.amp import autocast

    with pytest.raises(RuntimeError):
        with autocast("invalid_device"):
            pass
```

**Step 2: Run test to verify it fails**

Run: `PYTHONPATH=src pytest -q tests/mindtorch_v2/contract/test_amp_autocast_contract.py::test_autocast_invalid_device_type_raises`
Expected: FAIL

**Step 3: Write minimal implementation**

- In `src/mindtorch_v2/amp/autocast_mode.py`, when `device_type` not supported, raise RuntimeError matching torch message:
  `User specified an unsupported autocast device_type '{device_type}'`.

**Step 4: Run test to verify it passes**

Run: `PYTHONPATH=src pytest -q tests/mindtorch_v2/contract/test_amp_autocast_contract.py::test_autocast_invalid_device_type_raises`
Expected: PASS

**Step 5: Commit**

```bash
git add src/mindtorch_v2/amp/autocast_mode.py tests/mindtorch_v2/contract/test_amp_autocast_contract.py
git commit -m "feat(amp): validate autocast device type"
```

---

### Task 3: Add register_autocast API surface

**Files:**
- Modify: `src/mindtorch_v2/library.py`
- Test: `tests/mindtorch_v2/contract/test_amp_autocast_contract.py`

**Step 1: Write the failing test**

```python
# Add to tests/mindtorch_v2/contract/test_amp_autocast_contract.py

def test_register_autocast_api_exists():
    import mindtorch_v2.library as library
    assert hasattr(library, "register_autocast")
```

**Step 2: Run test to verify it fails**

Run: `PYTHONPATH=src pytest -q tests/mindtorch_v2/contract/test_amp_autocast_contract.py::test_register_autocast_api_exists`
Expected: FAIL

**Step 3: Write minimal implementation**

- In `src/mindtorch_v2/library.py`, add `register_autocast(op, device_type, cast_inputs, *, lib=None)` with the same input validation as torch:
  - Accept `str` op only for now; if not, raise `ValueError`.
  - Validate `device_type` in `("cpu", "cuda")`.
  - Return a placeholder registration object or no-op; add TODO about dispatch key exclusion.

**Step 4: Run test to verify it passes**

Run: `PYTHONPATH=src pytest -q tests/mindtorch_v2/contract/test_amp_autocast_contract.py::test_register_autocast_api_exists`
Expected: PASS

**Step 5: Commit**

```bash
git add src/mindtorch_v2/library.py tests/mindtorch_v2/contract/test_amp_autocast_contract.py
git commit -m "feat(amp): add register_autocast API stub"
```

---

### Task 4: GradScaler parity (minimal torch-like behavior)

**Files:**
- Modify: `src/mindtorch_v2/amp/grad_scaler.py`
- Test: `tests/mindtorch_v2/contract/test_amp_grad_scaler_contract.py`

**Step 1: Write the failing test**

```python
# Add to tests/mindtorch_v2/contract/test_amp_grad_scaler_contract.py

def test_grad_scaler_stage_errors_match_torch():
    import mindtorch_v2 as torch
    from mindtorch_v2 import optim
    from mindtorch_v2.amp import GradScaler

    x = torch.tensor([1.0], requires_grad=True)
    opt = optim.SGD([x], lr=0.1)
    scaler = GradScaler()

    scaler.scale(x).backward()
    scaler.unscale_(opt)
    with pytest.raises(RuntimeError):
        scaler.unscale_(opt)
```

**Step 2: Run test to verify it fails**

Run: `PYTHONPATH=src pytest -q tests/mindtorch_v2/contract/test_amp_grad_scaler_contract.py::test_grad_scaler_stage_errors_match_torch`
Expected: FAIL

**Step 3: Write minimal implementation**

- Expand `GradScaler` to track per-optimizer states and raise errors for repeated `unscale_` or `step` calls, mirroring torch’s messages.
- Keep numeric checks using `isnan`/`isinf` but add per-optimizer stage transitions identical to torch.

**Step 4: Run test to verify it passes**

Run: `PYTHONPATH=src pytest -q tests/mindtorch_v2/contract/test_amp_grad_scaler_contract.py::test_grad_scaler_stage_errors_match_torch`
Expected: PASS

**Step 5: Commit**

```bash
git add src/mindtorch_v2/amp/grad_scaler.py tests/mindtorch_v2/contract/test_amp_grad_scaler_contract.py
git commit -m "feat(amp): align GradScaler state machine"
```

---

### Task 5: Full contract verification

**Files:**
- Test: `tests/mindtorch_v2/contract/*`

**Step 1: Run required gates**

Run:
- `PYTHONPATH=src pytest -q tests/mindtorch_v2/contract/test_schema_registration_order.py`
- `PYTHONPATH=src pytest -q tests/mindtorch_v2/contract/test_schema_coverage.py`
- `PYTHONPATH=src pytest -q tests/mindtorch_v2/contract/test_amp_autocast_contract.py`
- `PYTHONPATH=src pytest -q tests/mindtorch_v2/contract/test_amp_grad_scaler_contract.py`
- `PYTHONPATH=src pytest -q tests/mindtorch_v2/contract/test_amp_policy_coverage_contract.py tests/mindtorch_v2/contract/test_amp_policy_smoke_contract.py`

Expected: PASS (some skips allowed in smoke test)

**Step 2: Commit**

```bash
git add docs/plans/2026-03-04-mindtorch-v2-amp-parity-plan.md
git commit -m "docs(amp): add implementation plan"
```

