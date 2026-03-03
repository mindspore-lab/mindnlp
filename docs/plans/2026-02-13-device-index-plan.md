# Device Index Support Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Add torch-like device indices (cpu/meta/npu) with per-device NPU runtimes and cross-device NPU copy.

**Architecture:** Extend the `device` object to parse and store indices; dispatch stays keyed by device type but tensors carry full device objects. NPU runtime becomes a pool keyed by device id, and cross-device NPU copy uses CPU staging.

**Tech Stack:** Python, NumPy, ACL (ctypes), pytest.

---

### Task 1: Device parsing + representation

**Files:**
- Create: `tests/mindtorch_v2/test_device_index.py`
- Modify: `src/mindtorch_v2/_device.py`

**Step 1: Write the failing test**

```python
import mindtorch_v2 as torch


def test_device_parsing_and_repr():
    dev = torch.Device("npu:1")
    assert dev.type == "npu"
    assert dev.index == 1
    assert repr(dev) == "npu:1"

    dev = torch.Device("cpu")
    assert dev.type == "cpu"
    assert dev.index is None
    assert repr(dev) == "cpu"

    dev = torch.Device("meta:1")
    assert dev.type == "meta"
    assert dev.index == 1
    assert repr(dev) == "meta:1"

    dev = torch.Device("cpu", 1)
    assert dev.type == "cpu"
    assert dev.index == 1
    assert repr(dev) == "cpu:1"
```

**Step 2: Run test to verify it fails**

Run: `pytest tests/mindtorch_v2/test_device_index.py -v`
Expected: FAIL (missing `index` or parsing)

**Step 3: Write minimal implementation**

```python
class device:
    def __init__(self, dev, index=None):
        if isinstance(dev, device):
            self.type = dev.type
            self.index = dev.index if index is None else index
            return
        if isinstance(dev, str) and ":" in dev:
            dev, idx = dev.split(":", 1)
            index = int(idx)
        self.type = str(dev)
        self.index = None if index is None else int(index)

    def __repr__(self):
        if self.index is None:
            return f"{self.type}"
        return f"{self.type}:{self.index}"
```

**Step 4: Run test to verify it passes**

Run: `pytest tests/mindtorch_v2/test_device_index.py -v`
Expected: PASS

**Step 5: Commit**

```bash
git add tests/mindtorch_v2/test_device_index.py src/mindtorch_v2/_device.py
git commit -m "feat: parse device indices"
```

---

### Task 2: Propagate device indices for cpu/meta + dispatch

**Files:**
- Modify: `src/mindtorch_v2/_storage.py`
- Modify: `src/mindtorch_v2/_functional.py`
- Modify: `src/mindtorch_v2/_dispatch/dispatcher.py`
- Modify: `src/mindtorch_v2/_backends/cpu/creation.py`
- Modify: `src/mindtorch_v2/_backends/meta/creation.py`
- Test: `tests/mindtorch_v2/test_creation.py`

**Step 1: Write the failing test**

Add to `tests/mindtorch_v2/test_creation.py`:

```python
def test_creation_device_index_cpu_meta():
    cpu_tensor = torch.ones((1,), device="cpu:1")
    assert cpu_tensor.device.type == "cpu"
    assert cpu_tensor.device.index == 1

    meta_tensor = torch.ones((1,), device="meta:1")
    assert meta_tensor.device.type == "meta"
    assert meta_tensor.device.index == 1
```

**Step 2: Run test to verify it fails**

Run: `pytest tests/mindtorch_v2/test_creation.py::test_creation_device_index_cpu_meta -v`
Expected: FAIL (index lost in storage/device)

**Step 3: Write minimal implementation**

- Update `_CPUUntypedStorage` and `_MetaUntypedStorage` to accept an optional `device` argument (defaulting to `Device("cpu")` / `Device("meta")`) and store it.
- Update `typed_storage_from_numpy`, `empty_cpu_typed_storage`, `meta_typed_storage_from_shape`, and `meta_typed_storage_from_size` to accept an optional `device` and pass it through.
- Update cpu/meta creation functions to pass the incoming `device` (already parsed to `Device`) into storage creation.
- Update `_functional._as_device` to return `Device` objects for strings including indices.
- Update `dispatch` to accept a `device` object and use `device.type` for registry lookup while preserving the full device for pending storage.

**Step 4: Run test to verify it passes**

Run: `pytest tests/mindtorch_v2/test_creation.py::test_creation_device_index_cpu_meta -v`
Expected: PASS

**Step 5: Commit**

```bash
git add src/mindtorch_v2/_storage.py src/mindtorch_v2/_functional.py src/mindtorch_v2/_dispatch/dispatcher.py \
        src/mindtorch_v2/_backends/cpu/creation.py src/mindtorch_v2/_backends/meta/creation.py \
        tests/mindtorch_v2/test_creation.py
git commit -m "feat: preserve device index for cpu/meta"
```

---

### Task 3: NPU runtime pool + device-indexed storage

**Files:**
- Modify: `src/mindtorch_v2/_backends/npu/runtime.py`
- Modify: `src/mindtorch_v2/_storage.py`
- Modify: `src/mindtorch_v2/_backends/npu/creation.py`
- Modify: `src/mindtorch_v2/_backends/npu/ops.py`
- Modify: `src/mindtorch_v2/_backends/common/convert.py`
- Test: `tests/mindtorch_v2/test_ops_npu.py`

**Step 1: Write the failing test**

Add to `tests/mindtorch_v2/test_ops_npu.py`:

```python
def test_npu_device_index_preserved():
    if not torch.npu.is_available():
        pytest.skip("NPU not available")
    out = torch.ones((1,), device="npu:0")
    assert out.device.type == "npu"
    assert out.device.index == 0
```

**Step 2: Run test to verify it fails**

Run: `pytest tests/mindtorch_v2/test_ops_npu.py::test_npu_device_index_preserved -v`
Expected: FAIL (device index lost)

**Step 3: Write minimal implementation**

- Add runtime pool:
  ```python
  _RUNTIMES = {}

  def get_runtime(device_id=0):
      runtime = _RUNTIMES.get(device_id)
      if runtime is None:
          runtime = _Runtime()
          runtime.init(device_id)
          _RUNTIMES[device_id] = runtime
      else:
          runtime.activate()  # calls acl.rt.set_device(device_id)
      return runtime
  ```
- Add `_Runtime.activate()` to call `acl.rt.set_device(self.device_id)`.
- Update `_alloc_device`, `_copy_cpu_to_npu`, `_copy_npu_to_cpu` to accept a `runtime` parameter and use its device id.
- Update `_NPUUntypedStorage` and `npu_typed_storage_from_ptr` to accept a `device` object and store it.
- Update NPU creation and ops to use the runtime for `device.index or 0` and to pass the device into storage.
- Update `convert.to_device` NPU branches to pass device index to copy helpers.

**Step 4: Run test to verify it passes**

Run: `pytest tests/mindtorch_v2/test_ops_npu.py::test_npu_device_index_preserved -v`
Expected: PASS

**Step 5: Commit**

```bash
git add src/mindtorch_v2/_backends/npu/runtime.py src/mindtorch_v2/_storage.py \
        src/mindtorch_v2/_backends/npu/creation.py src/mindtorch_v2/_backends/npu/ops.py \
        src/mindtorch_v2/_backends/common/convert.py tests/mindtorch_v2/test_ops_npu.py
git commit -m "feat: npu device index runtime pool"
```

---

### Task 4: Cross-device NPU copy + device count helper + print index

**Files:**
- Modify: `src/mindtorch_v2/_backends/npu/runtime.py`
- Modify: `src/mindtorch_v2/_C.py`
- Modify: `src/mindtorch_v2/npu.py`
- Modify: `src/mindtorch_v2/_backends/common/convert.py`
- Modify: `src/mindtorch_v2/_printing.py`
- Test: `tests/mindtorch_v2/test_ops_npu.py`
- Test: `tests/mindtorch_v2/test_tensor_print.py`

**Step 1: Write the failing test**

Add to `tests/mindtorch_v2/test_ops_npu.py`:

```python
def test_npu_cross_device_copy():
    if not torch.npu.is_available():
        pytest.skip("NPU not available")
    if torch._C._npu_device_count() < 2:
        pytest.skip("Need 2 NPUs")
    src = torch.ones((2,), device="npu:0")
    dst = src.to("npu:1")
    assert dst.device.index == 1
    assert dst.to("cpu").numpy().tolist() == [1.0, 1.0]
```

Add to `tests/mindtorch_v2/test_tensor_print.py`:

```python
def test_tensor_repr_npu_index():
    if not torch.npu.is_available():
        return
    if torch._C._npu_device_count() < 2:
        return
    t = torch.ones((1,), device="npu:1")
    rep = repr(t)
    assert "device='npu:1'" in rep
```

**Step 2: Run tests to verify they fail**

Run: `pytest tests/mindtorch_v2/test_ops_npu.py::test_npu_cross_device_copy -v`
Expected: FAIL (device count helper or cross-device copy missing)

Run: `pytest tests/mindtorch_v2/test_tensor_print.py::test_tensor_repr_npu_index -v`
Expected: FAIL (repr missing index)

**Step 3: Write minimal implementation**

- Add `device_count()` in `npu/runtime.py` using `acl.rt.get_device_count`.
- Expose via `torch._C._npu_device_count()` and `torch.npu.device_count()`.
- Update `convert.to_device`:
  - If `a.device.type == "npu"` and `dev.type == "npu"` with different indices, copy via CPU staging (`_copy_npu_to_cpu` using runtime A, `_copy_cpu_to_npu` using runtime B).
- Update `_printing.format_tensor` to use `repr(tensor.device)` for NPU devices so `npu:1` is shown, but keep cpu/meta as `device.type` so they remain unindexed.

**Step 4: Run tests to verify they pass**

Run: `pytest tests/mindtorch_v2/test_ops_npu.py::test_npu_cross_device_copy -v`
Expected: PASS

Run: `pytest tests/mindtorch_v2/test_tensor_print.py::test_tensor_repr_npu_index -v`
Expected: PASS

**Step 5: Commit**

```bash
git add src/mindtorch_v2/_backends/npu/runtime.py src/mindtorch_v2/_C.py src/mindtorch_v2/npu.py \
        src/mindtorch_v2/_backends/common/convert.py src/mindtorch_v2/_printing.py \
        tests/mindtorch_v2/test_ops_npu.py tests/mindtorch_v2/test_tensor_print.py
git commit -m "feat: cross-device npu copy and device_count"
```

---

Plan complete and saved to `docs/plans/2026-02-13-device-index-plan.md`. Two execution options:

1. Subagent-Driven (this session) — I dispatch fresh subagent per task, review between tasks, fast iteration
2. Parallel Session (separate) — Open new session with executing-plans, batch execution with checkpoints

Which approach?
