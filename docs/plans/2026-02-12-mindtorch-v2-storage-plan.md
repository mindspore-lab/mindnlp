# Storage Alignment Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Implement TypedStorage + UntypedStorage with CPU/NPU/Meta backends and replace existing storage usage.

**Architecture:** Introduce new storage classes, rewire Tensor and backends to use them, and add CPU‑only shared/file‑backed APIs.

**Tech Stack:** Python, pytest, numpy

---

### Task 1: Add UntypedStorage + TypedStorage skeletons

**Files:**
- Create: `src/mindtorch_v2/_storage_typed.py`
- Modify: `src/mindtorch_v2/_storage.py`
- Test: `tests/mindtorch_v2/test_storage.py::test_typed_untyped_basic`

**Step 1: Write the failing test**

```python
import mindtorch_v2 as torch


def test_typed_untyped_basic():
    t = torch.tensor([1.0, 2.0])
    st = t.storage()
    ust = t.untyped_storage()
    assert st.dtype.name == "float32"
    assert st.nbytes() == 8
    assert ust.nbytes() == st.nbytes()
```

**Step 2: Run test to verify it fails**

Run: `pytest tests/mindtorch_v2/test_storage.py::test_typed_untyped_basic -v`

Expected: FAIL because storage API not implemented.

**Step 3: Write minimal implementation**

Add `TypedStorage`/`UntypedStorage` base classes in `_storage_typed.py`, and update `Storage` wrappers to use them. Implement `Tensor.storage()` and `Tensor.untyped_storage()`.

**Step 4: Run test to verify it passes**

Run: `pytest tests/mindtorch_v2/test_storage.py::test_typed_untyped_basic -v`

Expected: PASS.

**Step 5: Commit**

```bash
git add src/mindtorch_v2/_storage_typed.py src/mindtorch_v2/_storage.py src/mindtorch_v2/_tensor.py tests/mindtorch_v2/test_storage.py
git commit -m "feat: add typed/untyped storage skeleton"
```

---

### Task 2: CPU untyped storage + shared memory + file backing

**Files:**
- Modify: `src/mindtorch_v2/_storage_typed.py`
- Test: `tests/mindtorch_v2/test_storage.py::test_cpu_share_memory` `tests/mindtorch_v2/test_storage.py::test_cpu_from_file`

**Step 1: Write failing tests**

```python
import os
import tempfile
import mindtorch_v2 as torch


def test_cpu_share_memory():
    t = torch.tensor([1.0, 2.0])
    ust = t.untyped_storage()
    ust.share_memory_()
    assert ust.is_shared() is True


def test_cpu_from_file():
    with tempfile.NamedTemporaryFile(delete=False) as f:
        f.write(b"\x00" * 16)
        fname = f.name
    ust = torch.UntypedStorage.from_file(fname, shared=False)
    assert ust.nbytes() == 16
    assert ust.filename() == fname
    os.unlink(fname)
```

**Step 2: Run tests to verify they fail**

Run: `pytest tests/mindtorch_v2/test_storage.py::test_cpu_share_memory -v`
Run: `pytest tests/mindtorch_v2/test_storage.py::test_cpu_from_file -v`

Expected: FAIL until CPU untyped supports shared/file.

**Step 3: Write minimal implementation**

Implement `CPUUntypedStorage` backed by `bytearray`/`memoryview` and `np.memmap` for file‑backed. Add `share_memory_()` + `is_shared()` + `filename()`.

**Step 4: Run tests to verify they pass**

Run: `pytest tests/mindtorch_v2/test_storage.py::test_cpu_share_memory -v`
Run: `pytest tests/mindtorch_v2/test_storage.py::test_cpu_from_file -v`

Expected: PASS.

**Step 5: Commit**

```bash
git add src/mindtorch_v2/_storage_typed.py tests/mindtorch_v2/test_storage.py
git commit -m "feat: add cpu shared and file-backed storage"
```

---

### Task 3: NPU untyped storage without CPU copy

**Files:**
- Modify: `src/mindtorch_v2/_storage_typed.py`
- Modify: `src/mindtorch_v2/_backends/ascend.py`
- Test: `tests/mindtorch_v2/test_storage.py::test_npu_storage_no_cpu_copy`

**Step 1: Write failing test**

```python
import mindtorch_v2 as torch


def test_npu_storage_no_cpu_copy():
    if not torch.npu.is_available():
        return
    t = torch.tensor([1.0, 2.0], device="npu")
    st = t.storage()
    assert st.device.type == "npu"
    assert st.untyped_storage().data_ptr() != 0
```

**Step 2: Run test to verify it fails**

Run: `pytest tests/mindtorch_v2/test_storage.py::test_npu_storage_no_cpu_copy -v`

Expected: FAIL until NPU storage uses untyped device memory.

**Step 3: Write minimal implementation**

Implement `NPUUntypedStorage` with device ptr + nbytes. Update NPU backend to create typed storage wrapping NPU untyped storage; remove any CPU data buffers.

**Step 4: Run test to verify it passes**

Run: `pytest tests/mindtorch_v2/test_storage.py::test_npu_storage_no_cpu_copy -v`

Expected: PASS.

**Step 5: Commit**

```bash
git add src/mindtorch_v2/_storage_typed.py src/mindtorch_v2/_backends/ascend.py tests/mindtorch_v2/test_storage.py
git commit -m "feat: add npu untyped storage"
```

---

### Task 4: Meta untyped storage + meta tensor integration

**Files:**
- Modify: `src/mindtorch_v2/_storage_typed.py`
- Modify: `src/mindtorch_v2/_tensor.py`
- Test: `tests/mindtorch_v2/test_storage.py::test_meta_storage_no_data_ptr`

**Step 1: Write failing test**

```python
import pytest
import mindtorch_v2 as torch


def test_meta_storage_no_data_ptr():
    t = torch.tensor([1.0, 2.0], device="meta")
    with pytest.raises(RuntimeError):
        t.untyped_storage().data_ptr()
```

**Step 2: Run test to verify it fails**

Run: `pytest tests/mindtorch_v2/test_storage.py::test_meta_storage_no_data_ptr -v`

Expected: FAIL until meta storage is wired.

**Step 3: Write minimal implementation**

Implement `MetaUntypedStorage` with `data_ptr()` raising, and update meta tensor creation to use typed/untyped storage.

**Step 4: Run test to verify it passes**

Run: `pytest tests/mindtorch_v2/test_storage.py::test_meta_storage_no_data_ptr -v`

Expected: PASS.

**Step 5: Commit**

```bash
git add src/mindtorch_v2/_storage_typed.py src/mindtorch_v2/_tensor.py tests/mindtorch_v2/test_storage.py
git commit -m "feat: add meta untyped storage"
```
