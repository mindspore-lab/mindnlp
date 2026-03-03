# Device + Memory Management Alignment Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Implement torch‑compatible device/memory management APIs (summary/snapshot/fraction, device props, peer access, pinned memory, stream priority).

**Architecture:** Extend `mindtorch_v2.npu` and NPU runtime/allocator with new API surfaces, add minimal state to allocator for per‑process fraction and snapshot reporting, and provide torch‑compatible summary/snapshot formatting with safe defaults for missing fields.

**Tech Stack:** Python, pytest, mindtorch_v2 NPU runtime/allocator.

---

### Task 1: memory_summary + memory_snapshot (format stubs)

**Files:**
- Modify: `src/mindtorch_v2/npu.py`
- Modify: `src/mindtorch_v2/_backends/npu/allocator.py`
- Test: `tests/mindtorch_v2/test_npu_memory_api.py` (create)

**Step 1: Write the failing tests**

Create `tests/mindtorch_v2/test_npu_memory_api.py`:
```python
import pytest
import mindtorch_v2 as torch


def test_memory_summary_contains_keys(monkeypatch):
    class DummyAlloc:
        def memory_stats(self):
            return {
                "allocated_bytes.all.current": 1,
                "allocated_bytes.all.peak": 2,
                "reserved_bytes.all.current": 3,
                "reserved_bytes.all.peak": 4,
                "active_bytes.all.current": 5,
                "active_bytes.all.peak": 6,
                "inactive_split_bytes.all.current": 7,
                "inactive_split_bytes.all.peak": 8,
            }
    monkeypatch.setattr(torch.npu, "_get_allocator", lambda device=None: DummyAlloc())
    text = torch.npu.memory_summary()
    assert "Allocated memory" in text
    assert "Reserved memory" in text


def test_memory_snapshot_schema(monkeypatch):
    class DummyAlloc:
        def snapshot(self):
            return {"segments": [], "device": 0, "allocator": "npu"}
    monkeypatch.setattr(torch.npu, "_get_allocator", lambda device=None: DummyAlloc())
    snap = torch.npu.memory_snapshot()
    assert "segments" in snap
    assert "device" in snap
```

**Step 2: Run test to verify it fails**
Run: `pytest -q tests/mindtorch_v2/test_npu_memory_api.py -k summary -k snapshot`
Expected: FAIL (missing APIs).

**Step 3: Write minimal implementation**

In `allocator.py`, add `snapshot()` returning a torch‑like dict with cached/active blocks (stub if none).
In `npu.py`, add `memory_summary()` and `memory_snapshot()`:
- `memory_summary` should format torch‑like sections using `memory_stats()` values and fill missing fields with 0.
- `memory_snapshot` should call allocator `snapshot()`.

**Step 4: Run tests to verify it passes**
Run: `pytest -q tests/mindtorch_v2/test_npu_memory_api.py -k summary -k snapshot`
Expected: PASS.

**Step 5: Commit**
```bash
git add tests/mindtorch_v2/test_npu_memory_api.py src/mindtorch_v2/npu.py src/mindtorch_v2/_backends/npu/allocator.py
git commit -m "feat: add npu memory summary and snapshot"
```

---

### Task 2: set_per_process_memory_fraction

**Files:**
- Modify: `src/mindtorch_v2/_backends/npu/allocator.py`
- Modify: `src/mindtorch_v2/npu.py`
- Test: `tests/mindtorch_v2/test_npu_memory_api.py`

**Step 1: Write the failing test**

Add:
```python
def test_memory_fraction_enforced(monkeypatch):
    import mindtorch_v2 as torch
    from mindtorch_v2._backends.npu import allocator

    alloc = allocator.NpuAllocator(device_id=0)
    monkeypatch.setattr(torch.npu, "_get_allocator", lambda device=None: alloc)
    monkeypatch.setattr(alloc, "_mem_get_info", lambda: (0, 100))

    torch.npu.set_per_process_memory_fraction(0.5)
    alloc._stats["reserved_bytes.all.current"] = 60
    with pytest.raises(RuntimeError):
        alloc._enforce_memory_fraction(50)
```

**Step 2: Run test to verify it fails**
Run: `pytest -q tests/mindtorch_v2/test_npu_memory_api.py -k fraction`
Expected: FAIL (missing methods).

**Step 3: Write minimal implementation**

- Store `memory_fraction` in allocator state.
- Add `_enforce_memory_fraction(requested)` called inside `malloc()` before `_raw_malloc`.
- Add `npu.set_per_process_memory_fraction(fraction, device=None)`.

**Step 4: Run test to verify it passes**
Run: `pytest -q tests/mindtorch_v2/test_npu_memory_api.py -k fraction`
Expected: PASS.

**Step 5: Commit**
```bash
git add tests/mindtorch_v2/test_npu_memory_api.py src/mindtorch_v2/npu.py src/mindtorch_v2/_backends/npu/allocator.py
git commit -m "feat: enforce per-process memory fraction"
```

---

### Task 3: device properties + capability

**Files:**
- Modify: `src/mindtorch_v2/npu.py`
- Modify: `src/mindtorch_v2/_backends/npu/runtime.py`
- Test: `tests/mindtorch_v2/test_npu_device_api.py` (create)

**Step 1: Write failing tests**

Create `tests/mindtorch_v2/test_npu_device_api.py`:
```python
import mindtorch_v2 as torch


def test_get_device_name_stub(monkeypatch):
    monkeypatch.setattr(torch.npu, "_get_device_name", lambda device=None: "Ascend")
    assert torch.npu.get_device_name("npu:0") == "Ascend"


def test_get_device_capability_stub(monkeypatch):
    monkeypatch.setattr(torch.npu, "_get_device_capability", lambda device=None: (0, 0))
    assert torch.npu.get_device_capability("npu:0") == (0, 0)
```

**Step 2: Run test to verify it fails**
Run: `pytest -q tests/mindtorch_v2/test_npu_device_api.py`
Expected: FAIL (missing APIs).

**Step 3: Write minimal implementation**

- Add `get_device_name`, `get_device_properties`, `get_device_capability` with stub defaults.
- In runtime, add internal helpers if ACL exposes device info (if not, return stubs + warn).

**Step 4: Run test to verify it passes**
Run: `pytest -q tests/mindtorch_v2/test_npu_device_api.py`
Expected: PASS.

**Step 5: Commit**
```bash
git add tests/mindtorch_v2/test_npu_device_api.py src/mindtorch_v2/npu.py src/mindtorch_v2/_backends/npu/runtime.py
git commit -m "feat: add npu device property APIs"
```

---

### Task 4: peer access + pinned memory + stream priority

**Files:**
- Modify: `src/mindtorch_v2/npu.py`
- Modify: `src/mindtorch_v2/_backends/npu/runtime.py`
- Modify: `src/mindtorch_v2/_storage.py`
- Test: `tests/mindtorch_v2/test_npu_device_api.py`

**Step 1: Write failing tests**

Add:
```python
def test_peer_access_unsupported(monkeypatch):
    import mindtorch_v2 as torch
    assert torch.npu.can_device_access_peer(0, 1) is False
    with pytest.raises(RuntimeError):
        torch.npu.enable_peer_access(1)


def test_stream_priority_range_fallback(monkeypatch):
    import mindtorch_v2 as torch
    assert torch.npu.stream_priority_range() == (0, 0)


def test_pinned_memory(monkeypatch):
    import mindtorch_v2 as torch
    t = torch.tensor([1.0, 2.0])
    tp = torch.npu.pin_memory(t)
    assert torch.npu.is_pinned(tp) is True
```

**Step 2: Run test to verify it fails**
Run: `pytest -q tests/mindtorch_v2/test_npu_device_api.py -k peer -k pinned -k stream_priority`
Expected: FAIL.

**Step 3: Write minimal implementation**

- `can_device_access_peer` returns False with warning.
- `enable_peer_access/disable_peer_access` raise RuntimeError.
- `stream_priority_range` returns `(0,0)` if no ACL support.
- Implement pinned memory by adding `_pinned` flag to CPU untyped storage and a helper in `npu.py`.

**Step 4: Run test to verify it passes**
Run: `pytest -q tests/mindtorch_v2/test_npu_device_api.py -k peer -k pinned -k stream_priority`
Expected: PASS.

**Step 5: Commit**
```bash
git add tests/mindtorch_v2/test_npu_device_api.py src/mindtorch_v2/npu.py src/mindtorch_v2/_backends/npu/runtime.py src/mindtorch_v2/_storage.py
git commit -m "feat: add peer access, pinned memory, stream priority APIs"
```

---

### Task 5: Full test run (mindspore env)

**Step 1: Run full tests**
Run: `source /home/lvyufeng/miniconda3/bin/activate mindspore && pytest -q tests/mindtorch_v2`
Expected: PASS
