# NPU Allocator OOM + Alloc Conf Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Add torch-aligned allocator OOM retry + GC threshold behavior with alloc conf parsing and mem_get_info API.

**Architecture:** Introduce a config parser for `MINDTORCH_NPU_ALLOC_CONF` / `PYTORCH_CUDA_ALLOC_CONF`, store parsed values in allocator instances, add `mem_get_info` via ACL runtime, and implement GC+OOM retry logic in the allocator allocation path with unit tests.

**Tech Stack:** Python, pytest, mindtorch_v2 NPU runtime/allocator.

---

### Task 1: Alloc conf parsing (precedence + warnings)

**Files:**
- Modify: `src/mindtorch_v2/_backends/npu/allocator.py`
- Test: `tests/mindtorch_v2/test_npu_allocator.py`

**Step 1: Write the failing tests**

Add tests:
```python
def test_alloc_conf_precedence(monkeypatch):
    from mindtorch_v2._backends.npu import allocator

    allocator._reset_alloc_conf_for_test()
    monkeypatch.setenv("PYTORCH_CUDA_ALLOC_CONF", "max_split_size_mb:4")
    monkeypatch.setenv("MINDTORCH_NPU_ALLOC_CONF", "max_split_size_mb:8")
    conf = allocator._load_alloc_conf(force=True)
    assert conf["max_split_size_mb"] == 8


def test_alloc_conf_unsupported_key_warns(monkeypatch):
    from mindtorch_v2._backends.npu import allocator

    allocator._reset_alloc_conf_for_test()
    monkeypatch.setenv("MINDTORCH_NPU_ALLOC_CONF", "unknown_key:1")
    with pytest.warns(UserWarning):
        conf = allocator._load_alloc_conf(force=True)
    assert conf == {}
```

**Step 2: Run tests to verify they fail**
Run: `pytest -q tests/mindtorch_v2/test_npu_allocator.py -k alloc_conf`
Expected: FAIL (missing functions).

**Step 3: Write minimal implementation**

In `allocator.py`, add:
```python
_ALLOC_CONF = None


def _reset_alloc_conf_for_test():
    global _ALLOC_CONF
    _ALLOC_CONF = None


def _load_alloc_conf(force=False):
    global _ALLOC_CONF
    if _ALLOC_CONF is not None and not force:
        return dict(_ALLOC_CONF)
    raw = os.getenv("MINDTORCH_NPU_ALLOC_CONF")
    if not raw:
        raw = os.getenv("PYTORCH_CUDA_ALLOC_CONF", "")
    conf = {}
    for item in raw.split(","):
        item = item.strip()
        if not item:
            continue
        if ":" in item:
            key, val = item.split(":", 1)
        elif "=" in item:
            key, val = item.split("=", 1)
        else:
            warnings.warn(f"Ignoring allocator config {item}")
            continue
        key = key.strip()
        val = val.strip()
        if key in ("max_split_size_mb", "garbage_collection_threshold"):
            conf[key] = val
        else:
            warnings.warn(f"Unsupported allocator config key: {key}")
    _ALLOC_CONF = conf
    return dict(conf)
```
Parse and validate values in later task (OOM/GC). Keep tests minimal for now.

**Step 4: Run tests to verify they pass**
Run: `pytest -q tests/mindtorch_v2/test_npu_allocator.py -k alloc_conf`
Expected: PASS.

**Step 5: Commit**
```bash
git add tests/mindtorch_v2/test_npu_allocator.py src/mindtorch_v2/_backends/npu/allocator.py
git commit -m "test: add alloc conf parsing coverage"
```

---

### Task 2: mem_get_info API

**Files:**
- Modify: `src/mindtorch_v2/_backends/npu/runtime.py`
- Modify: `src/mindtorch_v2/npu.py`
- Test: `tests/mindtorch_v2/test_runtime.py`

**Step 1: Write the failing test**

Add:
```python
def test_npu_mem_get_info(monkeypatch):
    from mindtorch_v2._backends.npu import runtime as npu_runtime

    class DummyRT:
        def set_device(self, device_id):
            return 0
        def set_context(self, ctx):
            return 0
        def get_mem_info(self, attr):
            return 10, 20, 0

    dummy_acl = types.SimpleNamespace(rt=DummyRT())
    monkeypatch.setattr(npu_runtime, "acl", dummy_acl)

    import mindtorch_v2.npu as npu
    free, total = npu.mem_get_info("npu:0")
    assert (free, total) == (10, 20)
```

**Step 2: Run test to verify it fails**
Run: `pytest -q tests/mindtorch_v2/test_runtime.py -k mem_get_info`
Expected: FAIL (missing API).

**Step 3: Write minimal implementation**

In `runtime.py`, add:
```python
def mem_get_info(device_id=0, attr=0):
    global acl
    if acl is None:
        acl = ensure_acl()
    runtime = get_runtime(device_id)
    runtime.activate()
    if not hasattr(acl.rt, "get_mem_info"):
        raise RuntimeError("acl.rt.get_mem_info not available")
    free, total, ret = acl.rt.get_mem_info(attr)
    if ret != ACL_ERROR_CODE:
        raise RuntimeError(f"acl.rt.get_mem_info failed: {ret}")
    return int(free), int(total)
```

In `npu.py`, add:
```python
def mem_get_info(device=None):
    dev = _normalize_npu_device(device)
    from ._backends.npu import runtime as npu_runtime
    return npu_runtime.mem_get_info(dev.index or 0)
```
Also add to `__all__`.

**Step 4: Run test to verify it passes**
Run: `pytest -q tests/mindtorch_v2/test_runtime.py -k mem_get_info`
Expected: PASS.

**Step 5: Commit**
```bash
git add tests/mindtorch_v2/test_runtime.py src/mindtorch_v2/_backends/npu/runtime.py src/mindtorch_v2/npu.py
git commit -m "feat: add npu mem_get_info"
```

---

### Task 3: Apply alloc conf values + max_split_size

**Files:**
- Modify: `src/mindtorch_v2/_backends/npu/allocator.py`
- Test: `tests/mindtorch_v2/test_npu_allocator.py`

**Step 1: Write the failing test**

Add:
```python
def test_alloc_conf_max_split_size_mb(monkeypatch):
    from mindtorch_v2._backends.npu import allocator

    allocator._reset_alloc_conf_for_test()
    monkeypatch.setenv("MINDTORCH_NPU_ALLOC_CONF", "max_split_size_mb:1")
    conf = allocator._load_alloc_conf(force=True)
    alloc = allocator.NpuAllocator(device_id=0)
    assert alloc.max_split_size == 1 * 1024 * 1024
```

**Step 2: Run test to verify it fails**
Run: `pytest -q tests/mindtorch_v2/test_npu_allocator.py -k max_split_size`
Expected: FAIL (attribute missing).

**Step 3: Write minimal implementation**

In `allocator.py`:
```python
_DEFAULT_MAX_SPLIT_SIZE = None

class NpuAllocator:
    def __init__(self, device_id):
        conf = _load_alloc_conf()
        self.max_split_size = _parse_max_split(conf.get("max_split_size_mb"))
```
And update `_split_block`:
```python
if self.max_split_size is not None and block.size > self.max_split_size:
    return block, None
```

**Step 4: Run test to verify it passes**
Run: `pytest -q tests/mindtorch_v2/test_npu_allocator.py -k max_split_size`
Expected: PASS.

**Step 5: Commit**
```bash
git add tests/mindtorch_v2/test_npu_allocator.py src/mindtorch_v2/_backends/npu/allocator.py
git commit -m "feat: apply max_split_size alloc conf"
```

---

### Task 4: GC threshold + OOM retry

**Files:**
- Modify: `src/mindtorch_v2/_backends/npu/allocator.py`
- Test: `tests/mindtorch_v2/test_npu_allocator.py`

**Step 1: Write the failing tests**

Add:
```python
def test_alloc_conf_gc_threshold_triggers(monkeypatch):
    from mindtorch_v2._backends.npu import allocator

    allocator._reset_alloc_conf_for_test()
    monkeypatch.setenv("MINDTORCH_NPU_ALLOC_CONF", "garbage_collection_threshold:0.5")
    allocator._load_alloc_conf(force=True)

    alloc = allocator.NpuAllocator(device_id=0)
    alloc._stats["reserved_bytes.all.current"] = 80
    monkeypatch.setattr(alloc, "_mem_get_info", lambda: (20, 100))

    freed = []
    monkeypatch.setattr(alloc, "_raw_free", lambda ptr: freed.append(ptr))

    block = allocator.Block(1234, 16, 16, "small_pool", None)
    alloc._cached["small_pool"].append(block)

    alloc._maybe_collect_garbage()
    assert freed


def test_oom_retry_increments_stats(monkeypatch):
    from mindtorch_v2._backends.npu import allocator

    alloc = allocator.NpuAllocator(device_id=0)
    calls = []

    def raw_malloc(size):
        calls.append(size)
        if len(calls) == 1:
            raise RuntimeError("acl.rt.malloc failed: 100")
        return (999, size)

    monkeypatch.setattr(alloc, "_raw_malloc", raw_malloc)
    monkeypatch.setattr(alloc, "_sync_device", lambda: None)
    monkeypatch.setattr(alloc, "_record_event", lambda stream: object())
    monkeypatch.setattr(alloc, "_event_complete", lambda event: True)

    ptr = alloc.malloc(512)
    assert ptr == 999
    assert alloc._stats["num_ooms"] == 1
    assert alloc._stats["num_alloc_retries"] == 1
```

**Step 2: Run tests to verify they fail**
Run: `pytest -q tests/mindtorch_v2/test_npu_allocator.py -k gc_threshold -k oom_retry`
Expected: FAIL (missing methods).

**Step 3: Write minimal implementation**

In `allocator.py`:
- Parse `garbage_collection_threshold` into `self.gc_threshold`.
- Add `_mem_get_info()` calling `npu_runtime.mem_get_info`.
- Add `_maybe_collect_garbage()` to free cached blocks if ratio exceeds threshold.
- Update `malloc()` to:
  - call `_maybe_collect_garbage()` before `_raw_malloc`.
  - on `_raw_malloc` failure, increment `num_ooms`, run GC + `empty_cache`, retry once, increment `num_alloc_retries` on success.

**Step 4: Run tests to verify they pass**
Run: `pytest -q tests/mindtorch_v2/test_npu_allocator.py -k gc_threshold -k oom_retry`
Expected: PASS.

**Step 5: Commit**
```bash
git add tests/mindtorch_v2/test_npu_allocator.py src/mindtorch_v2/_backends/npu/allocator.py
git commit -m "feat: allocator gc threshold and oom retry"
```

---

### Task 5: Full test run (mindspore env)

**Step 1: Run full tests**
Run: `source /home/lvyufeng/miniconda3/bin/activate mindspore && pytest -q tests/mindtorch_v2`
Expected: PASS

