# Dispatch + Pipeline Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Implement an explicit pipeline mode that defers execution across ops and integrates with the dispatcher.

**Architecture:** Extend the dispatcher registry to support meta/plan/impl kernels, add a pipeline queue with flush semantics and sync boundaries, and mark pending tensors until execution.

**Tech Stack:** Python, pytest

---

### Task 1: Add pipeline context + queue skeleton

**Files:**
- Create: `src/mindtorch_v2/_dispatch/pipeline.py`
- Modify: `src/mindtorch_v2/_dispatch/__init__.py`
- Test: `tests/mindtorch_v2/test_pipeline.py::test_pipeline_context_records_ops`

**Step 1: Write the failing test**

```python
import mindtorch_v2 as torch


def test_pipeline_context_records_ops():
    a = torch.tensor([1.0, 2.0])
    b = torch.tensor([3.0, 4.0])
    with torch.pipeline():
        c = a + b
        d = c * b
        assert c._pending is True
        assert d._pending is True
    # leaving context should flush
    assert c._pending is False
    assert d._pending is False
```

**Step 2: Run test to verify it fails**

Run: `pytest tests/mindtorch_v2/test_pipeline.py::test_pipeline_context_records_ops -v`

Expected: FAIL because pipeline context does not exist.

**Step 3: Write minimal implementation**

Create `src/mindtorch_v2/_dispatch/pipeline.py`:

```python
import contextlib

_CURRENT = None


class Pipeline:
    def __init__(self):
        self.queue = []

    def record(self, entry):
        self.queue.append(entry)

    def flush(self):
        for entry in self.queue:
            entry.execute()
        self.queue.clear()


@contextlib.contextmanager
def pipeline_context():
    global _CURRENT
    prev = _CURRENT
    _CURRENT = Pipeline()
    try:
        yield _CURRENT
    finally:
        _CURRENT.flush()
        _CURRENT = prev


def current_pipeline():
    return _CURRENT
```

Expose in `src/mindtorch_v2/_dispatch/__init__.py`:

```python
from .pipeline import pipeline_context, current_pipeline
```

Add a `torch.pipeline()` helper in `src/mindtorch_v2/__init__.py` that returns `pipeline_context()`.

**Step 4: Run test to verify it passes**

Run: `pytest tests/mindtorch_v2/test_pipeline.py::test_pipeline_context_records_ops -v`

Expected: PASS.

**Step 5: Commit**

```bash
git add src/mindtorch_v2/_dispatch/pipeline.py src/mindtorch_v2/_dispatch/__init__.py src/mindtorch_v2/__init__.py tests/mindtorch_v2/test_pipeline.py
git commit -m "feat: add pipeline context skeleton"
```

---

### Task 2: Extend dispatcher registry for meta/impl and pending tensors

**Files:**
- Modify: `src/mindtorch_v2/_dispatch/registry.py`
- Modify: `src/mindtorch_v2/_dispatch/dispatcher.py`
- Modify: `src/mindtorch_v2/_tensor.py`
- Test: `tests/mindtorch_v2/test_pipeline.py::test_pipeline_dispatch_marks_pending`

**Step 1: Write the failing test**

```python
import mindtorch_v2 as torch


def test_pipeline_dispatch_marks_pending():
    a = torch.tensor([1.0, 2.0])
    b = torch.tensor([3.0, 4.0])
    with torch.pipeline():
        c = a + b
        assert c._pending is True
    assert c._pending is False
```

**Step 2: Run test to verify it fails**

Run: `pytest tests/mindtorch_v2/test_pipeline.py::test_pipeline_dispatch_marks_pending -v`

Expected: FAIL because dispatcher does not integrate pipeline.

**Step 3: Write minimal implementation**

Update registry to store entries like `{meta, impl}`:

```python
class OpRegistry:
    def __init__(self):
        self._ops = {}

    def register(self, name, device, fn, meta=None):
        self._ops[(name, device)] = {"impl": fn, "meta": meta}

    def get(self, name, device):
        return self._ops[(name, device)]
```

Update dispatcher to check pipeline:

```python
from .pipeline import current_pipeline


def dispatch(name, device, *args, **kwargs):
    entry = registry.get(name, device)
    pipe = current_pipeline()
    if pipe is None or entry["meta"] is None:
        return entry["impl"](*args, **kwargs)
    out = entry["meta"](*args, **kwargs)
    out._pending = True
    pipe.record(_PendingOp(entry, args, kwargs, out))
    return out
```

Add `_pending` to Tensor and clear it after execution in `_PendingOp.execute()`.

**Step 4: Run test to verify it passes**

Run: `pytest tests/mindtorch_v2/test_pipeline.py::test_pipeline_dispatch_marks_pending -v`

Expected: PASS.

**Step 5: Commit**

```bash
git add src/mindtorch_v2/_dispatch/registry.py src/mindtorch_v2/_dispatch/dispatcher.py src/mindtorch_v2/_tensor.py tests/mindtorch_v2/test_pipeline.py
git commit -m "feat: integrate pipeline with dispatcher"
```

---

### Task 3: Add sync boundary on to()/numpy()

**Files:**
- Modify: `src/mindtorch_v2/_tensor.py`
- Modify: `src/mindtorch_v2/_storage.py`
- Test: `tests/mindtorch_v2/test_pipeline.py::test_pipeline_flush_on_to_cpu`

**Step 1: Write the failing test**

```python
import mindtorch_v2 as torch


def test_pipeline_flush_on_to_cpu():
    a = torch.tensor([1.0, 2.0])
    b = torch.tensor([3.0, 4.0])
    with torch.pipeline():
        c = a + b
        out = c.to("cpu")
        assert c._pending is False
    assert out.storage.data.tolist() == [4.0, 6.0]
```

**Step 2: Run test to verify it fails**

Run: `pytest tests/mindtorch_v2/test_pipeline.py::test_pipeline_flush_on_to_cpu -v`

Expected: FAIL because to("cpu") does not flush.

**Step 3: Write minimal implementation**

In `Tensor.to()` and `Tensor.numpy()`, if `_pending` then call `pipeline.flush()` before returning.

**Step 4: Run test to verify it passes**

Run: `pytest tests/mindtorch_v2/test_pipeline.py::test_pipeline_flush_on_to_cpu -v`

Expected: PASS.

**Step 5: Commit**

```bash
git add src/mindtorch_v2/_tensor.py src/mindtorch_v2/_storage.py tests/mindtorch_v2/test_pipeline.py
git commit -m "feat: flush pipeline on sync boundaries"
```

---

### Task 4: Add meta kernels for add/mul/relu/sum

**Files:**
- Modify: `src/mindtorch_v2/_backends/cpu.py`
- Modify: `src/mindtorch_v2/_backends/ascend.py`
- Modify: `src/mindtorch_v2/_functional.py`
- Test: `tests/mindtorch_v2/test_pipeline.py::test_pipeline_meta_shapes`

**Step 1: Write the failing test**

```python
import mindtorch_v2 as torch


def test_pipeline_meta_shapes():
    a = torch.tensor([[1.0, 2.0]])
    b = torch.tensor([[3.0, 4.0]])
    with torch.pipeline():
        c = a + b
        d = c.relu()
        e = d.sum()
        assert c.shape == a.shape
        assert d.shape == a.shape
        assert e.shape == ()
```

**Step 2: Run test to verify it fails**

Run: `pytest tests/mindtorch_v2/test_pipeline.py::test_pipeline_meta_shapes -v`

Expected: FAIL because meta kernels are missing.

**Step 3: Write minimal implementation**

Add `*_meta` kernels that return a Tensor with correct shape/stride/dtype but no data.

Register with `registry.register(name, device, impl, meta=meta)`.

**Step 4: Run test to verify it passes**

Run: `pytest tests/mindtorch_v2/test_pipeline.py::test_pipeline_meta_shapes -v`

Expected: PASS.

**Step 5: Commit**

```bash
git add src/mindtorch_v2/_backends/cpu.py src/mindtorch_v2/_backends/ascend.py src/mindtorch_v2/_functional.py tests/mindtorch_v2/test_pipeline.py
git commit -m "feat: add meta kernels for pipeline"
```

---

### Task 5: Add pipeline integration test for NPU

**Files:**
- Modify: `tests/mindtorch_v2/test_ops_npu.py`

**Step 1: Write the failing test**

```python
import mindtorch_v2 as torch


def test_npu_pipeline_chain():
    if not torch.npu.is_available():
        pytest.skip("NPU not available")
    with torch.pipeline():
        a = torch.tensor([1.0, 2.0], device="npu")
        b = torch.tensor([3.0, 4.0], device="npu")
        c = (a + b).relu()
        d = c.sum()
        out = d.to("cpu")
    assert out.numpy().tolist() == 10.0
```

**Step 2: Run test to verify it fails**

Run: `pytest tests/mindtorch_v2/test_ops_npu.py::test_npu_pipeline_chain -v`

Expected: FAIL until pipeline meta/flush are implemented.

**Step 3: Write minimal implementation**

No additional code changes beyond tasks above.

**Step 4: Run test to verify it passes**

Run: `pytest tests/mindtorch_v2/test_ops_npu.py::test_npu_pipeline_chain -v`

Expected: PASS.

**Step 5: Commit**

```bash
git add tests/mindtorch_v2/test_ops_npu.py
git commit -m "test: add npu pipeline chain coverage"
```
