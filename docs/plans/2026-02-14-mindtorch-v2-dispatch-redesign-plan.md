# Torch-Style Dispatcher + Pipeline Redesign Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Replace the current dispatcher/registry with Torch-style dispatch keys and key sets, then integrate the multi-stage pipeline as a wrapper key while preserving current behavior.

**Architecture:** Introduce DispatchKey/DispatchKeySet + OperatorEntry with schema, kernels, and fallthrough. Migrate registrations to key-based kernels, add wrapper keys for Autograd/Pipeline, then implement the pipeline kernel for multi-stage execution and flush boundaries.

**Tech Stack:** Python, existing mindtorch_v2 dispatch/pipeline/autograd.

---

### Task 1: Introduce DispatchKey + OperatorEntry (no behavior change)

**Files:**
- Create: `src/mindtorch_v2/_dispatch/keys.py`
- Modify: `src/mindtorch_v2/_dispatch/registry.py`
- Modify: `src/mindtorch_v2/_dispatch/dispatcher.py`
- Test: `tests/mindtorch_v2/test_dispatch_keys.py`

**Step 1: Write the failing test**

```python
import mindtorch_v2 as torch
from mindtorch_v2._dispatch.keys import DispatchKey, DispatchKeySet
from mindtorch_v2._dispatch.registry import registry


def test_dispatch_keyset_cpu():
    t = torch.ones((2,))
    keyset = DispatchKeySet.from_tensors((t,))
    assert DispatchKey.CPU in keyset


def test_registry_schema_and_kernel():
    registry.register_schema("aten::add", "add(Tensor a, Tensor b) -> Tensor")
    def cpu_impl(a, b):
        return a
    registry.register_kernel("aten::add", DispatchKey.CPU, cpu_impl)
    entry = registry.get("aten::add")
    assert entry.schema is not None
    assert entry.kernels[DispatchKey.CPU] is cpu_impl
```

**Step 2: Run test to verify it fails**

Run: `pytest tests/mindtorch_v2/test_dispatch_keys.py::test_dispatch_keyset_cpu -v`
Expected: FAIL (DispatchKey not defined).

**Step 3: Write minimal implementation**

```python
# keys.py
from enum import Enum, auto

class DispatchKey(Enum):
    CPU = auto()
    NPU = auto()
    Meta = auto()
    Autograd = auto()
    Pipeline = auto()

class DispatchKeySet(set):
    @classmethod
    def from_tensors(cls, tensors, *, grad_enabled=False, pipeline_enabled=False):
        keys = cls()
        # infer device keys
        # add Autograd and Pipeline as needed
        return keys
```

```python
# registry.py
class OperatorEntry:
    def __init__(self, name):
        self.name = name
        self.schema = None
        self.kernels = {}
        self.fallthrough = set()

class OpRegistry:
    def register_schema(self, name, schema): ...
    def register_kernel(self, name, key, fn): ...
    def register_fallthrough(self, name, key): ...
```

Update dispatcher to call `registry.get(name)` and resolve kernels by key order (temporarily CPU/NPU/Meta only).

**Step 4: Run test to verify it passes**

Run: `pytest tests/mindtorch_v2/test_dispatch_keys.py::test_dispatch_keyset_cpu -v`
Expected: PASS

**Step 5: Commit**

```bash
git add src/mindtorch_v2/_dispatch/keys.py src/mindtorch_v2/_dispatch/registry.py src/mindtorch_v2/_dispatch/dispatcher.py tests/mindtorch_v2/test_dispatch_keys.py
git commit -m "Add dispatch keys and operator entries"
```

---

### Task 2: Migrate backend registrations to DispatchKey

**Files:**
- Modify: `src/mindtorch_v2/_backends/cpu/__init__.py`
- Modify: `src/mindtorch_v2/_backends/npu/__init__.py`
- Modify: `src/mindtorch_v2/_backends/meta/__init__.py`
- Modify: `src/mindtorch_v2/_dispatch/registry.py`
- Test: `tests/mindtorch_v2/test_dispatch_registry.py`

**Step 1: Write the failing test**

```python
import mindtorch_v2 as torch
from mindtorch_v2._dispatch.keys import DispatchKey
from mindtorch_v2._dispatch.registry import registry


def test_backend_registration_uses_keys():
    entry = registry.get("aten::add")
    assert DispatchKey.CPU in entry.kernels
    assert DispatchKey.NPU in entry.kernels
    assert DispatchKey.Meta in entry.kernels
```

**Step 2: Run test to verify it fails**

Run: `pytest tests/mindtorch_v2/test_dispatch_registry.py::test_backend_registration_uses_keys -v`
Expected: FAIL (registry not populated with keys).

**Step 3: Write minimal implementation**

Update backend registrations:
- map op names to `aten::` namespace
- call `registry.register_kernel(op, DispatchKey.CPU/NPU/Meta, fn)`
- register meta kernels via `DispatchKey.Meta`

**Step 4: Run test to verify it passes**

Run: `pytest tests/mindtorch_v2/test_dispatch_registry.py::test_backend_registration_uses_keys -v`
Expected: PASS

**Step 5: Commit**

```bash
git add src/mindtorch_v2/_backends/cpu/__init__.py src/mindtorch_v2/_backends/npu/__init__.py src/mindtorch_v2/_backends/meta/__init__.py src/mindtorch_v2/_dispatch/registry.py tests/mindtorch_v2/test_dispatch_registry.py
git commit -m "Migrate backend registrations to dispatch keys"
```

---

### Task 3: Implement keyset resolution in dispatcher

**Files:**
- Modify: `src/mindtorch_v2/_dispatch/dispatcher.py`
- Modify: `src/mindtorch_v2/_dispatch/keys.py`
- Test: `tests/mindtorch_v2/test_dispatch_resolution.py`

**Step 1: Write the failing test**

```python
import mindtorch_v2 as torch
from mindtorch_v2._dispatch.keys import DispatchKey


def test_dispatch_prefers_meta_when_input_meta():
    a = torch.ones((2,), device="meta")
    b = torch.ones((2,), device="meta")
    c = torch.add(a, b)
    assert c.device.type == "meta"


def test_dispatch_prefers_npu_over_cpu():
    a = torch.ones((2,), device="npu")
    b = torch.ones((2,), device="npu")
    c = torch.add(a, b)
    assert c.device.type == "npu"
```

**Step 2: Run test to verify it fails**

Run: `pytest tests/mindtorch_v2/test_dispatch_resolution.py::test_dispatch_prefers_meta_when_input_meta -v`
Expected: FAIL

**Step 3: Write minimal implementation**

Implement DispatchKeySet.from_tensors that derives device keys + wrapper keys.
Update dispatcher to select kernel by priority order (Pipeline → Autograd → Meta → NPU → CPU), honoring fallthrough.

**Step 4: Run test to verify it passes**

Run: `pytest tests/mindtorch_v2/test_dispatch_resolution.py::test_dispatch_prefers_meta_when_input_meta -v`
Expected: PASS

**Step 5: Commit**

```bash
git add src/mindtorch_v2/_dispatch/dispatcher.py src/mindtorch_v2/_dispatch/keys.py tests/mindtorch_v2/test_dispatch_resolution.py
git commit -m "Implement torch-style dispatch key resolution"
```

---

### Task 4: Add Autograd and Pipeline keys (fallthrough)

**Files:**
- Modify: `src/mindtorch_v2/_dispatch/keys.py`
- Modify: `src/mindtorch_v2/_dispatch/dispatcher.py`
- Test: `tests/mindtorch_v2/test_dispatch_keys.py`

**Step 1: Write the failing test**

```python
import mindtorch_v2 as torch
from mindtorch_v2._dispatch.keys import DispatchKeySet
from mindtorch_v2._dispatch.pipeline import pipeline_context


def test_keyset_includes_autograd_when_needed():
    a = torch.ones((2,))
    a.requires_grad_(True)
    keyset = DispatchKeySet.from_tensors((a,), grad_enabled=True, pipeline_enabled=False)
    assert "Autograd" in {k.name for k in keyset}


def test_keyset_includes_pipeline_when_enabled():
    a = torch.ones((2,))
    with pipeline_context():
        keyset = DispatchKeySet.from_tensors((a,), grad_enabled=False, pipeline_enabled=True)
        assert "Pipeline" in {k.name for k in keyset}
```

**Step 2: Run test to verify it fails**

Run: `pytest tests/mindtorch_v2/test_dispatch_keys.py::test_keyset_includes_autograd_when_needed -v`
Expected: FAIL

**Step 3: Write minimal implementation**

Update keyset builder and dispatcher to include Autograd/Pipeline keys but mark them fallthrough by default (no behavior change).

**Step 4: Run test to verify it passes**

Run: `pytest tests/mindtorch_v2/test_dispatch_keys.py::test_keyset_includes_autograd_when_needed -v`
Expected: PASS

**Step 5: Commit**

```bash
git add src/mindtorch_v2/_dispatch/keys.py src/mindtorch_v2/_dispatch/dispatcher.py tests/mindtorch_v2/test_dispatch_keys.py
git commit -m "Add autograd and pipeline dispatch keys"
```

---

### Task 5: Implement Pipeline as a wrapper kernel

**Files:**
- Modify: `src/mindtorch_v2/_dispatch/pipeline.py`
- Modify: `src/mindtorch_v2/_dispatch/dispatcher.py`
- Modify: `src/mindtorch_v2/_dispatch/registry.py`
- Test: `tests/mindtorch_v2/test_dispatch_pipeline.py`

**Step 1: Write the failing test**

```python
import mindtorch_v2 as torch


def test_pipeline_defers_execution():
    with torch.pipeline() as pipe:
        a = torch.ones((2,))
        b = torch.ones((2,))
        c = torch.add(a, b)
        assert getattr(c, "_pending", False) is True
        pipe.flush()
        assert getattr(c, "_pending", False) is False
```

**Step 2: Run test to verify it fails**

Run: `pytest tests/mindtorch_v2/test_dispatch_pipeline.py::test_pipeline_defers_execution -v`
Expected: FAIL

**Step 3: Write minimal implementation**

Implement a Pipeline kernel for each op (default wrapper):
- Call Meta kernel for shape inference
- Return Pending Tensor
- Record PendingOp with plan/impl references (using backend impl)
Update dispatcher to invoke Pipeline kernel when Pipeline key is present.

**Step 4: Run test to verify it passes**

Run: `pytest tests/mindtorch_v2/test_dispatch_pipeline.py::test_pipeline_defers_execution -v`
Expected: PASS

**Step 5: Commit**

```bash
git add src/mindtorch_v2/_dispatch/pipeline.py src/mindtorch_v2/_dispatch/dispatcher.py src/mindtorch_v2/_dispatch/registry.py tests/mindtorch_v2/test_dispatch_pipeline.py
git commit -m "Add pipeline wrapper kernel for deferred execution"
```

---

### Task 6: Flush boundaries for autograd and data access

**Files:**
- Modify: `src/mindtorch_v2/_tensor.py`
- Modify: `src/mindtorch_v2/_autograd/engine.py`
- Test: `tests/mindtorch_v2/test_dispatch_pipeline.py`

**Step 1: Write the failing test**

```python
import mindtorch_v2 as torch


def test_pipeline_flushes_on_backward():
    with torch.pipeline():
        a = torch.ones((2,))
        a.requires_grad_(True)
        b = torch.sum(a)
        b.backward()
        assert a.grad is not None
```

**Step 2: Run test to verify it fails**

Run: `pytest tests/mindtorch_v2/test_dispatch_pipeline.py::test_pipeline_flushes_on_backward -v`
Expected: FAIL

**Step 3: Write minimal implementation**

Ensure `.backward()` and `autograd.grad` call `pipeline.flush()` before running autograd. Add flush in tensor data access (`numpy`, `item`, `repr`).

**Step 4: Run test to verify it passes**

Run: `pytest tests/mindtorch_v2/test_dispatch_pipeline.py::test_pipeline_flushes_on_backward -v`
Expected: PASS

**Step 5: Commit**

```bash
git add src/mindtorch_v2/_tensor.py src/mindtorch_v2/_autograd/engine.py tests/mindtorch_v2/test_dispatch_pipeline.py
git commit -m "Flush pipeline on backward and data access"
```

---

### Task 7: Full test run

**Step 1: Run full suite**

Run: `pytest -q tests/mindtorch_v2`
Expected: PASS

**Step 2: Commit (if needed)**

```bash
git status -sb
```

---

Plan complete.
