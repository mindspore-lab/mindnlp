# Base Framework Parity Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Align Dispatch, Storage, Autograd, and NPU API behavior with torch (including exact error messages) in a single PR.

**Architecture:** Contract test harness compares mindtorch_v2 behavior to PyTorch; fix implementation to satisfy parity without expanding ops.

**Tech Stack:** Python, PyTorch (test oracle), ACL runtime, existing dispatch/autograd/storage infrastructure.

---

### Task 1: Add contract test harness

**Files:**
- Create: `tests/mindtorch_v2/contract/__init__.py`
- Create: `tests/mindtorch_v2/contract/helpers.py`

**Step 1: Write the failing test**
```python
# tests/mindtorch_v2/contract/test_harness.py
from tests.mindtorch_v2.contract.helpers import assert_torch_error

def test_harness_asserts_exact_error():
    import mindtorch_v2 as torch
    import torch as pt

    def mt():
        raise RuntimeError("X")

    def th():
        raise RuntimeError("Y")

    try:
        assert_torch_error(mt, th)
    except AssertionError:
        assert True
    else:
        assert False
```

**Step 2: Run test to verify it fails**
Run: `pytest tests/mindtorch_v2/contract/test_harness.py::test_harness_asserts_exact_error -v`
Expected: FAIL (helper not defined).

**Step 3: Write minimal implementation**
```python
# helpers.py
import torch as pt


def assert_torch_error(fn_mt, fn_torch):
    try:
        fn_torch()
    except Exception as e_torch:
        torch_exc = e_torch
    else:
        torch_exc = None

    try:
        fn_mt()
    except Exception as e_mt:
        mt_exc = e_mt
    else:
        mt_exc = None

    assert type(mt_exc) is type(torch_exc)
    assert str(mt_exc) == str(torch_exc)
```

**Step 4: Run test to verify it passes**
Run: `pytest tests/mindtorch_v2/contract/test_harness.py::test_harness_asserts_exact_error -v`
Expected: PASS.

**Step 5: Commit**
```bash
git add tests/mindtorch_v2/contract
git commit -m "test: add contract test harness"
```

---

### Task 2: Dispatch parity tests

**Files:**
- Create: `tests/mindtorch_v2/contract/test_dispatch_contract.py`

**Step 1: Write failing tests**
```python
import mindtorch_v2 as torch
import torch as pt
from tests.mindtorch_v2.contract.helpers import assert_torch_error


def test_pipeline_requires_meta_kernel_error():
    def mt():
        from mindtorch_v2._dispatch import pipeline
        from mindtorch_v2._dispatch.dispatcher import dispatch
        p = pipeline.DispatchPipeline()
        with p:
            dispatch("unknown_op", "cpu")

    def th():
        raise RuntimeError("pipeline requires meta kernel for op unknown_op")

    assert_torch_error(mt, th)
```

**Step 2: Run test to verify it fails**
Run: `pytest tests/mindtorch_v2/contract/test_dispatch_contract.py::test_pipeline_requires_meta_kernel_error -v`
Expected: FAIL (error mismatch).

**Step 3: Minimal implementation**
Align error messages in dispatcher to match torch; fix any dispatch key ordering issues revealed.

**Step 4: Run test to verify it passes**
Run: same as Step 2.

**Step 5: Commit**
```bash
git add tests/mindtorch_v2/contract/test_dispatch_contract.py src/mindtorch_v2/_dispatch/dispatcher.py

git commit -m "test: add dispatch contract tests"
```

---

### Task 3: Storage parity tests

**Files:**
- Create: `tests/mindtorch_v2/contract/test_storage_contract.py`

**Step 1: Write failing tests**
```python
import mindtorch_v2 as torch
import torch as pt
from tests.mindtorch_v2.contract.helpers import assert_torch_error


def test_storage_resize_pinned_error():
    x = torch.tensor([1.0]).pin_memory()
    px = pt.tensor([1.0]).pin_memory()

    def mt():
        x.storage().resize_(0)

    def th():
        px.untyped_storage().resize_(0)

    assert_torch_error(mt, th)
```

**Step 2: Run test to verify it fails**
Run: `pytest tests/mindtorch_v2/contract/test_storage_contract.py::test_storage_resize_pinned_error -v`
Expected: FAIL (message mismatch).

**Step 3: Minimal implementation**
Align storage error messages and unsupported paths in `_storage.py`.

**Step 4: Run test to verify it passes**
Run: same as Step 2.

**Step 5: Commit**
```bash
git add tests/mindtorch_v2/contract/test_storage_contract.py src/mindtorch_v2/_storage.py

git commit -m "test: add storage contract tests"
```

---

### Task 4: Autograd parity tests

**Files:**
- Create: `tests/mindtorch_v2/contract/test_autograd_contract.py`

**Step 1: Write failing tests**
```python
import mindtorch_v2 as torch
import torch as pt
from tests.mindtorch_v2.contract.helpers import assert_torch_error


def test_inplace_view_version_error_message():
    def mt():
        x = torch.tensor([1.0], requires_grad=True)
        y = x.view((1,))
        y.add_(1.0)
        y.sum().backward()

    def th():
        x = pt.tensor([1.0], requires_grad=True)
        y = x.view((1,))
        y.add_(1.0)
        y.sum().backward()

    assert_torch_error(mt, th)
```

**Step 2: Run test to verify it fails**
Run: `pytest tests/mindtorch_v2/contract/test_autograd_contract.py::test_inplace_view_version_error_message -v`
Expected: FAIL.

**Step 3: Minimal implementation**
Align version counter checks and error messages in `_autograd`/`_tensor`.

**Step 4: Run test to verify it passes**
Run: same as Step 2.

**Step 5: Commit**
```bash
git add tests/mindtorch_v2/contract/test_autograd_contract.py src/mindtorch_v2/_autograd src/mindtorch_v2/_tensor.py

git commit -m "test: add autograd contract tests"
```

---

### Task 5: NPU API parity tests (core + diagnostics)

**Files:**
- Create: `tests/mindtorch_v2/contract/test_npu_api_contract.py`

**Step 1: Write failing tests**
```python
import mindtorch_v2 as torch
import torch as pt
from tests.mindtorch_v2.contract.helpers import assert_torch_error


def test_npu_memory_summary_signature():
    def mt():
        torch.npu.memory_summary()

    def th():
        pt.cuda.memory_summary()

    assert_torch_error(mt, th)
```

**Step 2: Run test to verify it fails**
Run: `pytest tests/mindtorch_v2/contract/test_npu_api_contract.py::test_npu_memory_summary_signature -v`
Expected: FAIL.

**Step 3: Minimal implementation**
Align npu API signatures/behaviors and error messages to torch.cuda.

**Step 4: Run test to verify it passes**
Run: same as Step 2.

**Step 5: Commit**
```bash
git add tests/mindtorch_v2/contract/test_npu_api_contract.py src/mindtorch_v2/npu.py

git commit -m "test: add npu api contract tests"
```

---

### Task 6: Full test suite

**Step 1: Run tests**
Run: `pytest -q tests/mindtorch_v2`
Expected: PASS.

**Step 2: Commit (if needed)**
```bash
git add -A
git commit -m "test: run mindtorch_v2 suite"
```
