# MindTorch v2 ACLNN Ctypes Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Implement a ctypes-based ACLNN backend for NPU ops `add/mul/relu/sum/matmul` without OM conversion or torch_npu.

**Architecture:** Add a dedicated ctypes wrapper module to load ACLNN shared libs and expose the minimal op APIs. The existing Ascend backend becomes a facade that initializes ACL runtime, validates inputs, and dispatches to the ctypes wrapper while keeping NPU storage semantics.

**Tech Stack:** Python, ctypes, ACL runtime (`acl`), Ascend ACLNN shared libraries.

---

### Task 1: Add ACLNN availability probe + test

**Files:**
- Create: `src/mindtorch_v2/_backends/ascend_ctypes.py`
- Modify: `src/mindtorch_v2/_backends/ascend.py`
- Test: `tests/mindtorch_v2/test_ops_npu.py`

**Step 1: Write the failing test**

```python
def test_npu_aclnn_available():
    if not torch.npu.is_available():
        pytest.skip("NPU not available")
    assert torch._C._npu_aclnn_available() is True
```

**Step 2: Run test to verify it fails**

Run: `pytest tests/mindtorch_v2/test_ops_npu.py::test_npu_aclnn_available -v`
Expected: FAIL (AttributeError)

**Step 3: Write minimal implementation**

- Implement `ascend_ctypes.is_available()` by attempting to load `libaclnn_ops_infer.so` and `libaclnn_math.so` via ctypes.
- Expose `_npu_aclnn_available()` in `_C` and wire through `ascend.py`.

**Step 4: Run test to verify it passes**

Run: `pytest tests/mindtorch_v2/test_ops_npu.py::test_npu_aclnn_available -v`
Expected: PASS (or skip)

**Step 5: Commit**

```bash
git add src/mindtorch_v2/_backends/ascend_ctypes.py src/mindtorch_v2/_backends/ascend.py tests/mindtorch_v2/test_ops_npu.py src/mindtorch_v2/_C.py
git commit -m "feat(mindtorch_v2): add aclnn ctypes availability probe"
```

---

### Task 2: Define ACLNN ctypes bindings for core ops

**Files:**
- Modify: `src/mindtorch_v2/_backends/ascend_ctypes.py`

**Step 1: Write the failing test**

Add a test that calls a no-op binding stub or checks symbol presence.

```python
def test_aclnn_symbols_present():
    if not torch.npu.is_available():
        pytest.skip("NPU not available")
    assert torch._C._npu_aclnn_symbols_ok() is True
```

**Step 2: Run test to verify it fails**

Run: `pytest tests/mindtorch_v2/test_ops_npu.py::test_aclnn_symbols_present -v`
Expected: FAIL

**Step 3: Write minimal implementation**

- In `ascend_ctypes.py`, define ctypes signatures for:
  - `aclnnAdd` / `aclnnMul` / `aclnnRelu` / `aclnnReduceSum` / `aclnnMatmul` (exact names from headers).
  - Any required tensor/descriptor create/destroy APIs.
- Implement `_npu_aclnn_symbols_ok()` in `_C`.

**Step 4: Run test to verify it passes**

Run: `pytest tests/mindtorch_v2/test_ops_npu.py::test_aclnn_symbols_present -v`
Expected: PASS (or skip)

**Step 5: Commit**

```bash
git add src/mindtorch_v2/_backends/ascend_ctypes.py src/mindtorch_v2/_C.py tests/mindtorch_v2/test_ops_npu.py
git commit -m "feat(mindtorch_v2): add aclnn ctypes bindings"
```

---

### Task 3: Wire NPU op add through ACLNN

**Files:**
- Modify: `src/mindtorch_v2/_backends/ascend.py`
- Modify: `src/mindtorch_v2/_backends/ascend_ctypes.py`
- Test: `tests/mindtorch_v2/test_ops_npu.py`

**Step 1: Write the failing test**

```python
def test_npu_add_execute():
    if not torch.npu.is_available():
        pytest.skip("NPU not available")
    a = torch.tensor([1.0, 2.0], device="npu")
    b = torch.tensor([3.0, 4.0], device="npu")
    out = a + b
    assert out.to("cpu").numpy().tolist() == [4.0, 6.0]
```

**Step 2: Run test to verify it fails**

Run: `pytest tests/mindtorch_v2/test_ops_npu.py::test_npu_add_execute -v`
Expected: FAIL

**Step 3: Write minimal implementation**

- In `ascend_ctypes.py`, implement an `add` wrapper that builds ACLNN tensors from device pointers and calls the ACLNN add API.
- In `ascend.py`, dispatch NPU add to the ctypes wrapper.

**Step 4: Run test to verify it passes**

Run: `pytest tests/mindtorch_v2/test_ops_npu.py::test_npu_add_execute -v`
Expected: PASS (or skip)

**Step 5: Commit**

```bash
git add src/mindtorch_v2/_backends/ascend.py src/mindtorch_v2/_backends/ascend_ctypes.py tests/mindtorch_v2/test_ops_npu.py
git commit -m "feat(mindtorch_v2): run npu add via aclnn ctypes"
```

---

### Task 4: Implement mul + relu via ACLNN

**Files:**
- Modify: `src/mindtorch_v2/_backends/ascend.py`
- Modify: `src/mindtorch_v2/_backends/ascend_ctypes.py`
- Test: `tests/mindtorch_v2/test_ops_npu.py`

**Step 1: Write the failing test**

```python
def test_npu_mul_relu():
    if not torch.npu.is_available():
        pytest.skip("NPU not available")
    a = torch.tensor([-1.0, 2.0], device="npu")
    b = torch.tensor([3.0, 4.0], device="npu")
    assert (a * b).to("cpu").numpy().tolist() == [-3.0, 8.0]
    assert a.relu().to("cpu").numpy().tolist() == [0.0, 2.0]
```

**Step 2: Run test to verify it fails**

Run: `pytest tests/mindtorch_v2/test_ops_npu.py::test_npu_mul_relu -v`
Expected: FAIL

**Step 3: Write minimal implementation**

- Implement `mul` and `relu` ctypes wrappers, add dispatch in `ascend.py`.

**Step 4: Run test to verify it passes**

Run: `pytest tests/mindtorch_v2/test_ops_npu.py::test_npu_mul_relu -v`
Expected: PASS (or skip)

**Step 5: Commit**

```bash
git add src/mindtorch_v2/_backends/ascend.py src/mindtorch_v2/_backends/ascend_ctypes.py tests/mindtorch_v2/test_ops_npu.py
git commit -m "feat(mindtorch_v2): add aclnn mul/relu"
```

---

### Task 5: Implement sum (dim/keepdim) via ACLNN

**Files:**
- Modify: `src/mindtorch_v2/_backends/ascend.py`
- Modify: `src/mindtorch_v2/_backends/ascend_ctypes.py`
- Test: `tests/mindtorch_v2/test_ops_npu.py`

**Step 1: Write the failing test**

```python
def test_npu_sum():
    if not torch.npu.is_available():
        pytest.skip("NPU not available")
    a = torch.tensor([[1.0, 2.0]], device="npu")
    assert a.sum().to("cpu").numpy().tolist() == 3.0
    assert a.sum(dim=1, keepdim=True).to("cpu").numpy().tolist() == [[3.0]]
```

**Step 2: Run test to verify it fails**

Run: `pytest tests/mindtorch_v2/test_ops_npu.py::test_npu_sum -v`
Expected: FAIL

**Step 3: Write minimal implementation**

- Implement `sum` ctypes wrapper, pass `dim` and `keepdim` attributes if supported by ACLNN.

**Step 4: Run test to verify it passes**

Run: `pytest tests/mindtorch_v2/test_ops_npu.py::test_npu_sum -v`
Expected: PASS (or skip)

**Step 5: Commit**

```bash
git add src/mindtorch_v2/_backends/ascend.py src/mindtorch_v2/_backends/ascend_ctypes.py tests/mindtorch_v2/test_ops_npu.py
git commit -m "feat(mindtorch_v2): add aclnn sum"
```

---

### Task 6: Implement matmul via ACLNN

**Files:**
- Modify: `src/mindtorch_v2/_backends/ascend.py`
- Modify: `src/mindtorch_v2/_backends/ascend_ctypes.py`
- Test: `tests/mindtorch_v2/test_ops_npu.py`

**Step 1: Write the failing test**

```python
def test_npu_matmul():
    if not torch.npu.is_available():
        pytest.skip("NPU not available")
    a = torch.tensor([[1.0, 2.0]], device="npu")
    b = torch.tensor([[3.0], [4.0]], device="npu")
    out = a @ b
    assert out.to("cpu").numpy().tolist() == [[11.0]]
```

**Step 2: Run test to verify it fails**

Run: `pytest tests/mindtorch_v2/test_ops_npu.py::test_npu_matmul -v`
Expected: FAIL

**Step 3: Write minimal implementation**

- Implement `matmul` ctypes wrapper and dispatch in `ascend.py`.

**Step 4: Run test to verify it passes**

Run: `pytest tests/mindtorch_v2/test_ops_npu.py::test_npu_matmul -v`
Expected: PASS (or skip)

**Step 5: Commit**

```bash
git add src/mindtorch_v2/_backends/ascend.py src/mindtorch_v2/_backends/ascend_ctypes.py tests/mindtorch_v2/test_ops_npu.py
git commit -m "feat(mindtorch_v2): add aclnn matmul"
```

---

Plan complete and saved to `docs/plans/2026-02-11-mindtorch-v2-aclnn-ctypes-plan.md`. Two execution options:

1. Subagent-Driven (this session) - I dispatch fresh subagent per task, review between tasks, fast iteration
2. Parallel Session (separate) - Open new session with executing-plans, batch execution with checkpoints

Which approach? (1/2)
