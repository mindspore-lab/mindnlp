# MindTorch v2 ACL Execute Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Implement real Ascend NPU ops in mindtorch_v2 via `acl.op.execute_v2` (no OM conversion) with clear errors when ops/models are unavailable.

**Architecture:** Add an Ascend backend runtime wrapper that initializes ACL, resolves a working model dir by probing toolkit OPP and local `acl_engine`, and dispatches `add/mul/relu/sum/matmul` via `execute_v2` with explicit H2D/D2H transfers and stream sync. CPU path stays NumPy-based.

**Tech Stack:** Python, `acl` Python API, NumPy, pytest.

---

### Task 1: Add a model-dir probe test (NPU-only)

**Files:**
- Modify: `tests/mindtorch_v2/test_ops_npu.py`

**Step 1: Write the failing test**

```python
def test_npu_model_dir_probe():
    if not torch.npu.is_available():
        pytest.skip("NPU not available")
    ok = torch._C._npu_probe_model_dirs()
    assert ok is True
```

**Step 2: Run test to verify it fails**

Run: `pytest tests/mindtorch_v2/test_ops_npu.py::test_npu_model_dir_probe -v`
Expected: FAIL with AttributeError/Not implemented

**Step 3: Write minimal implementation**

Expose a private probe helper in the NPU backend, wired through `_C` or `mindtorch_v2` package for test usage.

**Step 4: Run test to verify it passes**

Run: `pytest tests/mindtorch_v2/test_ops_npu.py::test_npu_model_dir_probe -v`
Expected: PASS (or skip if NPU unavailable)

**Step 5: Commit**

```bash
git add tests/mindtorch_v2/test_ops_npu.py src/mindtorch_v2/__init__.py src/mindtorch_v2/_backends/ascend.py
git commit -m "test(mindtorch_v2): add npu model dir probe"
```

---

### Task 2: Implement ACL runtime wrapper + model-dir resolution

**Files:**
- Modify: `src/mindtorch_v2/_backends/ascend.py`

**Step 1: Write the failing test**

Extend the probe test to call a backend probe and assert the selected model dir is one of the allowed paths.

```python
def test_npu_model_dir_selected():
    if not torch.npu.is_available():
        pytest.skip("NPU not available")
    path = torch._C._npu_model_dir()
    assert path in {"/usr/local/Ascend/ascend-toolkit/latest/opp", "/home/lvyufeng/lvyufeng/acl_engine"}
```

**Step 2: Run test to verify it fails**

Run: `pytest tests/mindtorch_v2/test_ops_npu.py::test_npu_model_dir_selected -v`
Expected: FAIL

**Step 3: Write minimal implementation**

Implement:
- `_ensure_init()` that calls `acl.init`, `acl.rt.set_device`, `acl.rt.create_context`, `acl.rt.create_stream`
- `_probe_model_dirs()` which tries `acl.op.set_model_dir` on each candidate and attempts a tiny op (e.g., add) to validate.
- `_model_dir()` getter that returns selected path or raises.

**Step 4: Run test to verify it passes**

Run: `pytest tests/mindtorch_v2/test_ops_npu.py::test_npu_model_dir_selected -v`
Expected: PASS (or skip)

**Step 5: Commit**

```bash
git add src/mindtorch_v2/_backends/ascend.py tests/mindtorch_v2/test_ops_npu.py
git commit -m "feat(mindtorch_v2): add acl runtime and model dir resolution"
```

---

### Task 3: Implement execute_v2 helpers (desc, malloc, memcpy, execute)

**Files:**
- Modify: `src/mindtorch_v2/_backends/ascend.py`

**Step 1: Write the failing test**

Add a small internal helper test that calls the add op via NPU backend on tiny tensors and checks output.

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

Add helpers:
- `_dtype_to_acl`, `_shape_to_desc` (ACL_FORMAT_ND)
- `_malloc_device`, `_free_device`
- `_memcpy_h2d`, `_memcpy_d2h`
- `_execute_v2(op_name, input_descs, input_bufs, output_descs, output_bufs, attrs=None)`

**Step 4: Run test to verify it passes**

Run: `pytest tests/mindtorch_v2/test_ops_npu.py::test_npu_add_execute -v`
Expected: PASS (or skip)

**Step 5: Commit**

```bash
git add src/mindtorch_v2/_backends/ascend.py tests/mindtorch_v2/test_ops_npu.py
git commit -m "feat(mindtorch_v2): add execute_v2 helpers"
```

---

### Task 4: Wire NPU tensor storage + to("npu")/to("cpu")

**Files:**
- Modify: `src/mindtorch_v2/_tensor.py`
- Modify: `src/mindtorch_v2/_backends/ascend.py`

**Step 1: Write the failing test**

```python
def test_npu_roundtrip():
    if not torch.npu.is_available():
        pytest.skip("NPU not available")
    t = torch.tensor([[1, 2], [3, 4]], dtype=torch.float32)
    n = t.to("npu")
    assert n.device.type == "npu"
    c = n.to("cpu")
    assert c.numpy().tolist() == [[1.0, 2.0], [3.0, 4.0]]
```

**Step 2: Run test to verify it fails**

Run: `pytest tests/mindtorch_v2/test_ops_npu.py::test_npu_roundtrip -v`
Expected: FAIL

**Step 3: Write minimal implementation**

Add NPU storage wrapper with device ptr + size. `to("npu")` should allocate device buffer and copy H2D. `to("cpu")` should copy D2H and build CPU tensor.

**Step 4: Run test to verify it passes**

Run: `pytest tests/mindtorch_v2/test_ops_npu.py::test_npu_roundtrip -v`
Expected: PASS (or skip)

**Step 5: Commit**

```bash
git add src/mindtorch_v2/_tensor.py src/mindtorch_v2/_backends/ascend.py tests/mindtorch_v2/test_ops_npu.py
git commit -m "feat(mindtorch_v2): add npu storage and device transfer"
```

---

### Task 5: Implement NPU ops add/mul/relu/sum/matmul

**Files:**
- Modify: `src/mindtorch_v2/_backends/ascend.py`
- Modify: `src/mindtorch_v2/_functional.py`

**Step 1: Write the failing tests**

Add tests for each op in `tests/mindtorch_v2/test_ops_npu.py` (or extend existing).

```python
def test_npu_ops():
    if not torch.npu.is_available():
        pytest.skip("NPU not available")
    a = torch.tensor([[1.0, 2.0]], device="npu")
    b = torch.tensor([[3.0, 4.0]], device="npu")
    assert (a + b).to("cpu").numpy().tolist() == [[4.0, 6.0]]
    assert (a * b).to("cpu").numpy().tolist() == [[3.0, 8.0]]
    assert (a @ b.T).to("cpu").numpy().tolist() == [[11.0]]
    assert a.relu().to("cpu").numpy().tolist() == [[1.0, 2.0]]
    assert a.sum().to("cpu").numpy().tolist() == 3.0
```

**Step 2: Run test to verify it fails**

Run: `pytest tests/mindtorch_v2/test_ops_npu.py::test_npu_ops -v`
Expected: FAIL

**Step 3: Write minimal implementation**

Implement NPU backend ops and dispatch in `_functional` based on device type. Ensure shapes/dtypes are validated and errors are explicit when op model missing.

**Step 4: Run test to verify it passes**

Run: `pytest tests/mindtorch_v2/test_ops_npu.py::test_npu_ops -v`
Expected: PASS (or skip)

**Step 5: Commit**

```bash
git add src/mindtorch_v2/_backends/ascend.py src/mindtorch_v2/_functional.py tests/mindtorch_v2/test_ops_npu.py
git commit -m "feat(mindtorch_v2): add npu ops via execute_v2"
```

---

### Task 6: Add backward coverage for NPU ops

**Files:**
- Modify: `tests/mindtorch_v2/test_ops_npu.py`

**Step 1: Write the failing test**

```python
def test_npu_autograd():
    if not torch.npu.is_available():
        pytest.skip("NPU not available")
    x = torch.tensor([1.0, 2.0], device="npu", requires_grad=True)
    y = (x * x).sum()
    y.backward()
    assert x.grad.to("cpu").numpy().tolist() == [2.0, 4.0]
```

**Step 2: Run test to verify it fails**

Run: `pytest tests/mindtorch_v2/test_ops_npu.py::test_npu_autograd -v`
Expected: FAIL

**Step 3: Write minimal implementation**

Ensure NPU ops integrate with autograd and gradients are computed on CPU (current engine) but respect device transfer for results.

**Step 4: Run test to verify it passes**

Run: `pytest tests/mindtorch_v2/test_ops_npu.py::test_npu_autograd -v`
Expected: PASS (or skip)

**Step 5: Commit**

```bash
git add tests/mindtorch_v2/test_ops_npu.py src/mindtorch_v2/_functional.py
git commit -m "test(mindtorch_v2): cover npu autograd"
```

---

### Task 7: Wire `torch.npu` availability in package init

**Files:**
- Modify: `src/mindtorch_v2/__init__.py`

**Step 1: Write the failing test**

```python
def test_npu_module_available():
    import mindtorch_v2 as torch
    assert hasattr(torch, "npu")
```

**Step 2: Run test to verify it fails**

Run: `pytest tests/mindtorch_v2/test_ops_npu.py::test_npu_module_available -v`
Expected: FAIL

**Step 3: Write minimal implementation**

Expose `npu` module in package init.

**Step 4: Run test to verify it passes**

Run: `pytest tests/mindtorch_v2/test_ops_npu.py::test_npu_module_available -v`
Expected: PASS

**Step 5: Commit**

```bash
git add src/mindtorch_v2/__init__.py tests/mindtorch_v2/test_ops_npu.py
git commit -m "feat(mindtorch_v2): expose torch.npu module"
```

---

### Task 8: Documentation note for NPU op availability

**Files:**
- Modify: `README.md`

**Step 1: Write the failing doc check**

No automated test.

**Step 2: Write minimal documentation**

Add a short section noting that NPU ops require a working op model dir and will error if unavailable.

**Step 3: Commit**

```bash
git add README.md
git commit -m "docs: note npu op model dependency"
```

---

Plan complete and saved to `docs/plans/2026-02-11-mindtorch-v2-acl-execute-plan.md`. Two execution options:

1. Subagent-Driven (this session) - I dispatch fresh subagent per task, review between tasks, fast iteration
2. Parallel Session (separate) - Open new session with executing-plans, batch execution with checkpoints

Which approach? (1/2)
