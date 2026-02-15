# Contiguous + To Autograd Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Add Autograd wrappers and backend kernels for `contiguous` and `to`, aligned with Torch behavior.

**Architecture:** Thin Tensor methods dispatch into Autograd wrappers which redispatch to backend kernels. Backend kernels provide device-specific contiguous creation and conversion.

**Tech Stack:** Python, NumPy, ACL NPU runtime, existing dispatch/autograd infrastructure.

---

### Task 1: Add failing tests for `contiguous` autograd

**Files:**
- Modify: `tests/mindtorch_v2/test_dispatch_autograd_wrappers.py`

**Step 1: Write the failing test**
```python
def test_contiguous_autograd_sets_grad_fn():
    import mindtorch_v2 as torch

    x = torch.tensor([[1.0, 2.0], [3.0, 4.0]], requires_grad=True)
    y = x.transpose(0, 1).contiguous()
    assert y.grad_fn is not None

    out = y.sum()
    out.backward()
    assert x.grad is not None
```

**Step 2: Run test to verify it fails**
Run: `pytest tests/mindtorch_v2/test_dispatch_autograd_wrappers.py::test_contiguous_autograd_sets_grad_fn -v`
Expected: FAIL with missing `contiguous` kernel or missing grad_fn.

**Step 3: Write minimal implementation**
(Do in later tasks.)

**Step 4: Run test to verify it passes**
(Do after implementation.)

**Step 5: Commit**
```bash
git add tests/mindtorch_v2/test_dispatch_autograd_wrappers.py
git commit -m "test: add contiguous autograd coverage"
```

---

### Task 2: Add failing tests for `to` autograd

**Files:**
- Modify: `tests/mindtorch_v2/test_dispatch_autograd_wrappers.py`

**Step 1: Write the failing test**
```python
def test_to_autograd_sets_grad_fn_cpu():
    import mindtorch_v2 as torch

    x = torch.tensor([1.0, 2.0, 3.0], requires_grad=True)
    y = x.to("cpu")
    assert y.grad_fn is not None or y is x

    out = y.sum()
    out.backward()
    assert x.grad is not None
```

**Step 2: Run test to verify it fails**
Run: `pytest tests/mindtorch_v2/test_dispatch_autograd_wrappers.py::test_to_autograd_sets_grad_fn_cpu -v`
Expected: FAIL with missing autograd wrapper or missing grad.

**Step 3: Write minimal implementation**
(Do in later tasks.)

**Step 4: Run test to verify it passes**
(Do after implementation.)

**Step 5: Commit**
```bash
git add tests/mindtorch_v2/test_dispatch_autograd_wrappers.py
git commit -m "test: add to autograd coverage"
```

---

### Task 3: Add CPU `contiguous` kernel and register

**Files:**
- Modify: `src/mindtorch_v2/_backends/cpu/ops.py`
- Modify: `src/mindtorch_v2/_backends/cpu/__init__.py`

**Step 1: Write minimal implementation**
```python
# in cpu/ops.py

def contiguous(a):
    if a.device.type != "cpu":
        raise ValueError("CPU contiguous expects CPU tensors")
    arr = np.ascontiguousarray(a._numpy_view())
    storage = typed_storage_from_numpy(arr, a.dtype, device=a.device)
    stride = tuple(np.array(arr.strides) // arr.itemsize)
    return Tensor(storage, arr.shape, stride)
```

**Step 2: Register kernel**
```python
# in cpu/__init__.py
registry.register("contiguous", "cpu", contiguous, meta=meta_infer.infer_unary)
```

**Step 3: Run targeted tests**
Run: `pytest tests/mindtorch_v2/test_dispatch_autograd_wrappers.py::test_contiguous_autograd_sets_grad_fn -v`
Expected: FAIL (autograd still missing), but kernel should resolve missing-op errors.

**Step 4: Commit**
```bash
git add src/mindtorch_v2/_backends/cpu/ops.py src/mindtorch_v2/_backends/cpu/__init__.py
git commit -m "feat: add cpu contiguous kernel"
```

---

### Task 4: Add NPU `contiguous` kernel and register

**Files:**
- Modify: `src/mindtorch_v2/_backends/npu/ops.py`
- Modify: `src/mindtorch_v2/_backends/npu/__init__.py`

**Step 1: Write minimal implementation**
```python
# in npu/ops.py

def contiguous(a):
    if a.device.type != "npu":
        raise ValueError("NPU contiguous expects NPU tensors")
    runtime = npu_runtime.get_runtime((a.device.index or 0))
    stream = npu_state.current_stream((a.device.index or 0))
    a_storage = _unwrap_storage(a)
    out_size = int(np.prod(a.shape)) * np.dtype(npu_runtime._dtype_to_numpy(a.dtype)).itemsize
    out_ptr = npu_runtime._alloc_device(out_size, runtime=runtime)
    ret = npu_runtime.acl.rt.memcpy(out_ptr, out_size, a_storage.data_ptr(), out_size, 3)
    if ret != npu_runtime.ACL_ERROR_CODE:
        raise RuntimeError(f"acl.rt.memcpy D2D failed: {ret}")
    storage = npu_typed_storage_from_ptr(out_ptr, a.shape, a.dtype, device=a.device)
    return Tensor(storage, a.shape, npu_runtime._contiguous_stride(a.shape))
```

**Step 2: Register kernel**
```python
# in npu/__init__.py
registry.register("contiguous", "npu", contiguous, meta=meta_infer.infer_unary)
```

**Step 3: Commit**
```bash
git add src/mindtorch_v2/_backends/npu/ops.py src/mindtorch_v2/_backends/npu/__init__.py
git commit -m "feat: add npu contiguous kernel"
```

---

### Task 5: Add Meta `contiguous` kernel and register

**Files:**
- Modify: `src/mindtorch_v2/_backends/meta/ops.py`
- Modify: `src/mindtorch_v2/_backends/meta/__init__.py`

**Step 1: Write minimal implementation**
```python
# in meta/ops.py

def _meta_contiguous_meta(a):
    return _meta_tensor(a.shape, a.dtype, a.device)
```

**Step 2: Register kernel**
```python
# in meta/__init__.py
registry.register("contiguous", "meta", _meta_contiguous_meta)
```

**Step 3: Commit**
```bash
git add src/mindtorch_v2/_backends/meta/ops.py src/mindtorch_v2/_backends/meta/__init__.py
git commit -m "feat: add meta contiguous kernel"
```

---

### Task 6: Add Autograd wrappers for `contiguous` and `to`

**Files:**
- Modify: `src/mindtorch_v2/_backends/autograd.py`

**Step 1: Write minimal implementation**
```python
# in autograd.py

def contiguous(a):
    keyset = current_dispatch_keyset().without(DispatchKey.Autograd)
    out = redispatch("contiguous", keyset, a)
    if GradMode.enabled and a.requires_grad:
        grad_fn = ContiguousBackward(a)
        out.grad_fn = grad_fn
        out.requires_grad = True
    return out


def to(a, device, non_blocking=False):
    keyset = current_dispatch_keyset().without(DispatchKey.Autograd)
    out = redispatch("to", keyset, a, device, non_blocking=non_blocking)
    if GradMode.enabled and a.requires_grad and out is not a:
        grad_fn = ToBackward(a)
        out.grad_fn = grad_fn
        out.requires_grad = True
    return out
```

**Step 2: Run tests**
Run: `pytest tests/mindtorch_v2/test_dispatch_autograd_wrappers.py::test_contiguous_autograd_sets_grad_fn -v`
Expected: PASS

Run: `pytest tests/mindtorch_v2/test_dispatch_autograd_wrappers.py::test_to_autograd_sets_grad_fn_cpu -v`
Expected: PASS

**Step 3: Commit**
```bash
git add src/mindtorch_v2/_backends/autograd.py
git commit -m "feat: add autograd wrappers for contiguous and to"
```

---

### Task 7: Clean `Tensor.contiguous` implementation

**Files:**
- Modify: `src/mindtorch_v2/_tensor.py`

**Step 1: Update implementation**
```python
    def contiguous(self, memory_format=None):
        if self.is_contiguous():
            return self
        return dispatch("contiguous", self)
```

**Step 2: Run targeted tests**
Run: `pytest tests/mindtorch_v2/test_dispatch_autograd_wrappers.py::test_contiguous_autograd_sets_grad_fn -v`
Expected: PASS

**Step 3: Commit**
```bash
git add src/mindtorch_v2/_tensor.py
git commit -m "refactor: simplify tensor.contiguous"
```

---

### Task 8: Full test suite

**Step 1: Run tests**
Run: `pytest -q tests/mindtorch_v2`
Expected: PASS

**Step 2: Commit (if needed)**
```bash
git add -A
git commit -m "test: run mindtorch_v2 suite"
```
