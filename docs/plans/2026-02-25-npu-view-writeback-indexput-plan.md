# NPU View Writeback via IndexPutImpl Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Add ACLNN IndexPutImpl writeback so non-contiguous NPU views are written back under functionalize, matching torch/torch-npu behavior.

**Architecture:** Keep the existing contiguous D2D memcpy path. For non-contiguous views, build NPU indices via aclnnArange and use aclnnIndexPutImpl to update the flattened base tensor. Update view metadata after writeback.

**Tech Stack:** Python, pytest, ACLNN via ctypes.

---

### Task 1: Add failing test for non-contiguous NPU view writeback

**Files:**
- Modify: `tests/mindtorch_v2/contract/test_functionalize_view_writeback.py`

**Step 1: Write the failing test**

```python
def test_functionalize_writeback_respects_noncontig_view_npu():
    if not torch.npu.is_available():
        pytest.skip("NPU not available")
    base = torch.tensor([0.0, 1.0, 2.0, 3.0, 4.0, 5.0], device="npu").view((2, 3))
    view = base.transpose(0, 1)
    with torch.functionalize():
        view.add_(torch.ones(view.shape, device="npu"))
    assert base.to("cpu").storage().data.tolist() == [1.0, 2.0, 3.0, 4.0, 5.0, 6.0]
```

**Step 2: Run test to verify it fails**

Run: `pytest -q tests/mindtorch_v2/contract/test_functionalize_view_writeback.py -k noncontig`
Expected: FAIL with `aten::add_ is not implemented for NPU`.

---

### Task 2: Add ACLNN bindings for arange and index_put_impl

**Files:**
- Modify: `src/mindtorch_v2/_backends/npu/aclnn.py`

**Step 1: Add bindings**

Add optional symbols for:
- `aclnnArangeGetWorkspaceSize`, `aclnnArange`
- `aclnnIndexPutImplGetWorkspaceSize`, `aclnnIndexPutImpl`

**Step 2: Add symbol checks**

```python
def arange_symbols_ok():
    return all([bindings.aclnn_arange_get_workspace, bindings.aclnn_arange])

def index_put_impl_symbols_ok():
    return all([bindings.aclnn_index_put_impl_get_workspace, bindings.aclnn_index_put_impl])
```

**Step 3: Implement arange helper**

```python
def arange(start, end, step, out_ptr, out_shape, out_stride, dtype, runtime, stream=None):
    # aclnnArangeGetWorkspaceSize + aclnnArange
```

**Step 4: Implement index_put_impl helper**

```python
def index_put_impl(self_ptr, self_shape, self_stride, dtype, index_ptrs, index_shapes, index_strides, index_dtypes,
                   values_ptr, values_shape, values_stride, values_dtype, accumulate, unsafe, runtime, stream=None):
    # create aclTensorList for indices, run aclnnIndexPutImpl
```

**Step 5: Run unit test**

Run: `pytest -q tests/mindtorch_v2/contract/test_functionalize_view_writeback.py -k noncontig`
Expected: still FAIL (implementation not wired yet).

---

### Task 3: Implement NPU non-contiguous view writeback

**Files:**
- Modify: `src/mindtorch_v2/_dispatch/functionalize.py`
- Modify: `src/mindtorch_v2/_backends/npu/ops.py`

**Step 1: Add NPU index helpers**

In `ops.py`, add helper functions:
- `_npu_arange_1d(size, device)` using aclnn.arange
- `_npu_broadcast_shape(tensor, target_shape)` that uses `reshape` + `expand` (or equivalent ops available)
- `_npu_linear_index(shape, stride, offset)` that returns an NPU int64 tensor of linear indices

**Step 2: Add NPU index_put_impl wrapper**

In `ops.py`, add `_index_put_impl(self_tensor, index_tensor, values_tensor)` calling aclnn.index_put_impl.

**Step 3: Wire into functionalize writeback**

In `functionalize.py`:
- For non-contiguous NPU views, call the new index_put_impl path.
- Keep contiguous memcpy path unchanged.

**Step 4: Run tests**

Run: `pytest -q tests/mindtorch_v2/contract/test_functionalize_view_writeback.py -k noncontig`
Expected: PASS.

---

### Task 4: Verify regression tests

**Step 1: Run full contract file**

Run: `pytest -q tests/mindtorch_v2/contract/test_functionalize_view_writeback.py`
Expected: PASS.

**Step 2: Commit**

```bash
git add tests/mindtorch_v2/contract/test_functionalize_view_writeback.py \
    src/mindtorch_v2/_backends/npu/aclnn.py \
    src/mindtorch_v2/_backends/npu/ops.py \
    src/mindtorch_v2/_dispatch/functionalize.py \
    docs/plans/2026-02-25-npu-view-writeback-indexput-design.md \
    docs/plans/2026-02-25-npu-view-writeback-indexput-plan.md

git commit -m "feat: add npu non-contig view writeback via index_put"
```
