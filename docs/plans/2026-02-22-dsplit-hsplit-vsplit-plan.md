# dsplit/hsplit/vsplit Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Add CPU + meta support for `dsplit`, `hsplit`, `vsplit` with PyTorch-aligned behavior and update ops coverage.

**Architecture:** Implement `dsplit/hsplit/vsplit` as thin wrappers around existing `split` in CPU backend; register ops for dispatch + meta. Meta backend adds wrapper ops and meta inference functions that select the correct dimension and enforce `dsplit` dimensionality. Functional APIs dispatch to backend kernels; `__init__` exports the functions.

**Tech Stack:** Python, NumPy, PyTest, mindtorch v2 dispatch/registry.

---

### Task 1: Verify tests are red (TDD baseline)

**Files:**
- Test: `tests/mindtorch_v2/test_ops_cpu.py`
- Test: `tests/mindtorch_v2/test_meta_device.py`

**Step 1: Add failing tests**

Add CPU tests:
- `test_vsplit_cpu` (1D and 2D)
- `test_hsplit_cpu` (1D uses dim=0; 2D uses dim=1)
- `test_dsplit_cpu` (3D split + 2D error)

Add meta tests:
- `test_meta_vsplit_shape`
- `test_meta_hsplit_shape`
- `test_meta_dsplit_shape`

**Step 2: Run tests (expect FAIL)**

Run:
`pytest tests/mindtorch_v2/test_ops_cpu.py::test_vsplit_cpu tests/mindtorch_v2/test_ops_cpu.py::test_hsplit_cpu tests/mindtorch_v2/test_ops_cpu.py::test_dsplit_cpu tests/mindtorch_v2/test_meta_device.py::test_meta_vsplit_shape tests/mindtorch_v2/test_meta_device.py::test_meta_hsplit_shape tests/mindtorch_v2/test_meta_device.py::test_meta_dsplit_shape -q`

Expected: FAIL with op not registered/implemented.

---

### Task 2: Add CPU kernels (vsplit/hsplit/dsplit)

**Files:**
- Modify: `src/mindtorch_v2/_backends/cpu/ops.py`

**Step 1: Write minimal implementations**

Add:

```python
def vsplit(a, split_size_or_sections):
    return split(a, split_size_or_sections, dim=0)


def hsplit(a, split_size_or_sections):
    dim = 0 if a.dim() == 1 else 1
    return split(a, split_size_or_sections, dim=dim)


def dsplit(a, split_size_or_sections):
    if a.dim() < 3:
        raise ValueError("dsplit expects input with at least 3 dimensions")
    return split(a, split_size_or_sections, dim=2)
```

**Step 2: Run CPU tests**

Run:
`pytest tests/mindtorch_v2/test_ops_cpu.py::test_vsplit_cpu tests/mindtorch_v2/test_ops_cpu.py::test_hsplit_cpu tests/mindtorch_v2/test_ops_cpu.py::test_dsplit_cpu -q`

Expected: PASS for CPU tests.

---

### Task 3: Register CPU ops

**Files:**
- Modify: `src/mindtorch_v2/_backends/cpu/__init__.py`

**Step 1: Add imports**

Add `vsplit`, `hsplit`, `dsplit` to imports.

**Step 2: Register**

Add:

```python
registry.register("vsplit", "cpu", vsplit, meta=meta_infer.infer_vsplit)
registry.register("hsplit", "cpu", hsplit, meta=meta_infer.infer_hsplit)
registry.register("dsplit", "cpu", dsplit, meta=meta_infer.infer_dsplit)
```

**Step 3: Re-run CPU tests**

Run:
`pytest tests/mindtorch_v2/test_ops_cpu.py::test_vsplit_cpu tests/mindtorch_v2/test_ops_cpu.py::test_hsplit_cpu tests/mindtorch_v2/test_ops_cpu.py::test_dsplit_cpu -q`

Expected: PASS.

---

### Task 4: Add meta infer (vsplit/hsplit/dsplit)

**Files:**
- Modify: `src/mindtorch_v2/_backends/meta/infer.py`

**Step 1: Implement infer_vsplit/infer_hsplit/infer_dsplit**

```python
def infer_vsplit(a, split_size_or_sections):
    return infer_split(a, split_size_or_sections, dim=0)


def infer_hsplit(a, split_size_or_sections):
    dim = 0 if len(a.shape) == 1 else 1
    return infer_split(a, split_size_or_sections, dim=dim)


def infer_dsplit(a, split_size_or_sections):
    if len(a.shape) < 3:
        raise ValueError("dsplit expects input with at least 3 dimensions")
    return infer_split(a, split_size_or_sections, dim=2)
```

**Step 2: Run meta tests**

Run:
`pytest tests/mindtorch_v2/test_meta_device.py::test_meta_vsplit_shape tests/mindtorch_v2/test_meta_device.py::test_meta_hsplit_shape tests/mindtorch_v2/test_meta_device.py::test_meta_dsplit_shape -q`

Expected: PASS.

---

### Task 5: Add meta op registration

**Files:**
- Modify: `src/mindtorch_v2/_backends/meta/ops.py`
- Modify: `src/mindtorch_v2/_backends/meta/__init__.py`

**Step 1: Add meta wrapper ops**

In `ops.py` add:

```python
def _meta_vsplit_meta(a, split_size_or_sections):
    return _meta_split_meta(a, split_size_or_sections, dim=0)


def _meta_hsplit_meta(a, split_size_or_sections):
    dim = 0 if len(a.shape) == 1 else 1
    return _meta_split_meta(a, split_size_or_sections, dim=dim)


def _meta_dsplit_meta(a, split_size_or_sections):
    if len(a.shape) < 3:
        raise ValueError("dsplit expects input with at least 3 dimensions")
    return _meta_split_meta(a, split_size_or_sections, dim=2)
```

**Step 2: Register**

In `meta/__init__.py` add imports and register:

```python
registry.register("vsplit", "meta", _meta_vsplit_meta)
registry.register("hsplit", "meta", _meta_hsplit_meta)
registry.register("dsplit", "meta", _meta_dsplit_meta)
```

---

### Task 6: Functional API + exports

**Files:**
- Modify: `src/mindtorch_v2/_functional.py`
- Modify: `src/mindtorch_v2/__init__.py`

**Step 1: Add functional wrappers**

```python
def vsplit(a, split_size_or_sections):
    return dispatch("vsplit", a.device.type, a, split_size_or_sections)


def hsplit(a, split_size_or_sections):
    return dispatch("hsplit", a.device.type, a, split_size_or_sections)


def dsplit(a, split_size_or_sections):
    return dispatch("dsplit", a.device.type, a, split_size_or_sections)
```

**Step 2: Export in `__init__.py`**

Add to import list and `__all__`: `vsplit`, `hsplit`, `dsplit`.

---

### Task 7: Update ops coverage doc

**Files:**
- Modify: `docs/plans/ops-coverage.md`

**Step 1: Add entries**

Add lines for:
- `aten::vsplit`
- `aten::hsplit`
- `aten::dsplit`

---

### Task 8: Commit

**Step 1: Git status**

Run: `git status -sb`

**Step 2: Commit**

Run:
```bash
git add tests/mindtorch_v2/test_ops_cpu.py tests/mindtorch_v2/test_meta_device.py \
  src/mindtorch_v2/_backends/cpu/ops.py src/mindtorch_v2/_backends/cpu/__init__.py \
  src/mindtorch_v2/_backends/meta/infer.py src/mindtorch_v2/_backends/meta/ops.py \
  src/mindtorch_v2/_backends/meta/__init__.py src/mindtorch_v2/_functional.py \
  src/mindtorch_v2/__init__.py docs/plans/ops-coverage.md

git commit -m "feat: add dsplit hsplit vsplit"
```

---

### Task 9: Verification + PR prep

**Step 1: Run focused tests**

Run:
`pytest tests/mindtorch_v2/test_ops_cpu.py::test_vsplit_cpu tests/mindtorch_v2/test_ops_cpu.py::test_hsplit_cpu tests/mindtorch_v2/test_ops_cpu.py::test_dsplit_cpu tests/mindtorch_v2/test_meta_device.py::test_meta_vsplit_shape tests/mindtorch_v2/test_meta_device.py::test_meta_hsplit_shape tests/mindtorch_v2/test_meta_device.py::test_meta_dsplit_shape -q`

Expected: PASS.

**Step 2: Rebase `ms/master`**

Run:
`git fetch`
`git rebase ms/master`

**Step 3: Push + create PR**

Use correct PR description with real newlines (heredoc).
