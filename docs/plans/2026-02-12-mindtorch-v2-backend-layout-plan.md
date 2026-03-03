# Backend Layout (Device-Centric) Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Reorganize `_backends/` by device (cpu/meta/npu) with shared helpers, while keeping dispatch registration and a compatibility shim.

**Architecture:** Move device-specific ops/creation/runtime into per-device packages, keep shared helpers in `common/`, and add a shim module to preserve existing imports. Update registrations and imports accordingly.

**Tech Stack:** Python, pytest

---

### Task 1: Add a failing layout/shim test

**Files:**
- Create: `tests/mindtorch_v2/test_backend_layout.py`

**Step 1: Write the failing test**

```python
import mindtorch_v2._backends.ascend as ascend
import mindtorch_v2._backends.npu as npu


def test_backend_shim_imports():
    assert ascend.is_available is npu.is_available
```

**Step 2: Run test to verify it fails**

Run: `pytest tests/mindtorch_v2/test_backend_layout.py::test_backend_shim_imports -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'mindtorch_v2._backends.npu'`

**Step 3: Commit**

```bash
git add tests/mindtorch_v2/test_backend_layout.py
git commit -m "test: add backend shim import check"
```

---

### Task 2: Move shared helpers into `common/`

**Files:**
- Create: `src/mindtorch_v2/_backends/common/__init__.py`
- Move: `src/mindtorch_v2/_backends/view.py` -> `src/mindtorch_v2/_backends/common/view.py`
- Move: `src/mindtorch_v2/_backends/convert.py` -> `src/mindtorch_v2/_backends/common/convert.py`
- Modify: `src/mindtorch_v2/_backends/cpu.py`
- Modify: `src/mindtorch_v2/_backends/ascend.py`

**Step 1: Update imports to use `common.view` and `common.convert`**

```python
from ..common import view as view_backend
from ..common import convert as convert_backend
```

**Step 2: Remove old top-level `view.py`/`convert.py`**

**Step 3: Run test to verify it still fails**

Run: `pytest tests/mindtorch_v2/test_backend_layout.py::test_backend_shim_imports -v`
Expected: FAIL (still no npu package)

**Step 4: Commit**

```bash
git add src/mindtorch_v2/_backends/common src/mindtorch_v2/_backends/cpu.py src/mindtorch_v2/_backends/ascend.py
git commit -m "refactor: move backend helpers to common"
```

---

### Task 3: Create `cpu/` and `meta/` packages

**Files:**
- Create: `src/mindtorch_v2/_backends/cpu/__init__.py`
- Create: `src/mindtorch_v2/_backends/cpu/ops.py`
- Create: `src/mindtorch_v2/_backends/cpu/creation.py`
- Create: `src/mindtorch_v2/_backends/cpu/meta.py`
- Create: `src/mindtorch_v2/_backends/meta/__init__.py`
- Create: `src/mindtorch_v2/_backends/meta/ops.py`
- Create: `src/mindtorch_v2/_backends/meta/creation.py`
- Delete: `src/mindtorch_v2/_backends/cpu.py`
- Modify: `src/mindtorch_v2/_backends/__init__.py`

**Step 1: Move CPU ops/creation into new modules**

```python
# cpu/ops.py
import numpy as np
from ..common.view import _contiguous_stride  # or duplicate helper
from ..._storage import typed_storage_from_numpy
from ..._tensor import Tensor

# add/mul/matmul/relu/sum_ functions
```

```python
# cpu/creation.py
import numpy as np
from ..._dtype import to_numpy_dtype
from ..._storage import typed_storage_from_numpy, meta_typed_storage_from_shape
from ..._tensor import Tensor

# tensor_create/zeros_create/ones_create/empty_create + meta creation
```

```python
# cpu/meta.py
# _meta_binary/_meta_unary/_meta_sum for CPU pipeline
# _meta_binary_meta/_meta_unary_meta/_meta_sum_meta/_meta_matmul_meta for meta device
```

**Step 2: Register CPU ops in `cpu/__init__.py`**

```python
from ..common import view as view_backend
from ..common import convert as convert_backend
from ..registry import registry
from .ops import add, mul, matmul, relu, sum_
from .creation import tensor_create, zeros_create, ones_create, empty_create
from .meta import (
    _meta_binary,
    _meta_unary,
    _meta_sum,
)

registry.register("add", "cpu", add, meta=_meta_binary)
# ... other cpu registrations
registry.register("reshape", "cpu", view_backend.reshape, meta=_meta_view)
registry.register("to", "cpu", convert_backend.to_device)
```

**Step 3: Register meta device ops in `meta/__init__.py`**

```python
from ..common import view as view_backend
from ..common import convert as convert_backend
from ..registry import registry
from .ops import _meta_binary_meta, _meta_unary_meta, _meta_sum_meta, _meta_matmul_meta
from .creation import tensor_create_meta, zeros_create_meta, ones_create_meta, empty_create_meta

registry.register("add", "meta", _meta_binary_meta)
# ... other meta registrations
registry.register("reshape", "meta", view_backend.reshape, meta=_meta_view_meta)
registry.register("to", "meta", convert_backend.to_device)
```

**Step 4: Update `_backends/__init__.py`**

```python
from . import cpu
from . import meta
from . import npu
```

**Step 5: Run test to verify it still fails**

Run: `pytest tests/mindtorch_v2/test_backend_layout.py::test_backend_shim_imports -v`
Expected: FAIL (still no npu package)

**Step 6: Commit**

```bash
git add src/mindtorch_v2/_backends/cpu src/mindtorch_v2/_backends/meta src/mindtorch_v2/_backends/__init__.py
git commit -m "refactor: split cpu/meta backends"
```

---

### Task 4: Create `npu/` package and shim

**Files:**
- Create: `src/mindtorch_v2/_backends/npu/__init__.py`
- Create: `src/mindtorch_v2/_backends/npu/runtime.py`
- Create: `src/mindtorch_v2/_backends/npu/ops.py`
- Create: `src/mindtorch_v2/_backends/npu/creation.py`
- Create: `src/mindtorch_v2/_backends/npu/aclnn.py`
- Move: `src/mindtorch_v2/_backends/acl_loader.py` -> `src/mindtorch_v2/_backends/npu/acl_loader.py`
- Delete: `src/mindtorch_v2/_backends/ascend.py`
- Delete: `src/mindtorch_v2/_backends/ascend_ctypes.py`
- Create: `src/mindtorch_v2/_backends/ascend.py` (shim)
- Modify: `src/mindtorch_v2/_C.py`
- Modify: `src/mindtorch_v2/npu.py`
- Modify: imports in `src/mindtorch_v2/_tensor.py`, `src/mindtorch_v2/_storage.py`, `src/mindtorch_v2/_backends/common/convert.py`

**Step 1: Move NPU runtime + ops into `npu/`**

```python
# npu/runtime.py
# move _Runtime, ensure_acl usage, model dir probe, _alloc_device, _copy_cpu_to_npu, _copy_npu_to_cpu
```

```python
# npu/ops.py
# move add/mul/relu/sum_ (import runtime + aclnn)
```

```python
# npu/creation.py
# move tensor_create/zeros_create/ones_create/empty_create
```

```python
# npu/aclnn.py
# move ascend_ctypes.py content and update ensure_acl import
```

**Step 2: Register NPU ops in `npu/__init__.py`**

```python
from ..common import view as view_backend
from ..common import convert as convert_backend
from ..registry import registry
from .ops import add, mul, relu, sum_
from .creation import tensor_create, zeros_create, ones_create, empty_create
from .runtime import is_available, _probe_model_dirs, _model_dir

registry.register("add", "npu", add)
# ... other npu registrations
registry.register("reshape", "npu", view_backend.reshape)
registry.register("to", "npu", convert_backend.to_device)
```

**Step 3: Add shim `ascend.py`**

```python
from .npu.runtime import *
from .npu.ops import *
from .npu.creation import *
```

**Step 4: Update imports**

- `_C.py` to import from `_backends.npu` and `_backends.npu.aclnn`.
- `npu.py` to import from `_backends.npu`.
- `_tensor.py`, `_storage.py`, `common/convert.py` to import from `_backends.npu.runtime` for alloc/copy helpers.

**Step 5: Run test to verify it passes**

Run: `pytest tests/mindtorch_v2/test_backend_layout.py::test_backend_shim_imports -v`
Expected: PASS

**Step 6: Run key regression tests**

- `pytest tests/mindtorch_v2/test_creation.py -v`
- `pytest tests/mindtorch_v2/test_view_dispatch.py -v`
- `PYTHONPATH=/usr/local/Ascend/ascend-toolkit/latest/python/site-packages:src pytest tests/mindtorch_v2/test_ops_npu.py -v`

**Step 7: Commit**

```bash
git add src/mindtorch_v2/_backends src/mindtorch_v2/_C.py src/mindtorch_v2/npu.py src/mindtorch_v2/_tensor.py src/mindtorch_v2/_storage.py tests/mindtorch_v2/test_backend_layout.py
git commit -m "refactor: organize backends by device"
```
