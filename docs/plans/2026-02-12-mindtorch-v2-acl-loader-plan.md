# ACL Lazy Loader (No LD_LIBRARY_PATH) Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Make ACL import lazy and self-contained by preloading CANN libs once and caching the `acl` module, avoiding per-op overhead and manual `LD_LIBRARY_PATH`.

**Architecture:** Add `acl_loader.ensure_acl()` with a fast-path cache and one-time preload via `ctypes.CDLL(..., RTLD_GLOBAL)`. Update `ascend.py` and `ascend_ctypes.py` to use `ensure_acl()` instead of importing `acl` at module import time. Tests stub the loader internals to validate caching/fast-path behavior without real ACL.

**Tech Stack:** Python, pytest

---

### Task 1: Add failing tests for lazy loader cache behavior

**Files:**
- Create: `tests/mindtorch_v2/test_acl_loader.py`
- Create: `src/mindtorch_v2/_backends/acl_loader.py`
- Modify: `src/mindtorch_v2/_backends/ascend.py`
- Modify: `src/mindtorch_v2/_backends/ascend_ctypes.py`
- Test: `tests/mindtorch_v2/test_acl_loader.py`

**Step 1: Write the failing tests**

```python
import types

import mindtorch_v2._backends.acl_loader as acl_loader


def test_ensure_acl_caches_module(monkeypatch):
    calls = {"import": 0}

    def fake_import_acl():
        calls["import"] += 1
        return types.SimpleNamespace(name="acl")

    monkeypatch.setattr(acl_loader, "_import_acl", fake_import_acl)
    monkeypatch.setattr(acl_loader, "_ACL_READY", False)
    monkeypatch.setattr(acl_loader, "_ACL_MODULE", None)

    first = acl_loader.ensure_acl()
    second = acl_loader.ensure_acl()

    assert first is second
    assert calls["import"] == 1


def test_ensure_acl_retries_after_failure(monkeypatch):
    calls = {"import": 0}

    def fake_import_acl():
        calls["import"] += 1
        if calls["import"] == 1:
            raise RuntimeError("boom")
        return types.SimpleNamespace(name="acl")

    monkeypatch.setattr(acl_loader, "_import_acl", fake_import_acl)
    monkeypatch.setattr(acl_loader, "_ACL_READY", False)
    monkeypatch.setattr(acl_loader, "_ACL_MODULE", None)

    try:
        acl_loader.ensure_acl()
    except RuntimeError:
        pass

    acl = acl_loader.ensure_acl()
    assert acl.name == "acl"
    assert calls["import"] == 2
```

**Step 2: Run tests to verify they fail**

Run: `pytest tests/mindtorch_v2/test_acl_loader.py -v`

Expected: FAIL because `acl_loader` does not exist yet.

**Step 3: Write minimal implementation**

Create `src/mindtorch_v2/_backends/acl_loader.py`:

```python
import ctypes
import os
import threading

_ACL_READY = False
_ACL_MODULE = None
_ACL_LOCK = threading.Lock()

_CANDIDATE_LIB_DIRS = (
    "/usr/local/Ascend/ascend-toolkit/latest/lib64",
    "/usr/local/Ascend/ascend-toolkit/8.3.RC2/aarch64-linux/lib64",
    "/usr/local/Ascend/latest/lib64",
    "/usr/local/Ascend/driver/lib64/driver",
)

_PRELOAD_LIBS = (
    "libascendcl.so",
    "libascendcl_impl.so",
)


def _existing_dirs():
    return [d for d in _CANDIDATE_LIB_DIRS if os.path.isdir(d)]


def _prepend_ld_library_path(dirs):
    if not dirs:
        return
    existing = os.environ.get("LD_LIBRARY_PATH", "")
    prefix = ":".join(dirs)
    os.environ["LD_LIBRARY_PATH"] = f"{prefix}:{existing}" if existing else prefix


def _preload_libs(dirs):
    for d in dirs:
        for lib in _PRELOAD_LIBS:
            candidate = os.path.join(d, lib)
            if os.path.exists(candidate):
                ctypes.CDLL(candidate, mode=ctypes.RTLD_GLOBAL)


def _import_acl():
    import acl

    return acl


def ensure_acl():
    global _ACL_READY, _ACL_MODULE
    if _ACL_READY:
        return _ACL_MODULE
    with _ACL_LOCK:
        if _ACL_READY:
            return _ACL_MODULE
        dirs = _existing_dirs()
        _prepend_ld_library_path(dirs)
        _preload_libs(dirs)
        _ACL_MODULE = _import_acl()
        _ACL_READY = True
        return _ACL_MODULE
```

Update `ascend.py`:
- Remove `import acl` at top.
- Add `from .acl_loader import ensure_acl`.
- In `_Runtime.init()` start with `acl = ensure_acl()` and assign to module-level `acl` if needed.
- In helper functions that use `acl`, add `acl = ensure_acl()` at start if `acl` is not already set.

Update `ascend_ctypes.py`:
- Remove `import acl` at top.
- Add `from .acl_loader import ensure_acl`.
- At the start of each op and any function calling `acl.*`, do `acl = ensure_acl()` and use that variable.

**Step 4: Run tests to verify they pass**

Run: `pytest tests/mindtorch_v2/test_acl_loader.py -v`

Expected: PASS.

**Step 5: Commit**

```bash
git add src/mindtorch_v2/_backends/acl_loader.py \
  src/mindtorch_v2/_backends/ascend.py \
  src/mindtorch_v2/_backends/ascend_ctypes.py \
  tests/mindtorch_v2/test_acl_loader.py
git commit -m "feat: lazy-load acl with ctypes preload"
```
