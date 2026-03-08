# MindTorch v2 CUDA Backend MVP Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Build the first usable CUDA backend for `mindtorch_v2`, covering runtime discovery, storage, transfer, and CUDA tensor creation without relying on any external deep learning framework.

**Architecture:** Add a thin CUDA runtime layer over `libcudart.so`, implement a dedicated `CudaStorage`, wire transfer and creation paths into existing tensor/storage APIs, and upgrade CUDA to explicit dispatch keys and register the backend against those keys. Keep the phase mechanism-focused and avoid broad operator work.

**Tech Stack:** Python, `ctypes`, CUDA Runtime API, existing `mindtorch_v2` dispatch/storage/tensor stack, `pytest`.

---

### Task 1: Confirm current CUDA surface and test baselines

**Files:**
- Inspect: `src/mindtorch_v2/_device.py`
- Inspect: `src/mindtorch_v2/_tensor.py`
- Inspect: `src/mindtorch_v2/_storage.py`
- Inspect: `src/mindtorch_v2/_creation.py`
- Inspect: `src/mindtorch_v2/_dispatch/registration.py`
- Inspect: `src/mindtorch_v2/_dispatch/keys.py`
- Inspect: `tests/mindtorch_v2/test_device_transfer.py`

**Steps:**
1. Read the files above and note all current CUDA placeholders and unsupported paths.
2. Identify which creation APIs already have schema coverage and which only need backend wiring.
3. Confirm which tests currently assume `.cuda()` is unsupported.
4. Write down the exact functions that need to change before touching code.

**Test command:**
Run: `PYTHONPATH=src pytest -q tests/mindtorch_v2/test_device_transfer.py`
Expected: current CUDA tests fail because support is not implemented, or assert unsupported behavior.

---

### Task 2: Add failing CUDA MVP tests

**Files:**
- Modify: `tests/mindtorch_v2/test_device_transfer.py`
- Create: `tests/mindtorch_v2/test_cuda_backend.py`

**Steps:**
1. Replace unsupported `.cuda()` assertions with behavior expected from the MVP.
2. Add tests for `tensor(..., device="cuda")`, `x.to("cuda")`, `x.to("cpu")`, and CUDA creation ops.
3. Add tests for `current_device`, `set_device`, and `synchronize`.
4. Guard runtime-dependent tests with skip conditions when CUDA is unavailable.

**Test command:**
Run: `PYTHONPATH=src pytest -q tests/mindtorch_v2/test_device_transfer.py tests/mindtorch_v2/test_cuda_backend.py`
Expected: FAIL with missing CUDA backend/runtime implementation.

---

### Task 3: Add CUDA runtime bindings

**Files:**
- Create: `src/mindtorch_v2/_backends/cuda/runtime.py`
- Create: `src/mindtorch_v2/_backends/cuda/__init__.py`

**Steps:**
1. Implement lazy runtime library loading for common CUDA runtime library names.
2. Define Python wrappers for device count, current device, set device, allocate/free, memcpy, memset, synchronize, stream, and event APIs.
3. Implement a shared CUDA error checker that raises descriptive Python exceptions.
4. Add a small availability probe that cleanly returns `False` when CUDA runtime is missing.

**Test command:**
Run: `PYTHONPATH=src pytest -q tests/mindtorch_v2/test_cuda_backend.py -k "availability or device"`
Expected: device/runtime tests move from import errors to behavior failures where storage/transfer is still missing.

---

### Task 4: Implement `CudaStorage`

**Files:**
- Create: `src/mindtorch_v2/_backends/cuda/storage.py`
- Modify: `src/mindtorch_v2/_storage.py`

**Steps:**
1. Add a storage type that owns a CUDA allocation and frees it on cleanup.
2. Store pointer, byte size, dtype, and device metadata.
3. Add host-device and device-host copy helpers.
4. Add typed storage factory helpers for CUDA allocation and upload/download.

**Test command:**
Run: `PYTHONPATH=src pytest -q tests/mindtorch_v2/test_cuda_backend.py -k "tensor_creation or roundtrip"`
Expected: transfer tests still fail until tensor/creation wiring is added, but storage paths import and allocate correctly.

---

### Task 5: Expose public `cuda` module

**Files:**
- Create: `src/mindtorch_v2/cuda.py`
- Modify: `src/mindtorch_v2/__init__.py`

**Steps:**
1. Implement `is_available`, `device_count`, `current_device`, `set_device`, and `synchronize`.
2. Add `Stream`, `Event`, and `device` context manager wrappers using the runtime layer.
3. Export the module from the package root.
4. Match the public API shape used by `npu.py` where possible without overbuilding.

**Test command:**
Run: `PYTHONPATH=src pytest -q tests/mindtorch_v2/test_cuda_backend.py -k "availability or current_device or stream or event"`
Expected: public CUDA API tests pass or are skipped when CUDA is unavailable.

---

### Task 6: Wire tensor transfer paths

**Files:**
- Modify: `src/mindtorch_v2/_tensor.py`
- Modify: `src/mindtorch_v2/_creation.py`
- Modify: `src/mindtorch_v2/_device.py`

**Steps:**
1. Implement `.cuda()` as a transfer convenience API.
2. Implement `.to("cuda")` and `.to("cpu")` using the new storage helpers.
3. Preserve shape, stride, dtype, and device metadata across transfers.
4. Keep `_numpy_view()` disallowed for CUDA-backed tensors unless moved to CPU first.

**Test command:**
Run: `PYTHONPATH=src pytest -q tests/mindtorch_v2/test_device_transfer.py tests/mindtorch_v2/test_cuda_backend.py -k "cuda or to"`
Expected: transfer tests pass for supported dtypes.

---

### Task 7: Add CUDA creation kernels

**Files:**
- Create: `src/mindtorch_v2/_backends/cuda/creation.py`
- Modify: `src/mindtorch_v2/_dispatch/schemas.py` (only if missing schema coverage is found)
- Modify: `src/mindtorch_v2/_dispatch/registry.py` or backend registration entrypoints as needed
- Modify: `src/mindtorch_v2/_creation.py`

**Steps:**
1. Confirm schemas already exist for `tensor`, `empty`, `zeros`, `ones`, `full`, and `to`.
2. If any schema is missing, add it first.
3. Register CUDA creation kernels through the existing registration helpers.
4. Implement `empty`, `zeros`, `ones`, `full`, and direct `tensor(..., device="cuda")` support.
5. Use the simplest correct fill path first, even if `ones/full` initially stage through CPU.

**Test command:**
Run: `PYTHONPATH=src pytest -q tests/mindtorch_v2/test_cuda_backend.py -k "creation or tensor_creation"`
Expected: CUDA creation tests pass.

---

### Task 8: Validate schema and contract invariants

**Files:**
- Inspect: `src/mindtorch_v2/_dispatch/schemas.py`
- Inspect: `tests/mindtorch_v2/contract/test_schema_registration_order.py`
- Inspect: `tests/mindtorch_v2/contract/test_schema_coverage.py`

**Steps:**
1. Confirm no CUDA kernel registration happens before schema registration.
2. Run required contract tests.
3. Fix only CUDA MVP-related schema or registration issues.

**Test commands:**
Run: `PYTHONPATH=src pytest -q tests/mindtorch_v2/contract/test_schema_registration_order.py`
Expected: PASS

Run: `PYTHONPATH=src pytest -q tests/mindtorch_v2/contract/test_schema_coverage.py`
Expected: PASS

---

### Task 9: Run focused regression tests

**Files:**
- Inspect: `tests/mindtorch_v2/test_creation.py`
- Inspect: `tests/mindtorch_v2/test_dtype_device.py`

**Steps:**
1. Run the CUDA-focused tests created in this plan.
2. Run adjacent creation/device tests to ensure CPU and existing behavior did not regress.
3. Fix only issues directly caused by CUDA MVP changes.

**Test commands:**
Run: `PYTHONPATH=src pytest -q tests/mindtorch_v2/test_device_transfer.py tests/mindtorch_v2/test_cuda_backend.py tests/mindtorch_v2/test_creation.py tests/mindtorch_v2/test_dtype_device.py`
Expected: PASS, with CUDA tests skipped on systems without CUDA runtime.

---

### Task 10: Document limitations and next phase

**Files:**
- Modify: `docs/plans/2026-03-08-mindtorch-v2-cuda-backend-mvp-design.md`
- Optionally modify: package docs that describe device support, if such docs exist in-scope

**Steps:**
1. Record what the CUDA MVP supports.
2. Record what is intentionally out of scope.
3. List the next recommended phase: pointwise ops, reductions, BLAS-backed ops.

**Validation:**
Review the document and verify that supported vs unsupported CUDA features are explicit.

