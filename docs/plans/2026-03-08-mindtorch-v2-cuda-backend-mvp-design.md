# MindTorch v2 CUDA Backend MVP Design

**Status:** Approved

**Goal:** Turn `mindtorch_v2` CUDA from a reserved device label into a real backend with working device APIs, storage, transfer, and tensor creation—without depending on any other framework.

**Scope:** This design covers the first usable CUDA backend milestone only. It does not attempt broad operator parity, full autograd parity, AMP, profiler support, or distributed support.

---

## Problem

`mindtorch_v2` already reserves a CUDA path in dispatch and device handling, but CUDA is not a real backend today:

- Dispatch registration maps `cuda` to placeholder keys in `src/mindtorch_v2/_dispatch/registration.py`.
- Dispatch key construction recognizes CUDA tensors in `src/mindtorch_v2/_dispatch/keys.py`.
- Tensor APIs still treat `.cuda()` as unsupported in current tests.

This means the mechanism is partially prepared, but there is no CUDA runtime layer, storage layer, or creation/transfer implementation behind it.

---

## Constraints

- Pure Python implementation.
- No dependency on external deep learning frameworks.
- Respect `mindtorch_v2` schema-first registration rules.
- Keep the first phase small and mechanism-focused.
- Use CUDA runtime primitives directly via Python FFI.

---

## Non-Goals

This MVP does not include:

- Broad math operator coverage.
- Full Torch CUDA semantic parity.
- Full autograd on CUDA.
- AMP/autocast/GradScaler support.
- CUDA profiler support.
- NCCL/distributed support.

Those will become follow-on phases once backend fundamentals are stable.

---

## Recommended Approach

Use a layered CUDA backend built directly on `ctypes` bindings to `libcudart.so`.

The first phase should only establish:

1. CUDA runtime discovery and device management.
2. GPU memory allocation/free and memory copy.
3. A dedicated `CudaStorage` implementation.
4. Tensor movement between CPU and CUDA.
5. CUDA creation ops: `empty`, `zeros`, `ones`, `full`, `tensor`, `to`.

This is the smallest design that turns CUDA into a real device in the system.

---

## Architecture

### 1. Runtime Layer

Add `src/mindtorch_v2/_backends/cuda/runtime.py`.

Responsibilities:

- Load `libcudart.so` lazily.
- Expose Python wrappers for:
  - `cudaGetDeviceCount`
  - `cudaGetDevice`
  - `cudaSetDevice`
  - `cudaMalloc`
  - `cudaFree`
  - `cudaMemcpy`
  - `cudaMemcpyAsync`
  - `cudaMemset`
  - `cudaDeviceSynchronize`
  - `cudaStreamCreate`
  - `cudaStreamDestroy`
  - `cudaStreamSynchronize`
  - `cudaEventCreate`
  - `cudaEventDestroy`
  - `cudaEventRecord`
  - `cudaEventSynchronize`
- Raise clear Python exceptions from CUDA error codes.

This layer is intentionally small and should not include operator logic.

### 2. Storage Layer

Add `src/mindtorch_v2/_backends/cuda/storage.py`.

Responsibilities:

- Represent device-backed raw storage.
- Own a CUDA pointer and its lifetime.
- Track `nbytes`, `dtype`, and `device`.
- Support conversion helpers for host/device copies.
- Provide enough storage API compatibility for `Tensor` to work.

The initial storage type can remain simple: contiguous allocation only, no custom allocator, no pooling, no IPC.

### 3. Storage Factory Integration

Extend `src/mindtorch_v2/_storage.py`.

Responsibilities:

- Add CUDA storage factory helpers.
- Support CPU numpy -> CUDA upload.
- Support CUDA -> CPU download.
- Route typed storage creation by device type.

This is the bridge between existing CPU/meta flows and the new CUDA flow.

### 4. Public CUDA API

Add `src/mindtorch_v2/cuda.py`.

Responsibilities:

- Mirror the shape of `src/mindtorch_v2/npu.py` where practical.
- Expose:
  - `is_available`
  - `device_count`
  - `current_device`
  - `set_device`
  - `synchronize`
  - `Stream`
  - `Event`
  - `device` context manager

This gives users a stable entry point for CUDA backend discovery and control.

### 5. Tensor Transfer and Creation

Update:

- `src/mindtorch_v2/_tensor.py`
- `src/mindtorch_v2/_creation.py`
- `src/mindtorch_v2/_backends/cuda/creation.py`

Responsibilities:

- Make `Tensor.cuda()` call into `to("cuda")`.
- Make `Tensor.to("cuda")` and `Tensor.to("cpu")` perform actual device transfer.
- Support direct creation on CUDA for `tensor`, `empty`, `zeros`, `ones`, `full`.

For `ones` and `full`, the first version may use a temporary host buffer plus upload if that is simpler than adding a fill kernel immediately.

---

## Dispatch Strategy

Do not redesign dispatch in this phase.

Make CUDA a first-class dispatch backend in this PR.

Required dispatch changes:

- add `DispatchKey.CUDA`
- add `DispatchKey.AutogradCUDA`
- update dispatch priority and keyset construction
- update registration helpers so `cuda` no longer maps to `PrivateUse1`
- keep `PrivateUse1` reserved for actual private-use backends

This aligns the implementation with the requirement that CUDA must not be represented as `PrivateUse1`.

---

## Testing Strategy

Follow the repo rule: schema first, then contract tests, then backend wiring.

### Required mechanism tests

- `PYTHONPATH=src pytest -q tests/mindtorch_v2/contract/test_schema_registration_order.py`
- `PYTHONPATH=src pytest -q tests/mindtorch_v2/contract/test_schema_coverage.py`

### New CUDA MVP tests

Add or update tests for:

- `torch.cuda.is_available()` style availability surface.
- `mt.tensor(..., device="cuda")` creation.
- `x.cuda()` success.
- `x.to("cuda")` and `x.to("cpu")` round-trip correctness.
- `zeros/ones/full/empty(..., device="cuda")` creation.
- `current_device` / `set_device` behavior.

Tests should gracefully skip when CUDA runtime is unavailable.

---

## Risks

### Runtime loading risk

CUDA library names can differ across systems. The runtime layer should attempt a small set of common library names and fail gracefully.

### Lifetime management risk

Leaking device memory is easy when Python owns raw pointers. `CudaStorage` should centralize ownership and cleanup.

### Shape/stride risk

This MVP should avoid pretending to support advanced non-contiguous CUDA storage semantics before they are actually implemented.

### Scope creep risk

Do not add math kernels in this phase unless they are strictly required to support creation/transfer semantics.

---

## Success Criteria

The MVP is complete when all of the following are true:

- `mindtorch_v2` exposes a working `cuda` module.
- CUDA availability can be queried without crashing on non-CUDA systems.
- Tensors can be created on CUDA.
- CPU <-> CUDA transfer works for supported dtypes.
- `.cuda()` no longer fails as an unsupported reserved path.
- CUDA backend changes respect schema-first and pass the required contract tests.

---

## Follow-On Phases

After this MVP, the next recommended phases are:

1. Pointwise math ops.
2. Reductions.
3. `matmul` and BLAS-backed operations.
4. Convolution/pooling via cuDNN.
5. Autograd correctness.
6. AMP, profiler, and distributed support.

