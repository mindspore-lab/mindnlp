# MindTorch v2 Base Framework Parity Design

## Goal
Freeze Dispatch, Storage, Autograd, and NPU API core+diagnostics behavior to match torch (including exact error messages) before expanding operator coverage.

## Scope
- Dispatcher contract (keyset order, redispatch, pipeline/meta requirements).
- Storage semantics (typed/untyped, resize rules, copy_ errors).
- Autograd semantics (view/inplace versioning, saved tensor hooks, no_grad/create_graph).
- NPU API parity with torch.cuda (core + diagnostics): device/stream/event, sync, memory stats, empty_cache, memory_summary/memory_snapshot.

## Non‑Goals
- Expanding operator coverage beyond what contract tests require.
- Performance tuning or allocator redesign.

## Torch Parity Strategy
Use PyTorch as an oracle to verify exact exception types and **exact error messages**. Tests call equivalent torch and mindtorch code paths and compare results or failure text 1:1. For NPU, use torch.cuda as the reference; if an NPU‑specific behavior has no CUDA analogue, document and test it separately.

## Design
### 1) Contract Test Harness
Add `tests/mindtorch_v2/contract/` with helpers:
- `assert_torch_error(fn_mt, fn_torch)` to assert type + exact message match.
- `assert_same_value(fn_mt, fn_torch)` for success paths that must align.
These helpers isolate comparison logic and allow minimal platform normalization only when torch behavior is inherently platform‑dependent.

### 2) Dispatcher Contracts
- Keyset ordering: Pipeline → Autograd → Meta → NPU → CPU.
- Meta required for pipeline; error message must match torch.
- redispatch should bypass Autograd key and honor device inference.

### 3) Storage Contracts
- Typed/untyped invariants and dtype sizes.
- Resize rules for pinned/shared/file‑backed storage with exact torch error text.
- copy_ cross‑device behavior and error messaging.

### 4) Autograd Contracts
- Inplace/view version counter enforcement with torch‑exact error messages.
- saved_tensors_hooks semantics and error behavior.
- no_grad/create_graph behavior alignment.

### 5) NPU API Contracts
- Device/stream/event APIs (default/current stream, context managers, synchronize).
- Memory stats + empty_cache behavior consistent with torch.cuda.
- Diagnostics: memory_summary/memory_snapshot parity (shape, key presence, error text).

## Testing
- New contract tests run in CPU‑only mode where possible.
- NPU checks guarded by `torch.npu.is_available()`; otherwise compare against torch.cuda.
- Base test suite: `pytest -q tests/mindtorch_v2`.

## Open Questions
None. Proceed with implementation plan.
