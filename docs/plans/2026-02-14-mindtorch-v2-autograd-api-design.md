# MindTorch v2 Autograd API + Graph Correctness Design

**Goal:** Fully align MindTorch v2 autograd behavior with PyTorch for training correctness (graph creation, in‑place semantics, versioning, backward flags, and functional `autograd.grad`).

**Scope:** Graph correctness (version counters, views, saved‑tensor checks), autograd API (`backward`, `requires_grad_`, `detach`, `detach_`, `retain_grad`, hooks), and `torch.autograd.grad`. Applies to CPU and NPU.

---

## Graph Correctness Model

### Version Counters and Views
- Each base tensor owns a shared `VersionCounter` object.
- Views created by `view/reshape/transpose` share the base’s `VersionCounter`.
- Views store `_base` and `_view_meta` (op, shape, stride, offset) for debugging and correctness checks.

### In‑place Semantics (Torch‑aligned)
- If grad is enabled and the tensor requires grad:
  - In‑place on a **leaf** is forbidden.
  - In‑place on a **view of a leaf** is forbidden.
  - Otherwise allowed, but increments the shared `VersionCounter`.
- Under `no_grad`, graph creation and in‑place checks are bypassed.

### Saved Tensor Validation
- `Node.save_for_backward(*tensors)` stores `SavedTensor` objects capturing the version at save time.
- `Node.saved_tensors()` validates current version vs saved; mismatch raises:
  - "one of the variables needed for gradient computation has been modified by an inplace operation"

---

## Autograd API Surface

### Tensor.backward
- Signature: `backward(gradient=None, retain_graph=False, create_graph=False)`.
- If `gradient` is None:
  - Scalar outputs use implicit ones‑like gradient.
  - Non‑scalar outputs raise (torch behavior).
- `retain_graph=True` keeps graph for repeated backward.
- `create_graph=True` builds higher‑order gradient graph.

### Leaf vs Non‑leaf Grad
- Leaf `.grad` accumulates additively.
- Non‑leaf `.grad` is **only** populated after `retain_grad()`.

### Detach / Requires Grad
- `requires_grad_()` toggles requires‑grad on leaf tensors.
- `detach()` returns a view sharing storage with no grad history.
- `detach_()` detaches in place.

### Hooks
- `Tensor.register_hook(fn)` runs `fn(grad)` when the tensor’s gradient is computed.
- Hook results can replace the gradient (torch behavior).

---

## torch.autograd.grad

Implement: `grad(outputs, inputs, grad_outputs=None, retain_graph=None, create_graph=False, allow_unused=False)`
- Returns grads without touching `.grad` unless explicitly requested.
- `allow_unused=False` raises if any input gradient is None.
- `retain_graph` defaults to `create_graph` when not provided (torch‑like).

---

## Testing Plan

**Core correctness**
- Version counter shared across views.
- In‑place on leaf / view‑of‑leaf raises.
- Saved‑tensor version mismatch raises with torch message.

**API behavior**
- Scalar vs non‑scalar backward behavior.
- `retain_graph` and `create_graph` semantics.
- `retain_grad` populates `.grad` on intermediates.
- `detach`/`detach_` behavior.
- `register_hook` runs and can modify gradient.

**autograd.grad**
- Correct gradients for multiple inputs.
- `allow_unused` behaviors.
- Higher‑order grad when `create_graph=True`.

**NPU coverage**
- In‑place version increment and saved‑tensor validation under NPU (guarded by `torch.npu.is_available()`).

---

## Non‑Goals (v1)
- Full custom autograd Functions.
- Graph‑level optimizations or JIT.
- Autograd for every op beyond core arithmetic.

---

## Implementation Notes
- Keep checks centralized in Tensor and Node to avoid backend divergence.
- All correctness rules must match torch even if backend operations are implemented in Python.
- Use minimal additional state to avoid runtime overhead.
