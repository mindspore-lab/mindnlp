# MindTorch V2 Dispatch Unification Design

## Goals
- Route all Tensor construction and methods (except `numpy()`/`item()`) through dispatcher.
- Centralize device selection, autograd, and pipeline behavior.
- Make view ops participate in autograd and preserve view semantics.

## Non-Goals
- No JIT or static graph compilation.
- No additional dtype conversions beyond existing behavior.

## Unified Entry Points
- **Creation ops**: `tensor/zeros/ones/empty` → `dispatch("tensor"|"zeros"|"ones"|"empty")`.
- **View ops**: `reshape/view/transpose` → `dispatch("reshape"|"view"|"transpose")`.
- **Compute ops**: `add/mul/matmul/relu/sum/to` → `_functional` → dispatch.
- **Excluded**: `numpy()`, `item()` remain direct data accessors.

## Autograd Rules
- **View ops**: create a grad_fn that maps gradients back through the view (shape/stride/offset only).
- **Creation ops**: no grad_fn unless `requires_grad` is set explicitly.
- **to()**: identity-like grad_fn; if device changes, propagate grad back to original device (or raise if unsupported).

## Dispatcher + Pipeline Integration
- All ops go through dispatcher, enabling meta/plan/impl stages.
- Pipeline mode records ops with meta outputs and flushes on sync boundaries.

## Notes
- View ops share storage; no data copies.
- Unified dispatch ensures consistent device rules and autograd tracking.
