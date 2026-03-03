# Dispatch Schema/Alias/Functionalize Design

## Goal
Align dispatcher schema binding, alias resolution, and functionalization semantics with PyTorch (including exact error messages), and make functionalization apply to all future in-place ops via schema mutation markers.

## Scope
- Schema registration and binding (positional/keyword/defaults/optional).
- Alias resolution (user name vs canonical name).
- Functionalization key (automatic for any `!` mutation in schema).

## Non-Goals
- Full dtype promotion rules (separate track).
- New operator coverage beyond the tests needed to validate dispatch semantics.

## Design
### 1) Schema Binding
- Introduce `OpSchema` with a Torch-style schema string.
- Add `bind(args, kwargs)` to validate inputs and produce a normalized arg map.
- All dispatch paths call schema binding before kernel resolution.
- Errors must match Torch exactly (type + message), using PyTorch as oracle tests.

### 2) Alias Resolution
- Add `registry.register_alias(alias, target)`.
- Dispatch resolves alias → canonical op for kernel lookup.
- Error messages retain the **original user-facing op name**.

### 3) Functionalization Key
- Add `DispatchKey.Functionalize` with priority: Pipeline → Functionalize → Autograd → Meta → NPU → CPU.
- If schema marks mutations (`Tensor(a!)`, `Tensor?` etc.), functionalize must be applied.
- For in-place ops with a clear functional name (`add_` → `add`), derive automatically.
- Otherwise require explicit functionalization rule at registration time; registration fails if missing.
- Functionalize executes out-of-place kernel, then writes back to mutated inputs, preserving version counters and user-visible in-place semantics.

## Error Handling
- Schema binding errors must match Torch exactly.
- Alias lookup errors must match Torch exactly.
- Missing functionalization rule errors must match Torch exactly.

## Testing
- New contract tests that compare mindtorch errors with Torch for:
  - unexpected/missing kwargs
  - bad positional args
  - alias resolution failures
  - missing functionalization rule
- Functionalization behavior tests validating:
  - in-place op updates input
  - version counter updated
  - autograd behavior matches Torch

## Open Questions
None. Proceed with implementation plan.
