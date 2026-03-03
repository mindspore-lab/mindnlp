# Dispatch Keyset Alignment Design (Torch 2.4)

**Goal**: Align mindtorch_v2 dispatch keyset semantics with torch 2.4 (bitmask + priority + TLS include/exclude), while preserving the custom Pipeline key ordering and existing fallthrough behavior.

## Scope
- Convert DispatchKeySet from Python `set` to an integer bitmask.
- Implement TLS include/exclude masks that apply to all dispatch entry points.
- Align dispatch key priority ordering with torch 2.4, inserting `Pipeline` after `BackendSelect`.
- Add Composite* and PrivateUse* keys to the enum and priority list.
- Keep the current global fallthrough behavior, extend it to new keys.
- Maintain current error behavior for missing kernels, but make it deterministic with the new key ordering.

## Architecture
### Key Representation
- `DispatchKey` becomes an `IntEnum` with fixed bit positions.
- `DispatchKeySet` stores an integer `mask` and exposes helpers:
  - `has(key)`, `add(key)`, `remove(key)`, `without(keys)`, `iter_keys()`
  - `iter_keys()` yields keys in priority order by filtering `mask` against the priority list.

### TLS Include/Exclude
- Thread-local state stores two masks: `include_mask`, `exclude_mask`.
- Effective keyset = `(base_mask | include_mask) & ~exclude_mask`.
- The TLS masks apply to all dispatch entry points:
  - `dispatch`, `dispatch_with_keyset`, and `redispatch`.
- Context managers push/pop include/exclude masks (internal only for now).

### Priority Ordering
- Use torch 2.4 dispatch key ordering as the baseline priority list.
- Insert `Pipeline` immediately after `BackendSelect`.
- New keys (CompositeImplicitAutograd, CompositeExplicitAutograd, PrivateUse1/2/3) are in the list but default to fallthrough.

### Fallthrough
- Global fallthrough keys remain supported.
- Newly added keys are added to global fallthrough by default to avoid behavior changes.

## Error Handling
- Missing kernel errors should be computed after TLS masking and priority filtering to reflect the effective dispatch behavior.
- Error messages remain consistent with existing behavior, but should include deterministic key ordering.

## Testing
Targeted contract tests in `tests/mindtorch_v2/contract`:
1. Priority ordering (Pipeline placed after BackendSelect).
2. TLS include/exclude applies to `dispatch` and `redispatch`.
3. Composite* and PrivateUse* keys do not affect kernel selection when fallthrough is set.
4. Effective keyset with Meta/NPU/CPU combined with TLS masking is deterministic.
5. Fallthrough + masking still selects the expected kernel.

## Non-Goals
- No public-facing include/exclude API yet; internal-only until semantics are stable.
- No changes to op coverage or backend kernels.
- No changes to schema aliasing (handled in separate P0 item).
