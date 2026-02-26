# Dispatch Registry Isolation Contract

## Scope

This document defines the required test-time isolation behavior for the mutable dispatch registry in `mindtorch_v2`.

It is intended for contributors and coding agents that register temporary schemas/kernels/aliases during tests.

## Problem

`mindtorch_v2._dispatch.registry` is global mutable state.

Without isolation, one test can change registered kernels/schemas and silently affect later tests.
This causes order-dependent failures and makes parallel agent development unreliable.

## Internal API

The dispatch registry now provides two internal helpers:

- `registry.snapshot()`
  - Returns a deep-copied state object containing:
    - operator table (`_ops`)
    - alias table (`_aliases`)
    - global fallthrough set (`_global_fallthrough`)
- `registry.restore(state)`
  - Restores the exact state from `snapshot()`.

These methods are for test isolation and developer tooling.
They are not a stable public runtime API.

## Required Test Policy

Any test that mutates dispatch registry state must be isolated.

The default repository policy is an autouse fixture in `tests/conftest.py`:

```python
@pytest.fixture(autouse=True)
def _dispatch_registry_isolation():
    state = registry.snapshot()
    try:
        yield
    finally:
        registry.restore(state)
```

This means all tests run with per-test registry rollback by default.

## Agent Guidance

When writing or modifying tests:

1. You may freely call `register_schema/register_kernel/register_alias/register_fallthrough`.
2. Do not add manual cleanup logic for registry mutations unless there is a strong reason.
3. If you intentionally bypass `tests/conftest.py` (custom harness), you must call `snapshot/restore` yourself.

Minimal manual pattern:

```python
state = registry.snapshot()
try:
    registry.register_kernel(...)
    # run assertions
finally:
    registry.restore(state)
```

## Expected Outcome

- No cross-test registry leakage.
- Deterministic behavior independent of test order.
- Safer multi-agent parallel development for operator registration work.
