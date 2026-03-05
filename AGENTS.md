# MindTorch v2 Agent Rules

This file defines mandatory development order and verification gates for all contributors and coding agents working on `mindtorch_v2`.

## Scope

- Applies to all changes under `src/mindtorch_v2/` and `tests/mindtorch_v2/`.
- Priority: mechanism alignment with Torch behavior over adding new operator count.

## Non-Negotiable Order

For any new operator or API path in `mindtorch_v2`, follow this order:

1. Register schema first in `src/mindtorch_v2/_dispatch/schemas.py`.
2. Add or update contract tests.
3. Register backend kernels (CPU/NPU/Meta/Autograd/Functionalize).
4. Add or update functional/tensor API exports.

Do not register a kernel before schema registration.

## Hard Invariant

`OpRegistry.register_kernel` enforces schema-first registration.

If schema is missing, registration must fail with:
- `schema must be registered before kernel registration for op ...`

Treat this as a design guardrail, not a temporary check.

## Required Tests Before PR

Every PR touching `mindtorch_v2` must pass:

- `PYTHONPATH=src pytest -q tests/mindtorch_v2/contract/test_schema_registration_order.py`
- `PYTHONPATH=src pytest -q tests/mindtorch_v2/contract/test_schema_coverage.py`

Recommended full gate for mechanism changes:

- `PYTHONPATH=src pytest -q tests/mindtorch_v2/contract`

## PR Scope Rule

- Keep PRs mechanism-focused and small.
- Do not mix unrelated features in one PR.
- If you add a new operator family, include only required schema/tests/registration/API for that family.

## Torch Alignment Rule

- Match Torch dispatch semantics first (schema binding, error class, dispatch path), then optimize implementation.
- Error message wording can differ slightly unless a contract test requires exact match.

## Worktree Rule

- Always develop in a dedicated git worktree rebased on latest `ms/master`.
- Rebase before opening PR to avoid conflict PRs.

## Exclusions

These rules do not require changing legacy v1 (`src/mindtorch/`) unless explicitly requested.
