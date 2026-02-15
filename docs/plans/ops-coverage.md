# MindTorch v2 Ops Coverage Collaboration Rules

## Goal
Provide a shared, low-overhead collaboration contract for parallel CPU/NPU op development.

## Scope
This document is about collaboration rules only (not defining the op scope).

## Rules
1. **File ownership**
   - CPU agent owns: `src/mindtorch_v2/_backends/cpu/*`, `src/mindtorch_v2/_backends/meta/*`, CPU tests.
   - NPU agent owns: `src/mindtorch_v2/_backends/npu/*`, NPU tests.
   - Shared files (`_backends/autograd.py`, `_functional.py`, common tests) must have a single owner per PR. If both need changes, split into separate PRs or agree on one owner.

2. **Single source of truth**
   - Add new ops by updating this doc with a short entry (one line per op).
   - Each op entry must list: owner (CPU/NPU), PR link, and status.

3. **PR size**
   - Small PRs only (3–5 ops max).
   - Rebase on `ms/master` before opening PRs.

4. **Tests**
   - Every new op must include: CPU test, meta test; NPU test guarded by `torch.npu.is_available()`.
   - Autograd ops must include a gradient test.

5. **Conflict prevention**
   - No concurrent edits to shared files unless pre‑agreed.
   - If a shared file must change, note it in the op entry and block other agent from editing it until merged.

## Op Tracking Table (append-only)

| Op | Owner | Status | PR | Notes |
|---|---|---|---|---|
| `aten::abs` | CPU | done | - | numpy impl + meta + tests |
| `aten::neg` | CPU | done | - | numpy impl + meta + tests |
| `aten::exp` | CPU | done | - | numpy impl + meta + tests |
| `aten::log` | CPU | done | - | numpy impl + meta + tests |
| `aten::sqrt` | CPU | done | - | numpy impl + meta + tests |
