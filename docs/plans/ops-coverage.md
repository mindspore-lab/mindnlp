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
| `aten::sin` | CPU | done | - | numpy impl + meta + tests |
| `aten::cos` | CPU | done | - | numpy impl + meta + tests |
| `aten::tan` | CPU | done | - | numpy impl + meta + tests |
| `aten::tanh` | CPU | done | - | numpy impl + meta + tests |
| `aten::sigmoid` | CPU | done | - | numpy impl + meta + tests |
| `aten::floor` | CPU | done | - | numpy impl + meta + tests |
| `aten::ceil` | CPU | done | - | numpy impl + meta + tests |
| `aten::round` | CPU | done | - | numpy impl + meta + tests |
| `aten::trunc` | CPU | done | - | numpy impl + meta + tests |
| `aten::frac` | CPU | done | - | numpy impl + meta + tests |
| `aten::pow` | CPU | done | - | numpy impl + meta + tests |
| `aten::log2` | CPU | done | - | numpy impl + meta + tests |
| `aten::log10` | CPU | done | - | numpy impl + meta + tests |
| `aten::exp2` | CPU | done | - | numpy impl + meta + tests |
| `aten::rsqrt` | CPU | done | - | numpy impl + meta + tests |
| `aten::sign` | CPU | done | - | numpy impl + meta + tests |
| `aten::signbit` | CPU | done | - | numpy impl + meta + tests |
| `aten::isnan` | CPU | done | - | numpy impl + meta + tests |
| `aten::isinf` | CPU | done | - | numpy impl + meta + tests |
| `aten::isfinite` | CPU | done | - | numpy impl + meta + tests |
| `aten::sinh` | CPU | done | - | numpy impl + meta + tests |
| `aten::cosh` | CPU | done | - | numpy impl + meta + tests |
| `aten::asinh` | CPU | done | - | numpy impl + meta + tests |
| `aten::acosh` | CPU | done | - | numpy impl + meta + tests |
| `aten::atanh` | CPU | done | - | numpy impl + meta + tests |
| `aten::erf` | CPU | done | - | numpy impl + meta + tests |
| `aten::erfc` | CPU | done | - | numpy impl + meta + tests |
| `aten::softplus` | CPU | done | - | numpy impl + meta + tests |
| `aten::clamp` | CPU | done | - | numpy impl + meta + tests |
| `aten::clamp_min` | CPU | done | - | numpy impl + meta + tests |
| `aten::clamp_max` | CPU | done | - | numpy impl + meta + tests |
| `aten::relu6` | CPU | done | - | numpy impl + meta + tests |
| `aten::hardtanh` | CPU | done | - | numpy impl + meta + tests |
| `aten::min` | CPU | done | - | numpy impl + meta + tests |
| `aten::max` | CPU | done | - | numpy impl + meta + tests |
| `aten::amin` | CPU | done | - | numpy impl + meta + tests |
| `aten::amax` | CPU | done | - | numpy impl + meta + tests |
| `aten::fmin` | CPU | done | - | numpy impl + meta + tests |
| `aten::fmax` | CPU | done | - | numpy impl + meta + tests |
| `aten::where` | CPU | done | - | numpy impl + meta + tests |
| `aten::atan` | CPU | done | - | numpy impl + meta + tests |
| `aten::atan2` | CPU | done | - | numpy impl + meta + tests |
| `aten::asin` | CPU | done | - | numpy impl + meta + tests |
| `aten::acos` | CPU | done | - | numpy impl + meta + tests |
| `aten::lerp` | CPU | done | - | numpy impl + meta + tests |
| `aten::addcmul` | CPU | done | - | numpy impl + meta + tests |
| `aten::addcdiv` | CPU | done | - | numpy impl + meta + tests |
| `aten::logaddexp` | CPU | done | - | numpy impl + meta + tests |
| `aten::logaddexp2` | CPU | done | - | numpy impl + meta + tests |
| `aten::hypot` | CPU | done | - | numpy impl + meta + tests |
| `aten::remainder` | CPU | done | - | numpy impl + meta + tests |
| `aten::fmod` | CPU | done | - | numpy impl + meta + tests |
| `aten::arange` | CPU | done | - | numpy impl + meta + tests |
| `aten::linspace` | CPU | done | - | numpy impl + meta + tests |
| `aten::full` | CPU | done | - | numpy impl + meta + tests |
| `aten::all` | CPU | done | - | numpy impl + meta + tests |
| `aten::any` | CPU | done | - | numpy impl + meta + tests |
| `aten::argmax` | CPU | done | - | numpy impl + meta + tests |
