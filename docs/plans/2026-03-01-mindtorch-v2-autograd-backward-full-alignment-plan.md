# MindTorch v2 Autograd Backward Full Alignment Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Fully remove storage-direct backward logic and `cpu_only` autograd gating for the target op set (`relu/relu_/abs/neg/silu/leaky_relu/elu/mish/prelu`) and align behavior with Torch dispatch-driven autograd semantics across CPU/NPU/Meta.

**Architecture:** Refactor `_backends/autograd.py` into a formula-driven backward layer where backward functions only compose tensor ops through `redispatch`, never numpy/storage payload access. Keep wrapper plumbing (`_autograd_unary/_binary/_inplace`) focused on graph wiring and keyset handling, then migrate target formulas and registrations to this unified mechanism.

**Tech Stack:** Python, pytest, mindtorch_v2 dispatch/autograd core (`_dispatch`, `_autograd`, `_backends/autograd.py`).

---

### Task 1: Add guard contracts to prevent storage-direct backward regressions

**Files:**
- Create: `tests/mindtorch_v2/contract/test_autograd_backward_storage_isolation.py`
- Test: `tests/mindtorch_v2/contract/test_schema_registration_order.py`
- Test: `tests/mindtorch_v2/contract/test_schema_coverage.py`

**Step 1: Write the failing tests**

Add tests that scan `src/mindtorch_v2/_backends/autograd.py` source and fail if:
- `storage().data` appears in backward formula sections
- `storage()._data` appears in backward formula sections

Also add a test for target ops (`relu`, `abs`, `neg`, `silu`, `leaky_relu`, `elu`, `mish`, `prelu`) asserting that `AutogradNPU` registration path is not marked with `cpu_only=True`.

**Step 2: Run tests to verify failure**

Run:
`PYTHONPATH=src pytest -q tests/mindtorch_v2/contract/test_autograd_backward_storage_isolation.py`

Expected: FAIL (current code still has storage-direct backward + cpu_only gates).

**Step 3: Commit failing tests checkpoint**

```bash
git add tests/mindtorch_v2/contract/test_autograd_backward_storage_isolation.py
git commit -m "test(mindtorch_v2): add autograd backward storage-isolation guard contracts"
```

---

### Task 2: Refactor relu/relu_ backward to pure dispatch formulas

**Files:**
- Modify: `src/mindtorch_v2/_backends/autograd.py`
- Test: `tests/mindtorch_v2/test_dispatch_autograd_wrappers.py`
- Test: `tests/mindtorch_v2/test_autograd_inplace.py`

**Step 1: Write/adjust failing behavioral tests**

Add/adjust tests for relu/relu_ asserting:
- `AutogradNPU` path attaches `grad_fn`
- backward runs without touching storage payload
- inplace relu still respects versioning contracts

**Step 2: Run tests to verify failure (red)**

Run:
`PYTHONPATH=src pytest -q tests/mindtorch_v2/test_dispatch_autograd_wrappers.py -k "relu"`

Expected: FAIL before refactor.

**Step 3: Implement minimal relu/relu_ formula refactor**

In `src/mindtorch_v2/_backends/autograd.py`:
- Replace storage/numpy logic in `_relu_backward` and `_inplace_relu_backward` with redispatch-only composition.
- Remove `cpu_only=True` for relu/relu_ registrations.
- Keep wrapper/keyset plumbing unchanged.

**Step 4: Run targeted tests (green)**

Run:
`PYTHONPATH=src pytest -q tests/mindtorch_v2/test_dispatch_autograd_wrappers.py -k "relu" tests/mindtorch_v2/test_autograd_inplace.py`

Expected: PASS.

**Step 5: Commit**

```bash
git add src/mindtorch_v2/_backends/autograd.py tests/mindtorch_v2/test_dispatch_autograd_wrappers.py tests/mindtorch_v2/test_autograd_inplace.py
git commit -m "refactor(mindtorch_v2): make relu backward formulas storage-free and key-consistent"
```

---

### Task 3: Migrate abs/neg backward formulas

**Files:**
- Modify: `src/mindtorch_v2/_backends/autograd.py`
- Modify/Test: `tests/mindtorch_v2/test_dispatch_autograd_wrappers.py`

**Step 1: Add failing tests for abs/neg AutogradNPU behavior**

Assertions:
- output has `grad_fn` when requires_grad input
- no cpu_only gating path is active for NPU

**Step 2: Run failing tests**

Run:
`PYTHONPATH=src pytest -q tests/mindtorch_v2/test_dispatch_autograd_wrappers.py -k "abs or neg"`

Expected: FAIL initially.

**Step 3: Implement minimal formula changes**

In `src/mindtorch_v2/_backends/autograd.py`:
- Replace storage/numpy formula in `_abs_backward` with redispatch-only ops.
- Keep `_neg_backward` dispatch-only.
- Remove `cpu_only=True` flags for abs/neg registrations.

**Step 4: Verify pass**

Run:
`PYTHONPATH=src pytest -q tests/mindtorch_v2/test_dispatch_autograd_wrappers.py -k "abs or neg"`

Expected: PASS.

**Step 5: Commit**

```bash
git add src/mindtorch_v2/_backends/autograd.py tests/mindtorch_v2/test_dispatch_autograd_wrappers.py
git commit -m "refactor(mindtorch_v2): align abs/neg autograd formulas with dispatch-only execution"
```

---

### Task 4: Migrate silu/leaky_relu/elu/mish/prelu formulas to dispatch-only

**Files:**
- Modify: `src/mindtorch_v2/_backends/autograd.py`
- Modify/Test: `tests/mindtorch_v2/test_nn_functional.py`
- Modify/Test: `tests/mindtorch_v2/test_dispatch_autograd_wrappers.py`

**Step 1: Add failing tests for target op autograd attachment and backward smoke**

For each op in scope:
- requires_grad input => output has grad_fn on CPU and NPU-like key path
- backward executes without storage-direct dependency

**Step 2: Run tests to verify red**

Run:
`PYTHONPATH=src pytest -q tests/mindtorch_v2/test_nn_functional.py -k "silu or leaky_relu or elu or mish or prelu"`

Expected: at least one failure before migration.

**Step 3: Implement formulas with redispatch primitives only**

In `src/mindtorch_v2/_backends/autograd.py`:
- Remove numpy/storage reads/writes for the five formulas.
- Express formulas with existing operators (`sigmoid`, `tanh`, `softplus`, `exp`, `where`, `mul`, `add`, `sub`, `div`, etc.)
- Remove `cpu_only=True` registrations for migrated ops.

**Step 4: Validate targeted tests (green)**

Run:
`PYTHONPATH=src pytest -q tests/mindtorch_v2/test_nn_functional.py -k "silu or leaky_relu or elu or mish or prelu" tests/mindtorch_v2/test_dispatch_autograd_wrappers.py`

Expected: PASS.

**Step 5: Commit**

```bash
git add src/mindtorch_v2/_backends/autograd.py tests/mindtorch_v2/test_nn_functional.py tests/mindtorch_v2/test_dispatch_autograd_wrappers.py
git commit -m "refactor(mindtorch_v2): migrate activation backward formulas to dispatch-only autograd"
```

---

### Task 5: Remove duplicate/legacy branches and normalize wrapper behavior

**Files:**
- Modify: `src/mindtorch_v2/_backends/autograd.py`
- Test: `tests/mindtorch_v2/test_autograd.py`
- Test: `tests/mindtorch_v2/contract/test_autograd_create_graph.py`

**Step 1: Add failing tests for create_graph and wrapper consistency**

Validate that migrated ops:
- keep grad graph under `create_graph=True`
- do not regress wrapper semantics across CPU/NPU/Meta

**Step 2: Run tests to verify red**

Run:
`PYTHONPATH=src pytest -q tests/mindtorch_v2/contract/test_autograd_create_graph.py -k "relu or abs or silu"`

Expected: fail if wrapper/formula inconsistency remains.

**Step 3: Implement cleanup**

- Remove dead branches and duplicated local variables (for example duplicate `ones = ...`).
- Ensure wrappers do not special-case backend numeric logic.
- Keep backward formulas as the only source of per-op gradient math.

**Step 4: Re-run focused tests**

Run:
`PYTHONPATH=src pytest -q tests/mindtorch_v2/test_autograd.py tests/mindtorch_v2/contract/test_autograd_create_graph.py`

Expected: PASS.

**Step 5: Commit**

```bash
git add src/mindtorch_v2/_backends/autograd.py tests/mindtorch_v2/test_autograd.py tests/mindtorch_v2/contract/test_autograd_create_graph.py
git commit -m "refactor(mindtorch_v2): normalize autograd wrapper behavior after formula migration"
```

---

### Task 6: Full verification gate and PR prep

**Files:**
- Verify only (no required file changes)

**Step 1: Run required schema gates**

Run:
`PYTHONPATH=src pytest -q tests/mindtorch_v2/contract/test_schema_registration_order.py tests/mindtorch_v2/contract/test_schema_coverage.py`

Expected: PASS.

**Step 2: Run contract + autograd regression suite**

Run:
`PYTHONPATH=src pytest -q tests/mindtorch_v2/contract/test_autograd_contract.py tests/mindtorch_v2/contract/test_autograd_create_graph.py tests/mindtorch_v2/test_autograd.py tests/mindtorch_v2/test_autograd_inplace.py tests/mindtorch_v2/test_dispatch_autograd_wrappers.py tests/mindtorch_v2/test_nn_functional.py`

Expected: PASS.

**Step 3: Optional NPU runtime checks (if hardware available)**

Run:
`PYTHONPATH=src pytest -q tests/mindtorch_v2/test_nn_functional_npu.py -k "silu or leaky_relu or elu or mish or prelu or relu"`

Expected: PASS (or explicit skip on unavailable NPU).

**Step 4: Final commit hygiene**

Run:
```bash
git status
git log --oneline -n 10
```

Ensure commits are scoped and message history is reviewable.

**Step 5: Open PR**

Create PR to `mindspore-lab/mindnlp:master` with:
- mechanism summary (storage-free backward + cpu_only removal + formula migration)
- full test plan and outputs

---

## Notes for Execution
- Do not add unrelated operator families in this PR.
- If an op lacks required backend primitives, keep explicit deterministic error and document it in tests.
- Any new differentiable op added later must follow this storage-free formula pattern.
