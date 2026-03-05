# MindTorch V2 Dynamic Pipeline Design

## 1. Background and Positioning

The `pipeline` in `mindtorch v2` is positioned as a **dynamic-graph dispatch acceleration layer**, not a static-graph compilation solution.

Core goals:

1. Reduce per-op host-side dispatch overhead without changing eager-observable semantics.
2. Convert multiple dynamic op calls from "submit one by one" into "windowed plan + submit".
3. Reserve architectural anchors for future concurrent submission, while prioritizing correctness and debuggability in phase 1.

Core non-goals:

1. No whole-graph capture and no static-graph closure requirements.
2. No change to Python control-flow execution model.
3. No cross-window global graph optimization in phase 1.

## 2. Semantic Contract

### 2.1 Strong Equivalence (Must match eager)

1. Numerical results must match: value/dtype/shape/stride.
2. Side-effect order must match: inplace, view-writeback, and version bump order.
3. Autograd behavior must match: gradients and graph behavior (including `retain_graph/create_graph`).
4. RNG consumption order must match: same seed yields same results.

### 2.2 Weak Equivalence (Allowed difference)

1. Exception timing may be delayed to a flush boundary.
2. Exception type and core message should align with eager as much as possible.

### 2.3 Failure Consistency

1. On flush failure, non-submitted ops must not pollute visible state.
2. Already-submitted ops are treated as eager side effects that already happened; no fake rollback.
3. Apply "first-error" policy: one flush raises only the first root-cause error, others are marked `suppressed`.

## 3. API and Enablement Strategy

### 3.1 Enablement Model

1. Globally supported, but disabled by default.
2. Recommended entrypoint: `with torch.pipeline():`.
3. Global config entrypoint: `torch.pipeline_config(...)` (merged with context config).

### 3.2 Auto-flush Strategy (Choice B + C)

Flush triggers:

1. Explicit trigger: `pipeline.flush()`.
2. Forced boundaries: observable APIs (see Section 4).
3. Threshold trigger: any hit on `max_ops / max_pending_bytes / max_wait_us`.

Priority: `forced boundary > explicit trigger > threshold trigger`.

### 3.3 Device Presets

1. CPU preset uses smaller windows to reduce latency jitter.
2. NPU preset uses larger windows to amortize host->device submission overhead.
3. User explicit config always overrides device presets.

## 4. Flush Boundary Contract

The following cases must flush first:

1. Value exposed to Host/Python: `item()`, `numpy()`, `to("cpu")`.
2. Observable output: `repr/print`.
3. Autograd boundaries: `backward()`, `autograd.grad()`.
4. Context boundary: exiting `with torch.pipeline():`.
5. User explicit boundary: `pipeline.flush()`.

By default, `is_pending()`-style state is not exposed to normal users; pending state is only available in debug APIs.

## 5. Runtime State Machine

Pipeline window states:

1. `recording`: collect `OpEntry`.
2. `frozen`: freeze window and reject new records.
3. `planning`: run plan stage and build submission plan.
4. `submitting`: submit in eager order.
5. `committed/failed`: publish visibility or raise error.

Phase-1 submission policy:

1. Strict sequential submit (correctness first).
2. Keep dependency edges and event slots internally for future concurrent submit.

## 6. Data Structures and Required Fields

### 6.1 `OpEntry`

Minimum recommended fields:

1. `op_seq`, `op_name`, `schema`, `backend`.
2. Input/output handles and device/dtype metadata.
3. `read_set`, `write_set`, `alias_set_id`.
4. `view_of`, `mutates_metadata`, `version_plan`.
5. `callsite` (file/line/function) for delayed-error localization.

### 6.2 `ErrorEnvelope` (Full level, default)

Use `dataclass + to_dict()` dual form. Recommended fields:

1. `error_id`, `flush_id`, `op_seq`, `op_name`, `schema`.
2. `phase` (plan/submit/commit), `backend`, `callsite`.
3. `read_set`, `write_set`, `alias_set`, `version_plan`.
4. `dependency_edges`, `runtime_code` (optional device runtime code).
5. `suppressed_errors` (chained error summary).

Companion APIs:

1. `pipeline.last_error()` returns structured error object.
2. `pipeline.format_error(style="short|full")` returns readable text.
3. `pipeline.debug_dump(failed_only=True)` exports failed-window context.

## 7. Alias/View/Version Contract (Top Priority)

1. Inplace must preserve object identity exactly as eager.
2. Each logical write bumps version once, in eager-consistent order.
3. Base/view sharing must be preserved; post-flush visibility must match eager.
4. Under functionalize path, writeback order must match eager.
5. Read/write conflicts must be represented as explicit dependencies; windowing must not break ordering.

## 8. Contract Test Strategy (CPU + NPU First)

Phase-1 gates run on `CPU + NPU` as primary lanes; Meta is supplemental:

1. `inplace identity`: object identity and values match.
2. `single bump per write`: version increments match.
3. `view writeback`: base/view visibility matches.
4. `base-view mixed writes`: order-sensitive behavior matches.
5. `functionalize + pipeline`: alias/writeback ordering matches.
6. `flush delayed error`: error is raised at flush with complete localization fields.

Phase-1 P0 gate: **NPU delayed-error localizability**.

Must satisfy:

1. Error can be traced to `op_seq + phase + callsite`.
2. Same input reproduces the same `error_id` (in deterministic replay mode).
3. Logs are programmatically consumable (`to_dict()` complete fields).

## 9. Execution Order (Pipeline Scope Only)

1. Freeze semantic and error contracts (docs + contract tests).
2. Add threshold auto-flush (including device presets and reason tagging).
3. Complete `OpEntry/ErrorEnvelope` and debug APIs.
4. Converge CPU/NPU semantic consistency under sequential submit.
5. Only after passing gates, consider dependency-aware concurrent submit (without changing semantic contract).
