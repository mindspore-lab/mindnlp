# MindTorch V2 Dynamic Pipeline Review Checklist

## Scope

- Target: `mindtorch v2` dynamic pipeline only.
- Excluded: static-graph compilation work, cross-window global optimization, unrelated MindSpore features.
- Review objective: confirm phase-1 design is semantically safe, debuggable, and implementation-ready.

## A. Positioning and Non-Goals

- [ ] Pipeline is explicitly defined as a dynamic-graph dispatch acceleration layer.
- [ ] Team agrees this is not a static-graph capture/compile path.
- [ ] Python control-flow behavior is unchanged.
- [ ] No phase-1 commitment to cross-window graph rewrites.

## B. Semantic Contract

- [ ] Strong equivalence is explicit: value/dtype/shape/stride must match eager.
- [ ] Side-effect order contract is explicit: inplace, view-writeback, version bumps match eager order.
- [ ] Autograd behavior parity is explicit, including `retain_graph/create_graph`.
- [ ] RNG consumption order parity is explicit.
- [ ] Weak equivalence is explicit: error timing may shift to flush boundary.
- [ ] Failure consistency is explicit: no pollution from non-submitted ops.
- [ ] First-error policy is explicit: one flush raises one root-cause error.

## C. Flush and Execution Model

- [ ] Flush triggers include all three sources: manual, forced boundary, threshold.
- [ ] Trigger priority is explicit: `forced > manual > threshold`.
- [ ] Forced boundaries include: `item/numpy/to(cpu)/repr-print/backward/autograd.grad/context-exit`.
- [ ] Auto-flush thresholds are explicit: `max_ops`, `max_pending_bytes`, `max_wait_us`.
- [ ] Device presets are explicit and asymmetric: smaller CPU windows, larger NPU windows.
- [ ] User config override precedence is explicit.
- [ ] Phase-1 submit policy is explicit: strict sequential submit.
- [ ] Dependency/event placeholders for future concurrent submit are preserved in internal design.

## D. Data Model and Debuggability

- [ ] `OpEntry` minimum fields are defined (`op_seq`, schema/op info, read/write/alias/view/version/callsite).
- [ ] `ErrorEnvelope` uses dual form: dataclass + `to_dict()`.
- [ ] `ErrorEnvelope` includes root localization tuple: `op_seq + phase + callsite`.
- [ ] `ErrorEnvelope` includes dependency and suppression context.
- [ ] Companion debug APIs are defined: `last_error`, `format_error`, `debug_dump`.
- [ ] Pending state is debug-only by default (no normal-user `is_pending` surface).

## E. Alias/View/Version (Top Priority)

- [ ] Inplace identity parity is explicitly required.
- [ ] Single version bump per logical write is explicitly required.
- [ ] Base/view visibility parity after flush is explicitly required.
- [ ] Functionalize writeback ordering parity is explicitly required.
- [ ] Read/write hazards must be represented as explicit dependencies.

## F. Test and Gate Plan

- [ ] Primary test lanes are `CPU + NPU`; Meta is supplemental.
- [ ] Contract tests cover: inplace identity, version bump parity, view writeback, mixed base-view writes, functionalize+pipeline, delayed error.
- [ ] Phase-1 P0 gate is explicitly set to NPU delayed-error localizability.
- [ ] P0 requires deterministic reproduction of stable `error_id`.
- [ ] P0 requires machine-consumable complete error payload via `to_dict()`.

## G. Rollout Order

- [ ] Step 1: lock semantic + error contracts in docs and contract tests.
- [ ] Step 2: add threshold auto-flush with reason tagging.
- [ ] Step 3: complete OpEntry/ErrorEnvelope and debug API surface.
- [ ] Step 4: converge semantic parity on CPU/NPU under sequential submit.
- [ ] Step 5: only then evaluate dependency-aware concurrent submit.

## Exit Criteria for Design Approval

- [ ] All checklist sections A-G pass.
- [ ] No unresolved contradictions against eager-equivalence requirements.
- [ ] Team agrees phase-1 success metric: correctness and localization quality first, throughput second.
