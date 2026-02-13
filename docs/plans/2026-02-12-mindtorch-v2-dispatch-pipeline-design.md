# MindTorch V2 Dispatcher + Pipeline Design

## Goals
- Provide a torch-like dispatcher that can integrate multi-stage execution.
- Reduce host-bound overhead by batching plan/execute across multiple ops.
- Keep explicit opt-in pipeline mode for backward compatibility.

## Non-Goals
- No static graph compilation yet.
- No background worker threads in the first iteration.
- No implicit pipeline mode by default.

## Key Concepts
- **Meta stage**: infer shape/stride/dtype and create output Tensor metadata.
- **Plan stage**: build ACLNN executor, compute workspace size, record dependencies.
- **Impl stage**: perform ACLNN execution and synchronize stream.

## Dispatcher Integration
- Extend registry entries to support `meta`, `plan`, and `impl` kernels per op.
- Default mode: dispatch calls `impl` immediately (eager execution).
- Pipeline mode: dispatch calls `meta`, returns a pending Tensor, and records `plan+impl` in a pipeline queue.

## Pipeline API
- `with torch.pipeline():` enables pipeline mode.
- `pipeline.flush()` forces execution of all pending ops.
- Sync boundaries trigger flush automatically (e.g., `to("cpu")`, `numpy()`, `item()`, `repr/print`, comparisons).

## Execution Flow (Pipeline Mode)
1. `meta` stage returns output Tensor (pending state).
2. Queue records `plan+impl` with input/output dependencies.
3. On flush: execute `plan` for all queued ops, then `impl` in order.
4. Mark output Tensors ready; errors abort flush and clear queue.

## Autograd Behavior
- Autograd nodes are created during flush when forward executes.
- `backward()` triggers a flush before running the existing eager backward path.
- Optional inference-only pipeline mode can skip autograd recording.

## Error Handling
- Fail fast in `plan` or `impl` with clear exceptions.
- On failure: stop execution, clear queue, leave pending tensors invalid.

## Notes
- Future work may add executor/workspace reuse and background execution.
- CPU ops can remain eager (or be trivially flushed in pipeline).
