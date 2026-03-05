# MindTorch V2 Pipeline NPU Benchmark Design

## Goals

- Provide reproducible NPU performance comparisons for pipeline vs eager.
- Focus on Transformer-style workloads where kernel launch overhead is visible.
- Produce structured, scriptable output without CI performance thresholds.

## Non-Goals

- No static-graph compilation.
- No strict performance gating in CI.
- No backward benchmarks in phase 1.

## Benchmark Coverage Matrix

### Layer A: Micro-bench (dispatch-sensitive)
- A1: `layernorm -> linear -> gelu -> linear -> residual`
  - B=1, S=128, H=1024, dtype=fp16
- A2: `qkv linear -> reshape -> matmul -> softmax -> matmul -> proj`
  - B=1, S=512, H=1024, heads=16, dtype=fp16
- A3: `ffn (linear -> silu -> linear)`
  - B=8, S=128, H=2048, dtype=fp16

### Layer B: Module-level (1–2 blocks)
- B1: 1 block, B=1, S=512, H=2048, heads=16, dtype=fp16
- B2: 1 block, B=4, S=128, H=1024, heads=8, dtype=bf16
- B3: 2 blocks, B=1, S=2048, H=1024, heads=16, dtype=fp16

### Layer C: Mini Transformer (4 blocks)
- C1: 4 blocks, B=1, S=512, H=1024, heads=16, dtype=fp16
- C2: 4 blocks, B=2, S=256, H=1024, heads=16, dtype=bf16

## Measurement Strategy

- Compare: eager vs `with torch.pipeline(max_ops=..., max_pending_bytes=...)`.
- Synchronize: `torch.npu.synchronize()` before/after timing windows.
- Timing: `time.perf_counter()`; collect mean/median/p95.
- Iterations: warmup=5, iters=20 by default (override via env vars).

## Output Format

- JSON line per case with:
  - `case_id`, `dtype`, `batch`, `seq`, `hidden`, `heads`
  - `mean_ms`, `median_ms`, `p95_ms`
  - `op_count` (from `pipe.debug_dump()`)

## Script Layout

- `benchmarks/pipeline_npu/bench.py` (entry point)
- `benchmarks/pipeline_npu/cases.py` (case definitions)
- `benchmarks/pipeline_npu/utils.py` (timing/stat helpers)

## Error Handling

- On failure: output JSON with `case_id` + `error` and continue to next case.

## Testing

- Add a smoke test that runs 1 tiny case and asserts JSON schema only.
- No performance thresholds in CI.

## Benchmark Results (NPU)

Environment: `conda env mindspore`, warmup=2, iters=5, device=`npu`.

Baseline (original A2/B1 shapes)
- A2 eager 14.930 ms, pipeline 16.069 ms, op_count=19 (pipeline slower ~7.6%)
- B1 eager 25.655 ms, pipeline 26.496 ms, op_count=26 (pipeline slower ~3.3%)

Additional cases (A3/B2/C1/C2)
- A3 eager 0.350 ms, pipeline 0.306 ms, op_count=3 (pipeline faster ~14.2%)
- B2 eager 0.830 ms, pipeline 0.528 ms, op_count=7 (pipeline faster ~57.1%)
- C1 eager 1.129 ms, pipeline 0.640 ms, op_count=9 (pipeline faster ~76.5%)
- C2 eager 1.324 ms, pipeline 0.826 ms, op_count=11 (pipeline faster ~60.3%)

Workspace-safe shapes and long-chain cases
- A2s eager 13.279 ms, pipeline 14.651 ms, op_count=19 (pipeline slower ~10.3%)
- B1s eager 20.383 ms, pipeline 22.461 ms, op_count=26 (pipeline slower ~10.2%)
- D1 eager 162.408 ms, pipeline 173.328 ms, op_count=16 (pipeline slower ~6.7%)
- D2 eager 36.590 ms, pipeline 34.960 ms, op_count=48 (pipeline faster ~4.5%)

Notes
- D2 is a light-ops chain; pipeline reduces dispatch overhead and shows a small gain.
- In matmul-heavy chains, pipeline overhead can offset benefits. Further gains likely need
  larger `max_ops`, fewer flushes, or lower meta/schema overhead.

## Interpretation and Next Steps

Interpretation
- Pipeline helps most when dispatch overhead dominates (light ops + long chains).
- Matmul-heavy chains currently pay extra metadata/flush overhead that can erase gains.
- Contiguity and workspace constraints in large batched matmul can skew results; prefer
  workspace-safe shapes for repeatable comparisons.

Next steps
- Sweep `max_ops` and `max_pending_bytes` to quantify sensitivity to flush frequency.
- Add a micro-profiler around pipeline record vs. flush to isolate metadata cost.
- Explore batching submits or deferring schema validation to reduce per-op overhead.
