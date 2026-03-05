# MindTorch V2 Pipeline NPU Bench Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Add transformer-style NPU benchmark scripts (pipeline vs eager) with a CPU-safe smoke test and structured JSON output.

**Architecture:** Provide a small benchmark package (`benchmarks/pipeline_npu`) with reusable timing utilities, case definitions for transformer micro/block/mini models, and a CLI runner. Tests use CPU fallback to validate schema without requiring NPU availability.

**Tech Stack:** Python, mindtorch_v2 (NPU/CPU), pytest.

---

### Task 1: Create benchmark package skeleton

**Files:**
- Create: `benchmarks/pipeline_npu/__init__.py`
- Create: `benchmarks/pipeline_npu/utils.py`
- Create: `benchmarks/pipeline_npu/cases.py`
- Create: `benchmarks/pipeline_npu/bench.py`
- Test: `tests/mindtorch_v2/test_pipeline_npu_bench_smoke.py`

**Step 1: Write the failing test**

```python
import mindtorch_v2 as torch
from benchmarks.pipeline_npu.bench import run_case, CASES


def test_pipeline_bench_smoke_cpu():
    case = CASES["A1"]
    result = run_case(case, device="cpu", pipeline=False, warmup=1, iters=1)
    assert "mean_ms" in result
    assert "p95_ms" in result
    assert result["case_id"] == "A1"
```

**Step 2: Run test to verify it fails**

Run: `PYTHONPATH=src pytest -q tests/mindtorch_v2/test_pipeline_npu_bench_smoke.py -q`
Expected: FAIL with `ImportError` or `AttributeError` (missing benchmark package).

**Step 3: Write minimal implementation**

- `utils.py`: timing helper `measure(fn, warmup, iters, sync)` returning stats.
- `cases.py`: define `CASES` dict with placeholder A1 only (CPU-safe ops).
- `bench.py`: implement `run_case(case, device, pipeline, warmup, iters)` returning JSON dict.

**Step 4: Run test to verify it passes**

Run: `PYTHONPATH=src pytest -q tests/mindtorch_v2/test_pipeline_npu_bench_smoke.py -q`
Expected: PASS.

**Step 5: Commit**

```bash
git add benchmarks/pipeline_npu tests/mindtorch_v2/test_pipeline_npu_bench_smoke.py
git commit -m "bench: add pipeline NPU benchmark skeleton"
```

### Task 2: Implement full transformer case matrix

**Files:**
- Modify: `benchmarks/pipeline_npu/cases.py`
- Modify: `benchmarks/pipeline_npu/bench.py`
- Test: `tests/mindtorch_v2/test_pipeline_npu_bench_smoke.py`

**Step 1: Write the failing test**

```python
from benchmarks.pipeline_npu.bench import CASES


def test_pipeline_bench_cases_matrix():
    assert all(k in CASES for k in ["A1", "A2", "A3", "B1", "B2", "B3", "C1", "C2"])
```

**Step 2: Run test to verify it fails**

Run: `PYTHONPATH=src pytest -q tests/mindtorch_v2/test_pipeline_npu_bench_smoke.py -q`
Expected: FAIL (missing cases).

**Step 3: Write minimal implementation**

- Add full A/B/C case list with transformer micro/block/mini definitions.
- Implement softmax, layernorm, and silu from available ops.
- Ensure device is parameterized (`device` argument) and dtype set.

**Step 4: Run test to verify it passes**

Run: `PYTHONPATH=src pytest -q tests/mindtorch_v2/test_pipeline_npu_bench_smoke.py -q`
Expected: PASS.

**Step 5: Commit**

```bash
git add benchmarks/pipeline_npu tests/mindtorch_v2/test_pipeline_npu_bench_smoke.py
git commit -m "bench: add transformer case matrix for pipeline NPU"
```

### Task 3: Add pipeline vs eager measurement and JSON output

**Files:**
- Modify: `benchmarks/pipeline_npu/bench.py`
- Modify: `benchmarks/pipeline_npu/utils.py`

**Step 1: Write the failing test**

```python
from benchmarks.pipeline_npu.bench import run_case, CASES


def test_pipeline_bench_json_output():
    result = run_case(CASES["A1"], device="cpu", pipeline=True, warmup=1, iters=1)
    for key in ["case_id", "mean_ms", "median_ms", "p95_ms", "op_count"]:
        assert key in result
```

**Step 2: Run test to verify it fails**

Run: `PYTHONPATH=src pytest -q tests/mindtorch_v2/test_pipeline_npu_bench_smoke.py -q`
Expected: FAIL (missing fields).

**Step 3: Write minimal implementation**

- For pipeline mode, wrap execution with `torch.pipeline(...)` and record `op_count` via `debug_dump()`.
- Output JSON dict with `case_id`, `dtype`, `batch`, `seq`, `hidden`, `heads`, `mean_ms`, `median_ms`, `p95_ms`, `op_count`.

**Step 4: Run test to verify it passes**

Run: `PYTHONPATH=src pytest -q tests/mindtorch_v2/test_pipeline_npu_bench_smoke.py -q`
Expected: PASS.

**Step 5: Commit**

```bash
git add benchmarks/pipeline_npu tests/mindtorch_v2/test_pipeline_npu_bench_smoke.py
git commit -m "bench: add pipeline vs eager JSON metrics"
```
