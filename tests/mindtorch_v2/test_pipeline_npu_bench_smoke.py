import mindtorch_v2 as torch

from benchmarks.pipeline_npu.bench import run_case, CASES


def test_pipeline_bench_smoke_cpu():
    case = CASES["A1"]
    result = run_case(case, device="cpu", pipeline=False, warmup=1, iters=1)
    assert result["case_id"] == "A1"
    assert "mean_ms" in result
    assert "p95_ms" in result
    assert isinstance(result["mean_ms"], float)
    assert isinstance(result["p95_ms"], float)


def test_pipeline_bench_cases_matrix():
    for key in ["A1", "A2", "A2s", "A3", "B1", "B1s", "B2", "B3", "C1", "C2", "D1", "D2"]:
        assert key in CASES
