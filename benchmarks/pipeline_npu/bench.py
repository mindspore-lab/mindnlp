import mindtorch_v2 as torch

from .cases import CASES
from .utils import measure, summarize


def _resolve_dtype(dtype_name):
    return getattr(torch, dtype_name)


def _sync_for(device):
    if device == "npu" and hasattr(torch, "npu") and torch.npu.is_available():
        return torch.npu.synchronize
    return None


def run_case(case, *, device="cpu", pipeline=False, warmup=5, iters=20):
    dtype = _resolve_dtype(case["dtype"])
    forward = case["builder"](device, dtype)
    sync = _sync_for(device)

    def _run_once():
        if pipeline:
            with torch.pipeline(max_ops=64):
                forward()
        else:
            forward()

    samples = measure(_run_once, warmup=warmup, iters=iters, sync=sync)
    mean, median, p95 = summarize(samples)

    op_count = 0
    if pipeline:
        with torch.pipeline(max_ops=64) as pipe:
            forward()
            pipe.flush()
            dump = pipe.debug_dump()
            op_count = len(dump.get("entries", []))

    return {
        "case_id": case["case_id"],
        "batch": case["batch"],
        "seq": case["seq"],
        "hidden": case["hidden"],
        "heads": case["heads"],
        "dtype": case["dtype"],
        "mean_ms": float(mean),
        "median_ms": float(median),
        "p95_ms": float(p95),
        "op_count": int(op_count),
    }


def run():
    results = {}
    for name, case in CASES.items():
        results[name] = run_case(case, device="cpu", pipeline=False, warmup=1, iters=1)
    return results


__all__ = ["CASES", "run_case", "run"]
