import time


def measure(fn, *, warmup=5, iters=20, sync=None):
    for _ in range(warmup):
        if sync is not None:
            sync()
        fn()
        if sync is not None:
            sync()

    samples = []
    for _ in range(iters):
        if sync is not None:
            sync()
        start = time.perf_counter()
        fn()
        if sync is not None:
            sync()
        samples.append((time.perf_counter() - start) * 1000.0)
    return samples


def summarize(samples):
    if not samples:
        return 0.0, 0.0, 0.0
    values = sorted(samples)
    mean = sum(values) / len(values)
    median = values[len(values) // 2]
    p95 = values[int(len(values) * 0.95) - 1]
    return mean, median, p95
