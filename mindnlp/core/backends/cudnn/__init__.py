from contextlib import contextmanager

@contextmanager
def flags(
    enabled=False,
    benchmark=False,
    benchmark_limit=10,
    deterministic=False,
    allow_tf32=True,
    fp32_precision="none",
):
    try:
        yield
    finally:
        pass


def is_acceptable(tensor):
    return True
