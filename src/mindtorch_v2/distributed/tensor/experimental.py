"""torch.distributed.tensor.experimental stub - not available in mindtorch_v2."""

from contextlib import contextmanager


@contextmanager
def context_parallel(*args, **kwargs):
    yield


def __getattr__(name):
    raise AttributeError(
        f"module 'torch.distributed.tensor.experimental' has no attribute '{name}'. "
        "Tensor experimental is not available in mindtorch_v2."
    )
