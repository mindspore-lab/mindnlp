"""torch.distributed.tensor.parallel stub - not available in mindtorch_v2."""


class SequenceParallel:
    pass


def parallelize_module(*args, **kwargs):
    raise NotImplementedError("parallelize_module is not available in mindtorch_v2.")


def __getattr__(name):
    raise AttributeError(
        f"module 'torch.distributed.tensor.parallel' has no attribute '{name}'. "
        "Tensor parallel is not available in mindtorch_v2."
    )
