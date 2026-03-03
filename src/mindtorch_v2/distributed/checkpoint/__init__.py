"""torch.distributed.checkpoint stub - not available in mindtorch_v2."""


def __getattr__(name):
    raise AttributeError(
        f"module 'torch.distributed.checkpoint' has no attribute '{name}'. "
        "Distributed checkpoint is not available in mindtorch_v2."
    )
