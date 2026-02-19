"""torch.distributed.fsdp stub - not available in mindtorch_v2."""

def __getattr__(name):
    raise AttributeError(
        f"module 'torch.distributed.fsdp' has no attribute '{name}'. "
        "FSDP is not available in mindtorch_v2."
    )
