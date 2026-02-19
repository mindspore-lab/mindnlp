"""torch.distributed.device_mesh stub - not available in mindtorch_v2."""

def __getattr__(name):
    raise AttributeError(
        f"module 'torch.distributed.device_mesh' has no attribute '{name}'. "
        "DeviceMesh is not available in mindtorch_v2."
    )
