"""torch.distributed.device_mesh stub - not available in mindtorch_v2."""


class DeviceMesh:
    pass


def init_device_mesh(*args, **kwargs):
    raise NotImplementedError("DeviceMesh is not available in mindtorch_v2.")


def __getattr__(name):
    raise AttributeError(
        f"module 'torch.distributed.device_mesh' has no attribute '{name}'. "
        "DeviceMesh is not available in mindtorch_v2."
    )
