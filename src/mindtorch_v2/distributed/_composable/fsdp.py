"""torch.distributed._composable.fsdp stub - not available in mindtorch_v2."""


def fully_shard(*args, **kwargs):
    raise NotImplementedError("fully_shard is not available in mindtorch_v2.")


def register_fsdp_forward_method(*args, **kwargs):
    raise NotImplementedError("register_fsdp_forward_method is not available in mindtorch_v2.")


def __getattr__(name):
    raise AttributeError(
        f"module 'torch.distributed._composable.fsdp' has no attribute '{name}'. "
        "Composable FSDP is not available in mindtorch_v2."
    )
