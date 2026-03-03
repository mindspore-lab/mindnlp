"""torch.distributed.fsdp.wrap stub - not available in mindtorch_v2."""


def transformer_auto_wrap_policy(*args, **kwargs):
    raise NotImplementedError("transformer_auto_wrap_policy is not available in mindtorch_v2.")


def __getattr__(name):
    raise AttributeError(
        f"module 'torch.distributed.fsdp.wrap' has no attribute '{name}'. "
        "FSDP wrap is not available in mindtorch_v2."
    )
