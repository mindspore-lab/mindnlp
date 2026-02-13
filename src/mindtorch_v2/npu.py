from ._backends.npu import is_available
from ._backends.npu.runtime import device_count

__all__ = ["is_available", "device_count"]
