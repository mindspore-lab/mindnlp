"""NPU (Ascend) device support module for mindtorch_v2.

Provides torch.npu compatibility for libraries that check for NPU
availability via torch_npu conventions.
"""

from ..configs import DEVICE_TARGET, SOC

_npu_available = DEVICE_TARGET == 'Ascend'


def is_available() -> bool:
    """Check if NPU (Ascend) is available."""
    return _npu_available


def device_count() -> int:
    """Return the number of NPU devices available."""
    if not _npu_available:
        return 0
    # For now, assume 1 device; can be extended to query actual device count
    return 1


def current_device() -> int:
    """Return the index of the current NPU device."""
    if not _npu_available:
        raise RuntimeError("NPU is not available")
    return 0


def get_device_name(device=None) -> str:
    """Return the name of the NPU device."""
    if not _npu_available:
        raise RuntimeError("NPU is not available")
    return f"Ascend {SOC}" if SOC else "Ascend NPU"


def get_device_capability(device=None):
    """Return device capability (placeholder for compatibility)."""
    if not _npu_available:
        raise RuntimeError("NPU is not available")
    # Return a tuple similar to CUDA capability format
    return (9, 0)  # Ascend 910


def set_device(device):
    """Set the current NPU device (placeholder for single-device setup)."""
    if not _npu_available:
        raise RuntimeError("NPU is not available")
    # Currently single-device, so this is a no-op
    pass


def synchronize(device=None):
    """Synchronize the NPU device."""
    if not _npu_available:
        raise RuntimeError("NPU is not available")
    # MindSpore handles synchronization internally
    pass


class device:
    """Context manager for NPU device."""
    def __init__(self, device_id):
        self.device_id = device_id

    def __enter__(self):
        return self

    def __exit__(self, *args):
        pass


# Memory management stubs
def memory_allocated(device=None) -> int:
    """Return memory allocated on NPU (placeholder)."""
    return 0


def memory_reserved(device=None) -> int:
    """Return memory reserved on NPU (placeholder)."""
    return 0


def max_memory_allocated(device=None) -> int:
    """Return max memory allocated on NPU (placeholder)."""
    return 0


def empty_cache():
    """Empty NPU cache (placeholder)."""
    pass


def reset_peak_memory_stats(device=None):
    """Reset peak memory stats (placeholder)."""
    pass


def manual_seed(seed):
    """Set the random seed for NPU (placeholder)."""
    import numpy as np
    np.random.seed(seed)


def manual_seed_all(seed):
    """Set the random seed for all NPU devices (placeholder)."""
    manual_seed(seed)


# Flash attention stubs (some libraries check for these)
def npu_fusion_attention(*args, **kwargs):
    """NPU fusion attention (stub - not implemented)."""
    raise NotImplementedError("npu_fusion_attention is not implemented in mindtorch_v2")


# Version info
__version__ = "2.0.0"
