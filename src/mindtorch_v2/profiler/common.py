from enum import Enum


class ProfilerActivity(Enum):
    """Profiler activity enum for mindtorch_v2 profiler."""

    CPU = "CPU"
    NPU = "NPU"
    CUDA = "NPU"
    GPU = "NPU"
