from enum import Enum


class ProfilerActivity(Enum):
    """Profiler activity enum for mindtorch_v2 profiler."""

    CPU = "CPU"
    NPU = "NPU"
    CUDA = "NPU"
    GPU = "NPU"


class ProfilerAction(Enum):
    """Profiler schedule action enum."""

    NONE = 0
    WARMUP = 1
    RECORD = 2
    RECORD_AND_SAVE = 3
