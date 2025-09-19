from enum import Enum

class ProfilerActivity(Enum):
    """The profiler activity enum."""

    NPU = "NPU"
    GPU = "GPU"
    CPU = "CPU"
    CUDA = "GPU"