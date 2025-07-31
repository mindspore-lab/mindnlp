from contextlib import contextmanager
from .profiler import profile, tensorboard_trace_handler
from .scheduler import Schedule as schedule
from .experimental_config import AiCMetrics, ProfilerLevel, _ExperimentalConfig, ExportType
from .common import ProfilerActivity

__all__ = ["profile", "ProfilerActivity", "tensorboard_trace_handler", "schedule",
           "_ExperimentalConfig", "ProfilerLevel", "AiCMetrics", "ExportType"]


@contextmanager
def record_function(name):
    yield