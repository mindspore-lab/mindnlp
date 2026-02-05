"""Stub for torch.profiler module."""

from enum import Enum
from contextlib import contextmanager


class ProfilerActivity(Enum):
    """Profiler activity types."""
    CPU = 0
    CUDA = 1
    XPU = 2
    MTIA = 3


class ProfilerAction(Enum):
    """Profiler actions."""
    NONE = 0
    WARMUP = 1
    RECORD = 2
    RECORD_AND_SAVE = 3


class schedule:
    """Profiler schedule."""
    def __init__(self, wait=0, warmup=0, active=0, repeat=0, skip_first=0):
        self.wait = wait
        self.warmup = warmup
        self.active = active
        self.repeat = repeat
        self.skip_first = skip_first

    def __call__(self, step):
        return ProfilerAction.NONE


def tensorboard_trace_handler(dir_name, worker_name=None, use_gzip=False):
    """Tensorboard trace handler - stub."""
    def handler(prof):
        pass
    return handler


class profile:
    """Profiler context manager - stub implementation."""
    def __init__(self, activities=None, schedule=None, on_trace_ready=None,
                 record_shapes=False, profile_memory=False, with_stack=False,
                 with_flops=False, with_modules=False, experimental_config=None,
                 use_cuda=None):
        self.activities = activities
        self.schedule_fn = schedule
        self.on_trace_ready = on_trace_ready
        self.record_shapes = record_shapes
        self.profile_memory = profile_memory
        self.with_stack = with_stack
        self.with_flops = with_flops
        self.with_modules = with_modules

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        pass

    def step(self):
        pass

    def start(self):
        pass

    def stop(self):
        pass

    def export_chrome_trace(self, path):
        pass

    def key_averages(self, group_by_input_shape=False, group_by_stack_n=0):
        return _KeyAverages()


class _KeyAverages:
    """Key averages stub."""
    def table(self, sort_by=None, row_limit=100, max_src_column_width=75,
              max_name_column_width=55, max_shapes_column_width=80,
              header=None, top_level_events_only=False):
        return "Profiler not available with mindtorch_v2"


class record_function:
    """Record function context manager - stub."""
    def __init__(self, name, args=None):
        self.name = name
        self.args = args

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        pass


# Export key classes
__all__ = [
    'ProfilerActivity',
    'ProfilerAction',
    'schedule',
    'tensorboard_trace_handler',
    'profile',
    'record_function',
]
