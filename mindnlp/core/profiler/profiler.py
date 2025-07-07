from mindspore.profiler import ProfilerActivity
from typing import Optional, Iterable, Callable, Any

from mindspore import Profiler as Profiler
try:
    from mindspore.profiler import tensor_board_trace_handler
except:
    from mindspore.profiler import tensorboard_trace_handler
    tensor_board_trace_handler = None

from mindspore.profiler.schedule import ProfilerAction
from .scheduler import Schedule
from .experimental_config import _ExperimentalConfig


if tensor_board_trace_handler is not None:
    def tensorboard_trace_handler(dir_name: str = None, worker_name: str = None,
                                analyse_flag: bool = True, async_mode: bool = False):
        def voidfunc():
            pass
        if analyse_flag:
            return (tensor_board_trace_handler, dir_name)
        else:
            return (voidfunc, dir_name)


class profile:
    def __init__(
        self,
        *,
        activities: Optional[Iterable[ProfilerActivity]] = None,
        schedule: Optional[Schedule] = None,
        on_trace_ready: Optional[tuple] = None,
        record_shapes: bool = False,
        profile_memory: bool = False,
        with_stack: bool = False,
        with_flops: bool = False,
        with_modules: bool = False,
        experimental_config: Optional[_ExperimentalConfig] = None,
        # deprecated:
        use_cuda: Optional[bool] = None,
    ):
        if on_trace_ready is not None:
            if isinstance(on_trace_ready, tuple):
                (on_trace_ready, dir_name) = on_trace_ready
            else:
                dir_name = ".data"
        else:
            dir_name = ".data"

        self.profiler = Profiler(
            start_profile = False,
            output_path = dir_name,
            profiler_level = experimental_config._profiler_level,
            activities = activities,
            schedule = schedule.scheduler,
            on_trace_ready = on_trace_ready,
            profile_memory = profile_memory,
            aicore_metrics = experimental_config._aic_metrics,
            with_stack = with_stack,
            data_simplification = experimental_config._data_simplification,
            l2_cache = experimental_config._l2_cache,
            mstx = experimental_config._msprof_tx
        )
    
    def start(self):
        self.profiler.start()

    def stop(self):
        self.profiler.stop()

    def step(self):
        self.profiler.step()

def analyse(profiler_path: str, max_process_number:int):
    Profiler.analyse(profiler_path)
