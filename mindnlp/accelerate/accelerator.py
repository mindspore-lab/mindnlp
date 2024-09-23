"""accelerate"""
import os
from contextlib import contextmanager
from typing import Optional

import mindspore
from mindspore import nn
from mindspore.communication import init

from .state import AcceleratorState
from .utils import (
    DistributedType,
    MindFormersPlugin,
    is_mindformers_available,
    wait_for_everyone
)
from ..utils import logging

if is_mindformers_available():
    from .utils import (
        MindFormersEngine,
        MindFormersOptimizerWrapper,
        MindFormersSchedulerWrapper,
        mindformers_initialize,
        mindformers_prepare_data_loader,
        mindformers_prepare_model_optimizer_scheduler
    )

logger = logging.get_logger(__name__)


class Accelerator:
    """
    Creates an instance of an accelerator for distributed training (on Ascend)

    Args:
        mindformers_plugin (`MindFormersPlugin`, *optional*):
            This argument is optional and can be configured directly using *accelerate config*
    """

    def __init__(
            self,
            mindformers_plugin: Optional[MindFormersPlugin] = None,
    ):
        # init mindformers_plugin from env variables
        if mindformers_plugin is None:
            mindformers_plugin = (
                MindFormersPlugin() if os.environ.get("ACCELERATE_USE_MINDFORMERS", "false") == "true" else None
            )
        else:
            os.environ["ACCELERATE_USE_MINDFORMERS"] = "true"
        self.state = AcceleratorState(mindformers_plugin=mindformers_plugin)

        if mindformers_plugin:
            if not is_mindformers_available():
                raise ImportError("MindFormers is not installed. Please install it")
            # The distributed backend required to initialize the communication service.
            # Should be placed before Tensor and Parameter are created.
            mindspore.set_context(mode=mindspore.PYNATIVE_MODE, device_target="Ascend")
            init()

        # Internal references to the training objects
        self._optimizers = []
        self._models = []
        self._schedulers = []
        self._dataloaders = []
        self._custom_objects = []

    @property
    def use_distributed(self):
        return self.state.use_distributed

    @property
    def distributed_type(self):
        return self.state.distributed_type

    @property
    def num_processes(self):
        return self.state.num_processes

    @property
    def process_index(self):
        return self.state.process_index

    @property
    def is_main_process(self):
        """True for one process only."""
        return self.state.is_main_process

    def prepare(self, *args):
        """
        Prepare all objects passed in `args` for  distributed training. Then return them in the same order.
        Args:
            *args (list of objects):
                Any of the following type of objects:

                - `mindspore.dataset.GeneratorDataset`: MindSpore Dataloader
                - `mindspore.nn.Cell`: MindSpore Module
                - `mindspore.nn.optim.Optimizer`: MindSpore Optimizer
                - `mindspore.nn.learning_rate_schedule.LearningRateSchedule`: MindSpore Scheduler

        Returns: Prepared objects in the same order.

        """
        result = []

        # Only support mindsormers now
        if self.distributed_type == DistributedType.MINDFORMERS:
            result = self._prepare_mindformers(*args)

        return result

    def _prepare_mindformers(self, *args):
        mindformers_plugin = self.state.mindformers_plugin

        model = None
        optimizer = None
        scheduler = None
        batch_data = None
        for obj in args:
            if isinstance(obj, mindspore.dataset.GeneratorDataset) and batch_data is None:
                batch_data = obj
            elif isinstance(obj, mindspore.nn.Cell):
                model = obj
            elif isinstance(obj, mindspore.nn.optim.Optimizer):
                optimizer = obj
            elif isinstance(obj, mindspore.nn.learning_rate_schedule.LearningRateSchedule):
                scheduler = obj

        # Config is not correct now
        if model is not None:
            mindformers_plugin.set_model_args(model, batch_data)
        if optimizer is not None:
            mindformers_plugin.set_optimizer_args(optimizer)
        if scheduler is not None:
            mindformers_plugin.set_paralle_args(scheduler)
        mindformers_plugin.set_training_args()

        # initialize mindformers
        mindformers_initialize(self, args_defaults=mindformers_plugin.mindformers_defualt_args)

        (model, optimizer, scheduler) = mindformers_prepare_model_optimizer_scheduler(self)
        self.wait_for_everyone()

        counter = 0
        result = []
        for obj in args:
            if isinstance(obj, mindspore.dataset.GeneratorDataset):
                data_loader = mindformers_prepare_data_loader(self, obj)
                result.append(data_loader)
                counter += 1
            else:
                result.append(obj)

        if model is not None:
            model = MindFormersEngine(self, model, optimizer)
        if optimizer is not None:
            optimizer = MindFormersOptimizerWrapper(optimizer)
        if scheduler is not None:
            scheduler = MindFormersSchedulerWrapper(scheduler, optimizer)

        for i in range(len(result)):
            if isinstance(result[i], nn.Cell):
                result[i] = model
            elif isinstance(result[i], nn.Optimizer):
                result[i] = optimizer
            elif isinstance(result[i], nn.learning_rate_schedule.LearningRateSchedule):
                result[i] = scheduler

        if model is not None:
            self._models.append(model)
            if len(self._models) > 1:
                raise AssertionError(
                    "You can't use same `Accelerator()` instance with multiple models when using MindFormers."
                )
        if optimizer is not None:
            self._optimizers.append(optimizer)

        return tuple(result)

    def backward(self, loss, **kwargs):
        pass

    @contextmanager
    def main_process_first(self):
        """
        Lets the main process go first inside a with block.

        The other processes will enter the with block after the main process exits.

        Example:

        ```python
        >>> from accelerate import Accelerator

        >>> accelerator = Accelerator()
        >>> with accelerator.main_process_first():
        ...     # This will be printed first by process 0 then in a seemingly
        ...     # random order by the other processes.
        ...     print(f"This will be printed by process {accelerator.process_index}")
        ```
        """
        with self.state.main_process_first():
            yield

    def wait_for_everyone(self):
        """
        Will stop the execution of the current process until every other process has reached that point (so this does
        nothing when the script is only run in one process). Useful to do before saving a model.

        Example:

        ```python
        >>> # Assuming two GPU processes
        >>> import time
        >>> from accelerate import Accelerator

        >>> accelerator = Accelerator()
        >>> if accelerator.is_main_process:
        ...     time.sleep(2)
        >>> else:
        ...     print("I'm waiting for the main process to finish its sleep...")
        >>> accelerator.wait_for_everyone()
        >>> # Should print on every process at the same time
        >>> print("Everyone is here")
        ```
        """
        wait_for_everyone()

    def print(self, *args, **kwargs):
        """
        Drop in replacement of `print()` to only print once per server.

        Example:

        ```python
        >>> from accelerate import Accelerator

        >>> accelerator = Accelerator()
        >>> accelerator.print("Hello world!")
        ```
        """
        self.state.print(*args, **kwargs)
