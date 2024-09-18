import os
import mindspore
from mindspore import nn
from mindspore.communication import init
from typing import Optional
from mindspore.experimental.optim.lr_scheduler import LRScheduler
from .state import AcceleratorState
from .utils import (
    DistributedType,
    MindFormersPlugin,
    is_mindformers_available,
    wait_for_everyone
)

if is_mindformers_available():
    from .utils import (
        MindFormersEngine,
        MindFormersDummyDataLoader,
        MindFormersDummyScheduler,
        MindFormersOptimizerWrapper,
        MindFormersSchedulerWrapper,
        mindformers_initialize,
        mindformers_prepare_data_loader,
        mindformers_prepare_model_optimizer_scheduler
)


from ..utils import logging

logger = logging.get_logger(__name__)

class Accelerator:
    """
    xx
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
        self.state = AcceleratorState()

        if mindformers_plugin:
            if not is_mindformers_available():
                raise ImportError("MindFormers is not installed. Please install it")
            # The distributed backend required to initialize the communication service.
            # Should be placed before Tensor and Parameter are created.
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

    def prepare(self, *args):
        if self.distributed_type == DistributedType.MINDFORMERS:
            result = self._prepare_mindformers(*args)
    
    def _prepare_mindformers(self, *args):
        
        mindformers_plugin = self.state.mindformers_plugin
        micro_batch_size = None
        # TODO: Check if MindFormers has this functionality.
        if not mindformers_plugin.mindformers_dataset_flag:
            batch_sizes = [obj.batch_size for obj in args if hasattr(obj, "batch_size")]
            if len(batch_sizes) == 0:
                raise ValueError(
                    "You must specify a training or evaluation dataloader in `accelerate.prepare()` when using MindFormers."
                ) 
            
            micro_batch_size = min(batch_sizes) if mindformers_plugin.is_train_batch_min else max(batch_sizes)
            if len(batch_sizes) > 1:
                logger.info(
                    "Since you passed both train and evaluation dataloader, `is_train_batch_min` (here "
                    f"{mindformers_plugin.is_train_batch_min} will decide the `train_batch_size` ({micro_batch_size})."
                )
            pass
        else:
            for obj in args:
                if isinstance(obj, MindFormersDummyDataLoader):
                    micro_batch_size = obj.dataset["micro_batch_size"]
                    break
        
        if micro_batch_size is not None:
            dp_degree = self.num_processes // (mindformers_plugin.tp_degree * mindformers_plugin.pp_degree)
            mindformers_plugin.set_training_args(micro_batch_size, dp_degree)
        else:
            raise ValueError(
                "When you do not pass the dataloader parameter, the `data_parallel_size`, "
                "`micro_batch_size`, and `global_batch_size` mindformers parameters will not be updated."
            )
        
        model = None
        optimizer = None
        scheduler = None
        batch_data = None
        for obj in args:
            if isinstance(obj, mindspore.dataset.GeneratorDataset) and batch_data is None:
                batch_data = obj
            elif isinstance(obj, mindspore.nn.Cell):
                model = obj
            elif isinstance(obj, mindspore.nn.Optimizer):
                optimizer = obj
            elif isinstance(obj, (LRScheduler, MindFormersDummyScheduler)):
                scheduler = obj

        if model is not None:
            mindformers_plugin.set_network_size_args(model, batch_data)
        if optimizer is not None:
            mindformers_plugin.set_optimizer_type(optimizer)
        if scheduler is not None:
            if not isinstance(scheduler, MindFormersDummyScheduler):
                raise ValueError(
                    "You can't use a custom scheduler with MindFormers. Please use the `accelerate.utils.MindFromersDummyScheduler` instead."
                )
            mindformers_plugin.set_scheduler_args(scheduler)

        # initialize mindformers
        mindformers_initialize(self, args_defaults=mindformers_plugin.mindformers_defualt_args)

        (model, optimizer) = mindformers_prepare_model_optimizer_scheduler(self)
        self.wait_for_everyone()

        counter = 0
        result = []
        for obj in args:
            if isinstance(obj, mindspore.dataset.GeneratorDataset):
                result.append(mindformers_prepare_data_loader(self, obj))
                counter += 1
            elif isinstance(obj, MindFormersDummyDataLoader):
                if counter == 0:
                    obj.set_mindformers_data_args()
                    dataloaders = mindformers_prepare_data_loader(self, obj)
                result.append(dataloaders[counter])
                counter += 1
            else:
                result.append(obj)

        if model is not None:
            model = MindFormersEngine(self, model, optimizer, scheduler)
        if optimizer is not None:
            optimizer = MindFormersOptimizerWrapper(optimizer)
        if scheduler is not None:
            scheduler = MindFormersSchedulerWrapper(scheduler)

        for i in range(len(result)):
            if isinstance(result[i], mindspore.nn.Cell):
                result[i] = model
            elif isinstance(result[i], mindspore.nn.Optimizer):
                result[i] = optimizer
            elif isinstance(result[i], MindFormersDummyScheduler):
                result[i] = scheduler
        
        if model is not None:
            self._models.append(model)
            if len(self._models) > 1:
                raise AssertionError(
                    "You can't use same `Accelerator()` instance with multiple models when using MindFormers."
                )
        if optimizer is not None:
            self._optimizers.append(optimizer)
        if scheduler is not None:
            self._schedulers.append(scheduler)

    def backward(self, loss, **kwargs):   
        pass

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

