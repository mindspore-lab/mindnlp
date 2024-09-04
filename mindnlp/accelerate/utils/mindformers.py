import mindspore
from ..optimizer import AcceleratedOptimizer
from ..scheduler import AcceleratedScheduler

def prepare_model_optimizer_scheduler(accelerator):
    accelerator.print("Preparing model, optimizer, and scheduler...")

def prepare_data_loader(accelerator, dataloader):
    accelerator.print("Preparing data loader...")

# optimizer utilities
class MindFormersOptimizerWrapper(AcceleratedOptimizer):

    def __init__(self, optimizer):
        super().__init__(optimizer, device_placement=False, scaler=None)

    def zero_grad(self, set_to_none=None):
        pass  # `model(**batch)` is doing that automatically. Therefore, it's implementation is not needed

    def step(self):
        pass  # `model(**batch)` is doing that automatically. Therefore, it's implementation is not needed


class MindFormersSchedulerWrapper(AcceleratedScheduler):

    def __init__(self, scheduler, optimizers):
        super().__init__(scheduler, optimizers)

    def step(self, *args, **kwargs):
        return  # `model(**batch)` is doing that automatically. Therefore, it's implementation is not needed



class MindFromersDummyDataLoader:
    ...


class MindFormersDummyScheduler:
    ...

# intialize mindformers setup
def initialize(accelerator, extra_args_provider=None, args_defaults={}):
    accelerator.print("Initializing MindFormers...")

class MindFormersEngine(mindspore.nn.Cell):

    def __init__(self, accelerator, model, optimizer, scheduler):
        super().__init__()