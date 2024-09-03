import mindspore

def prepare_model_optimizer_scheduler(accelerator):
    pass

def prepare_data_loader(accelerator, dataloader):
    pass

# optimizer utilities
class MindFormersOptimizerWrapper:
    ...

class MindFormersSchedulerWrapper:
    ...

class MindFromersDummyDataLoader:
    ...


class MindFormersDummyScheduler:
    ...

# intialize mindformers setup
def initialize(accelerator, extra_args_provider=None, args_defaults={}):
    pass

class MindFormersEngine(mindspore.nn.Cell):

    def __init__(self, accelerator, model, optimizer, scheduler):
        super().__init__()