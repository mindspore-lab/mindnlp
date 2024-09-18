import functools

from mindformers import LlamaConfig
from mindnlp.core import nn

from ..optimizer import AcceleratedOptimizer
from ..scheduler import AcceleratedScheduler
from .imports import is_mindformers_available
from ...utils import logging

if is_mindformers_available():
    from mindformers.experimental.model import LlamaForCausalLM
    from mindformers.experimental.distri_cores.config import init_configs_from_yaml
    from mindformers.experimental.distri_cores.create_comm import initialize_model_parallel
    from mindformers.experimental.distri_cores.training import get_model, TrainOneStepCell
    from mindformers.experimental.distri_cores import (
        get_optimizer,
    )

logger = logging.get_logger(__name__)

def prepare_model_optimizer_scheduler(accelerator):
    accelerator.print("Preparing model, optimizer...")

    # load mindformers config
    _CONFIG_PATH = accelerator.state.mindformers_plugin.config_path
    all_config = init_configs_from_yaml(_CONFIG_PATH)
    parallel_config = all_config.parallel_config
    optimizer_config = all_config.optimizer_config

    # get model and optimizer
    model_type = accelerator.state.mindformers_plugin.model_type
    model_provider_func = MODEL_PROVIDER_FUNC[model_type]
    model = get_model(model_provider_func, parallel_config)
    optimizer = get_optimizer(optimizer_config, model.trainable_params(), model)

    return model, optimizer


def prepare_data_loader(accelerator, dataloader):
    accelerator.print("Preparing data loader...")

def prepare_config(accelerator, args_default=None):
    """
    Prepare the configuration of mindformers. Create config yaml locally.
    """
    accelerator.print("Preparing config...")

    if args_default is None:
        args_default = {}



# optimizer utilities
class MindFormersOptimizerWrapper(AcceleratedOptimizer):

    def __init__(self, optimizer):
        super().__init__(optimizer)

    def zero_grad(self, set_to_none=None):
        pass  # `model(**batch)` is doing that automatically. Therefore, it's implementation is not needed

    def step(self):
        pass  # `model(**batch)` is doing that automatically. Therefore, it's implementation is not needed


class MindFormersSchedulerWrapper(AcceleratedScheduler):

    def __init__(self, scheduler, optimizers):
        super().__init__(scheduler, optimizers)

    def step(self, *args, **kwargs):
        return  # `model(**batch)` is doing that automatically. Therefore, it's implementation is not needed



class MindFormersDummyDataLoader:
    ...


class MindFormersDummyScheduler:
    ...

# intialize mindformers setup
def initialize(accelerator, extra_args_provider=None, args_defaults=None):
    if args_defaults is None:
        args_defaults = {}
    accelerator.print("Initializing MindFormers...")

    initialize_model_parallel()


class MindFormersEngine(nn.Module):

    def __init__(self, accelerator, model, optimizer, scheduler):
        super().__init__()
        self.moddel = model
        self.optimizer = optimizer

        _CONFIG_PATH = accelerator.state.mindformers_plugin.config_path
        all_config = init_configs_from_yaml(_CONFIG_PATH)
        training_config = all_config.training_config
        model_config = all_config.model_config

        self.train_one_step = TrainOneStepCell(model, optimizer, training_config, model_config)

    def forward(self, **batch_data):
        if self.model.training:
            loss, is_finite, loss_scale, learning_rate = self.train_one_step_cell(**batch_data)
            return loss
        else:
            raise RuntimeError("Eval mode is not implemented")


MODEL_PROVIDER_FUNC = {}

def add_model_provider_func(model_type: str):
    def add_model_provier_func_parser_helper(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            return func(*args, **kwargs)

        MODEL_PROVIDER_FUNC[model_type] = func
        return wrapper

    return add_model_provier_func_parser_helper

@add_model_provider_func("llama")
def provider_llama(pre_process=True, post_process=True):

    # load model config, then create model in mindformers
    # TODO: how to get model config
    config: LlamaConfig = LlamaConfig()
    model = LlamaForCausalLM(config=config, pre_process=pre_process, post_process=post_process)
    return model