"""accelerate mindformers core."""
import functools

from mindspore import nn, Tensor

from ..optimizer import AcceleratedOptimizer
from ..scheduler import AcceleratedScheduler
from .imports import is_mindformers_available
from ...utils import logging

if is_mindformers_available():
    try:
        from mindformers.experimental.model import LlamaForCausalLM  # pylint: disable=import-error
        from mindformers.experimental.parallel_core.pynative.config import init_configs_from_dict  # pylint: disable=import-error
        from mindformers.experimental.parallel_core.pynative.training import get_model, TrainOneStepCell  # pylint: disable=import-error
        from mindformers.experimental.parallel_core.pynative.parallel_state import initialize_model_parallel  # pylint: disable=import-error
        from mindformers.experimental.parallel_core.pynative import get_optimizer  # pylint: disable=import-error
    except Exception as e:
        raise ValueError('cannot found `mindformers.experimental`, please install dev version by\n'
                         '`pip install git+https://gitee.com/mindspore/mindformers` \n'
                         'or remove mindformers by \n'
                         '`pip uninstall mindformers`')

logger = logging.get_logger(__name__)

_GLOBAL_CONFIG_DICT: dict


def prepare_model_optimizer_scheduler(accelerator):
    """
    Prepare mindformers model and optimizer

    Args:
        accelerator: accelerator

    Returns: model, optimizer

    """
    accelerator.print("Preparing model, optimizer...")

    # load mindformers config
    _CONFIG_DICT = accelerator.state.mindformers_plugin.config_dict
    all_config = init_configs_from_dict(_CONFIG_DICT)
    model_config = all_config.model_config
    parallel_config = all_config.parallel_config
    optimizer_config = all_config.optimizer_config

    # get model and optimizer
    model_type = accelerator.state.mindformers_plugin.model_type
    model_provider_func = MODEL_PROVIDER_FUNC[model_type](model_config, True, True)
    model = get_model(model_provider_func, parallel_config)
    optimizer = get_optimizer(optimizer_config, model.trainable_params(), model)
    scheduler = None

    return model, optimizer, scheduler


def prepare_data_loader(accelerator, dataloader):
    """
    Prepare dataloader in mindformers

    Args:
        accelerator: accelerator
        dataloader: original dataloader

    Returns: dataloader

    """
    accelerator.print("Preparing data loader...")

    all_config = init_configs_from_dict(_GLOBAL_CONFIG_DICT)
    dataset_config = all_config.dataset_config

    # calculate global batch size
    global_batch_size = dataset_config.batch_size * dataset_config.micro_batch_num
    batch_dataloader = dataloader.batch(global_batch_size)

    return batch_dataloader


# optimizer utilities
class MindFormersOptimizerWrapper(AcceleratedOptimizer):
    #
    # def __init__(self, optimizer):
    #     super().__init__(optimizer)

    def zero_grad(self, set_to_none=None):
        pass  # `model(**batch)` is doing that automatically. Therefore, it's implementation is not needed

    def step(self):
        pass  # `model(**batch)` is doing that automatically. Therefore, it's implementation is not needed


class MindFormersSchedulerWrapper(AcceleratedScheduler):

    # def __init__(self, scheduler, optimizers):
    #     super().__init__(scheduler, optimizers)

    def step(self, *args, **kwargs):
        return  # `model(**batch)` is doing that automatically. Therefore, it's implementation is not needed


class MindFormersDummyDataLoader:
    ...


class MindFormersDummyScheduler:
    ...


def initialize(accelerator, args_defaults=None):
    """
    Intialize mindformers setup

    Args:
        accelerator: accelerator
        args_defaults: args mindformers needed

    """
    if args_defaults is None:
        args_defaults = {}
    accelerator.print("Initializing MindFormers...")

    global _GLOBAL_CONFIG_DICT
    if _GLOBAL_CONFIG_DICT is None:
        _GLOBAL_CONFIG_DICT = args_defaults

    all_config = init_configs_from_dict(_GLOBAL_CONFIG_DICT)
    parallel_config = all_config.parallel_config

    initialize_model_parallel(
        tensor_model_parallel_size=parallel_config.tensor_parallel,
        pipeline_model_parallel_size=parallel_config.pipeline_stage,
        virtual_pipeline_model_parallel_size=parallel_config.virtual_pipeline_model_parallel_size,
        context_parallel=parallel_config.context_parallel,
        expert_model_parallel_size=parallel_config.expert_parallel
    )


class MindFormersEngine(nn.Cell):
    """
    MindFormers model wrapper

    Args:
        accelerator (:class:`~accelerate.Accelerator`): The accelerator object to use.
        model: MindFormers model
        optimizer: MindFormers optimizer
    """

    def __init__(self, accelerator, model, optimizer):
        super().__init__()
        self.moddel = model
        self.optimizer = optimizer

        _CONFIG_DICT = accelerator.state.mindformers_plugin.config_dict
        all_config = init_configs_from_dict(_CONFIG_DICT)
        training_config = all_config.training_config
        model_config = all_config.model_config

        self.train_one_step = TrainOneStepCell(model, optimizer, training_config, model_config)

    def construct(self, tuple_data):
        if self.model.training:
            self.train_one_step.set_train(True)
            set_input_data = [
                Tensor(shape=(None,) * len(input_data.shape), dtype=input_data.dtype) for input_data in tuple_data
            ]
            self.train_one_step_cell.set_inputs(*set_input_data)
            self.train_one_step.set_inputs()
            loss, is_finite, loss_scale, learning_rate = self.train_one_step_cell(**tuple_data)
            return loss
        else:
            self.train_one_step.set_train(False)
            self.train_one_step.forward_backward_func(forward_only=True, **tuple_data)

            self.train_one_step.set_train(False)


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
def provider_llama(config, pre_process=True, post_process=True):
    # load model config, then create model in mindformers
    def model_provider(inner_pre_process=pre_process, inner_post_process=post_process):
        model = LlamaForCausalLM(config=config, pre_process=pre_process, post_process=post_process)
        return model

    return model_provider
