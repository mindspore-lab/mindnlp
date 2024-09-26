"""accelerate dataclasses"""
import enum
import functools
from dataclasses import dataclass, asdict

from mindnlp.accelerate.utils.config import (
    MindformersTrainningConfig,
    MindFormersModelParallelConfig,
    MindForemrsOptimizerConfig,
    MindFormersTransformerConfig
)


class DistributedType(str, enum.Enum):
    """
    Represents a type of distributed environment.

    Values:
        - **MINDFORMERS** -- Using mindformers
    """

    MINDFORMERS = "MINDFORMERS"
    NO = "NO"


@dataclass
class MindFormersPlugin:
    """
    Plugin for MindFormersLM to enable tensor, pipeline, sequence and data parallelism.
    """

    def __post_init__(self):
        self.mindformers_default_args = {
            "trainning_config": {},
            "parallel_config": {},
            "model_config": {},
            "dataset_config": {},
            "optimizer_config": {}
        }

    def set_trainning_args(self):
        trainning_config = MindformersTrainningConfig()
        self.mindformers_default_args["trainning_config"] = asdict(trainning_config)

    def set_optimizer_args(self):
        optimizer_config = MindForemrsOptimizerConfig()
        self.mindformers_default_args["optimizer_config"] = asdict(optimizer_config)

    def set_paralle_args(self):
        parallel_config = MindFormersModelParallelConfig()
        self.mindformers_default_args["parallel_config"] = asdict(parallel_config)

    def set_model_args(self, model, batch_data):
        model_config_type = model.config.model_type.lower()
        MODEL_CONFIGS_TO_MINDFORMERS_PARSERS[model_config_type](self, model, batch_data)

    @property
    def config_dict(self):
        return self.mindformers_default_args

    @property
    def model_type(self):
        model_type = "llama"
        return model_type


MODEL_CONFIGS_TO_MINDFORMERS_PARSERS = {}


def add_model_config_to_mindformers_parser(model_type: str):
    def add_model_config_parser_helper(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            return func(*args, **kwargs)

        MODEL_CONFIGS_TO_MINDFORMERS_PARSERS[model_type] = func
        return wrapper

    return add_model_config_parser_helper


@add_model_config_to_mindformers_parser("llama")
def parse_llama_config(mindformers_plugin, model, batch_data):
    model_config = MindFormersTransformerConfig(
        vocab_size=1200,
        hidden_size=128,
        ffn_hidden_size=512,
        num_layers=2,
        num_heads=8,
    )
    mindformers_plugin.mindformers_default_args["model_config"] = asdict(model_config)
