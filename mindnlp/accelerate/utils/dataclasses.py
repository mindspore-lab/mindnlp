import enum
import os
import functools
from .environment import str_to_bool
from dataclasses import dataclass, field


class DistributedType(str, enum.Enum):
    """
    Represents a type of distributed environment.

    Values:
        - **MINDFORMERS** -- Using mindformers
    """

    MINDFORMERS = "MINDFORMERS"


@dataclass
class MindFormersPlugin:
    """
    Plugin for MindFormersLM to enable tensor, pipeline, sequence and data parallelism.
    """

    tp_degree: int = field(default=None, metadata={"help": "tensor parallelism degree."})
    pp_degree: int = field(default=None, metadata={"help": "pipeline parallelism degree."})
    num_micro_batches: int = field(default=None, metadata={"help": "number of micro-batches."})

    def __post_init__(self):
        prefix = "MINDFORMERS_"
        if self.tp_degree is None:
            self.tp_degree = int(os.environ.get(prefix + "TP_DEGREE", 1))
        if self.pp_degree is None:
            self.pp_degree = int(os.environ.get(prefix + "PP_DEGREE", 1))
        if self.num_micro_batches is None:
            self.num_micro_batches = int(os.environ.get(prefix + "NUM_MICRO_BATCHES", 1))
        if self.gradient_clipping is None:
            self.gradient_clipping = float(os.environ.get(prefix + "GRADIENT_CLIPPING", 1.0))
        if self.recompute_activations is None:
            self.recompute_activations = str_to_bool(os.environ.get(prefix + "RECOMPUTE_ACTIVATIONS", "False")) == 1
        if self.use_distributed_optimizer is None:
            self.use_distributed_optimizer = (
                str_to_bool(os.environ.get(prefix + "USE_DISTRIBUTED_OPTIMIZER", "False")) == 1
            )
        if self.sequence_parallelism is None:
            self.sequence_parallelism = str_to_bool(os.environ.get(prefix + "SEQUENCE_PARALLELISM", "False")) == 1

        if self.pp_degree > 1 or self.use_distributed_optimizer:
            self.DDP_impl = "local"
        else:
            self.DDP_impl = "torch"

        if self.consumed_samples is not None:
            if len(self.consumed_samples) == 1:
                self.consumed_samples.extend([0, 0])
            elif len(self.consumed_samples) == 2:
                self.consumed_samples.append(0)

        self.mindformers_default_args = {
            "tensor_model_parallel_size": self.tp_degree,
            "pipeline_model_parallel_size": self.pp_degree,
            "pipeline_model_parallel_split_rank": self.pipeline_model_parallel_split_rank,
            "num_layers_per_virtual_pipeline_stage": self.num_layers_per_virtual_pipeline_stage,
            "DDP_impl": self.DDP_impl,
            "use_distributed_optimizer": self.use_distributed_optimizer,
            "sequence_parallel": self.sequence_parallelism,
            "clip_grad": self.gradient_clipping,
            "num_micro_batches": self.num_micro_batches,
            "consumed_samples": self.consumed_samples,
            "no_wd_decay_cond": self.no_wd_decay_cond,
            "scale_lr_cond": self.scale_lr_cond,
            "lr_mult": self.lr_mult,
            "megatron_dataset_flag": self.megatron_dataset_flag, # TODO whether mindformers support
            "eval_iters": self.eval_iters,
            "eval_interval": self.eval_interval,
        }
        if self.recompute_activations:
            self.mindformers_default_args["recompute_granularity"] = "selective"
        if self.tensorboard_dir is not None:
            self.mindformers_default_args["tensorboard_dir"] = self.tensorboard_dir
            if self.set_all_logging_options:
                self.set_tensorboard_logging_options()
        if self.other_mindformers_args is not None:
            self.mindformers_default_args.update(self.other_mindformers_args)

        def set_training_args(self, micro_batch_size, dp_degree):
            return
        
        def set_optimizer_type(self, optimizer):
            return

        def set_scheduler_args(self, scheduler):
            return

MODEL_CONFIGS_TO_MEGATRON_PARSERS = {}

def add_model_config_to_megatron_parser(model_type: str):
    def add_model_config_parser_helper(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            return func(*args, **kwargs)

        MODEL_CONFIGS_TO_MEGATRON_PARSERS[model_type] = func
        return wrapper

    return add_model_config_parser_helper