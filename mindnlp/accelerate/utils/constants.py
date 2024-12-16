"""constants"""
import os
import mindspore
import numpy
from .dataclasses import DistributedType


_random_seed = numpy.random.randint(1000)


def _prepare_data_parallel_native_minspore():
    # initialize data parallel hcc backend for data_loader and Trainer API
    mindspore.set_auto_parallel_context(parallel_mode=mindspore.ParallelMode.DATA_PARALLEL, gradients_mean=True)
    mindspore.communication.init()
    mindspore.set_seed(_random_seed)


def detect_accelerate_distributed_type():
    """
    detect distributed_type

    Returns:
        _type_: According to the factors such as the available parallel software and hardware environment of the current system and the user-specified parallel scheme,
          the optimal parallel strategy is comprehensively decided in different situations.
    """
    if os.environ.get("MULTI_NPU", None) == "true":
        _prepare_data_parallel_native_minspore()
        return DistributedType.MULTI_NPU
    if os.environ.get("ACCELERATE_USE_MINDFORMERS", "false") == "true":
        return DistributedType.MINDFORMERS
    else:
        return DistributedType.NO

accelerate_distributed_type = detect_accelerate_distributed_type()