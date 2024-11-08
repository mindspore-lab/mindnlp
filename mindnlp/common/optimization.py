# coding=utf-8
# Copyright 2018 The Google AI Language Team Authors and The HuggingFace Inc. team.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""MindSpore optimization for BERT model."""

import math
from functools import partial
from typing import Optional, Union

from mindspore import ops
from mindnlp.core.optim import Optimizer
from mindnlp.core.optim.lr_scheduler import LRScheduler, LambdaLR, ReduceLROnPlateau

from ..engine.utils import SchedulerType
from ..utils import logging


logger = logging.get_logger(__name__)

class LayerWiseDummyOptimizer(Optimizer):
    """
    For Layer-wise optimizers such as GaLoRE optimizer, the optimization
    step is already done through the post gradient hooks. Therefore
    the trick is to create a dummy optimizer that can take arbitrary
    args and kwargs and return a no-op during training.

    Initial idea from @hiyouga in LLaMA-Factory:
    https://github.com/hiyouga/LLaMA-Factory/commit/8664262cde3919e10eaecbd66e8c5d356856362e#diff-ebe08ab14496dfb9e06075f0fdd36799ef6d1535cc4dd4715b74c4e3e06fe3ba
    """
    def __init__(self, *args, optimizer_dict=None, **kwargs):
        r"""
        __init__
        
        Args:
            self (object): The instance of the class.
            *args: Variable length argument list.
            optimizer_dict (dict, optional): A dictionary containing optimizer settings. Defaults to None.
            **kwargs: Arbitrary keyword arguments. Here, it is used to extract the learning rate ('lr') from the keyword arguments.
        
        Returns:
            None: This method does not return any value.
        
        Raises:
            None
        """
        dummy_tensor = ops.randn(1, 1)
        self.optimizer_dict = optimizer_dict
        super().__init__([dummy_tensor], {"lr": kwargs.get("lr", 1e-03)})


class LayerWiseDummyScheduler(LRScheduler):
    """
    For Layer-wise optimizers such as GaLoRE optimizer, the optimization and scheduling step
    are already done through the post gradient hooks. Therefore
    the trick is to create a dummy scheduler that can take arbitrary
    args and kwargs and return a no-op during training.
    """
    def __init__(self, *args, **kwargs):
        r"""
        Initializes a new instance of the LayerWiseDummyScheduler class.
        
        Args:
            self: The instance of the LayerWiseDummyScheduler class.
        
        Returns:
            None. This method does not return any value.
        
        Raises:
            N/A. This method does not raise any exceptions.
        """
        optimizer = LayerWiseDummyOptimizer()
        last_epoch = -1
        super().__init__(optimizer, last_epoch)

    def get_lr(self):
        r"""
        Get the learning rates of all parameter groups in the optimizer.
        
        Args:
            self (LayerWiseDummyScheduler): The object instance.
        
        Returns:
            list: A list of learning rates for each parameter group in the optimizer.
        
        Raises:
            None.
        
        '''
        
        This docstring describes the 'get_lr' method in the 'LayerWiseDummyScheduler' class. The method takes one parameter, 'self', which is an instance of the 'LayerWiseDummyScheduler' class. The purpose of
this method is to retrieve the learning rates of all parameter groups in the optimizer.
        
        The method returns a list, where each element represents the learning rate of a parameter group in the optimizer. The type of the return value is a list.
        
        No exceptions are raised by this method.
        """
        return [group["lr"] for group in self.optimizer.param_groups]

    def _get_closed_form_lr(self):
        r"""
        This method _get_closed_form_lr in the class LayerWiseDummyScheduler computes the closed form learning rate (LR).
        
        Args:
            self: An instance of the LayerWiseDummyScheduler class.
        
        Returns:
            None. The method returns the computed base learning rates.
        
        Raises:
            This method does not raise any exceptions.
        """
        return self.base_lrs


def _get_constant_lambda(_=None):
    r"""
    This function returns a constant lambda value of 1.
    
    Args:
        _: This parameter is not used and can be ignored.
    
    Returns:
        None: This function does not return any value.
    
    Raises:
        None: This function does not raise any exceptions.
    """
    return 1


def get_constant_schedule(optimizer: Optimizer, last_epoch: int = -1):
    """
    Create a schedule with a constant learning rate, using the learning rate set in optimizer.

    Args:
        optimizer ([`~torch.optim.Optimizer`]):
            The optimizer for which to schedule the learning rate.
        last_epoch (`int`, *optional*, defaults to -1):
            The index of the last epoch when resuming training.

    Return:
        `torch.optim.lr_scheduler.LambdaLR` with the appropriate schedule.
    """
    return LambdaLR(optimizer, _get_constant_lambda, last_epoch=last_epoch)


def get_reduce_on_plateau_schedule(optimizer: Optimizer, **kwargs):
    """
    Create a schedule with a constant learning rate that decreases when a metric has stopped improving.

    Args:
        optimizer ([`~torch.optim.Optimizer`]):
            The optimizer for which to schedule the learning rate.
        kwargs (`dict`, *optional*):
            Extra parameters to be passed to the scheduler. See `torch.optim.lr_scheduler.ReduceLROnPlateau`
            for possible parameters.

    Return:
        `torch.optim.lr_scheduler.ReduceLROnPlateau` with the appropriate schedule.
    """
    return ReduceLROnPlateau(optimizer, **kwargs)


def _get_constant_schedule_with_warmup_lr_lambda(current_step: int, *, num_warmup_steps: int):
    r"""
    Args:
        current_step (int): The current step in the learning rate schedule.
        num_warmup_steps (int): The number of warmup steps to gradually increase the learning rate.
    
    Returns:
        None: This function does not return any value.
    
    Raises:
        None.
    """
    if current_step < num_warmup_steps:
        return float(current_step) / float(max(1.0, num_warmup_steps))
    return 1.0


def get_constant_schedule_with_warmup(optimizer: Optimizer, num_warmup_steps: int, last_epoch: int = -1):
    """
    Create a schedule with a constant learning rate preceded by a warmup period during which the learning rate
    increases linearly between 0 and the initial lr set in the optimizer.

    Args:
        optimizer ([`~torch.optim.Optimizer`]):
            The optimizer for which to schedule the learning rate.
        num_warmup_steps (`int`):
            The number of steps for the warmup phase.
        last_epoch (`int`, *optional*, defaults to -1):
            The index of the last epoch when resuming training.

    Return:
        `torch.optim.lr_scheduler.LambdaLR` with the appropriate schedule.
    """
    lr_lambda = partial(_get_constant_schedule_with_warmup_lr_lambda, num_warmup_steps=num_warmup_steps)
    return LambdaLR(optimizer, lr_lambda, last_epoch=last_epoch)


def _get_linear_schedule_with_warmup_lr_lambda(current_step: int, *, num_warmup_steps: int, num_training_steps: int):
    r"""
    Args:
        current_step (int): The current step in the training process.
        num_warmup_steps (int): The number of warm-up steps at the beginning of the training.
        num_training_steps (int): The total number of training steps.
    
    Returns:
        None. The function does not return a value, but it updates the learning rate schedule.
    
    Raises:
        None.
    """
    if current_step < num_warmup_steps:
        return float(current_step) / float(max(1, num_warmup_steps))
    return max(0.0, float(num_training_steps - current_step) / float(max(1, num_training_steps - num_warmup_steps)))


def get_linear_schedule_with_warmup(optimizer, num_warmup_steps, num_training_steps, last_epoch=-1):
    """
    Create a schedule with a learning rate that decreases linearly from the initial lr set in the optimizer to 0, after
    a warmup period during which it increases linearly from 0 to the initial lr set in the optimizer.

    Args:
        optimizer ([`~torch.optim.Optimizer`]):
            The optimizer for which to schedule the learning rate.
        num_warmup_steps (`int`):
            The number of steps for the warmup phase.
        num_training_steps (`int`):
            The total number of training steps.
        last_epoch (`int`, *optional*, defaults to -1):
            The index of the last epoch when resuming training.

    Return:
        `torch.optim.lr_scheduler.LambdaLR` with the appropriate schedule.
    """
    lr_lambda = partial(
        _get_linear_schedule_with_warmup_lr_lambda,
        num_warmup_steps=num_warmup_steps,
        num_training_steps=num_training_steps,
    )
    return LambdaLR(optimizer, lr_lambda, last_epoch)


def _get_cosine_schedule_with_warmup_lr_lambda(
    current_step: int, *, num_warmup_steps: int, num_training_steps: int, num_cycles: float
):
    """
    Args:
        current_step (int): The current step in the training process.
        num_warmup_steps (int): The number of warmup steps before the learning rate reaches its maximum value.
        num_training_steps (int): The total number of training steps.
        num_cycles (float): The number of cosine cycles for the schedule.
    
    Returns:
        None: This function does not return any value.
    
    Raises:
        None
    """
    if current_step < num_warmup_steps:
        return float(current_step) / float(max(1, num_warmup_steps))
    progress = float(current_step - num_warmup_steps) / float(max(1, num_training_steps - num_warmup_steps))
    return max(0.0, 0.5 * (1.0 + math.cos(math.pi * float(num_cycles) * 2.0 * progress)))


def get_cosine_schedule_with_warmup(
    optimizer: Optimizer, num_warmup_steps: int, num_training_steps: int, num_cycles: float = 0.5, last_epoch: int = -1
):
    """
    Create a schedule with a learning rate that decreases following the values of the cosine function between the
    initial lr set in the optimizer to 0, after a warmup period during which it increases linearly between 0 and the
    initial lr set in the optimizer.

    Args:
        optimizer ([`~torch.optim.Optimizer`]):
            The optimizer for which to schedule the learning rate.
        num_warmup_steps (`int`):
            The number of steps for the warmup phase.
        num_training_steps (`int`):
            The total number of training steps.
        num_cycles (`float`, *optional*, defaults to 0.5):
            The number of waves in the cosine schedule (the defaults is to just decrease from the max value to 0
            following a half-cosine).
        last_epoch (`int`, *optional*, defaults to -1):
            The index of the last epoch when resuming training.

    Return:
        `torch.optim.lr_scheduler.LambdaLR` with the appropriate schedule.
    """
    lr_lambda = partial(
        _get_cosine_schedule_with_warmup_lr_lambda,
        num_warmup_steps=num_warmup_steps,
        num_training_steps=num_training_steps,
        num_cycles=num_cycles,
    )
    return LambdaLR(optimizer, lr_lambda, last_epoch)


def _get_cosine_with_hard_restarts_schedule_with_warmup_lr_lambda(
    current_step: int, *, num_warmup_steps: int, num_training_steps: int, num_cycles: int
):
    r"""
    Calculates the learning rate lambda value for a cosine schedule with hard restarts and warm-up.
    
    Args:
        current_step (int): The current step in the training process.
    
    Returns:
        float: The learning rate lambda value.
    
    Raises:
        None.
    
    This function calculates the learning rate lambda value based on the current step in the training process. It uses a cosine schedule with hard restarts and warm-up. The learning rate lambda value is used
to adjust the learning rate during training.
    
    The function takes the following parameters:
    - current_step: The current step in the training process. It should be an integer.
    
    The function returns the learning rate lambda value as a float. The lambda value is used to adjust the learning rate for the current step.
    
    No exceptions are raised by this function.
    """
    if current_step < num_warmup_steps:
        return float(current_step) / float(max(1, num_warmup_steps))
    progress = float(current_step - num_warmup_steps) / float(max(1, num_training_steps - num_warmup_steps))
    if progress >= 1.0:
        return 0.0
    return max(0.0, 0.5 * (1.0 + math.cos(math.pi * ((float(num_cycles) * progress) % 1.0))))


def get_cosine_with_hard_restarts_schedule_with_warmup(
    optimizer: Optimizer, num_warmup_steps: int, num_training_steps: int, num_cycles: int = 1, last_epoch: int = -1
):
    """
    Create a schedule with a learning rate that decreases following the values of the cosine function between the
    initial lr set in the optimizer to 0, with several hard restarts, after a warmup period during which it increases
    linearly between 0 and the initial lr set in the optimizer.

    Args:
        optimizer ([`~torch.optim.Optimizer`]):
            The optimizer for which to schedule the learning rate.
        num_warmup_steps (`int`):
            The number of steps for the warmup phase.
        num_training_steps (`int`):
            The total number of training steps.
        num_cycles (`int`, *optional*, defaults to 1):
            The number of hard restarts to use.
        last_epoch (`int`, *optional*, defaults to -1):
            The index of the last epoch when resuming training.

    Return:
        `torch.optim.lr_scheduler.LambdaLR` with the appropriate schedule.
    """
    lr_lambda = partial(
        _get_cosine_with_hard_restarts_schedule_with_warmup_lr_lambda,
        num_warmup_steps=num_warmup_steps,
        num_training_steps=num_training_steps,
        num_cycles=num_cycles,
    )
    return LambdaLR(optimizer, lr_lambda, last_epoch)


def _get_polynomial_decay_schedule_with_warmup_lr_lambda(
    current_step: int,
    *,
    num_warmup_steps: int,
    num_training_steps: int,
    lr_end: float,
    power: float,
    lr_init: int,
):
    """
    Args:
        current_step (int): The current step in the training process.
                            It represents the progress of the training.
        num_warmup_steps (int): The number of warmup steps at the beginning of training.
                                Determines the portion of training steps used for warmup.
        num_training_steps (int): The total number of training steps.
                                  Represents the duration of the entire training process.
        lr_end (float): The final learning rate value to decay towards.
                        Specifies the target learning rate at the end of training.
        power (float): The power factor used in the polynomial decay calculation.
                       Influences the rate of decay of the learning rate.
        lr_init (int): The initial learning rate value at the start of training.
                       Represents the starting learning rate value.
    
    Returns:
        None: This function does not return a value explicitly, but modifies the learning rate.
    
    Raises:
        ValueError: If the current_step is negative or if the lr_init is zero.
    """
    if current_step < num_warmup_steps:
        return float(current_step) / float(max(1, num_warmup_steps))
    elif current_step > num_training_steps:
        return lr_end / lr_init  # as LambdaLR multiplies by lr_init
    else:
        lr_range = lr_init - lr_end
        decay_steps = num_training_steps - num_warmup_steps
        pct_remaining = 1 - (current_step - num_warmup_steps) / decay_steps
        decay = lr_range * pct_remaining**power + lr_end
        return decay / lr_init  # as LambdaLR multiplies by lr_init


def get_polynomial_decay_schedule_with_warmup(
    optimizer, num_warmup_steps, num_training_steps, lr_end=1e-7, power=1.0, last_epoch=-1
):
    """
    Create a schedule with a learning rate that decreases as a polynomial decay from the initial lr set in the
    optimizer to end lr defined by *lr_end*, after a warmup period during which it increases linearly from 0 to the
    initial lr set in the optimizer.

    Args:
        optimizer ([`~torch.optim.Optimizer`]):
            The optimizer for which to schedule the learning rate.
        num_warmup_steps (`int`):
            The number of steps for the warmup phase.
        num_training_steps (`int`):
            The total number of training steps.
        lr_end (`float`, *optional*, defaults to 1e-7):
            The end LR.
        power (`float`, *optional*, defaults to 1.0):
            Power factor.
        last_epoch (`int`, *optional*, defaults to -1):
            The index of the last epoch when resuming training.

    Note: *power* defaults to 1.0 as in the fairseq implementation, which in turn is based on the original BERT
    implementation at
    https://github.com/google-research/bert/blob/f39e881b169b9d53bea03d2d341b31707a6c052b/optimization.py#L37

    Return:
        `torch.optim.lr_scheduler.LambdaLR` with the appropriate schedule.

    """
    lr_init = optimizer.defaults["lr"]
    if not (lr_init > lr_end):
        raise ValueError(f"lr_end ({lr_end}) must be be smaller than initial lr ({lr_init})")

    lr_lambda = partial(
        _get_polynomial_decay_schedule_with_warmup_lr_lambda,
        num_warmup_steps=num_warmup_steps,
        num_training_steps=num_training_steps,
        lr_end=lr_end,
        power=power,
        lr_init=lr_init,
    )
    return LambdaLR(optimizer, lr_lambda, last_epoch)


def _get_inverse_sqrt_schedule_lr_lambda(current_step: int, *, num_warmup_steps: int, timescale: int = None):
    r"""
    This function calculates the learning rate decay based on the inverse square root schedule.
    
    Args:
        current_step (int): The current step in the learning process.
    
    Keyword Args:
        num_warmup_steps (int): The number of warm-up steps before the learning rate starts decaying.
        timescale (int, optional): The timescale parameter used in the decay calculation. Defaults to None.
    
    Returns:
        float: The decayed learning rate value.
    
    Raises:
        None.
    
    This function returns the decayed learning rate value based on the inverse square root schedule. If the current step is less than the number of warm-up steps, it returns the current step divided by the
maximum of 1 and the number of warm-up steps. Otherwise, it calculates the decayed learning rate using the inverse square root formula.
    """
    if current_step < num_warmup_steps:
        return float(current_step) / float(max(1, num_warmup_steps))
    shift = timescale - num_warmup_steps
    decay = 1.0 / math.sqrt((current_step + shift) / timescale)
    return decay


def get_inverse_sqrt_schedule(
    optimizer: Optimizer, num_warmup_steps: int, timescale: int = None, last_epoch: int = -1
):
    """
    Create a schedule with an inverse square-root learning rate, from the initial lr set in the optimizer, after a
    warmup period which increases lr linearly from 0 to the initial lr set in the optimizer.

    Args:
        optimizer ([`~torch.optim.Optimizer`]):
            The optimizer for which to schedule the learning rate.
        num_warmup_steps (`int`):
            The number of steps for the warmup phase.
        timescale (`int`, *optional*, defaults to `num_warmup_steps`):
            Time scale.
        last_epoch (`int`, *optional*, defaults to -1):
            The index of the last epoch when resuming training.

    Return:
        `torch.optim.lr_scheduler.LambdaLR` with the appropriate schedule.
    """
    # Note: this implementation is adapted from
    # https://github.com/google-research/big_vision/blob/f071ce68852d56099437004fd70057597a95f6ef/big_vision/utils.py#L930

    if timescale is None:
        timescale = num_warmup_steps or 10_000

    lr_lambda = partial(_get_inverse_sqrt_schedule_lr_lambda, num_warmup_steps=num_warmup_steps, timescale=timescale)
    return LambdaLR(optimizer, lr_lambda, last_epoch=last_epoch)


def _get_cosine_schedule_with_warmup_lr_lambda(
    current_step: int, *, num_warmup_steps: int, num_training_steps: int, num_cycles: float, min_lr_rate: float = 0.0
):
    r"""
    This function implements a cosine learning rate schedule with warmup for a given current step. The learning rate is adjusted based on the progress of the training.
    
    Args:
        current_step (int): The current step in the training process.
    
    Returns:
        float: The adjusted learning rate at the current step.
    
    Raises:
        None
    
    The function calculates the learning rate adjustment based on the number of warmup steps, training steps, number of cycles, and minimum learning rate rate. If the current step is less than the number of
warmup steps, the learning rate is linearly increased. Otherwise, the learning rate is adjusted using a cosine function with the given number of cycles. The learning rate is then scaled by the minimum learning
rate rate.
    
    The function returns the maximum of 0 and the adjusted learning rate factor, ensuring a non-negative learning rate.
    """
    if current_step < num_warmup_steps:
        return float(current_step) / float(max(1, num_warmup_steps))
    progress = float(current_step - num_warmup_steps) / float(max(1, num_training_steps - num_warmup_steps))
    factor = 0.5 * (1.0 + math.cos(math.pi * float(num_cycles) * 2.0 * progress))
    factor = factor * (1 - min_lr_rate) + min_lr_rate
    return max(0, factor)


def get_cosine_with_min_lr_schedule_with_warmup(
    optimizer: Optimizer,
    num_warmup_steps: int,
    num_training_steps: int,
    num_cycles: float = 0.5,
    last_epoch: int = -1,
    min_lr: float = None,
    min_lr_rate: float = None,
):
    """
    Create a schedule with a learning rate that decreases following the values of the cosine function between the
    initial lr set in the optimizer to min_lr, after a warmup period during which it increases linearly between 0 and the
    initial lr set in the optimizer.

    Args:
        optimizer ([`~torch.optim.Optimizer`]):
            The optimizer for which to schedule the learning rate.
        num_warmup_steps (`int`):
            The number of steps for the warmup phase.
        num_training_steps (`int`):
            The total number of training steps.
        num_cycles (`float`, *optional*, defaults to 0.5):
            The number of waves in the cosine schedule (the defaults is to just decrease from the max value to 0
            following a half-cosine).
        last_epoch (`int`, *optional*, defaults to -1):
            The index of the last epoch when resuming training.
        min_lr (`float`, *optional*):
            The minimum learning rate to reach after the cosine schedule.
        min_lr_rate (`float`, *optional*):
            The minimum learning rate as a ratio of the initial learning rate. If set, `min_lr` should not be set.

    Return:
        `torch.optim.lr_scheduler.LambdaLR` with the appropriate schedule.
    """
    if min_lr is not None and min_lr_rate is not None:
        raise ValueError("Only one of min_lr or min_lr_rate should be set")
    elif min_lr is not None:
        min_lr_rate = min_lr / optimizer.defaults["lr"]
    elif min_lr_rate is None:
        raise ValueError("One of min_lr or min_lr_rate should be set through the `lr_scheduler_kwargs`")

    lr_lambda = partial(
        _get_cosine_schedule_with_warmup_lr_lambda,
        num_warmup_steps=num_warmup_steps,
        num_training_steps=num_training_steps,
        num_cycles=num_cycles,
        min_lr_rate=min_lr_rate,
    )
    return LambdaLR(optimizer, lr_lambda, last_epoch)


TYPE_TO_SCHEDULER_FUNCTION = {
    SchedulerType.LINEAR: get_linear_schedule_with_warmup,
    SchedulerType.COSINE: get_cosine_schedule_with_warmup,
    SchedulerType.COSINE_WITH_RESTARTS: get_cosine_with_hard_restarts_schedule_with_warmup,
    SchedulerType.POLYNOMIAL: get_polynomial_decay_schedule_with_warmup,
    SchedulerType.CONSTANT: get_constant_schedule,
    SchedulerType.CONSTANT_WITH_WARMUP: get_constant_schedule_with_warmup,
    SchedulerType.INVERSE_SQRT: get_inverse_sqrt_schedule,
    SchedulerType.REDUCE_ON_PLATEAU: get_reduce_on_plateau_schedule,
    SchedulerType.COSINE_WITH_MIN_LR: get_cosine_with_min_lr_schedule_with_warmup,
}


def get_scheduler(
    name: Union[str, SchedulerType],
    optimizer: Optimizer,
    num_warmup_steps: Optional[int] = None,
    num_training_steps: Optional[int] = None,
    scheduler_specific_kwargs: Optional[dict] = None,
):
    """
    Unified API to get any scheduler from its name.

    Args:
        name (`str` or `SchedulerType`):
            The name of the scheduler to use.
        optimizer (`torch.optim.Optimizer`):
            The optimizer that will be used during training.
        num_warmup_steps (`int`, *optional*):
            The number of warmup steps to do. This is not required by all schedulers (hence the argument being
            optional), the function will raise an error if it's unset and the scheduler type requires it.
        num_training_steps (`int``, *optional*):
            The number of training steps to do. This is not required by all schedulers (hence the argument being
            optional), the function will raise an error if it's unset and the scheduler type requires it.
        scheduler_specific_kwargs (`dict`, *optional*):
            Extra parameters for schedulers such as cosine with restarts. Mismatched scheduler types and scheduler
            parameters will cause the scheduler function to raise a TypeError.
    """
    name = SchedulerType(name)
    schedule_func = TYPE_TO_SCHEDULER_FUNCTION[name]

    # If a `LayerWiseDummyOptimizer` is passed we extract the optimizer dict and
    # recursively call `get_scheduler` to get the proper schedulers on each parameter
    if optimizer is not None and isinstance(optimizer, LayerWiseDummyOptimizer):
        optimizer_dict = optimizer.optimizer_dict
        scheduler_dict = {}

        for param in optimizer_dict.keys():
            scheduler_dict[param] = get_scheduler(
                name,
                optimizer=optimizer_dict[param],
                num_warmup_steps=num_warmup_steps,
                num_training_steps=num_training_steps,
            )

        def scheduler_hook(param):
            # Since the optimizer hook has been already attached we only need to
            # attach the scheduler hook
            if param.grad is not None:
                scheduler_dict[param].step()

        for param in optimizer_dict.keys():
            if param.requires_grad:
                param.register_post_accumulate_grad_hook(scheduler_hook)

        return LayerWiseDummyScheduler()

    if name == SchedulerType.CONSTANT:
        return schedule_func(optimizer)

    if scheduler_specific_kwargs is None:
        scheduler_specific_kwargs = {}

    if name == SchedulerType.REDUCE_ON_PLATEAU:
        return schedule_func(optimizer, **scheduler_specific_kwargs)

    # All other schedulers require `num_warmup_steps`
    if num_warmup_steps is None:
        raise ValueError(f"{name} requires `num_warmup_steps`, please provide that argument.")

    if name == SchedulerType.CONSTANT_WITH_WARMUP:
        return schedule_func(optimizer, num_warmup_steps=num_warmup_steps)

    if name == SchedulerType.INVERSE_SQRT:
        return schedule_func(optimizer, num_warmup_steps=num_warmup_steps)

    # All other schedulers require `num_training_steps`
    if num_training_steps is None:
        raise ValueError(f"{name} requires `num_training_steps`, please provide that argument.")

    return schedule_func(
        optimizer,
        num_warmup_steps=num_warmup_steps,
        num_training_steps=num_training_steps,
        **scheduler_specific_kwargs,
    )

class AdafactorSchedule(LambdaLR):
    """
    Since [`~optimization.Adafactor`] performs its own scheduling, if the training loop relies on a scheduler (e.g.,
    for logging), this class creates a proxy object that retrieves the current lr values from the optimizer.

    It returns `initial_lr` during startup and the actual `lr` during stepping.
    """
    def __init__(self, optimizer, initial_lr=0.0):
        r"""
        Initialize the AdafactorSchedule class.
        
        Args:
            self (object): The instance of the AdafactorSchedule class.
            optimizer (object): The optimizer to be used for updating parameters.
            initial_lr (float, optional): The initial learning rate. Default is 0.0.
        
        Returns:
            None. This method initializes the AdafactorSchedule class.
        
        Raises:
            None.
        """
        def lr_lambda(_):
            return initial_lr

        for group in optimizer.param_groups:
            group["initial_lr"] = initial_lr
        super().__init__(optimizer, lr_lambda)
        for group in optimizer.param_groups:
            del group["initial_lr"]

    def get_lr(self):
        r"""
        This method retrieves the learning rates for the optimizer associated with the AdafactorSchedule class.
        
        Args:
            self: AdafactorSchedule - The instance of the AdafactorSchedule class.
            
        Returns:
            List - A list of learning rates associated with the optimizer's parameter groups.
        
        Raises:
            None
        """
        opt = self.optimizer
        lrs = [
            opt._get_lr(group, opt.state[group["params"][0]])
            for group in opt.param_groups
            if group["params"][0].grad is not None
        ]
        if len(lrs) == 0:
            lrs = self.base_lrs  # if called before stepping
        return lrs


def get_adafactor_schedule(optimizer, initial_lr=0.0):
    """
    Get a proxy schedule for [`~optimization.Adafactor`]

    Args:
        optimizer ([`~torch.optim.Optimizer`]):
            The optimizer for which to schedule the learning rate.
        initial_lr (`float`, *optional*, defaults to 0.0):
            Initial lr

    Return:
        [`~optimization.Adafactor`] proxy schedule object.


    """
    return AdafactorSchedule(optimizer, initial_lr)
