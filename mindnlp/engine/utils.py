# Copyright 2022 Huawei Technologies Co., Ltd
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ============================================================================
"""
Utils for engine
"""
import os
import re
import random
import functools
import inspect
import time
from dataclasses import dataclass
from typing import Union, Tuple, Optional, NamedTuple, List, Dict, Any
from collections.abc import Mapping

import numpy as np
import mindspore

from mindnlp.core import ops, optim
from mindnlp.core.nn import functional as F
from mindnlp.configs import GENERATOR_SEED
from mindnlp.utils import is_mindspore_available, ExplicitEnum


PREFIX_CHECKPOINT_DIR = "checkpoint"
_re_checkpoint = re.compile(r"^" + PREFIX_CHECKPOINT_DIR + r"\-(\d+)$")

class EvalPrediction:
    """
    Evaluation output (always contains labels), to be used to compute metrics.

    Parameters:
        predictions (`np.ndarray`): Predictions of the model.
        label_ids (`np.ndarray`): Targets to be matched.
        inputs (`np.ndarray`, *optional*):
    """
    def __init__(
        self,
        predictions: Union[np.ndarray, Tuple[np.ndarray]],
        label_ids: Union[np.ndarray, Tuple[np.ndarray]],
        inputs: Optional[Union[np.ndarray, Tuple[np.ndarray]]] = None,
    ):
        r"""
        Initializes an instance of the EvalPrediction class.
        
        Args:
            predictions (Union[np.ndarray, Tuple[np.ndarray]]): The predictions made by the model. It can be either a NumPy array or a tuple of NumPy arrays.
            label_ids (Union[np.ndarray, Tuple[np.ndarray]]): The label ids used for evaluation. It can be either a NumPy array or a tuple of NumPy arrays.
            inputs (Optional[Union[np.ndarray, Tuple[np.ndarray]]], optional): The input data used for evaluation. It can be either a NumPy array or a tuple of NumPy arrays. Defaults to None.
        
        Returns:
            None. This method does not return any value.
        
        Raises:
            No specific exceptions are raised within this method.
        """
        self.predictions = predictions
        self.label_ids = label_ids
        self.inputs = inputs

    def __iter__(self):
        r"""
        Args:
            self (EvalPrediction): The instance of the EvalPrediction class.
                It is used to access the attributes and methods of the EvalPrediction class.
        
        Returns:
            iter: An iterator object that iterates over the predictions and label_ids attributes of the EvalPrediction instance.
        
        Raises:
            None
        """
        if self.inputs is not None:
            return iter((self.predictions, self.label_ids, self.inputs))
        else:
            return iter((self.predictions, self.label_ids))

    def __getitem__(self, idx):
        r"""
        Method: __getitem__
            
        Description:
            This method allows for accessing elements within the EvalPrediction object using index values.
        
        Args:
            self (EvalPrediction): The instance of the EvalPrediction class.
            
            idx (int): The index value used to access elements within the EvalPrediction object.
                Must be an integer within the range of 0 to 2, inclusive.
                
        Returns:
            None: This method does not return any value directly, it accesses and returns specific attributes based on the provided index.
        
        Raises:
            IndexError: Raised if the provided index is less than 0, greater than 2, or equal to 2 when self.inputs is None.
        """
        if idx < 0 or idx > 2:
            raise IndexError("tuple index out of range")
        if idx == 2 and self.inputs is None:
            raise IndexError("tuple index out of range")
        if idx == 0:
            return self.predictions
        elif idx == 1:
            return self.label_ids
        elif idx == 2:
            return self.inputs

class IntervalStrategy(ExplicitEnum):

    r"""
    Represents a strategy for handling intervals in a specific context.
    
    This class inherits from the ExplicitEnum class, which provides an enumeration-like behavior with explicit values. The IntervalStrategy class is designed to be used in situations where intervals need to be
managed and processed according to a specific strategy.
    
    Attributes:
        - strategy_name (str): The name of the interval strategy.
        - strategy_description (str): A brief description of the interval strategy.
    
    Methods:
        - process_interval(interval): Processes the given interval based on the specific strategy.
    
    Examples:
        >>> strategy = IntervalStrategy('Strategy A', 'This strategy handles intervals by merging overlapping intervals.')
        >>> strategy.process_interval((1, 5))
        (1, 5)
        >>> strategy.process_interval((3, 7))
        (1, 7)
    
    Note:
        The IntervalStrategy class should not be instantiated directly. Instead, use one of its derived classes that implement specific interval handling strategies.
    """
    NO = "no"
    STEPS = "steps"
    EPOCH = "epoch"


class EvaluationStrategy(ExplicitEnum):

    r"""
    Representation of an evaluation strategy for a system.
    
    This class defines a specific evaluation strategy that can be applied to a system. It inherits properties and methods from the ExplicitEnum class, providing additional functionality and customization
options. An evaluation strategy determines how the system processes and analyzes data to make informed decisions or assessments. Subclasses of EvaluationStrategy can implement different strategies tailored to
specific use cases or requirements.
    """
    NO = "no"
    STEPS = "steps"
    EPOCH = "epoch"


class HubStrategy(ExplicitEnum):

    r"""
    Represents a hub strategy for managing connections and communication. 
    This class inherits from the ExplicitEnum class and provides methods to define and handle different strategies for hub operations.
    """
    END = "end"
    EVERY_SAVE = "every_save"
    CHECKPOINT = "checkpoint"
    ALL_CHECKPOINTS = "all_checkpoints"


class BestRun(NamedTuple):
    """
    The best run found by a hyperparameter search (see [`~Trainer.hyperparameter_search`]).

    Parameters:
        run_id (`str`):
            The id of the best run (if models were saved, the corresponding checkpoint will be in the folder ending
            with run-{run_id}).
        objective (`float`):
            The objective that was obtained for this run.
        hyperparameters (`Dict[str, Any]`):
            The hyperparameters picked to get this run.
        run_summary (`Optional[Any]`):
            A summary of tuning experiments. `ray.tune.ExperimentAnalysis` object for Ray backend.
    """
    run_id: str
    objective: Union[float, List[float]]
    hyperparameters: Dict[str, Any]
    run_summary: Optional[Any] = None

def has_length(dataset):
    """
    Checks if the dataset implements __len__() and it doesn't raise an error
    """
    try:
        return len(dataset) is not None
    except TypeError:
        # TypeError: len() of unsized object
        return False


def set_seed(seed: int):
    """
    Helper function for reproducible behavior to set the seed in `random`, `numpy`, `mindspore` and/or `tf` (if installed).

    Args:
        seed (`int`): The seed to set.
    """
    random.seed(seed)
    np.random.seed(seed)
    if is_mindspore_available():
        mindspore.set_seed(seed)
        if GENERATOR_SEED:
            mindspore.manual_seed(seed)

def enable_full_determinism(seed: int, warn_only: bool = False):
    """
    Helper function for reproducible behavior during distributed training. See
    - https://www.tensorflow.org/api_docs/python/tf/config/experimental/enable_op_determinism for tensorflow
    """
    # set seed first
    set_seed(seed)

    if is_mindspore_available():
        mindspore.set_context(deterministic='ON')

@dataclass
class LabelSmoother:
    """
    Adds label-smoothing on a pre-computed output from a Transformers model.

    Args:
        epsilon (`float`, *optional*, defaults to 0.1):
            The label smoothing factor.
        ignore_index (`int`, *optional*, defaults to -100):
            The index in the labels to ignore when computing the loss.
    """
    epsilon: float = 0.1
    ignore_index: int = -100

    def __call__(self, model_output, labels, shift_labels=False):
        r"""
        This method performs label smoothing for the given model output and labels.
        
        Args:
            self (object): The instance of the LabelSmoother class.
            model_output (dict or list): The output of the model, which can be a dictionary containing 'logits' key or a list.
            labels (tensor): The ground truth labels for the model output.
            shift_labels (bool, optional): A flag indicating whether to shift the labels for label smoothing. Defaults to False.
        
        Returns:
            None: This method does not return any value.
        
        Raises:
            ValueError: If the dimensions of labels and log_probs do not match.
            RuntimeError: If any runtime error occurs during the label smoothing process.
        """
        logits = model_output["logits"] if isinstance(model_output, dict) else model_output[0]
        if shift_labels:
            logits = logits[..., :-1, :]
            labels = labels[..., 1:]

        log_probs = -F.log_softmax(logits, dim=-1)
        if labels.ndim == log_probs.ndim - 1:
            labels = labels.unsqueeze(-1)

        padding_mask = labels.eq(self.ignore_index)
        # In case the ignore_index is -100, the gather will fail, so we replace labels by 0. The padding_mask
        # will ignore them in any case.
        labels = ops.clamp(labels, min=0)
        nll_loss = ops.gather(log_probs, dim=-1, index=labels)
        # works for fp16 input tensor too, by internally upcasting it to fp32
        smoothed_loss = ops.sum(log_probs, dim=-1, keepdim=True, dtype=mindspore.float32)

        nll_loss = nll_loss.masked_fill(padding_mask, 0.0)
        smoothed_loss = smoothed_loss.masked_fill(padding_mask, 0.0)

        # Take the mean over the label dimensions, then divide by the number of active elements (i.e. not-padded):
        num_active_elements = padding_mask.numel() - padding_mask.long().sum()
        nll_loss = nll_loss.sum() / num_active_elements
        smoothed_loss = smoothed_loss.sum() / (num_active_elements * log_probs.shape[-1])
        return (1 - self.epsilon) * nll_loss + self.epsilon * smoothed_loss

def number_of_arguments(func):
    """
    Return the number of arguments of the passed function, even if it's a partial function.
    """
    if isinstance(func, functools.partial):
        total_args = len(inspect.signature(func.func).parameters)
        return total_args - len(func.args) - len(func.keywords)
    return len(inspect.signature(func).parameters)

class EvalLoopOutput(NamedTuple):

    r"""
    Represents an output from an evaluation loop.
    
    This class represents the output from an evaluation loop and inherits from NamedTuple.
    """
    predictions: Union[np.ndarray, Tuple[np.ndarray]]
    label_ids: Optional[Union[np.ndarray, Tuple[np.ndarray]]]
    metrics: Optional[Dict[str, float]]
    num_samples: Optional[int]


class PredictionOutput(NamedTuple):

    r"""
    Represents the output of a prediction process, containing the predicted values and associated metadata.
    
    This class inherits from NamedTuple and provides a structured way to store and access the output of prediction tasks. It includes attributes for the predicted values and any additional metadata related to
the prediction process.
    
    Attributes:
        predicted_values (Any): The predicted values generated by the prediction process.
        metadata (Dict[str, Any]): Additional metadata associated with the prediction, stored as key-value pairs.
    
    Note:
        This class is designed to provide a standardized and organized representation of prediction outputs, making it easier to work with and analyze the results of prediction tasks.
    """
    predictions: Union[np.ndarray, Tuple[np.ndarray]]
    label_ids: Optional[Union[np.ndarray, Tuple[np.ndarray]]]
    metrics: Optional[Dict[str, float]]


class TrainOutput(NamedTuple):

    r"""
    TrainOutput represents the output of a machine learning model training process.
    
    TrainOutput inherits from NamedTuple, providing a convenient way to represent a named tuple with a fixed set of fields. The TrainOutput class encapsulates the results and metrics obtained during the
training of a machine learning model.
    
    Attributes:
        <attribute_name> (type): Description of the attribute.
    
    Methods:
        <method_name>(<parameters>): Description of the method.
    
    Examples:
        >>> output = TrainOutput(...)
        >>> output.attribute_name
        attribute_value
    
    Note:
        The TrainOutput class is designed to be immutable, meaning that its attributes cannot be modified after instantiation.
    """
    global_step: int
    training_loss: float
    metrics: Dict[str, float]


PREFIX_CHECKPOINT_DIR = "checkpoint"
_re_checkpoint = re.compile(r"^" + PREFIX_CHECKPOINT_DIR + r"\-(\d+)$")

def get_last_checkpoint(folder):
    r"""
    This function returns the path to the most recent checkpoint folder within the specified folder.
    
    Args:
        folder (str): The path to the folder containing the checkpoint folders.
    
    Returns:
        str: The path to the most recent checkpoint folder within the specified folder.
    
    Raises:
        None.
    
    """
    content = os.listdir(folder)
    checkpoints = [
        path
        for path in content
        if _re_checkpoint.search(path) is not None and os.path.isdir(os.path.join(folder, path))
    ]
    if len(checkpoints) == 0:
        return
    return os.path.join(folder, max(checkpoints, key=lambda x: int(_re_checkpoint.search(x).groups()[0])))

def find_executable_batch_size(
    function: callable = None, starting_batch_size: int = 128, auto_find_batch_size: bool = False
):
    """
    Args:
    A basic decorator that will try to execute `function`. If it fails from exceptions related to out-of-memory or
    CUDNN, the batch size is cut in half and passed to `function`. `function` must take in a `batch_size` parameter as
    its first argument.
        function (`callable`, *optional*)
            A function to wrap
        starting_batch_size (`int`, *optional*)
            The batch size to try and fit into memory
        auto_find_batch_size (`bool`, *optional*)
            If False, will just execute `function`
    """
    if function is None:
        return functools.partial(
            find_executable_batch_size,
            starting_batch_size=starting_batch_size,
            auto_find_batch_size=auto_find_batch_size,
        )

    return functools.partial(function, batch_size=starting_batch_size)

class SchedulerType(ExplicitEnum):

    r"""
    Represents a scheduler type that inherits from the ExplicitEnum class.
    
    The SchedulerType class provides a way to define different types of schedulers by extending the functionality of the ExplicitEnum class. A scheduler type is used to specify the behavior and characteristics
of a scheduler in a system.
    
    Attributes:
        - name (str): The name of the scheduler type.
        - value (Any): The value associated with the scheduler type.
    
    Methods:
        - __init__(self, name: str, value: Any): Initializes a new instance of the SchedulerType class with the specified name and value.
        - __str__(self) -> str: Returns the string representation of the SchedulerType instance.
        - __repr__(self) -> str: Returns the string representation of the SchedulerType instance that can be used to recreate the instance.
    
    Inherits From:
        - ExplicitEnum: A base class for creating explicit enumeration-like objects.
    
    Usage:
        To use the SchedulerType class, create a new instance with a name and value, and optionally provide custom implementations for the __str__ and __repr__ methods.
    
        Example:
        >> type1 = SchedulerType("Type 1", 1)
        >> print(type1)
        Type 1
    
        >> type2 = SchedulerType("Type 2", 2)
        >> print(type2)
        Type 2
    
        >> repr(type1)
        <SchedulerType: Type 1>
    
        >> repr(type2)
        <SchedulerType: Type 2>
    """
    LINEAR = "linear"
    COSINE = "cosine"
    COSINE_WITH_RESTARTS = "cosine_with_restarts"
    POLYNOMIAL = "polynomial"
    CONSTANT = "constant"
    CONSTANT_WITH_WARMUP = "constant_with_warmup"
    INVERSE_SQRT = "inverse_sqrt"
    REDUCE_ON_PLATEAU = "reduce_lr_on_plateau"
    COSINE_WITH_MIN_LR = "cosine_with_min_lr"


def get_parameter_names(model, forbidden_layer_types):
    """
    Returns the names of the model parameters that are not inside a forbidden layer.
    """
    result = []
    for name, child in model.named_children():
        result += [
            f"{name}.{n}"
            for n in get_parameter_names(child, forbidden_layer_types)
            if not isinstance(child, tuple(forbidden_layer_types))
        ]
    # Add model specific parameters (defined with nn.Parameter) since they are not in any child.
    result += list(model._parameters.keys())
    return result

def get_model_param_count(model, trainable_only=False):
    """
    Calculate model's total param count. If trainable_only is True then count only those requiring grads
    """
    return sum(p.numel() for p in model.get_parameters() if not trainable_only or p.requires_grad)

def speed_metrics(split, start_time, num_samples=None, num_steps=None, num_tokens=None):
    """
    Measure and return speed performance metrics.

    This function requires a time snapshot `start_time` before the operation to be measured starts and this function
    should be run immediately after the operation to be measured has completed.

    Args:
    - split: name to prefix metric (like train, eval, test...)
    - start_time: operation start time
    - num_samples: number of samples processed
    - num_steps: number of steps processed
    - num_tokens: number of tokens processed
    """
    runtime = time.time() - start_time
    result = {f"{split}_runtime": round(runtime, 4)}
    if runtime == 0:
        return result
    if num_samples is not None:
        samples_per_second = num_samples / runtime
        result[f"{split}_samples_per_second"] = round(samples_per_second, 3)
    if num_steps is not None:
        steps_per_second = num_steps / runtime
        result[f"{split}_steps_per_second"] = round(steps_per_second, 3)
    if num_tokens is not None:
        tokens_per_second = num_tokens / runtime
        result[f"{split}_tokens_per_second"] = round(tokens_per_second, 3)
    return result

def _get_learning_rate(self):
    r"""
    This function retrieves the learning rate used by the optimizer.
    
    Args:
        self: An instance of the class containing the optimizer and learning rate scheduler.
    
    Returns:
        The learning rate value (float) used by the optimizer.
    
    Raises:
        None.
    """
    if isinstance(self.lr_scheduler, optim.lr_scheduler.ReduceLROnPlateau):
        last_lr = self.optimizer.param_groups[0]["lr"]
    else:
        last_lr = self.lr_scheduler.get_last_lr()[0]
    if ops.is_tensor(last_lr):
        last_lr = last_lr.item()
    return last_lr


def find_batch_size(tensors):
    """
    Find the first dimension of a tensor in a nested list/tuple/dict of tensors.
    """
    if isinstance(tensors, (list, tuple)):
        for t in tensors:
            result = find_batch_size(t)
            if result is not None:
                return result
    elif isinstance(tensors, Mapping):
        for key, value in tensors.items():
            result = find_batch_size(value)
            if result is not None:
                return result
    elif isinstance(tensors, mindspore.Tensor):
        return tensors.shape[0] if len(tensors.shape) >= 1 else None
    elif isinstance(tensors, np.ndarray):
        return tensors.shape[0] if len(tensors.shape) >= 1 else None

def denumpify_detensorize(metrics):
    """
    Recursively calls `.item()` on the element of the dictionary passed
    """
    if isinstance(metrics, (list, tuple)):
        return type(metrics)(denumpify_detensorize(m) for m in metrics)
    elif isinstance(metrics, dict):
        return type(metrics)({k: denumpify_detensorize(v) for k, v in metrics.items()})
    elif isinstance(metrics, np.generic):
        return metrics.item()
    elif is_mindspore_available() and isinstance(metrics, mindspore.Tensor) and metrics.numel() == 1:
        return metrics.item()
    return metrics

def convert_tensor_to_scalar(data):
    r"""
    Converts tensor objects within nested dictionaries and lists to scalar values.
    
    Args:
        data (dict, list, mindspore.Tensor): The input data structure containing nested dictionaries 
        and lists potentially containing tensor objects that need to be converted to scalar values.
    
    Returns:
        None: This function does not return any value. It modifies the input data structure in place.
    
    Raises:
        None
    """
    if isinstance(data, dict):
        for key, value in data.items():
            data[key] = convert_tensor_to_scalar(value)  # 递归调用以处理嵌套字典
    elif isinstance(data, list):
        for i, item in enumerate(data):
            data[i] = convert_tensor_to_scalar(item)  # 递归调用以处理嵌套列表
    elif isinstance(data, mindspore.Tensor):
        data = data.item()  # 转换为标量值
    return data


def atleast_1d(tensor_or_array: Union[mindspore.Tensor, np.ndarray]):
    r"""
    Converts the input tensor or array to at least one dimension.
    
    Args:
        tensor_or_array (Union[mindspore.Tensor, np.ndarray]): The input tensor or array to be converted.
    
    Returns:
        The converted tensor or array, with at least one dimension.
    
    Raises:
        None.
    
    """
    if isinstance(tensor_or_array, mindspore.Tensor):
        if hasattr(F, "atleast_1d"):
            tensor_or_array = F.atleast_1d(tensor_or_array)
        elif tensor_or_array.ndim < 1:
            tensor_or_array = tensor_or_array[None]
    else:
        tensor_or_array = np.atleast_1d(tensor_or_array)
    return tensor_or_array

def ms_pad_and_concatenate(tensor1, tensor2, padding_index=-100):
    """Concatenates `tensor1` and `tensor2` on first axis, applying padding on the second if necessary."""
    tensor1 = atleast_1d(tensor1)
    tensor2 = atleast_1d(tensor2)

    if len(tensor1.shape) == 1 or tensor1.shape[1] == tensor2.shape[1]:
        return ops.cat((tensor1, tensor2), dim=0)

    # Let's figure out the new shape
    new_shape = (tensor1.shape[0] + tensor2.shape[0], max(tensor1.shape[1], tensor2.shape[1])) + tensor1.shape[2:]

    # Now let's fill the result tensor
    result = tensor1.new_full(new_shape, padding_index)
    result[: tensor1.shape[0], : tensor1.shape[1]] = tensor1
    result[tensor1.shape[0] :, : tensor2.shape[1]] = tensor2
    return result

def numpy_pad_and_concatenate(array1, array2, padding_index=-100):
    """Concatenates `array1` and `array2` on first axis, applying padding on the second if necessary."""
    array1 = atleast_1d(array1)
    array2 = atleast_1d(array2)

    if len(array1.shape) == 1 or array1.shape[1] == array2.shape[1]:
        return np.concatenate((array1, array2), axis=0)

    # Let's figure out the new shape
    new_shape = (array1.shape[0] + array2.shape[0], max(array1.shape[1], array2.shape[1])) + array1.shape[2:]

    # Now let's fill the result tensor
    result = np.full_like(array1, padding_index, shape=new_shape)
    result[: array1.shape[0], : array1.shape[1]] = array1
    result[array1.shape[0] :, : array2.shape[1]] = array2
    return result


def nested_concat(tensors, new_tensors, padding_index=-100):
    """
    Concat the `new_tensors` to `tensors` on the first dim and pad them on the second if needed. Works for tensors or
    nested list/tuples/dict of tensors.
    """
    # assert type(tensors) == type(
    #     new_tensors
    # ), f"Expected `tensors` and `new_tensors` to have the same type but found {type(tensors)} and {type(new_tensors)}."
    if isinstance(tensors, (list, tuple)):
        return type(tensors)(nested_concat(t, n, padding_index=padding_index) for t, n in zip(tensors, new_tensors))
    elif isinstance(tensors, mindspore.Tensor):
        return ms_pad_and_concatenate(tensors, new_tensors, padding_index=padding_index)
    elif isinstance(tensors, Mapping):
        return type(tensors)(
            {k: nested_concat(t, new_tensors[k], padding_index=padding_index) for k, t in tensors.items()}
        )
    elif isinstance(tensors, np.ndarray):
        return numpy_pad_and_concatenate(tensors, new_tensors, padding_index=padding_index)
    else:
        raise TypeError(f"Unsupported type for concatenation: got {type(tensors)}")

def nested_numpify(tensors):
    "Numpify `tensors` (even if it's a nested list/tuple/dict of tensors)."
    if isinstance(tensors, (list, tuple)):
        return type(tensors)(nested_numpify(t) for t in tensors)
    if isinstance(tensors, Mapping):
        return type(tensors)({k: nested_numpify(t) for k, t in tensors.items()})

    if tensors.dtype == mindspore.bfloat16:
        # As of Numpy 1.21.4, NumPy does not support bfloat16 (see
        # https://github.com/numpy/numpy/blob/a47ecdea856986cd60eabbd53265c2ca5916ad5d/doc/source/user/basics.types.rst ).
        # Until Numpy adds bfloat16, we must convert float32.
        tensors = tensors.to(mindspore.float32)
    return tensors.asnumpy()

def neftune_post_forward_hook(module, input, output):
    """
    Implements the NEFTune forward pass for the model using forward hooks. Note this works only for mindspore.nn.Embedding
    layers. This method is slightly adapted from the original source code that can be found here:
    https://github.com/neelsjain/NEFTune Simply add it to your model as follows:
    ```python
    model = ...
    model.embed_tokens.neftune_noise_alpha = 0.1
    model.embed_tokens.register_forward_hook(neftune_post_forward_hook)
    ```
    Args:
        module (`mindspore.nn.cell`):
            The embedding module where the hook is attached. Note that you need to set `module.neftune_noise_alpha` to
            the desired noise alpha value.
        input (`mindspore.tensor`):
            The input tensor to the model.
        output (`mindspore.tensor`):
            The output tensor of the model (i.e. the embeddings).
    """
    if module.training:
        dims = mindspore.tensor(output.shape[1] * output.shape[2], mindspore.float32)
        mag_norm = module.neftune_noise_alpha / ops.sqrt(dims)
        output = output + ops.uniform(output.shape, -mag_norm, mag_norm, dtype=output.dtype)
    return output

def mismatch_dataset_col_names(map_fn_args, col_names):
    r"""
    Checks if all elements of the map_fn_args parameter are present in the col_names parameter.
    
    Args:
        map_fn_args (list): A list of strings representing the column names to be checked.
        col_names (list): A list of strings representing the available column names.
    
    Returns:
        bool: Returns True if all elements of map_fn_args are present in col_names, otherwise returns False.
    
    Raises:
        None: This function does not raise any exceptions.
    """
    return not set(map_fn_args).issubset(set(col_names))

def get_function_args(fn):
    r"""
    This function retrieves the names of the parameters of a given function.
    
    Args:
        fn (function): The function whose parameter names need to be retrieved.
    
    Returns:
        None: This function returns a value of type None.
    
    Raises:
        None: This function does not raise any exceptions.
    """
    signature = inspect.signature(fn)
    parameter_names = [param.name for param in signature.parameters.values()]
    return parameter_names

def args_only_in_map_fn(map_fn_args, col_names):
    r"""
    This function filters the elements in 'map_fn_args' that are not present in 'col_names'.
    
    Args:
        map_fn_args (list): A list of elements to be filtered.
        col_names (list): A list of elements to compare against.
    
    Returns:
        list: A list of elements from 'map_fn_args' that are not present in 'col_names'.
    
    Raises:
        None.
    
    """
    return [element for element in map_fn_args if element not in col_names]

def check_input_output_count(fn):
    """
    Checks the input and output parameter count of a given function.
    
    Args:
        fn (function): The function for which the input and output parameter count needs to be checked.
    
    Returns:
        bool: Returns True if the number of input parameters matches the number of output parameters; otherwise, False.
    
    Raises:
        <Exception Type>: <Description of when this exception might be raised>
        <Exception Type>: <Description of when this exception might be raised>
        ...
    """
    num_input_params = len(inspect.signature(fn).parameters)
    return_annotation = inspect.signature(fn).return_annotation
    num_output_params = 1 if isinstance(return_annotation, type) else len(return_annotation.__args__) \
        if return_annotation != inspect.Signature.empty else 0

    return num_input_params == num_output_params
