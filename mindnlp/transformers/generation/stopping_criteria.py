# Copyright 2023 Huawei Technologies Co., Ltd
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
Stopping criteria
"""
import time
import warnings
from copy import deepcopy
from typing import Optional

import mindspore


class StoppingCriteria():
    """Abstract base class for all stopping criteria that can be applied during generation."""
    def __call__(self, input_ids: mindspore.Tensor, scores: mindspore.Tensor, **kwargs) -> bool:
        """
        This method is the call method for the StoppingCriteria class.
        
        Args:
            self (StoppingCriteria): The instance of the StoppingCriteria class.
            input_ids (mindspore.Tensor): The input tensor containing the IDs.
            scores (mindspore.Tensor): The input tensor containing the scores.
            
        Returns:
            bool: Returns a boolean value indicating whether the stopping criteria has been met.
            
        Raises:
            NotImplementedError:
                This exception is raised if the StoppingCriteria class is used directly instead of being subclassed.
        """
        raise NotImplementedError("StoppingCriteria needs to be subclassed")


class MaxLengthCriteria(StoppingCriteria):
    """
    This class can be used to stop generation whenever the full generated number of tokens exceeds `max_length`. Keep
    in mind for decoder-only type of transformers, this will include the initial prompted tokens.

    Args:
        max_length (`int`):
            The maximum length that the output sequence can have in number of tokens.
    """
    def __init__(self, max_length: int):
        """
        Initializes a MaxLengthCriteria object with the specified maximum length.
        
        Args:
            self (MaxLengthCriteria): The instance of the MaxLengthCriteria class.
            max_length (int): The maximum length to be set for the criteria. It must be a positive integer.
        
        Returns:
            None.
        
        Raises:
            ValueError: If max_length is not a positive integer.
        """
        self.max_length = max_length

    def __call__(self, input_ids: mindspore.Tensor, scores: mindspore.Tensor, **kwargs) -> bool:
        """
        This method evaluates whether the length of input_ids meets the maximum length criteria.
        
        Args:
            self (MaxLengthCriteria): The instance of the MaxLengthCriteria class.
            input_ids (mindspore.Tensor): The input tensor representing the input ids.
            scores (mindspore.Tensor): The tensor containing scores associated with the input_ids.
            **kwargs: Additional keyword arguments.
        
        Returns:
            bool: Returns True if the length of input_ids meets the maximum length criteria, otherwise False.
        
        Raises:
            TypeError: If input_ids or scores are not of type mindspore.Tensor.
            ValueError: If input_ids or scores are empty or have incompatible shapes.
        """
        return input_ids.shape[-1] >= self.max_length


class MaxNewTokensCriteria(StoppingCriteria):
    """
    This class can be used to stop generation whenever the generated number of tokens exceeds `max_new_tokens`. Keep in
    mind for decoder-only type of transformers, this will **not** include the initial prompted tokens. This is very
    close to `MaxLengthCriteria` but ignores the number of initial tokens.

    Args:
        start_length (`int`):
            The number of initial tokens.
        max_new_tokens (`int`):
            The maximum number of tokens to generate.
    """
    def __init__(self, start_length: int, max_new_tokens: int):
        """
        Initializes an instance of the MaxNewTokensCriteria class.
        
        Args:
            self: The instance of the MaxNewTokensCriteria class.
            start_length (int): The starting length value for the criteria.
            max_new_tokens (int): The maximum number of new tokens allowed.
        
        Returns:
            None.
        
        Raises:
            FutureWarning: If the MaxNewTokensCriteria class is deprecated.
                Suggests using MaxLengthCriteria with max_length equal to start_length plus max_new_tokens instead.
        """
        warnings.warn(
            "The class `MaxNewTokensCriteria` is deprecated. "
            f"Please use `MaxLengthCriteria(max_length={start_length + max_new_tokens})` "
            "with `max_length = start_length + max_new_tokens` instead.",
            FutureWarning,
        )
        self.start_length = start_length
        self.max_new_tokens = max_new_tokens
        self.max_length = start_length + max_new_tokens

    def __call__(self, input_ids: mindspore.Tensor, scores: mindspore.Tensor, **kwargs) -> bool:
        """
        This method evaluates the condition for the maximum number of new tokens based on the input ids and scores.
        
        Args:
            self (MaxNewTokensCriteria): The instance of the MaxNewTokensCriteria class.
            input_ids (mindspore.Tensor): A tensor containing the input ids.
            scores (mindspore.Tensor): A tensor containing the scores associated with the input ids.
            **kwargs: Additional keyword arguments that are not used in this method.
        
        Returns:
            bool: Returns True if the length of the input_ids is greater than or equal to the max_length defined in
                the MaxNewTokensCriteria instance; otherwise, returns False.
        
        Raises:
            None.
        """
        return input_ids.shape[-1] >= self.max_length


class MaxTimeCriteria(StoppingCriteria):
    """
    This class can be used to stop generation whenever the full generation exceeds some amount of time. By default, the
    time will start being counted when you initialize this function. You can override this by passing an
    `initial_time`.

    Args:
        max_time (`float`):
            The maximum allowed time in seconds for the generation.
        initial_time (`float`, *optional*, defaults to `time.time()`):
            The start of the generation allowed time.
    """
    def __init__(self, max_time: float, initial_timestamp: Optional[float] = None):
        """
        Initialize a MaxTimeCriteria object.
        
        Args:
            max_time (float): The maximum time value for the criteria.
                Must be a non-negative float representing the maximum time in seconds.
            initial_timestamp (Optional[float]): The initial timestamp for the criteria.
                If provided, it should be a float representing the initial timestamp in seconds.
                Defaults to None, in which case the current time will be used.
        
        Returns:
            None.
        
        Raises:
            None.
        """
        self.max_time = max_time
        self.initial_timestamp = time.time() if initial_timestamp is None else initial_timestamp

    def __call__(self, input_ids: mindspore.Tensor, scores: mindspore.Tensor, **kwargs) -> bool:
        """
        This method represents the call functionality of the MaxTimeCriteria class.
        
        Args:
            self (MaxTimeCriteria): The instance of the MaxTimeCriteria class.
            input_ids (mindspore.Tensor): The tensor containing input IDs.
                This parameter is used to pass input IDs to the method.
            scores (mindspore.Tensor): The tensor containing scores.
                This parameter is used to pass scores to the method.
            **kwargs: Additional keyword arguments that may be passed but are not used in this method.
        
        Returns:
            bool: A boolean value indicating whether the time elapsed since the initial timestamp
                exceeds the maximum allowed time defined by self.max_time.
        
        Raises:
            None.
        """
        return time.time() - self.initial_timestamp > self.max_time


class StoppingCriteriaList(list):
    """StoppingCriteriaList"""
    def __call__(self, input_ids: mindspore.Tensor, scores: mindspore.Tensor, **kwargs) -> bool:
        """
        This method '__call__' in the class 'StoppingCriteriaList' evaluates a list of stopping criteria against the input data.
        
        Args:
            self: Represents the instance of the StoppingCriteriaList class.
            input_ids (mindspore.Tensor): Tensor containing input IDs for evaluation.
            scores (mindspore.Tensor): Tensor containing scores for evaluation.

        Returns:
            bool: Returns a boolean value indicating whether any of the stopping criteria have been met.

        Raises:
            None.
        """
        return any(criteria(input_ids, scores) for criteria in self)

    @property
    def max_length(self) -> Optional[int]:
        """return max length"""
        for stopping_criterium in self:
            if isinstance(stopping_criterium, MaxLengthCriteria):
                return stopping_criterium.max_length
            if isinstance(stopping_criterium, MaxNewTokensCriteria):
                return stopping_criterium.max_length
        return None


def validate_stopping_criteria(stopping_criteria: StoppingCriteriaList, max_length: int) -> StoppingCriteriaList:
    """validate stopping criteria"""
    stopping_max_length = stopping_criteria.max_length
    new_stopping_criteria = deepcopy(stopping_criteria)
    if stopping_max_length is not None and stopping_max_length != max_length:
        warnings.warn("You set different `max_length` for stopping criteria and `max_length` parameter", UserWarning)
    elif stopping_max_length is None:
        new_stopping_criteria.append(MaxLengthCriteria(max_length=max_length))
    return new_stopping_criteria
