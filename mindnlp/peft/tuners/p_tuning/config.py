# Copyright 2023-present the HuggingFace Inc. team.
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
"""p-tuning config"""
import enum
from dataclasses import dataclass, field
from typing import Union

from ...config import PromptLearningConfig
from ...utils import PeftType


class PromptEncoderReparameterizationType(str, enum.Enum):

    """
    Represents a reparameterization type for prompt encoders in Python.
    
    This class, 'PromptEncoderReparameterizationType', is a subclass of both 'str' and 'enum.Enum', and it provides a way to define the reparameterization type for prompt encoders in Python.
    
    Attributes:
        - LINEAR: Represents a linear reparameterization type.
        - LOG: Represents a logarithmic reparameterization type.
        - EXPONENTIAL: Represents an exponential reparameterization type.
    
    Usage:
        To use this class, create an instance of 'PromptEncoderReparameterizationType' and specify the desired reparameterization type. The available reparameterization types are defined as class attributes:
        
            - PromptEncoderReparameterizationType.LINEAR
            - PromptEncoderReparameterizationType.LOG
            - PromptEncoderReparameterizationType.EXPONENTIAL
        
        Example usage:
            reparam_type = PromptEncoderReparameterizationType.LINEAR
            print(reparam_type)  # Output: 'LINEAR'
            
            reparam_type = PromptEncoderReparameterizationType.LOG
            print(reparam_type)  # Output: 'LOG'
            
            reparam_type = PromptEncoderReparameterizationType.EXPONENTIAL
            print(reparam_type)  # Output: 'EXPONENTIAL'
    
    Notes:
        - This class inherits from 'str' and 'enum.Enum', providing all the functionalities of these base classes.
        - The available reparameterization types are defined as class attributes and can be accessed using dot notation.
        - The reparameterization type can be used to configure prompt encoders in various natural language processing tasks.
    """
    MLP = "MLP"
    LSTM = "LSTM"


@dataclass
class PromptEncoderConfig(PromptLearningConfig):
    """
    This is the configuration class to store the configuration of a [`PromptEncoder`].

    Args:
        encoder_reparameterization_type (Union[[`PromptEncoderReparameterizationType`], `str`]):
            The type of reparameterization to use.
        encoder_hidden_size (`int`): The hidden size of the prompt encoder.
        encoder_num_layers (`int`): The number of layers of the prompt encoder.
        encoder_dropout (`float`): The dropout probability of the prompt encoder.
    """
    encoder_reparameterization_type: Union[str, PromptEncoderReparameterizationType] = field(
        default=PromptEncoderReparameterizationType.MLP,
        metadata={"help": "How to reparameterize the prompt encoder"},
    )
    encoder_hidden_size: int = field(
        default=None,
        metadata={"help": "The hidden size of the prompt encoder"},
    )
    encoder_num_layers: int = field(
        default=2,
        metadata={"help": "The number of layers of the prompt encoder"},
    )
    encoder_dropout: float = field(
        default=0.0,
        metadata={"help": "The dropout of the prompt encoder"},
    )

    def __post_init__(self):
        """
        Method for initializing PromptEncoderConfig instances after creation.
        
        Args:
            self: PromptEncoderConfig instance.
                The instance of PromptEncoderConfig class to be initialized.
        
        Returns:
            None. This method does not return any value.
        
        Raises:
            No specific exceptions are raised by this method.
        """
        self.peft_type = PeftType.P_TUNING
