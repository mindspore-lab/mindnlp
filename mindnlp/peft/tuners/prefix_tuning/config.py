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
"""prefix tuning config"""
from dataclasses import dataclass, field

from ...config import PromptLearningConfig
from ...utils import PeftType


@dataclass
class PrefixTuningConfig(PromptLearningConfig):
    """
    This is the configuration class to store the configuration of a [`PrefixEncoder`].

    Args:
        encoder_hidden_size (`int`): The hidden size of the prompt encoder.
        prefix_projection (`bool`): Whether to project the prefix embeddings.
    """
    encoder_hidden_size: int = field(
        default=None,
        metadata={"help": "The hidden size of the encoder"},
    )
    prefix_projection: bool = field(
        default=False,
        metadata={"help": "Whether to project the prefix tokens"},
    )

    def __post_init__(self):
        """
        The '__post_init__' method is a special method in the 'PrefixTuningConfig' class that is automatically called after the initialization of a new instance of the class.
        
        Args:
            self: An instance of the 'PrefixTuningConfig' class.
        
        Returns:
            None. This method does not return any value.
        
        Raises:
            This method does not raise any exceptions.
        """
        self.peft_type = PeftType.PREFIX_TUNING
