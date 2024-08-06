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
"""prompt tuning config."""
import enum
from dataclasses import dataclass, field
from typing import Optional, Union

from ...config import PromptLearningConfig
from ...utils import PeftType


class PromptTuningInit(str, enum.Enum):

    r"""
    Represents an initialization state for prompt tuning in a Python class named 'PromptTuningInit'. 
    This class inherits from the 'str' class and the 'enum.Enum' class.
    
    PromptTuningInit is used to define and manage the initialization state for prompt tuning. 
    It provides functionality to set and retrieve the initialization state, and inherits 
    all the methods and attributes of the 'str' class and the 'enum.Enum' class.
    
    Attributes:
        - None
    
    Methods:
        - None
    
    Inherited Attributes from the 'str' class:
        - capitalize()
        - casefold()
        - center()
        - count()
        - encode()
        - endswith()
        - expandtabs()
        - find()
        - format()
        - format_map()
        - index()
        - isalnum()
        - isalpha()
        - isascii()
        - isdecimal()
        - isdigit()
        - isidentifier()
        - islower()
        - isnumeric()
        - isprintable()
        - isspace()
        - istitle()
        - isupper()
        - join()
        - ljust()
        - lower()
        - lstrip()
        - maketrans()
        - partition()
        - replace()
        - rfind()
        - rindex()
        - rjust()
        - rpartition()
        - rsplit()
        - rstrip()
        - split()
        - splitlines()
        - startswith()
        - strip()
        - swapcase()
        - title()
        - translate()
        - upper()
        - zfill()
    
    Inherited Attributes from the 'enum.Enum' class:
        - name
        - value
    
    Inherited Methods from the 'enum.Enum' class:
        - __class__
        - __contains__
        - __delattr__
        - __dir__
        - __eq__
        - __format__
        - __ge__
        - __getattribute__
        - __getitem__
        - __gt__
        - __hash__
        - __init__
        - __init_subclass__
        - __iter__
        - __le__
        - __len__
        - __lt__
        - __members__
        - __module__
        - __ne__
        - __new__
        - __reduce__
        - __reduce_ex__
        - __repr__
        - __setattr__
        - __sizeof__
        - __str__
        - __subclasshook__
    
    """
    TEXT = "TEXT"
    RANDOM = "RANDOM"


@dataclass
class PromptTuningConfig(PromptLearningConfig):
    """
    This is the configuration class to store the configuration of a [`PromptEmbedding`].

    Args:
        prompt_tuning_init (Union[[`PromptTuningInit`], `str`]): The initialization of the prompt embedding.
        prompt_tuning_init_text (`str`, *optional*):
            The text to initialize the prompt embedding. Only used if `prompt_tuning_init` is `TEXT`.
        tokenizer_name_or_path (`str`, *optional*):
            The name or path of the tokenizer. Only used if `prompt_tuning_init` is `TEXT`.
        tokenizer_kwargs (`dict`, *optional*):
            The keyword arguments to pass to `AutoTokenizer.from_pretrained`. Only used if `prompt_tuning_init` is
            `TEXT`.
    """
    prompt_tuning_init: Union[PromptTuningInit, str] = field(
        default=PromptTuningInit.RANDOM,
        metadata={"help": "How to initialize the prompt tuning parameters"},
    )
    prompt_tuning_init_text: Optional[str] = field(
        default=None,
        metadata={
            "help": "The text to use for prompt tuning initialization. Only used if prompt_tuning_init is `TEXT`"
        },
    )
    tokenizer_name_or_path: Optional[str] = field(
        default=None,
        metadata={
            "help": "The tokenizer to use for prompt tuning initialization. Only used if prompt_tuning_init is `TEXT`"
        },
    )

    tokenizer_kwargs: Optional[dict] = field(
        default=None,
        metadata={
            "help": (
                "The keyword arguments to pass to `AutoTokenizer.from_pretrained`. Only used if prompt_tuning_init is "
                "`TEXT`"
            ),
        },
    )

    def __post_init__(self):
        r"""
        This method initializes the PromptTuningConfig object after its creation.
        
        Args:
            self: The instance of the PromptTuningConfig class.
        
        Returns:
            None. This method does not return any value.
        
        Raises:
            - ValueError: If the prompt_tuning_init is set to TEXT and tokenizer_name_or_path is not provided.
            - ValueError: If the prompt_tuning_init is set to TEXT and prompt_tuning_init_text is not provided.
            - ValueError: If tokenizer_kwargs is provided but prompt_tuning_init is not set to TEXT.
        """
        self.peft_type = PeftType.PROMPT_TUNING
        if (self.prompt_tuning_init == PromptTuningInit.TEXT) and not self.tokenizer_name_or_path:
            raise ValueError(
                f"When prompt_tuning_init='{PromptTuningInit.TEXT.value}', "
                f"tokenizer_name_or_path can't be {self.tokenizer_name_or_path}."
            )
        if (self.prompt_tuning_init == PromptTuningInit.TEXT) and self.prompt_tuning_init_text is None:
            raise ValueError(
                f"When prompt_tuning_init='{PromptTuningInit.TEXT.value}', "
                f"prompt_tuning_init_text can't be {self.prompt_tuning_init_text}."
            )
        if self.tokenizer_kwargs and (self.prompt_tuning_init != PromptTuningInit.TEXT):
            raise ValueError(
                f"tokenizer_kwargs only valid when using prompt_tuning_init='{PromptTuningInit.TEXT.value}'."
            )
