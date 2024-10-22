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
# ============================================================================
# pylint: disable=C0301
"IA3 Config"
from dataclasses import dataclass, field
from typing import List, Optional, Union

from mindnlp.peft.config import PeftConfig
from mindnlp.peft.utils import PeftType


@dataclass
class IA3Config(PeftConfig):
    """
    This is the configuration class to store the configuration of a [`IA3Model`].

    Args:
        target_modules (`Optional[Union[List[str], str]]`):
            The names of the cells to apply the adapter to. If this is specified, only the cells with the specified
            names will be replaced. When passing a string, a regex match will be performed. When passing a list of
            strings, either an exact match will be performed or it is checked if the name of the cell ends with any
            of the passed strings. If this is specified as 'all-linear', then all linear/Conv1D cells are chosen,
            excluding the output layer. If this is not specified, cells will be chosen according to the model
            architecture. If the architecture is not known, an error will be raised -- in this case, you should specify
            the target cells manually.
        feedforward_cells (`Optional[Union[List[str], str]]`):
            The names of the cells to be treated as feedforward cells, as in the original paper. These cells will
            have (IA)³ vectors multiplied to the input, instead of the output. `feedforward_cells` must be a name or
            a subset of names present in `target_modules`.
        fan_in_fan_out (`bool`):
            Set this to True if the layer to replace stores weight like (fan_in, fan_out). For example, gpt-2 uses
            `Conv1D` which stores weights like (fan_in, fan_out) and hence this should be set to `True`.
        modules_to_save (`Optional[List[str]]`):
            List of cells apart from (IA)³ layers to be set as trainable and saved in the final checkpoint.
        init_ia3_weights (`bool`):
            Whether to initialize the vectors in the (IA)³ layers, defaults to `True`. Setting this to `False` is
            discouraged.
    """
    target_modules: Optional[Union[List[str], str]] = field(
        default=None,
        metadata={
            "help": (
                "List of cell names or regex expression of the cell names to replace with (IA)³."
                "For example, ['q', 'v'] or '.*decoder.*(SelfAttention|EncDecAttention).*(q|v)$'."
                "This can also be a wildcard 'all-linear' which matches all linear/Conv1D layers except the output layer."
                "If not specified, cells will be chosen according to the model architecture, If the architecture is "
                "not known, an error will be raised -- in this case, you should specify the target cells manually."
            ),
        },
    )
    feedforward_cells: Optional[Union[List[str], str]] = field(
        default=None,
        metadata={
            "help": "List of cell names or a regex expression of cell names which are feedforward"
            "For example, ['output.dense']"
        },
    )
    fan_in_fan_out: bool = field(
        default=False,
        metadata={"help": "Set this to True if the layer to replace stores weight like (fan_in, fan_out)"},
    )
    modules_to_save: Optional[List[str]] = field(
        default=None,
        metadata={
            "help": "List of cells apart from (IA)^3 layers to be set as trainable and saved in the final checkpoint. "
            "For example, in Sequence Classification or Token Classification tasks, "
            "the final layer `classifier/score` are randomly initialized and as such need to be trainable and saved."
        },
    )
    init_ia3_weights: bool = field(
        default=True,
        metadata={"help": "Whether to initialize the vectors in the (IA)^3 layers."},
    )

    def __post_init__(self):
        r"""
        This method initializes the IA3Config class after its instance has been created.
        
        Args:
            self: An instance of the IA3Config class.
        
        Returns:
            None.
        
        Raises:
            ValueError: If the `feedforward_cells` parameter is not a subset of the `target_modules` parameter.
        
        Description:
            The __post_init__ method sets default values for the IA3Config instance. It assigns the PeftType.IA3 value to the
            peft_type attribute. The target_modules and feedforward_cells attributes are converted to sets if they are provided as
            lists, or left unchanged if they are already sets.
        
            The method then performs a check to ensure that if both target_modules and feedforward_cells are sets, the
            feedforward_cells subset is a subset of the target_modules set. If this check fails, a ValueError exception is raised
            with the message '`feedforward_cells` should be a subset of `target_modules`'.
        """
        self.peft_type = PeftType.IA3
        self.target_modules = (
            set(self.target_modules) if isinstance(self.target_modules, list) else self.target_modules
        )
        self.feedforward_cells = (
            set(self.feedforward_cells) if isinstance(self.feedforward_cells, list) else self.feedforward_cells
        )

        # check if feedforward_cells is a subset of target_modules. run the check only if both are sets
        if isinstance(self.feedforward_cells, set) and isinstance(self.target_modules, set):
            if not self.feedforward_cells.issubset(self.target_modules):
                raise ValueError("`feedforward_cells` should be a subset of `target_modules`")
