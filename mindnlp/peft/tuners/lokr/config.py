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
"""lokr."""
from dataclasses import dataclass, field
from typing import List, Optional, Union

from ...config import PeftConfig
from ...utils import PeftType


@dataclass
class LoKrConfig(PeftConfig):
    """
    This is the configuration class to store the configuration of a [`LoraModel`].

    Args:
        r (`int`): lokr attention dimension.
        target_modules (`Union[List[str],str]`): The names of the cells to apply Lora to.
        lora_alpha (`float`): The alpha parameter for Lokr scaling.
        rank_dropout (`float`):The dropout probability for rank dimension during training.
        cell_dropout (`float`): The dropout probability for LoKR layers.
        use_effective_conv2d (`bool`):
            Use parameter effective decomposition for
            Conv2d with ksize > 1 ("Proposition 3" from FedPara paper).
        decompose_both (`bool`):Perform rank decomposition of left kronecker product matrix.
        decompose_factor (`int`):Kronecker product decomposition factor.

        bias (`str`): Bias type for Lora. Can be 'none', 'all' or 'lora_only'
        modules_to_save (`List[str]`):
            List of cells apart from LoRA layers to be set as trainable
            and saved in the final checkpoint.
        init_weights (`bool`):
            Whether to perform initialization of adapter weights. This defaults to `True`, 
            passing `False` is discouraged.
        layers_to_transform (`Union[List[int],int]`):
            The layer indexes to transform, if this argument is specified, it will apply the LoRA transformations on
            the layer indexes that are specified in this list. If a single integer is passed, it will apply the LoRA
            transformations on the layer at this index.
        layers_pattern (`str`):
            The layer pattern name, used only if `layers_to_transform` is different from `None` and if the layer
            pattern is not in the common layers pattern.
        rank_pattern (`dict`):
            The mapping from layer names or regexp expression to ranks which are different from the default rank
            specified by `r`.
        alpha_pattern (`dict`):
            The mapping from layer names or regexp expression to alphas which are different from the default alpha
            specified by `alpha`.
    """
    r: int = field(default=8, metadata={"help": "lokr attention dimension"})
    target_modules: Optional[Union[List[str], str]] = field(
        default=None,
        metadata={
            "help": "List of cell names or regex expression of the cell names to replace with Lora."
            "For example, ['q', 'v'] or '.*decoder.*(SelfAttention|EncDecAttention).*(q|v)$' "
        },
    )
    lora_alpha: int = field(default=8, metadata={"help": "lokr alpha"})
    rank_dropout: float = field(
        default=0.0,
        metadata={"help": "The dropout probability for rank dimension during training"},
    )
    cell_dropout: float = field(default=0.0, metadata={"help": "lokr dropout"})
    use_effective_conv2d: bool = field(
        default=False,
        metadata={
            "help": 'Use parameter effective decomposition for Conv2d 3x3 with ksize > 1 ("Proposition 3" from FedPara paper)'
        },
    )
    decompose_both: bool = field(
        default=False,
        metadata={
            "help": "Perform rank decomposition of left kronecker product matrix."
        },
    )
    decompose_factor: int = field(
        default=-1, metadata={"help": "Kronecker product decomposition factor."}
    )

    bias: str = field(
        default="none",
        metadata={"help": "Bias type for Lora. Can be 'none', 'all' or 'lora_only'"},
    )
    modules_to_save: Optional[List[str]] = field(
        default=None,
        metadata={
            "help": "List of cells apart from LoRA layers to be set as trainable and saved in the final checkpoint. "
            "For example, in Sequence Classification or Token Classification tasks, "
            "the final layer `classifier/score` are randomly initialized and as such need to be trainable and saved."
        },
    )
    init_weights: bool = field(
        default=True,
        metadata={"help": "Whether to initialize the weights of the Lora layers."},
    )
    layers_to_transform: Optional[Union[List, int]] = field(
        default=None,
        metadata={
            "help": "The layer indexes to transform, is this argument is specified, \
                PEFT will transform only the layers indexes that are specified inside this list. \
                If a single integer is passed, PEFT will transform only the layer at this index."
        },
    )
    layers_pattern: Optional[str] = field(
        default=None,
        metadata={
            "help": "The layer pattern name, used only if `layers_to_transform` is different to None and \
                  if the layer pattern is not in the common layers pattern."
        },
    )
    rank_pattern: Optional[dict] = field(
        default_factory=dict,
        metadata={
            "help": (
                "The mapping from layer names or regexp expression to ranks which are different from the default rank specified by `r`. "
                "For example, `{model.decoder.layers.0.encoder_attn.k_proj: 8`}"
            )
        },
    )
    alpha_pattern: Optional[dict] = field(
        default_factory=dict,
        metadata={
            "help": (
                "The mapping from layer names or regexp expression to alphas which are different from the default alpha specified by `alpha`. "
                "For example, `{model.decoder.layers.0.encoder_attn.k_proj: 32`}"
            )
        },
    )

    def __post_init__(self):
        r"""
        Method to initialize the attributes of the LoKrConfig class after object creation.
        
        Args:
            self: Instance of the LoKrConfig class.
        
        Returns:
            None. This method performs attribute initialization within the class.
        
        Raises:
            No specific exceptions are raised within this method.
        """
        self.peft_type = PeftType.LOKR

    @property
    def is_prompt_learning(self):
        r"""
        Utility method to check if the configuration is for prompt learning.
        """
        return False
