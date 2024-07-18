# Copyright 2023 The HuggingFace Team. All rights reserved.
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
"""
FastSpeech2-Conformer Model.
"""
from typing import TYPE_CHECKING

from . import (
    configuration_fastspeech2_conformer,
    modeling_fastspeech2_conformer,
    tokenization_fastspeech2_conformer
)
from .configuration_fastspeech2_conformer import *
from .modeling_fastspeech2_conformer import *
from .tokenization_fastspeech2_conformer import *

from ....utils import (
    OptionalDependencyNotAvailable,
    is_mindspore_available,
)

_import_structure = {
    "configuration_fastspeech2_conformer": [
        "FastSpeech2ConformerConfig",
        "FastSpeech2ConformerHifiGanConfig",
        "FastSpeech2ConformerWithHifiGanConfig",
    ],
    "tokenization_fastspeech2_conformer": ["FastSpeech2ConformerTokenizer"],
}

try:
    if not is_mindspore_available():
        raise OptionalDependencyNotAvailable()
except OptionalDependencyNotAvailable:
    pass
else:
    _import_structure["modeling_fastspeech2_conformer"] = [
        "FastSpeech2ConformerWithHifiGan",
        "FastSpeech2ConformerHifiGan",
        "FastSpeech2ConformerModel",
        "FastSpeech2ConformerPreTrainedModel",
    ]

if TYPE_CHECKING:
    from .configuration_fastspeech2_conformer import (
        FastSpeech2ConformerConfig,
        FastSpeech2ConformerHifiGanConfig,
        FastSpeech2ConformerWithHifiGanConfig,
    )
    from .tokenization_fastspeech2_conformer import FastSpeech2ConformerTokenizer

    try:
        if not is_mindspore_available():
            raise OptionalDependencyNotAvailable()
    except OptionalDependencyNotAvailable:
        pass
    else:
        from .modeling_fastspeech2_conformer import (
            FastSpeech2ConformerHifiGan,
            FastSpeech2ConformerModel,
            FastSpeech2ConformerPreTrainedModel,
            FastSpeech2ConformerWithHifiGan,
        )

__all__ = []
__all__.extend(configuration_fastspeech2_conformer.__all__)
__all__.extend(modeling_fastspeech2_conformer.__all__)
__all__.extend(tokenization_fastspeech2_conformer.__all__)
