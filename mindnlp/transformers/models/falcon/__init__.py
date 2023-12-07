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
Falcon Model.
"""

from . import falcon, config_falcon

from .falcon import *
from .config_falcon import *

__all__ = []
__all__.extend(falcon.__all__)
__all__.extend(config_falcon.__all__)

from typing import TYPE_CHECKING

from mindnlp.utils import (
    OptionalDependencyNotAvailable,
    _LazyModule,
    is_mindspore_available,
)


_import_structure = {
    "config_falcon": ["FALCON_PRETRAINED_CONFIG_ARCHIVE_MAP", "FalconConfig"],
}

try:
    if not is_mindspore_available():
        raise OptionalDependencyNotAvailable()
except OptionalDependencyNotAvailable:
    pass
else:
    _import_structure["falcon"] = [
        "FALCON_PRETRAINED_MODEL_ARCHIVE_LIST",
        "FalconForCausalLM",
        "FalconModel",
        "FalconPreTrainedModel",
        "FalconForSequenceClassification",
        "FalconForTokenClassification",
        "FalconForQuestionAnswering",
    ]


if TYPE_CHECKING:
    from .config_falcon import FALCON_PRETRAINED_CONFIG_ARCHIVE_MAP, FalconConfig

    try:
        if not is_mindspore_available():
            raise OptionalDependencyNotAvailable()
    except OptionalDependencyNotAvailable:
        pass
    else:
        from .falcon import (
            FALCON_PRETRAINED_MODEL_ARCHIVE_LIST,
            FalconForCausalLM,
            FalconForQuestionAnswering,
            FalconForSequenceClassification,
            FalconForTokenClassification,
            FalconModel,
            FalconPreTrainedModel,
        )


else:
    import sys

    sys.modules[__name__] = _LazyModule(
        __name__, globals()["__file__"], _import_structure, module_spec=__spec__
    )
