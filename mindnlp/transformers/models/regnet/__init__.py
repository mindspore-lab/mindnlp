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

from typing import TYPE_CHECKING

from ....utils import (
    OptionalDependencyNotAvailable,
    _LazyModule,
    is_mindspore_available,
)

# 定义导入结构
_import_structure = {"configuration_regnet": ["REGNET_PRETRAINED_CONFIG_ARCHIVE_MAP", "RegNetConfig"]}

# 检查是否有 mindspore 库可用
try:
    if not is_mindspore_available():
        raise OptionalDependencyNotAvailable()
except OptionalDependencyNotAvailable:
    pass
else:
    # 如果 mindspore 可用，添加额外的导入结构
    _import_structure["modeling_regnet"] = [
        "REGNET_PRETRAINED_MODEL_ARCHIVE_LIST",
        "RegNetForImageClassification",
        "RegNetModel",
        "RegNetPreTrainedModel",
    ]

# 如果是类型检查阶段，进行额外的导入
if TYPE_CHECKING:
    from .configuration_regnet import REGNET_PRETRAINED_CONFIG_ARCHIVE_MAP, RegNetConfig
    try:
        if not is_mindspore_available():
            raise OptionalDependencyNotAvailable()
    except OptionalDependencyNotAvailable:
        pass
    else:
        from .modeling_regnet import (
            REGNET_PRETRAINED_MODEL_ARCHIVE_LIST,
            RegNetForImageClassification,
            RegNetModel,
            RegNetPreTrainedModel,
        )

# 如果不是类型检查阶段，将模块设置为懒加载模块
else:
    import sys
    sys.modules[__name__] = _LazyModule(__name__, globals()["__file__"], _import_structure)
