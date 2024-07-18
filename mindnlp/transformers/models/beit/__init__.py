# Copyright 2024 Huawei Technologies Co., Ltd
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
Beit Model.
"""
from . import configuration_beit, image_processing_beit, modeling_beit
from .configuration_beit import *
from .image_processing_beit import *
from .modeling_beit import *

__all__ = []
__all__.extend(configuration_beit.__all__)
__all__.extend(image_processing_beit.__all__)
__all__.extend(modeling_beit.__all__)
