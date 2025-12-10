# Copyright 2021 Huawei Technologies Co., Ltd
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
"""
Compatibility layer for mindnlp.diffusers - redirects to mindhf.diffusers
"""
from mindhf.diffusers import *
import warnings

warnings.warn(
    "The usage 'from mindnlp.diffusers import xx' is deprecated. "
    "Please use 'import mindhf; from diffusers import xxx' instead.",
    DeprecationWarning,
    stacklevel=2
)
