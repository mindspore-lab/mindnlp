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
# ============================================================================
"""MindNLP Activations"""

from collections import OrderedDict
from mindspore import nn


class ClassInstantier(OrderedDict):
    r"""
    Class Instantier
    """

    def __getitem__(self, key):
        content = super().__getitem__(key)
        cls, kwargs = content if isinstance(content, tuple) else (content, {})
        return cls(**kwargs)


ACT2CLS = {
    'relu': nn.ReLU,
    'gelu': (nn.GELU, {"approximate=": False}),
    'gelu_new': nn.GELU,
    'gelu_approximate': (nn.GELU, {"approximate=": True}),
    "swish": nn.SiLU,
}
ACT2FN = ClassInstantier(ACT2CLS)
