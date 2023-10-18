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
# pylint: disable=C0413
# pylint: disable=C0412
# pylint: disable=C0411

"""
dataset processing transforms
"""

from mindnlp.utils import less_min_pynative_first

if less_min_pynative_first:
    from mindnlp._legacy.transforms import Truncate, AddToken
else:
    from mindspore.dataset.text import Truncate, AddToken

from .lookup import Lookup
from .basic_tokenizer import BasicTokenizer
from .pad_transform import PadTransform

__all__ = [
    'Truncate', 'AddToken', 'Lookup', 'PadTransform', 'BasicTokenizer',
]
