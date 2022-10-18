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
"""
TextGeneration dataset init
"""

from .wikitext2 import WikiText2, WikiText2_Process
from .wikitext103 import WikiText103, WikiText103_Process
from .penntreebank import PennTreebank, PennTreebank_Process
from .lcsts import LCSTS, LCSTS_Process
