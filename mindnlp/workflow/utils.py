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
Workflow Utils
"""

import re

def cut_chinese_sent(para):
    """
    Cut the Chinese sentences more precisely.
    """
    para = re.sub(r"([。！？\?])([^”’])", r"\1\n\2", para)
    para = re.sub(r"(\.{6})([^”’])", r"\1\n\2", para)
    para = re.sub(r"(\…{2})([^”’])", r"\1\n\2", para)
    para = re.sub(r"([。！？\?][”’])([^，。！？\?])", r"\1\n\2", para)
    para = para.rstrip()
    return para.split("\n")
