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
# ============================================================================
"""
MindNLP library.
"""

from mindnlp.utils import less_min_pynative_first
if less_min_pynative_first:
    from mindspore import context
    from mindspore import ms_function as ms_jit
    context.set_context(mode=context.PYNATIVE_MODE)
else:
    from mindspore import jit as ms_jit

__all__ = ['ms_jit']
