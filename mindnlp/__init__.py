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
# pylint: disable=wrong-import-position
"""
MindNLP library.
"""
import os
import platform
if os.environ.get('HF_ENDPOINT', None) is None:
    os.environ["HF_ENDPOINT"] = 'https://hf-mirror.com'
os.environ["MS_DEV_FORCE_ACL"] = '1'
os.environ['TOKENIZERS_PARALLELISM'] = 'false'

from mindspore import context
from mindspore._c_expression import MSContext # pylint: disable=no-name-in-module, import-error

if platform.system().lower() == 'linux':
    SOC = MSContext.get_instance().get_ascend_soc_version()
    if '910' in SOC:
        os.environ["MS_ALLOC_CONF"] = 'enable_vmm:True,vmm_align_size:2MB'

    if SOC == 'ascend910':
        context.set_context(ascend_config={"precision_mode": "allow_mix_precision"})

from mindspore import jit as ms_jit
from mindnlp import transformers
from mindnlp import dataset
from mindnlp import evaluate
from mindnlp import core

__all__ = ['ms_jit', 'transformers']
