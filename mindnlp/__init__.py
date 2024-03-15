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
if os.environ.get('HF_ENDPOINT', None) is None:
    os.environ["HF_ENDPOINT"] = 'https://hf-mirror.com'
os.environ["MS_DEV_FORCE_ACL"] = '1'

import mindspore
from mindspore import jit as ms_jit
from mindnlp import injection
from mindnlp import transformers
from mindnlp.dataset import load_dataset
from mindnlp.workflow.workflow import Workflow
from mindnlp.vocab import Vocab

__all__ = ['ms_jit', 'load_dataset', 'Workflow', 'Vocab']
