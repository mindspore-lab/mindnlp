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

# huggingface env
if os.environ.get('HF_ENDPOINT', None) is None:
    os.environ["HF_ENDPOINT"] = 'https://hf-mirror.com'

# set mindnlp.core to torch
import mindtorch
from .utils.safetensors_patch import setup_safetensors_patch
setup_safetensors_patch()

from . import transformers
from . import diffusers

__version__ = '0.5.0rc2'
