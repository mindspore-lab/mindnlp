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
Global configs
"""
import os
from packaging import version
import mindspore
from mindspore._c_expression import MSContext # pylint: disable=no-name-in-module, import-error

SOC = MSContext.get_instance().get_ascend_soc_version()
DEVICE_TARGET = mindspore.get_context('device_target')
GENERATOR_SEED = version.parse(mindspore.__version__) >= version.parse('2.3.0')
SUPPORT_VIEW = GENERATOR_SEED
SUPPORT_BF16 = GENERATOR_SEED and '910b' in SOC
USE_PYBOOST = version.parse(mindspore.__version__) >= version.parse('2.3.0') and DEVICE_TARGET == 'Ascend'
DEFAULT_DTYPE = mindspore.float32

WEIGHTS_NAME = "mindspore_model.ckpt"
PT_WEIGHTS_NAME = "pytorch_model.bin"
WEIGHTS_INDEX_NAME = "mindspore_model.ckpt.index.json"
PT_WEIGHTS_INDEX_NAME = "pytorch_model.bin.index.json"
SAFE_WEIGHTS_NAME = "model.safetensors"
SAFE_WEIGHTS_INDEX_NAME = "model.safetensors.index.json"

CONFIG_NAME = "config.json"
GENERATION_CONFIG_NAME = "generation_config.json"
TOKENIZER_CONFIG_FILE = "tokenizer_config.json"

FEATURE_EXTRACTOR_NAME = "preprocessor_config.json"
IMAGE_PROCESSOR_NAME = FEATURE_EXTRACTOR_NAME
PROCESSOR_NAME = "processor_config.json"

ADAPTER_CONFIG_NAME = "adapter_config.json"
ADAPTER_WEIGHTS_NAME = "adapter_model.bin"
ADAPTER_SAFE_WEIGHTS_NAME = "adapter_model.safetensors"

DEFAULT_ROOT = os.path.join(os.getcwd(), ".mindnlp")
# for modelscope models
MS_URL_BASE = "https://modelscope.cn/api/v1/models/{}/repo?Revision={}&FilePath={}"
# for huggingface url
HF_ENDPOINT = os.environ.get('HF_ENDPOINT', 'https://hf-mirror.com')
HF_URL_BASE = HF_ENDPOINT + '/{}/resolve/{}/{}?download=true'

ENV_VARS_TRUE_VALUES = {"1", "ON", "YES", "TRUE"}
MINDNLP_CACHE = os.getenv("MINDNLP_CACHE", DEFAULT_ROOT)

REPO_TYPE_DATASET = "dataset"
REPO_TYPE_MODEL = "model"
REPO_TYPES = [None, REPO_TYPE_MODEL, REPO_TYPE_DATASET]

# Token
HF_TOKEN = os.environ.get('HF_TOKEN', None)

# Values
OPENAI_CLIP_MEAN = [0.48145466, 0.4578275, 0.40821073]
OPENAI_CLIP_STD = [0.26862954, 0.26130258, 0.27577711]
IMAGENET_DEFAULT_MEAN = [0.485, 0.456, 0.406]
IMAGENET_DEFAULT_STD = [0.229, 0.224, 0.225]
IMAGENET_STANDARD_MEAN = [0.5, 0.5, 0.5]
IMAGENET_STANDARD_STD = [0.5, 0.5, 0.5]
