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

WEIGHTS_NAME = "mindspore.ckpt"
PT_WEIGHTS_NAME = "pytorch_model.bin"
WEIGHTS_INDEX_NAME = "mindspore.ckpt.index.json"
PT_WEIGHTS_INDEX_NAME = "pytorch_model.bin.index.json"

CONFIG_NAME = "config.json"
GENERATION_CONFIG_NAME = "generation_config.json"
TOKENIZER_CONFIG_FILE = "tokenizer_config.json"

FEATURE_EXTRACTOR_NAME = "preprocessor_config.json"

DEFAULT_ROOT = os.path.join(os.getcwd(), ".mindnlp")
# for modelscope models
MS_URL_BASE = "https://modelscope.cn/api/v1/models/mindnlp/{}/repo?Revision=master&FilePath={}"
# for huggingface url
HF_URL_BASE = os.environ.get('HF_ENDPOINT', 'https://hf-mirror.com') + '/{}/resolve/main/{}'

ENV_VARS_TRUE_VALUES = {"1", "ON", "YES", "TRUE"}
MINDNLP_CACHE = os.getenv("MINDNLP_CACHE", DEFAULT_ROOT)

REPO_TYPE_DATASET = "dataset"
REPO_TYPE_MODEL = "model"
REPO_TYPES = [None, REPO_TYPE_MODEL, REPO_TYPE_DATASET]
