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

CONFIG_NAME = "config.json"
WEIGHTS_NAME = "mindspore.ckpt"
GENERATION_CONFIG_NAME = "generation_config.json"
TOKENIZER_CONFIG_FILE = "tokenizer_config.json"

DEFAULT_ROOT = os.path.join(os.getcwd(), ".mindnlp")
# for modelscope models
MS_CONFIG_URL_BASE = "https://modelscope.cn/api/v1/models/mindnlp/{}/repo?Revision=master&FilePath=config.json"
MS_TOKENIZER_CONFIG_URL_BASE = "https://modelscope.cn/api/v1/models/mindnlp/{}/repo?Revision=master&FilePath=tokenizer.json"
MS_MODEL_URL_BASE = "https://modelscope.cn/api/v1/models/mindnlp/{}/repo?Revision=master&FilePath=mindspore.ckpt"
# for huggingface url
HF_CONFIG_URL_BASE = 'https://hf-mirror.com/{}/resolve/main/config.json'
HF_TOKENIZER_CONFIG_URL_BASE = 'https://hf-mirror.com/{}/resolve/main/tokenizer.json'
HF_MODEL_URL_BASE = 'https://hf-mirror.com/{}/resolve/main/pytorch_model.bin'
HF_VOCAB_URL_BASE = 'https://hf-mirror.com/{}/resolve/main/vocab.txt'
# for mindnlp obs storage
MINDNLP_CONFIG_URL_BASE =  "https://download.mindspore.cn/toolkits/mindnlp/models/{}/{}/config.json"
MINDNLP_MODEL_URL_BASE =  "https://download.mindspore.cn/toolkits/mindnlp/models/{}/{}/mindspore.ckpt"
MINDNLP_TOKENIZER_CONFIG_URL_BASE =  "https://download.mindspore.cn/toolkits/mindnlp/models/{}/{}/tokenizer.json"
MINDNLP_VOCAB_URL_BASE =  "https://download.mindspore.cn/toolkits/mindnlp/models/{}/{}/vocab.txt"
