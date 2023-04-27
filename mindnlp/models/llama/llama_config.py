# Copyright 2023 Huawei Technologies Co., Ltd
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
"""llama model configuration"""

from mindnlp.abc import PreTrainedConfig

class LlamaConfig(PreTrainedConfig):
    """
    Configuration for Llama
    """
    def __init__(
            self,
            dim=512,
            n_layers=8,
            n_heads=8,
            vocab_size=-1,  # defined later by tokenizer
            multiple_of=256,  # make SwiGLU hidden layer size multiple of large power of 2
            norm_eps=1e-5,
            max_batch_size=32,
            max_seq_len=2048,
            **kwargs):
        super().__init__(**kwargs)
        self.vocab_size = vocab_size
        self.n_layers = n_layers
        self.max_seq_len = max_seq_len
        self.dim = dim
        self.n_heads = n_heads
        self.multiple_of = multiple_of
        self.max_batch_size = max_batch_size
        self.norm_eps = norm_eps

__all__ = ['LlamaConfig']
