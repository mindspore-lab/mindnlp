# coding=utf-8
# Copyright 2024 BlinkDL, et al.
# Copyright 2024 yuunnn-w, et al.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Modifications copyright 2024 [Huawei Technologies Co., Ltd]
# Changes: Migrated to MindSpore interface
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ============================================================================
import time
import mindspore
from mindspore import ops
from mindnlp.experimental.rwkv6.modeling_rwkv6 import RWKV_RNN
from mindnlp.experimental.rwkv6.tokenizer_rwkv6 import RWKV_TOKENIZER
from mindnlp.experimental.rwkv6.sampler_rwkv6 import sample_logits

if __name__ == '__main__':
    args = {
        'MODEL_NAME': 'RWKV-x060-World-1B6-v2.1-20240328-ctx4096', #模型文件的名字，ckpt结尾的权重文件。
        'vocab_size': 65536, #词表大小
        'batch_size': 3
    }

    # 加载模型和分词器
    print("Loading model and tokenizer...")
    model = RWKV_RNN(args)
    tokenizer = RWKV_TOKENIZER()
    print("Done.")

    # 设置续写的初始字符串和参数
    BATCH_SIZE = args['batch_size']
    initial_string = "User: 帮我用python写一个打印字符三角形的代码.\n\nAssistant: "
    TEMPERATURE = 2.5  # 温度参数
    TOP_P = 0.1  # Top-p采样参数
    LENGTH_PER_TRIAL = 50  # 生成的长度

    # 编码初始字符串
    token = mindspore.Tensor(tokenizer.encode([initial_string] * BATCH_SIZE), dtype=mindspore.int64)
    for t in ops.unbind(token, dim=-1):
        out = model(t)
    else:
        token_sampled = sample_logits(out, TEMPERATURE, TOP_P).type_as(token)
        token = ops.cat((token, token_sampled.unsqueeze(1)), 1)

    start_time = time.time() # 开始计时
    for step in range(LENGTH_PER_TRIAL):  # 生成指定数量的token
        out = model(token_sampled)
        token_sampled = sample_logits(out, TEMPERATURE, TOP_P).type_as(token)
        token = ops.cat((token, token_sampled.unsqueeze(1)), 1)
    end_time = time.time() # 结束计时

    # 打印结果
    decoded_sequences = tokenizer.decode(token.tolist())
    for i, seq in enumerate(decoded_sequences):
        print(f"Batch {i+1}: {seq}")
    
    total_time = end_time - start_time
    tokens_generated = LENGTH_PER_TRIAL * BATCH_SIZE
    speed = tokens_generated / total_time
    speed_per_batch = speed / BATCH_SIZE
    print(f"\nTotal time: {total_time:.2f} seconds")
    print(f"Tokens generated: {tokens_generated}")
    print(f"Token generation speed: {speed:.2f} tokens/second")
    print(f"Token generation speed per batch: {speed_per_batch:.2f} tokens/second")
    