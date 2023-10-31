#!/usr/bin/env python
# coding: utf-8
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
# pylint: disable=C0103
"""test longformer for sequence classification."""
import time
import numpy as np
import mindspore
from mindspore import nn, Tensor
from mindnlp.transformers import LongformerConfig, LongformerForSequenceClassification

def test_train_longformer_pynative():
    """test train longformer."""
    epochs = 1
    batch_size = 2
    lr = 1e-4
    seq_len = 512
    steps = 10

    config = LongformerConfig(attention_window=[8, 8], num_hidden_layers=2, max_position_embeddings=seq_len+1, vocab_size=30, num_layers=2,
                              num_labels=3, pad_token_id=0)
    model = LongformerForSequenceClassification(config)
    optimizer = nn.AdamWeightDecay(model.trainable_params(), lr)

    def forward_fn(input_ids, labels):
        outputs = model(input_ids, labels=labels)
        return outputs[0]

    grad_fn = mindspore.value_and_grad(forward_fn, None, optimizer.parameters)

    for _ in range(epochs):
        s = time.time()
        for i in range(steps):
            input_ids = Tensor(np.random.randint(1, config.vocab_size, (batch_size, seq_len)), dtype=mindspore.int32)
            labels = Tensor(np.random.randint(0, config.num_labels, (batch_size,)), mindspore.int32)
            loss, grads = grad_fn(input_ids, labels)
            t = time.time()
            print(f"loss: {loss}, cost time: {(t - s) / (i + 1):.3f} s/step")
            optimizer(grads)
