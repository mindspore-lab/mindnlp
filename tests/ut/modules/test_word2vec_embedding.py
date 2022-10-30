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
# pylint: disable=import-outside-toplevel
"""Test Word2vec_embedding"""

import unittest
import pytest
import numpy as np
import mindspore
from mindspore import Tensor
from mindspore.dataset.text import Vocab
from mindnlp.modules.embeddings.word2vec_embedding import Word2vec


class TestWord2vec(unittest.TestCase):
    r"""
    Test module Word2vec
    """

    def setUp(self):
        self.input = None

    @pytest.mark.skip(reason="this ut has already tested")
    def test_word2vec_embedding(self):
        r"""
        Unit test for word2vec embedding.
        """
        import torch
        from torch.nn.modules.sparse import Embedding as Embedding

        # pytorch embedding
        embed_pt = torch.nn.Embedding(10, 3)
        input_pt = torch.LongTensor([0, 1, 2])

        # mindnlp embedding
        vocab_ms = Vocab.from_list(["one", "two", "three"])
        init_embed = Tensor(embed_pt.weight.detach().numpy())
        embed_ms = Word2vec(vocab_ms, init_embed, dropout=0.0)
        input_ms= Tensor([0, 1, 2])

        # forward
        import time
        pt_s = time.time()
        output_pt = embed_pt(input_pt)
        pt_t = time.time() - pt_s

        ms_s = time.time()
        output_ms = embed_ms(input_ms)
        ms_t = time.time() - ms_s

        print("pytorch:", pt_t)
        print("mindnlp:", ms_t)

        assert np.allclose(output_ms.asnumpy(), output_pt.detach().numpy(), 1e-3, 1e-3)

        # backward
        grad_fn = mindspore.grad(embed_ms, grad_position=0, weights=embed_ms.trainable_params(), has_aux=False)
        embed_ms_grads = grad_fn(input_ms)

        output_pt.backward(torch.ones_like(output_pt), retain_graph=True)
        embed_pt_grads = [param.grad for param in embed_pt.parameters()]

        for ms_grad, pt_grad in zip(embed_ms_grads, embed_pt_grads):
            assert np.mean(ms_grad.asnumpy() - pt_grad.detach().numpy()) < 1e-3
