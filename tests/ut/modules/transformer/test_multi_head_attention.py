# pylint: disable=C0103
"""test multi head attention."""
import unittest
from unittest import skipUnless
import numpy as np
from ddt import ddt, data, unpack
import mindspore
from mindspore import Tensor
from mindnlp import ms_jit
from mindnlp.utils import less_min_pynative_first
from mindnlp.modules import MultiheadAttention
from ....common import MindNLPTestCase

@ddt
class TestMultiHeadAttention(MindNLPTestCase):
    """ut for multi head attention"""
    @unpack
    @data(
        {'dtype': mindspore.float32, 'jit':True},
        {'dtype': mindspore.float16, 'jit':True},
        {'dtype': mindspore.float32, 'jit':False},
        {'dtype': mindspore.float16, 'jit':False},
    )
    @skipUnless(less_min_pynative_first, 'Use MindSpore API')
    def test_multihead_attention(self, dtype, jit):
        """test different dtype with jit."""
        embed_dim = 128
        num_heads = 8
        sl = 10
        bs = 8
        model = MultiheadAttention(embed_dim, num_heads).to_float(dtype)
        q = Tensor(np.random.randn(sl, bs, embed_dim), dtype)
        k = Tensor(np.random.randn(sl, bs, embed_dim), dtype)
        v = Tensor(np.random.randn(sl, bs, embed_dim), dtype)

        def forward(q, k, v):
            out = model(q, k, v)
            return out
        if jit:
            forward = ms_jit(forward)
        out = forward(q, k, v)
        self.assertEqual(q.shape, out[0].shape)

    @skipUnless(less_min_pynative_first, 'Use MindSpore API')
    @data(True, False)
    def test_multihead_attention_dtype_batch_first(self, jit):
        """test batch first."""
        embed_dim = 128
        num_heads = 8
        sl = 10
        bs = 8
        # With batch_first=True, we have the possibility of hitting
        # the native fast path if we call .eval() and enable inference
        # mode. Test both paths.
        model = MultiheadAttention(embed_dim, num_heads, batch_first=True)

        def forward(q, k, v, need_weights=False):
            out = model(q, k, v, need_weights=need_weights)
            return out

        if jit:
            forward = ms_jit(forward)

        for training in (True, False):
            if not training:
                model = model.set_train(False)
            q = Tensor(np.random.randn(sl, bs, embed_dim), mindspore.float32)
            k = Tensor(np.random.randn(sl, bs, embed_dim), mindspore.float32)
            v = Tensor(np.random.randn(sl, bs, embed_dim), mindspore.float32)
            # fast path currently doesn't support weights
            out = forward(q, k, v)
            self.assertEqual(q.shape, out[0].shape)

    @skipUnless(less_min_pynative_first, 'Use MindSpore API')
    def test_multihead_attn_same_qkv(self):
        """test same qkv."""
        mha = MultiheadAttention(4, 4, batch_first=True).set_train(False)
        query = Tensor(np.random.randn(4, 4, 4), mindspore.float32)
        mha(query, query, query)

    @skipUnless(less_min_pynative_first, 'Use MindSpore API')
    def test_multihead_attn_in_proj_bias_none(self):
        """test in_proj bias is none."""
        mha = MultiheadAttention(2, 2, bias=False)
        query = Tensor(np.random.randn(2, 2, 2), mindspore.float32)
        mha(query, query, query)

    @skipUnless(less_min_pynative_first, 'Use MindSpore API')
    def test_multihead_attn_in_proj_weight_none(self):
        """test in_proj weight is none."""
        # Setting kdim == vdim == 2 means that vdim != embed_dim
        # will cause the logic to use per-input project weights, thereby
        # forcing self.in_proj_weight = None
        mha = MultiheadAttention(4, 4, vdim=2, kdim=2)
        query = Tensor(np.random.randn(4, 4, 4), mindspore.float32)
        key = Tensor(np.random.randn(4, 4, 2), mindspore.float32)
        mha(query, key, key)
