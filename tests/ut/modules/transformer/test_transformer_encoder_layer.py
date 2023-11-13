"""test transformer encoder layer."""
import unittest
from unittest import skipUnless
import numpy as np
from ddt import ddt, data, unpack
import mindspore
from mindspore import Tensor
from mindnlp import ms_jit
from mindnlp.modules import TransformerEncoderLayer
from mindnlp.utils.compatibility import less_min_pynative_first
from ....common import MindNLPTestCase

@ddt
class TestTransformerEncoderLayer(MindNLPTestCase):
    """ut for transformer encoder layer."""
    @unpack
    @data(
        {'nhead': 1, 'jit': False},
        {'nhead': 4, 'jit': True},
        {'nhead': 8, 'jit': False}
    )
    @skipUnless(less_min_pynative_first, 'Use MindSpore API')
    def test_transformerencoderlayer_src_mask(self, nhead, jit):
        """test nhead setting forward with jit."""
        batch_size = 2
        seqlen = 4
        d_model = 8
        dim_feedforward = 32

        model = TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            batch_first=True)
        src = Tensor(np.random.rand(batch_size, seqlen, d_model), mindspore.float32)
        src_mask = Tensor(np.zeros((seqlen, seqlen)), mindspore.bool_)

        def forward(src, src_mask):
            return model(src, src_mask=src_mask)

        if jit:
            forward = ms_jit(forward)
        forward(src, src_mask)
        model.set_train(False)
        forward(src, src_mask)
