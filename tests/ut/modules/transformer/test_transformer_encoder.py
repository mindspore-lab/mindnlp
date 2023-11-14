# pylint: disable=C0103
"""test transformer encoder."""

import unittest
from unittest import skipUnless
import numpy as np
from ddt import ddt, data
import mindspore
from mindspore import ops
from mindnlp.utils.compatibility import less_min_pynative_first
from mindnlp.modules import TransformerEncoder, TransformerEncoderLayer

@ddt
class TestTransformerEncoder(unittest.TestCase):
    """ut for transformer encoder."""
    @data(True, False)
    @skipUnless(less_min_pynative_first, 'Use MindSpore API')
    def test_transformerencoder_square_input(self, training):
        """
        Test for edge cases when input of shape (batch size, sequence length, embedding dimension) has
        batch size == sequence length
        """
        model = TransformerEncoder(
            TransformerEncoderLayer(d_model=4, nhead=2, dim_feedforward=16, dropout=0.0, batch_first=True),
            num_layers=2)

        # set constant weights of the model
        for _, p in model.parameters_and_names():
            x = p.data
            sz = x.view(-1).shape[0]
            shape = x.shape
            x = ops.cos(ops.arange(0, sz).astype(mindspore.float32).view(shape))
            p.set_data(x)

        if training:
            model = model.set_train()
        else:
            model = model.set_train(False)
        x = ops.arange(0, 16).reshape(2, 2, 4).astype(mindspore.float32)
        src_mask = mindspore.Tensor([[0, 1], [0, 0]]).astype(mindspore.bool_)

        result = model(x, src_mask=src_mask)

        ref_output = mindspore.Tensor([[[2.420306205749512, 0.017629241570830, -0.607857942581177, -0.085519507527351],
                                    [2.420306205749512, 0.017629241570830, -0.607857942581177, -0.085519507527351]],
                                   [[2.419836044311523, 0.017548924311996, -0.608187675476074, -0.085347734391689],
                                    [2.419836044311523, 0.017548924311996, -0.608187675476074, -0.085347734391689]]]
                                  , mindspore.float32)
        self.assertEqual(tuple(result.shape), tuple(ref_output.shape))
        np.allclose(result.asnumpy(), ref_output.asnumpy(), rtol=1e-7, atol=1e-5)
