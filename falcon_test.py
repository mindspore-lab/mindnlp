# pylint: disable=E1101
import numpy as np
import mindspore
import torch
from mindspore import Tensor
# from mindspore import nn

from transformers.models.falcon import modeling_falcon as falcon_pt
from mindnlp.transformers.models.falcon import falcon as falcon_ms


dtype_list = [(mindspore.float32, torch.float32)]


class falcon_test():

    def __init__(self):
        print("<----------falcon_test init---------->\n")

    def test_FalconLinear(self, ms_dtype, pt_dtype):
        print(">==========test_FalconLinear")
        # init model
        pt_model = falcon_pt.FalconLinear(3, 4)
        ms_model = falcon_ms.FalconLinear(3, 4)
        # load parameters
        pt_params = pt_model.state_dict()
        for key, param in ms_model.parameters_and_names():
            param.set_data(Tensor(pt_params.get(key).detach().numpy()))
        # set eval mode
        ms_model.set_train(False)
        pt_model.eval()
        # prepare data
        x = np.random.randn(4, 3)
        ms_x = Tensor(x, dtype=ms_dtype)
        pt_x = torch.tensor(x, dtype=pt_dtype)
        # output
        pt_out = pt_model(pt_x)
        ms_out = ms_model(ms_x)
        # shape & loss
        assert ms_out.shape == pt_out.shape
        assert np.allclose(
            ms_out.asnumpy(), pt_out.detach().numpy(), 1e-5, 1e-5)
        print(">===========test_FalconLinear_end\n")

    def test_FalconRotaryEmbedding(self, ms_dtype, pt_dtype):
        print(">==========test_FalconRotaryEmbedding")
        seq_len = 12
        batch_size = 32
        embed_dim = 128
        # init model
        pt_model = falcon_pt.FalconRotaryEmbedding(128)
        ms_model = falcon_ms.FalconRotaryEmbedding(128)
        # set eval mode
        ms_model.set_train(False)
        pt_model.eval()
        # data
        query = np.random.randn(batch_size, seq_len, embed_dim)
        key = np.random.randn(batch_size, seq_len, embed_dim)
        # prepare data
        pt_query = torch.tensor(query, dtype=pt_dtype)
        pt_key = torch.tensor(key, dtype=pt_dtype)
        pt_position_ids = torch.tensor(  # position_ids
            np.arange(seq_len).reshape(1, seq_len),
            dtype=torch.int64)
        ms_query = Tensor(query, dtype=ms_dtype)
        ms_key = Tensor(key, dtype=ms_dtype)
        ms_position_ids = Tensor(  # position_ids
            np.arange(seq_len).reshape(1, seq_len),
            dtype=mindspore.int64)
        assert np.allclose(pt_query.detach().numpy(),
                           ms_query.asnumpy(), 1e-5, 1e-5)
        # output
        pt_out = pt_model(pt_query, pt_key, seq_len, pt_position_ids)
        ms_out = ms_model(ms_query, ms_key, seq_len, ms_position_ids)
        # shape & loss
        assert ms_out[0].shape == pt_out[0].shape
        assert ms_out[1].shape == pt_out[1].shape
        assert np.allclose(
            ms_out[0].asnumpy(), pt_out[0].detach().numpy(), 1e-5, 1e-5)
        assert np.allclose(
            ms_out[1].asnumpy(), pt_out[1].detach().numpy(), 1e-5, 1e-5)
        print(">===========test_FalconRotaryEmbedding_end\n")

    def test_FalconAttention(self, ms_dtype, pt_dtype):
        print(">==========test_FalconAttention")
        batch_size = 1
        seq_len = 4672
        hidden_size = 4544
        # init config
        ms_config = falcon_ms.FalconConfig()
        pt_config = falcon_pt.FalconConfig()
        # init model
        ms_model = falcon_ms.FalconAttention(ms_config)
        pt_model = falcon_pt.FalconAttention(pt_config)
        # load parameters
        pt_params = pt_model.state_dict()
        for key, param in ms_model.parameters_and_names():
            if key in pt_params:
                param.set_data(Tensor(pt_params.get(key).detach().numpy()))
        # set eval mode
        ms_model.set_train(False)
        pt_model.eval()
        # prepare data
        hidden_states = np.random.randn(batch_size, seq_len, hidden_size)
        alibi = np.random.randn(batch_size, 8, 1, seq_len)
        attention_mask = np.random.randint(low=0, high=1, size=(batch_size, seq_len))
        ms_hidden_states = Tensor(hidden_states, dtype=ms_dtype)
        ms_alibi = Tensor(alibi, dtype=ms_dtype)
        ms_attention_mask = Tensor(attention_mask, dtype=mindspore.bool_)
        pt_hidden_states = torch.tensor(hidden_states, dtype=pt_dtype)
        pt_alibi = torch.tensor(alibi, dtype=pt_dtype)
        pt_attention_mask = torch.tensor(attention_mask, dtype=pt_dtype)
        # output
        ms_out = ms_model(ms_hidden_states, ms_alibi, ms_attention_mask)
        pt_out = pt_model(pt_hidden_states, pt_alibi, pt_attention_mask)
        # shape & loss
        assert ms_out.shape == pt_out.shape
        assert np.allclose(
            ms_out.asnumpy(), pt_out.detach().numpy(), 1e-5, 1e-5)
        print(">===========test_FalconAttention_end\n")



if __name__ == "__main__":

    t = falcon_test()
    t.test_FalconLinear(*dtype_list[0])
    t.test_FalconRotaryEmbedding(*dtype_list[0])
    t.test_FalconAttention(*dtype_list[0])
