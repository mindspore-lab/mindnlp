# pylint: disable=E1101
import numpy as np
import mindspore
import torch
from mindspore import Tensor, context
# from mindspore import nn

from transformers.models.falcon import modeling_falcon as falcon_pt
from mindnlp.transformers.models.falcon import falcon as falcon_ms
from transformers.models.t5 import modeling_t5 as t5_pt
from transformers

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
        print(">==========test_FalconRotaryEmbedding_begin")
        seq_len = 1024
        batch_size = 2
        num_attn_heads = 71
        head_size = 4544
        # init model
        pt_model = falcon_pt.FalconRotaryEmbedding(128)
        ms_model = falcon_ms.FalconRotaryEmbedding(128)
        # set eval mode
        ms_model.set_train(False)
        pt_model.eval()
        # data
        x = np.random.randn(batch_size, num_attn_heads, seq_len, head_size)
        # prepare data
        ms_x = Tensor(x, dtype=ms_dtype)
        pt_x = torch.tensor(x, dtype=pt_dtype)
        # output
        pt_out = pt_model(pt_x, seq_len)
        ms_out = ms_model(ms_x, seq_len)
        # shape & loss
        assert ms_out[0].shape == pt_out[0].shape
        assert ms_out[1].shape == pt_out[1].shape
        assert np.allclose(
            ms_out[0].asnumpy(), pt_out[0].detach().numpy(), 1e-5, 1e-5)
        assert np.allclose(
            ms_out[1].asnumpy(), pt_out[1].detach().numpy(), 1e-5, 1e-5)
        print(">===========test_FalconRotaryEmbedding_end\n")

    def test_FalconAttention(self, ms_dtype, pt_dtype):
        print(">==========test_FalconAttention_begin")
        batch_size = 1
        seq_len = 1024
        hidden_size = 4544
        max_length = 1420
        num_attention_heads = 71
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
        # alibi = None
        attention_mask = np.random.choice([0, 1], size=(batch_size, seq_len))
        ms_attention_mask = Tensor(attention_mask, dtype=mindspore.bool_)
        ms_hidden_states = Tensor(hidden_states, dtype=ms_dtype)
        ms_alibi = falcon_ms.build_alibi_tensor(ms_attention_mask, num_attention_heads, dtype_list[0][0])
        # ms_alibi = None
        pt_attention_mask = torch.tensor(attention_mask, dtype=pt_dtype)
        pt_hidden_states = torch.tensor(hidden_states, dtype=pt_dtype)
        pt_alibi = falcon_pt.build_alibi_tensor(pt_attention_mask, num_attention_heads, dtype_list[0][1])
        # pt_alibi = None
        # output
        pt_out = pt_model(pt_hidden_states, pt_alibi, pt_attention_mask)
        ms_out = ms_model(ms_hidden_states, ms_alibi, ms_attention_mask)
        # shape & loss
        assert ms_out[0].shape == pt_out[0].shape
        assert np.allclose(
            ms_out[0].asnumpy(), pt_out[0].detach().numpy(), 1e-5, 1e-5)
        print(">===========test_FalconAttention_end\n")
        
    def test_FalconFlashAttention2(self, ms_dtype, pt_dtype):
        print(">==========test_FalconFlashAttention2_begin")
        batch_size = 1
        seq_len = 1024
        hidden_size = 4544
        max_length = 1420
        num_attention_heads = 71
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
        # alibi = None
        attention_mask = np.random.choice([0, 1], size=(batch_size, seq_len))
        ms_attention_mask = Tensor(attention_mask, dtype=mindspore.bool_)
        ms_hidden_states = Tensor(hidden_states, dtype=ms_dtype)
        ms_alibi = falcon_ms.build_alibi_tensor(ms_attention_mask, num_attention_heads, dtype_list[0][0])
        # ms_alibi = None
        pt_attention_mask = torch.tensor(attention_mask, dtype=pt_dtype)
        pt_hidden_states = torch.tensor(hidden_states, dtype=pt_dtype)
        pt_alibi = falcon_pt.build_alibi_tensor(pt_attention_mask, num_attention_heads, dtype_list[0][1])
        # pt_alibi = None
        # output
        pt_out = pt_model(pt_hidden_states, pt_alibi, pt_attention_mask)
        ms_out = ms_model(ms_hidden_states, ms_alibi, ms_attention_mask)
        # shape & loss
        assert ms_out[0].shape == pt_out[0].shape
        assert np.allclose(
            ms_out[0].asnumpy(), pt_out[0].detach().numpy(), 1e-5, 1e-5)
        print(">===========test_FalconFlashAttention2_end\n")


if __name__ == "__main__":

    t = falcon_test()
    t.test_FalconLinear(*dtype_list[0])
    # t.test_FalconRotaryEmbedding(*dtype_list[0])
    t.test_FalconAttention(*dtype_list[0])

