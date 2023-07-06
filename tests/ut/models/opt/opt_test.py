import torch
import mindspore
import numpy as np
import transformers
from transformers.models.opt import modeling_opt as opt_pt, configuration_opt
from mindnlp.models.opt import opt as opt_ms

class OPT_test():
    def __init__(self):
        print("this is OPT model test")
    def test_OPTLearnedPositionalEmbedding(self):
        print(">===========test_OPTLearnedPositionalEmbedding_begin")
        # model instance
        num_embeddings = 30000
        embedding_dim = 64
        model_ms = opt_ms.OPTLearnedPositionalEmbedding(num_embeddings, embedding_dim)
        model_pt = opt_pt.OPTLearnedPositionalEmbedding(num_embeddings, embedding_dim)
        #load the parameters of pytorch network to mindspore
        params_pt = model_pt.state_dict()
        ''' print to check the pytorch model's parameter name
        for key in params_pt.keys():
            print(key)
        #'''
        ''' print to check the mindspore model's parameter name
        for key, _ in model_ms.parameters_and_names():
            print(key)
        #'''
        #pytorch: weight, mindspore: embedding_table
        for key, param in model_ms.parameters_and_names():
            if 'embedding_table' in key:
                key = key.replace('embedding_table', 'weight')
            param.set_data(mindspore.Tensor(params_pt.get(key).detach().numpy()))
        model_ms.set_train(False)
        model_pt.eval()
        input = np.random.randint(1, 10, (num_embeddings, embedding_dim))
        #print(input)
        input_ms = mindspore.Tensor(input, dtype = mindspore.int32)
        input_pt = torch.tensor(input, dtype = torch.int32)
        output_ms = model_ms(input_ms)
        output_pt = model_pt(input_pt)
        assert output_ms.shape == output_pt.shape
        assert np.allclose(output_ms.asnumpy(), output_pt.detach().numpy(), 1e-5, 1e-5)
        print("PASS!")

    def test_OPTAttention(self):
        print(">===========test_OPTAttention_begin")
        # model instance
        model_ms = opt_ms.OPTAttention(embed_dim=768,num_heads=12,dropout=0,is_decoder=True,bias=True)
        model_pt = opt_pt.OPTAttention(embed_dim=768,num_heads=12,dropout=0,is_decoder=True,bias=True)
        #load the parameters of pytorch network to mindspore
        params_pt = model_pt.state_dict()
        ''' print to check the pytorch model's parameter name
        for key in params_pt.keys():
            print(key)
        #'''
        ''' print to check the mindspore model's parameter name
        for key, _ in model_ms.parameters_and_names():
            print(key)
        #'''
        # checked, no difference
        for key, param in model_ms.parameters_and_names():
            param.set_data(mindspore.Tensor(params_pt.get(key).detach().numpy()))
        model_ms.set_train(False)
        model_pt.eval()
        input = np.random.random((2,3,768))
        input_ms = mindspore.Tensor(input, dtype = mindspore.float32)
        input_pt = torch.tensor(input, dtype = torch.float32) 
        output_ms = model_ms(input_ms)
        output_pt = model_pt(input_pt)
        #OPTAttention return attn_output, attn_weights_reshaped, past_key_value
        #output is a tuple(attn_output, attn_weights_reshaped, past_key_value), 
        #attn_weights_reshaped is NONE, past_key_value is tuple(key_states, value_states)
        assert output_ms[0].shape == output_pt[0].shape
        #type NONE has no shape
        #assert output_ms[1].shape == output_pt[1].shape
        assert output_ms[2][0].shape == output_pt[2][0].shape
        assert output_ms[2][1].shape == output_pt[2][1].shape
        #assert np.allclose(output_ms[0].asnumpy(), output_pt[0].detach().numpy(), 1e-5, 1e-5)
        assert np.allclose(output_ms[0].asnumpy(), output_pt[0].detach().numpy(), 1e-5, 1e-5)   
        #assert np.allclose(output_ms[1].asnumpy(), output_pt[1].detach().numpy(), 1e-5, 1e-5)
        assert np.allclose(output_ms[2][0].asnumpy(), output_pt[2][0].detach().numpy(), 1e-5, 1e-5)
        assert np.allclose(output_ms[2][1].asnumpy(), output_pt[2][1].detach().numpy(), 1e-5, 1e-5)
        print("PASS!")    
         
    def test_OPTDecoderLayer(self):
        print(">===========test_OPTDecoderLayer_begin")
        # model instance
        config_pytorch = configuration_opt.OPTConfig()
        config_mindspore = opt_ms.OPTConfig()
        model_ms = opt_ms.OPTDecoderLayer(config_mindspore)
        model_pt = opt_pt.OPTDecoderLayer(config_pytorch)
        #load the parameters of pytorch network to mindspore
        params_pt = model_pt.state_dict()
        ''' print to check the pytorch model's parameter name
        for key in params_pt.keys():
            print(key)
        #'''
        #print("---")
        ''' print to check the mindspore model's parameter name
        for key, _ in model_ms.parameters_and_names():
            print(key)
        #'''
        # self_attn_layer_norm.weight self_attn_layer_norm.bias ->self_attn_layer_norm.gamma self_attn_layer_norm.beta
        #final_layer_norm.weight final_layer_norm.bias -> final_layer_norm.gamma final_layer_norm.beta
        for key, param in model_ms.parameters_and_names():
            if 'self_attn_layer_norm.gamma' in key:
                key = key.replace('self_attn_layer_norm.gamma', 'self_attn_layer_norm.weight')
            if 'self_attn_layer_norm.beta' in key:
                key = key.replace('self_attn_layer_norm.beta', 'self_attn_layer_norm.bias')
            if 'final_layer_norm.gamma' in key:
                key = key.replace('final_layer_norm.gamma', 'final_layer_norm.weight')
            if 'final_layer_norm.beta' in key:
                key = key.replace('final_layer_norm.beta', 'final_layer_norm.bias')
            param.set_data(mindspore.Tensor(params_pt.get(key).detach().numpy()))
        model_ms.set_train(False)
        model_pt.eval()
        input = np.random.random((2,3,768))
        input_ms = mindspore.Tensor(input, dtype = mindspore.float32)
        input_pt = torch.tensor(input, dtype = torch.float32) 
        output_ms = model_ms(input_ms)
        output_pt = model_pt(input_pt)   
        print(output_ms[0])
        print("---")
        print(output_pt[0])
        assert output_ms[0].shape == output_pt[0].shape
        assert np.allclose(output_ms[0].asnumpy(), output_pt[0].detach().numpy(), 1e-5, 1e-5)
        print("PASS!")    

    def test_OPTDecoder(self):
        print(">===========test_OPTDecoder_begin")
        # model instance
        config_pytorch = configuration_opt.OPTConfig(num_hidden_layers = 12)
        config_mindspore = opt_ms.OPTConfig(num_hidden_layers = 12)     
        model_ms = opt_ms.OPTDecoder(config_mindspore)
        model_pt = opt_pt.OPTDecoder(config_pytorch)
        #load the parameters of pytorch network to mindspore
        params_pt = model_pt.state_dict()
        ''' print to check the pytorch model's parameter name
        for key in params_pt.keys():
            print(key)
        #'''
        print("---")
        ''' print to check the mindspore model's parameter name
        for key, _ in model_ms.parameters_and_names():
            print(key)
        #'''
        # self_attn_layer_norm.weight self_attn_layer_norm.bias ->self_attn_layer_norm.gamma self_attn_layer_norm.beta
        #final_layer_norm.weight final_layer_norm.bias -> final_layer_norm.gamma final_layer_norm.beta
        for key, param in model_ms.parameters_and_names():
            #print(key)
            if 'embedding_table' in key:
                key = key.replace('embedding_table', 'weight')
            if 'layer_norm.gamma' in key:
                key = key.replace('layer_norm.gamma', 'layer_norm.weight')
            if 'layer_norm.beta' in key:
                key = key.replace('layer_norm.beta', 'layer_norm.bias')
            #print(key)
            param.set_data(mindspore.Tensor(params_pt.get(key).detach().numpy()))
        model_ms.set_train(False)
        model_pt.eval()
        input = np.random.randint(0,10,size=(3,768))
        input_ms = mindspore.Tensor(input, dtype = mindspore.int32)
        input_pt = torch.tensor(input, dtype = torch.int32)
        output_ms = model_ms(input_ms)
        output_pt = model_pt(input_pt)
        assert output_ms[0].shape == output_pt[0].shape
        assert np.allclose(output_ms[0].asnumpy(), output_pt[0].detach().numpy(), 1e-5, 1e-5)
        print("PASS!")

if __name__ == "__main__":
    test = OPT_test()
    test.test_OPTLearnedPositionalEmbedding()
    test.test_OPTAttention()
    test.test_OPTDecoderLayer()
    test.test_OPTDecoder()
