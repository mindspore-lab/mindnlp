import mindspore
import mindspore.nn as nn
import mindspore.ops as ops
import math
#from miniCPM_config import MiniCPMConfig
from .miniCPM_config import MiniCPMConfig



class RMSNorm(nn.Cell):
    def __init__(self, hidden_size, eps=1e-5):
        super().__init__()
        self.eps = eps
        self.weight = mindspore.Parameter(ops.ones(hidden_size), name="rmsnorm_weight")

    def construct(self, hidden_states):
        norm = hidden_states.pow(2).mean(-1, keep_dims=True).add(self.eps).sqrt()
        return self.weight * hidden_states / norm


class MLP(nn.Cell):
    def __init__(self, config: MiniCPMConfig):
        super().__init__()
        self.gate_proj = nn.Dense(config.hidden_size, config.intermediate_size)
        self.up_proj = nn.Dense(config.hidden_size, config.intermediate_size)
        self.down_proj = nn.Dense(config.intermediate_size, config.hidden_size)
        self.act = nn.SiLU()

    def construct(self, x):
        return self.down_proj(self.act(self.gate_proj(x)) * self.up_proj(x))


class Attention(nn.Cell):
    def __init__(self, config: MiniCPMConfig):
        super().__init__()
        self.num_heads = config.num_attention_heads
        self.head_dim = config.hidden_size // config.num_attention_heads
        self.scale = self.head_dim ** -0.5

        self.q_proj = nn.Dense(config.hidden_size, config.hidden_size)
        self.k_proj = nn.Dense(config.hidden_size, config.hidden_size)
        self.v_proj = nn.Dense(config.hidden_size, config.hidden_size)
        self.out_proj = nn.Dense(config.hidden_size, config.hidden_size)

        self.softmax = nn.Softmax(axis=-1)

    def construct(self, x):
        B, T, C = x.shape
        q = self.q_proj(x).view(B, T, self.num_heads, self.head_dim).transpose(0, 2, 1, 3)
        k = self.k_proj(x).view(B, T, self.num_heads, self.head_dim).transpose(0, 2, 1, 3)
        v = self.v_proj(x).view(B, T, self.num_heads, self.head_dim).transpose(0, 2, 1, 3)

        attn_scores = ops.matmul(q, k.transpose(0, 1, 3, 2)) * self.scale
        attn_weights = self.softmax(attn_scores)
        attn_output = ops.matmul(attn_weights, v)

        attn_output = attn_output.transpose(0, 2, 1, 3).view(B, T, C)
        return self.out_proj(attn_output)


class DecoderLayer(nn.Cell):
    def __init__(self, config: MiniCPMConfig):
        super().__init__()
        self.ln1 = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.self_attn = Attention(config)
        self.ln2 = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.mlp = MLP(config)

    def construct(self, x):
        x = x + self.self_attn(self.ln1(x))
        x = x + self.mlp(self.ln2(x))
        return x


class MiniCPMModel(nn.Cell):
    def __init__(self, config: MiniCPMConfig):
        super().__init__()
        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size)
        self.layers = nn.CellList([DecoderLayer(config) for _ in range(config.num_hidden_layers)])
        self.ln_f = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)

    def construct(self, input_ids):
        x = self.embed_tokens(input_ids)
        for layer in self.layers:
            x = layer(x)
        x = self.ln_f(x)
        return x


class MiniCPMForCausalLM(nn.Cell):
    def __init__(self, config: MiniCPMConfig):
        super().__init__()
        self.model = MiniCPMModel(config)
        self.lm_head = nn.Dense(config.hidden_size, config.vocab_size, has_bias=False)

    def construct(self, input_ids):
        hidden_states = self.model(input_ids)
        logits = self.lm_head(hidden_states)
        return logits
