"""
vit part
"""
from argparse import Namespace
import mindspore
from mindspore import nn,ops
from ...activations import ACT2FN


class PatchEmbedding(nn.Cell):
    """
    patch embedding
    """
    def __init__(self, config):
        super().__init__()
        self.proj = nn.Conv2d(config.in_channels, config.hidden_size, kernel_size=config.patch_size,
                              stride=config.patch_size,pad_mode='valid',has_bias=True)
        self.cls_embedding = mindspore.Parameter(ops.zeros(1, config.hidden_size))
        self.position_embedding = nn.Embedding(config.num_positions, config.hidden_size)

    def construct(self, images: "tensor(B, C, H, W)") -> "tensor(B, L, D)":
        x = self.proj(images)
        x = x.flatten(start_dim=2).swapaxes(1, 2)
        cls_token = self.cls_embedding.broadcast_to((x.shape[0], -1, -1))
        x = ops.cat((cls_token, x), axis=1)
        x += self.position_embedding.weight.unsqueeze(0)
        return x


class Attention(nn.Cell):
    """
    attention
    """
    def __init__(self, config):
        super().__init__()
        self.num_heads = config.num_heads
        head_dim = config.hidden_size // config.num_heads
        self.scale = head_dim ** -0.5
        self.query_key_value = nn.Dense(config.hidden_size, config.hidden_size * 3)
        self.dense = nn.Dense(config.hidden_size, config.hidden_size)
        self.output_dropout = nn.Dropout(p = config.dropout_prob)

    def construct(self, x: "tensor(B, L, D)") -> "tensor(B, L, D)":
        B, L, _ = x.shape
        qkv = self.query_key_value(x)
        qkv = qkv.reshape(B, L, 3, self.num_heads, -1).permute(2, 0, 3, 1, 4)  # 3, B, H, L, D
        q, k, v = qkv[0], qkv[1], qkv[2]
        out = self.attention(q,k,v)
        output = self.dense(out.view(B, L, -1))
        output = self.output_dropout(output)
        return output

    def attention(self, q, k, v):
        attn_weights = ops.matmul(q * self.scale, k.swapaxes(-2, -1))
        attn_weights = ops.softmax(attn_weights,axis=-1)
        output = ops.matmul(attn_weights, v)
        output = output.transpose(0,2,1,3)
        return output


class MLP(nn.Cell):
    """
    MLP
    """
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.activation_fn = ACT2FN[config.hidden_act]
        self.fc1 = nn.Dense(config.hidden_size, config.intermediate_size)
        self.fc2 = nn.Dense(config.intermediate_size, config.hidden_size)

    def construct(self, x: mindspore.Tensor) -> mindspore.Tensor:
        x = self.fc1(x)
        x = self.activation_fn(x)
        x = self.fc2(x)
        return x


class TransformerLayer(nn.Cell):
    """
    transformer layer
    """
    def __init__(self, config):
        super().__init__()
        self.input_layernorm = nn.LayerNorm(config.hidden_size, epsilon=config.layer_norm_eps)
        self.attention = Attention(config)
        self.mlp = MLP(config)
        self.post_attention_layernorm = nn.LayerNorm(config.hidden_size, epsilon=config.layer_norm_eps)

    def construct(self, hidden_states):
        attention_input = hidden_states
        attention_output = self.attention(attention_input)
        layernorm = self.input_layernorm(attention_output)
        hidden_states = attention_input + layernorm
        mlp_input = hidden_states
        mlp_output = self.post_attention_layernorm(self.mlp(mlp_input))
        output = mlp_input + mlp_output
        return output


class Transformer(nn.Cell):
    def __init__(self, config):
        super().__init__()
        self.layers = nn.CellList([TransformerLayer(config) for _ in range(config.num_hidden_layers)])

    def construct(self, hidden_states):
        for layer_module in self.layers:
            hidden_states = layer_module(hidden_states)
        return hidden_states


class GLU(nn.Cell):
    def __init__(self, config, in_features):
        super().__init__()
        self.linear_proj = nn.Dense(in_features, config.hidden_size, has_bias=False)
        self.norm1 = nn.LayerNorm(config.hidden_size)
        self.act1 = nn.GELU(approximate=False)
        self.act2 = ops.silu
        self.dense_h_to_4h = nn.Dense(config.hidden_size, config.intermediate_size, has_bias=False)
        self.gate_proj = nn.Dense(config.hidden_size, config.intermediate_size, has_bias=False)
        self.dense_4h_to_h = nn.Dense(config.intermediate_size, config.hidden_size, has_bias=False)

    def construct(self, x):
        x = self.linear_proj(x)
        x1 = self.act1(self.norm1(x))
        x2 = self.act2(self.gate_proj(x1)) * self.dense_h_to_4h(x1)
        x3 = self.dense_4h_to_h(x2)
        return x3


class EVA2CLIPModel(nn.Cell):
    def __init__(self, config):
        super().__init__()
        vision_config = Namespace(**config.vision_config)
        self.patch_embedding = PatchEmbedding(vision_config)
        self.transformer = Transformer(vision_config)
        self.linear_proj = GLU(config, in_features=vision_config.hidden_size)
        self.boi = mindspore.Parameter(ops.zeros((1, 1, config.hidden_size)))
        self.eoi = mindspore.Parameter(ops.zeros((1, 1, config.hidden_size)))

    def construct(self, images: "tensor(B, C, H, W)") -> "tensor(B, L, D)":
        x = self.patch_embedding(images)
        x1 = self.transformer(x)
        x2 = x1[:, 1:]
        x3 = self.linear_proj(x2)
        boi = self.boi.broadcast_to((x3.shape[0], -1, -1))
        eoi = self.eoi.broadcast_to((x3.shape[0], -1, -1))
        x4 = ops.cat((boi, x3, eoi), axis=1)
        return x4

__all__ = [
    'EVA2CLIPModel',
    'TransformerLayer'
]
