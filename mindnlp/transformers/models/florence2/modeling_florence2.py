# coding=utf-8
# Copyright 2024 Microsoft and the HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

""" MindNLP Florence-2 model."""
import math
from dataclasses import dataclass
from typing import List, Optional, Tuple, Union


from collections import OrderedDict
from mindspore import Tensor
import mindspore
from mindnlp.core import nn, ops
import mindnlp.core.nn.functional as F
from mindnlp.core.nn import CrossEntropyLoss
from mindnlp.core.nn.init import trunc_normal_
from mindnlp.transformers.modeling_utils import PreTrainedModel
from mindnlp.utils import (
    ModelOutput,
    logging,
    is_einops_available
)

from mindnlp.transformers.activations import ACT2FN
from mindnlp.transformers.modeling_attn_mask_utils import (
    _prepare_4d_attention_mask,
    _prepare_4d_causal_attention_mask,
)
from mindnlp.transformers.modeling_outputs import (
    BaseModelOutput,
    BaseModelOutputWithPastAndCrossAttentions,
    Seq2SeqLMOutput,
    Seq2SeqModelOutput,
)

from .configuration_florence2 import Florence2Config
from .configuration_florence2 import Florence2LanguageConfig
from .configuration_florence2 import Florence2VisionConfig

if is_einops_available():
    from einops import rearrange


logger = logging.get_logger(__name__)

_CONFIG_FOR_DOC = "Florence2Config"


class DropPath(nn.Module):
    """DropPath (Stochastic Depth) regularization layers"""

    def __init__(
        self,
        drop_prob: float = 0.0,
        scale_by_keep: bool = True,
    ) -> None:
        super().__init__()
        self.keep_prob = 1.0 - drop_prob
        self.scale_by_keep = scale_by_keep
        self.dropout = nn.Dropout(p=drop_prob)

    def forward(self, x: Tensor) -> Tensor:
        if self.keep_prob == 1.0 or not self.training:
            return x
        shape = (x.shape[0],) + (1,) * (x.ndim - 1)
        random_tensor = self.dropout(ops.ones(shape))
        if not self.scale_by_keep:
            random_tensor = ops.mul(random_tensor, self.keep_prob)
        return x * random_tensor


class LearnedAbsolutePositionEmbedding2D(nn.Module):
    """
    This module learns positional embeddings up to a fixed maximum size.
    """

    def __init__(self, embedding_dim=256, num_pos=50):
        super().__init__()
        self.row_embeddings = nn.Embedding(num_pos, embedding_dim // 2)
        self.column_embeddings = nn.Embedding(num_pos, embedding_dim - (embedding_dim // 2))

    def forward(self, pixel_values):
        """
        pixel_values: (batch_size, height, width, num_channels) 
        returns: (batch_size, height, width, embedding_dim * 2)
        """
        if len(pixel_values.shape) != 4:
            raise ValueError('pixel_values must be a 4D tensor')
        height, width = pixel_values.shape[1:3]
        width_values = ops.arange(width)
        height_values = ops.arange(height)
        x_emb = self.column_embeddings(width_values)
        y_emb = self.row_embeddings(height_values)
        # (height, width, embedding_dim * 2)
        pos = ops.cat([x_emb.unsqueeze(0).tile((height, 1, 1)), y_emb.unsqueeze(1).tile((1, width, 1))], dim=-1)
        # (embedding_dim * 2, height, width)
        pos = pos.permute(2, 0, 1)
        pos = pos.unsqueeze(0)
        # (batch_size, embedding_dim * 2, height, width)
        pos = pos.tile((pixel_values.shape[0], 1, 1, 1))
        # (batch_size, height, width, embedding_dim * 2)
        pos = pos.permute(0, 2, 3, 1)
        return pos


class PositionalEmbeddingCosine1D(nn.Module):
    """
    This class implements a very simple positional encoding. It follows closely
    the encoder from the link below:
    https://pytorch.org/tutorials/beginner/translation_transformer.html

    Args:
        embed_dim: The dimension of the embeddings.
        dropout_prob: The dropout probability.
        max_seq_len: The maximum length to precompute the positional encodings.
    """
    def __init__(
            self,
            embed_dim: int = 512,
            max_seq_len: int = 1024) -> None:
        super(PositionalEmbeddingCosine1D, self).__init__()
        self.embed_dim = embed_dim
        self.max_seq_len = max_seq_len
        # Generate the sinusoidal arrays.
        factor = math.log(10000)
        denominator = ops.exp(
            -factor * ops.arange(0, self.embed_dim, 2) / self.embed_dim)
        # Matrix where rows correspond to a positional embedding as a function
        # of the position index (i.e., the row index).
        frequencies = \
            ops.arange(0, self.max_seq_len) \
            .reshape(self.max_seq_len, 1) * denominator
        pos_idx_to_embed = ops.zeros((self.max_seq_len, self.embed_dim))
        # Populate uneven entries.
        pos_idx_to_embed[:, 0::2] = ops.sin(frequencies)
        pos_idx_to_embed[:, 1::2] = ops.cos(frequencies)
        # Save the positional embeddings in a constant buffer.
        self.register_buffer("pos_idx_to_embed", pos_idx_to_embed)

    def forward(self, seq_embeds: Tensor) -> Tensor:
        """
        Args:
            seq_embeds: The sequence embeddings in order. Allowed size:
                1. [T, D], where T is the length of the sequence, and D is the
                frame embedding dimension.
                2. [B, T, D], where B is the batch size and T and D are the
                same as above.

        Returns a tensor of with the same dimensions as the input: i.e.,
        [1, T, D] or [T, D].
        """
        shape_len = len(seq_embeds.shape)
        assert 2 <= shape_len <= 3
        len_seq = seq_embeds.shape[-2]
        assert len_seq <= self.max_seq_len
        pos_embeds = self.pos_idx_to_embed[0:seq_embeds.shape[-2], :]
        # Adapt pre-computed positional embeddings to the input.
        if shape_len == 3:
            pos_embeds = pos_embeds.view(
                (1, pos_embeds.shape[0], pos_embeds.shape[1]))
        return pos_embeds


class LearnedAbsolutePositionEmbedding1D(nn.Module):
    """
    Learnable absolute positional embeddings for 1D sequences.

    Args:
        embedding_dim: The dimension of the embeddings.
        max_seq_len: The maximum length to precompute the positional encodings.
    """
    def __init__(
            self,
            embedding_dim: int = 512,
            num_pos: int = 1024) -> None:
        super(LearnedAbsolutePositionEmbedding1D, self).__init__()
        self.embeddings = nn.Embedding(num_pos, embedding_dim)
        self.num_pos = num_pos

    def forward(self, seq_embeds: Tensor) -> Tensor:
        """
        Args:
            seq_embeds: The sequence embeddings in order. Allowed size:
                1. [T, D], where T is the length of the sequence, and D is the
                frame embedding dimension.
                2. [B, T, D], where B is the batch size and T and D are the
                same as above.

        Returns a tensor of with the same dimensions as the input: i.e.,
        [1, T, D] or [T, D].
        """
        shape_len = len(seq_embeds.shape)
        assert 2 <= shape_len <= 3
        len_seq = seq_embeds.shape[-2]
        assert len_seq <= self.num_pos
        # [T, D]
        pos_embeds = self.embeddings(ops.arange(len_seq))
        # Adapt pre-computed positional embeddings to the input.
        if shape_len == 3:
            pos_embeds = pos_embeds.view(
                (1, pos_embeds.shape[0], pos_embeds.shape[1]))
        return pos_embeds


class MySequential(nn.Sequential):
    def forward(self, *inputs):
        for module in self._modules.values():
            if type(inputs) == tuple:
                inputs = module(*inputs)
            else:
                inputs = module(inputs)
        return inputs


class PreNorm(nn.Module):
    def __init__(self, norm, fn, drop_path=None):
        super().__init__()
        self.norm = norm
        self.fn = fn
        self.drop_path = drop_path

    def forward(self, x, *args, **kwargs):
        shortcut = x

        if self.norm is not None:
            x, size = self.fn(self.norm(x), *args, **kwargs)
        else:
            x, size = self.fn(x, *args, **kwargs)

        if self.drop_path:
            x = self.drop_path(x)

        x = shortcut + x

        return x, size


class Mlp(nn.Module):
    def __init__(
        self,
        in_features,
        hidden_features=None,
        out_features=None,
        act_layer=nn.GELU,
    ):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.net = nn.Sequential(OrderedDict([
            ("fc1", nn.Linear(in_features, hidden_features)),
            ("act", act_layer()),
            ("fc2", nn.Linear(hidden_features, out_features))
        ]))

    def forward(self, x, size):
        return self.net(x), size


class DepthWiseConv2d(nn.Module):
    def __init__(
        self,
        dim_in,
        kernel_size,
        padding,
        stride,
        bias=True,
    ):
        super().__init__()
        self.dw = nn.Conv2d(
            dim_in, dim_in,
            kernel_size=kernel_size,
            padding=padding,
            groups=dim_in,
            stride=stride,
            bias=bias
        )

    def forward(self, x, size):
        B, N, C = x.shape
        H, W = size
        assert N == H * W

        x = self.dw(ops.transpose(x, 1, 2).view(B, C, H, W))
        size = (x.shape[-2], x.shape[-1])
        x = ops.transpose(x.flatten(start_dim=2), 1, 2)
        return x, size


class ConvEmbed(nn.Module):
    """ Image to Patch Embedding
    """

    def __init__(
        self,
        patch_size=7,
        in_chans=3,
        embed_dim=64,
        stride=4,
        padding=2,
        norm_layer=None,
        pre_norm=True
    ):
        super().__init__()
        self.patch_size = patch_size

        self.proj = nn.Conv2d(
            in_chans, embed_dim,
            kernel_size=patch_size,
            stride=stride,
            padding=padding
        )

        dim_norm = in_chans if pre_norm else embed_dim
        self.norm = norm_layer(dim_norm)

        self.pre_norm = pre_norm

    def forward(self, x, size):
        H, W = size
        if len(x.shape) == 3:
            if self.norm and self.pre_norm:
                x = self.norm(x)
            x = rearrange(
                x, 'b (h w) c -> b c h w',
                h=H, w=W
            )
        x = self.proj(x)

        _, _, H, W = x.shape
        x = rearrange(x, 'b c h w -> b (h w) c')
        if self.norm and not self.pre_norm:
            x = self.norm(x)

        return x, (H, W)


class ChannelAttention(nn.Module):

    def __init__(self, dim, groups=8, qkv_bias=True):
        super().__init__()

        self.groups = groups
        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.proj = nn.Linear(dim, dim)

    def forward(self, x, size):
        B, N, C = x.shape

        qkv = self.qkv(x).reshape(B, N, 3, self.groups, C // self.groups).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]

        q = q * (float(N) ** -0.5)
        attention = ops.transpose(q, -1, -2) @ k
        # attention = attention.softmax(dim=-1)
        attention = ops.softmax(attention, dim=-1)
        x = ops.transpose(attention @ ops.transpose(v, -1, -2), -1, -2)
        x = ops.transpose(x, 1, 2).reshape(B, N, C)
        x = self.proj(x)
        return x, size


class ChannelBlock(nn.Module):

    def __init__(self, dim, groups, mlp_ratio=4., qkv_bias=True,
                 drop_path_rate=0., act_layer=nn.GELU, norm_layer=nn.LayerNorm,
                 conv_at_attn=True, conv_at_ffn=True):
        super().__init__()

        drop_path = DropPath(drop_path_rate) if drop_path_rate > 0. else nn.Identity()

        self.conv1 = PreNorm(None, DepthWiseConv2d(dim, 3, 1, 1)) if conv_at_attn else None
        self.channel_attn = PreNorm(
            norm_layer(dim),
            ChannelAttention(dim, groups=groups, qkv_bias=qkv_bias),
            drop_path
        )
        self.conv2 = PreNorm(None, DepthWiseConv2d(dim, 3, 1, 1)) if conv_at_ffn else None
        self.ffn = PreNorm(
            norm_layer(dim),
            Mlp(in_features=dim, hidden_features=int(dim*mlp_ratio), act_layer=act_layer),
            drop_path
        )

    def forward(self, x, size):
        if self.conv1:
            x, size = self.conv1(x, size)
        x, size = self.channel_attn(x, size)

        if self.conv2:
            x, size = self.conv2(x, size)
        x, size = self.ffn(x, size)

        return x, size


def window_partition(x, window_size: int):
    B, H, W, C = x.shape
    x = x.view(B, H // window_size, window_size, W // window_size, window_size, C)
    windows = x.permute(0, 1, 3, 2, 4, 5).view(-1, window_size, window_size, C)
    return windows


def window_reverse(windows, batch_size: int, window_size: int, H: int, W: int):
    B = batch_size
    # this will cause onnx conversion failed for dynamic axis, because treated as constant
    # int(windows.shape[0] / (H * W / window_size / window_size))
    x = windows.view(B, H // window_size, W // window_size, window_size, window_size, -1)
    x = x.permute(0, 1, 3, 2, 4, 5).view(B, H, W, -1)
    return x


class WindowAttention(nn.Module):
    def __init__(self, dim, num_heads, window_size, qkv_bias=True):

        super().__init__()
        self.dim = dim
        self.window_size = window_size
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = float(head_dim) ** -0.5

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.proj = nn.Linear(dim, dim)

        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x, size):

        H, W = size
        B, L, C = x.shape
        assert L == H * W, "input feature has wrong size"

        x = x.view(B, H, W, C)

        pad_l = pad_t = 0
        pad_r = (self.window_size - W % self.window_size) % self.window_size
        pad_b = (self.window_size - H % self.window_size) % self.window_size
        x = F.pad(x, (0, 0, pad_l, pad_r, pad_t, pad_b))
        _, Hp, Wp, _ = x.shape

        x = window_partition(x, self.window_size)
        x = x.view(-1, self.window_size * self.window_size, C)

        # W-MSA/SW-MSA
        # attn_windows = self.attn(x_windows)

        B_, N, C = x.shape
        qkv = self.qkv(x).reshape(B_, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]

        q = q * self.scale
        attn = (q @ ops.transpose(k, -2, -1))
        attn = self.softmax(attn)

        x = ops.transpose(attn @ v, 1, 2).reshape(B_, N, C)
        x = self.proj(x)

        # merge windows
        x = x.view(
            -1, self.window_size, self.window_size, C
        )
        x = window_reverse(x, B, self.window_size, Hp, Wp)

        if pad_r > 0 or pad_b > 0:
            x = x[:, :H, :W, :]

        x = x.view(B, H * W, C)

        return x, size


class SpatialBlock(nn.Module):

    def __init__(self, dim, num_heads, window_size,
                 mlp_ratio=4., qkv_bias=True, drop_path_rate=0., act_layer=nn.GELU,
                 norm_layer=nn.LayerNorm, conv_at_attn=True, conv_at_ffn=True):
        super().__init__()

        drop_path = DropPath(drop_path_rate) if drop_path_rate > 0. else nn.Identity()

        self.conv1 = PreNorm(None, DepthWiseConv2d(dim, 3, 1, 1)) if conv_at_attn else None
        self.window_attn = PreNorm(
            norm_layer(dim),
            WindowAttention(dim, num_heads, window_size, qkv_bias=qkv_bias),
            drop_path
        )
        self.conv2 = PreNorm(None, DepthWiseConv2d(dim, 3, 1, 1)) if conv_at_ffn else None
        self.ffn = PreNorm(
            norm_layer(dim),
            Mlp(in_features=dim, hidden_features=int(dim*mlp_ratio), act_layer=act_layer),
            drop_path
        )

    def forward(self, x, size):
        if self.conv1:
            x, size = self.conv1(x, size)
        x, size = self.window_attn(x, size)

        if self.conv2:
            x, size = self.conv2(x, size)
        x, size = self.ffn(x, size)
        return x, size


class DaViT(nn.Module):
    """ DaViT: Dual-Attention Transformer

    Args:
        in_chans (int): Number of input image channels. Default: 3.
        num_classes (int): Number of classes for classification head. Default: 1000.
        patch_size (tuple(int)): Patch size of convolution in different stages. Default: (7, 2, 2, 2).
        patch_stride (tuple(int)): Patch stride of convolution in different stages. Default: (4, 2, 2, 2).
        patch_padding (tuple(int)): Patch padding of convolution in different stages. Default: (3, 0, 0, 0).
        patch_prenorm (tuple(bool)): If True, perform norm before convlution layer. Default: (True, False, False, False).
        embed_dims (tuple(int)): Patch embedding dimension in different stages. Default: (64, 128, 192, 256).
        num_heads (tuple(int)): Number of spatial attention heads in different stages. Default: (4, 8, 12, 16).
        num_groups (tuple(int)): Number of channel groups in different stages. Default: (4, 8, 12, 16).
        window_size (int): Window size. Default: 7.
        mlp_ratio (float): Ratio of mlp hidden dim to embedding dim. Default: 4.
        qkv_bias (bool): If True, add a learnable bias to query, key, value. Default: True.
        drop_path_rate (float): Stochastic depth rate. Default: 0.1.
        norm_layer (nn.Module): Normalization layer. Default: nn.LayerNorm.
        enable_checkpoint (bool): If True, enable checkpointing. Default: False.
        conv_at_attn (bool): If True, performe depthwise convolution before attention layer. Default: True.
        conv_at_ffn (bool): If True, performe depthwise convolution before ffn layer. Default: True.
    """

    def __init__(
        self,
        in_chans=3,
        num_classes=1000,
        depths=(1, 1, 3, 1),
        patch_size=(7, 2, 2, 2),
        patch_stride=(4, 2, 2, 2),
        patch_padding=(3, 0, 0, 0),
        patch_prenorm=(False, False, False, False),
        embed_dims=(64, 128, 192, 256),
        num_heads=(3, 6, 12, 24),
        num_groups=(3, 6, 12, 24),
        window_size=7,
        mlp_ratio=4.,
        qkv_bias=True,
        drop_path_rate=0.1,
        norm_layer=nn.LayerNorm,
        enable_checkpoint=False,
        conv_at_attn=True,
        conv_at_ffn=True,
     ):
        super().__init__()

        self.num_classes = num_classes
        self.embed_dims = embed_dims
        self.num_heads = num_heads
        self.num_groups = num_groups
        self.num_stages = len(self.embed_dims)
        self.enable_checkpoint = enable_checkpoint
        assert self.num_stages == len(self.num_heads) == len(self.num_groups)

        num_stages = len(embed_dims)
        dpr = [x.item() for x in ops.linspace(0, drop_path_rate, sum(depths)*2)]

        depth_offset = 0
        convs = []
        blocks = []
        for i in range(num_stages):
            conv_embed = ConvEmbed(
                patch_size=patch_size[i],
                stride=patch_stride[i],
                padding=patch_padding[i],
                in_chans=in_chans if i == 0 else self.embed_dims[i - 1],
                embed_dim=self.embed_dims[i],
                norm_layer=norm_layer,
                pre_norm=patch_prenorm[i]
            )
            convs.append(conv_embed)

            block = MySequential(
                *[
                    MySequential(OrderedDict([
                        (
                            'spatial_block', SpatialBlock(
                                embed_dims[i],
                                num_heads[i],
                                window_size,
                                drop_path_rate=dpr[depth_offset+j*2],
                                qkv_bias=qkv_bias,
                                mlp_ratio=mlp_ratio,
                                conv_at_attn=conv_at_attn,
                                conv_at_ffn=conv_at_ffn,
                            )
                        ),
                        (
                            'channel_block', ChannelBlock(
                                embed_dims[i],
                                num_groups[i],
                                drop_path_rate=dpr[depth_offset+j*2+1],
                                qkv_bias=qkv_bias,
                                mlp_ratio=mlp_ratio,
                                conv_at_attn=conv_at_attn,
                                conv_at_ffn=conv_at_ffn,
                            )
                        )
                    ])) for j in range(depths[i])
                ]
            )
            blocks.append(block)
            depth_offset += depths[i]*2

        self.convs = nn.ModuleList(convs)
        self.blocks = nn.ModuleList(blocks)

        self.norms = norm_layer(self.embed_dims[-1])
        self.avgpool = nn.AdaptiveAvgPool1d(1)
        self.head = nn.Linear(self.embed_dims[-1], num_classes) if num_classes > 0 else nn.Identity()

        self.apply(self._init_weights)

    @property
    def dim_out(self):
        return self.embed_dims[-1]

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=0.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.Conv2d):
            nn.init.normal_(m.weight, std=0.02)
            for name, _ in m.named_parameters():
                if name in ['bias']:
                    nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.weight, 1.0)
            nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.BatchNorm2d):
            nn.init.constant_(m.weight, 1.0)
            nn.init.constant_(m.bias, 0)

    def forward_features_unpool(self, x):
        """
        forward until avg pooling 
        Args:
            x (_type_): input image tensor
        """
        input_size = (x.shape[2], x.shape[3])
        for conv, block in zip(self.convs, self.blocks):
            x, input_size = conv(x, input_size)
            x, input_size = block(x, input_size)
        return x

    def forward_features(self, x):
        x = self.forward_features_unpool(x)

        # (batch_size, num_tokens, token_dim)
        x = self.avgpool(ops.transpose(x, 1, 2))
        # (batch_size, 1, num_tokens)
        x = ops.flatten(x, 1)
        x = self.norms(x)

        return x

    def forward(self, x):
        x = self.forward_features(x)
        x = self.head(x)
        return x

    @classmethod
    def from_config(cls, config):
        return cls(
            depths=config.depths,
            embed_dims=config.dim_embed,
            num_heads=config.num_heads,
            num_groups=config.num_groups,
            patch_size=config.patch_size,
            patch_stride=config.patch_stride,
            patch_padding=config.patch_padding,
            patch_prenorm=config.patch_prenorm,
            drop_path_rate=config.drop_path_rate,
            window_size=config.window_size,
        )

# Copied from transformers.models.llama.modeling_llama._get_unpad_data
def _get_unpad_data(attention_mask):
    seqlens_in_batch = attention_mask.sum(dim=-1, dtype=mindspore.int32)
    indices = ops.nonzero(attention_mask.flatten(), as_tuple=False).flatten()
    max_seqlen_in_batch = seqlens_in_batch.max().item()
    cu_seqlens = F.pad(ops.cumsum(seqlens_in_batch, dim=0, dtype=mindspore.int32), (1, 0))
    return (
        indices,
        cu_seqlens,
        max_seqlen_in_batch,
    )


def shift_tokens_right(input_ids: Tensor, pad_token_id: int, decoder_start_token_id: int):
    """
    Shift input ids one token to the right.
    """
    shifted_input_ids = input_ids.new_zeros(input_ids.shape)
    shifted_input_ids[:, 1:] = input_ids[:, :-1].clone()
    shifted_input_ids[:, 0] = decoder_start_token_id

    if pad_token_id is None:
        raise ValueError("self.model.config.pad_token_id has to be defined.")
    # replace possible -100 values in labels by `pad_token_id`
    shifted_input_ids.masked_fill_(shifted_input_ids == -100, pad_token_id)

    return shifted_input_ids


class Florence2LearnedPositionalEmbedding(nn.Embedding):
    """
    This module learns positional embeddings up to a fixed maximum size.
    """

    def __init__(self, num_embeddings: int, embedding_dim: int):
        # Florence2 is set up so that if padding_idx is specified then offset the embedding ids by 2
        # and adjust num_embeddings appropriately. Other models don't have this hack
        self.offset = 2
        super().__init__(num_embeddings + self.offset, embedding_dim)

    def forward(self, input_ids: Tensor, past_key_values_length: int = 0):
        """`input_ids' shape is expected to be [bsz x seqlen]."""

        bsz, seq_len = input_ids.shape[:2]
        positions = ops.arange(
            past_key_values_length, past_key_values_length + seq_len, dtype=mindspore.int64
        ).broadcast_to((bsz, -1))

        return super().forward(positions + self.offset)


class Florence2ScaledWordEmbedding(nn.Embedding):
    """
    This module overrides nn.Embeddings' forward by multiplying with embeddings scale.
    """

    def __init__(self, num_embeddings: int, embedding_dim: int, padding_idx: int, embed_scale: Optional[float] = 1.0):
        super().__init__(num_embeddings, embedding_dim, padding_idx)
        self.embed_scale = embed_scale

    def forward(self, input_ids: Tensor):
        return super().forward(input_ids) * self.embed_scale


class Florence2Attention(nn.Module):
    """Multi-headed attention from 'Attention Is All You Need' paper"""

    def __init__(
        self,
        embed_dim: int,
        num_heads: int,
        dropout: float = 0.0,
        is_decoder: bool = False,
        bias: bool = True,
        is_causal: bool = False,
        config: Optional[Florence2LanguageConfig] = None,
    ):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.dropout = dropout
        self.head_dim = embed_dim // num_heads
        self.config = config

        if (self.head_dim * num_heads) != self.embed_dim:
            raise ValueError(
                f"embed_dim must be divisible by num_heads (got `embed_dim`: {self.embed_dim}"
                f" and `num_heads`: {num_heads})."
            )
        self.scaling = self.head_dim**-0.5
        self.is_decoder = is_decoder
        self.is_causal = is_causal

        self.k_proj = nn.Linear(embed_dim, embed_dim, bias=bias)
        self.v_proj = nn.Linear(embed_dim, embed_dim, bias=bias)
        self.q_proj = nn.Linear(embed_dim, embed_dim, bias=bias)
        self.out_proj = nn.Linear(embed_dim, embed_dim, bias=bias)

    def _shape(self, tensor: Tensor, seq_len: int, bsz: int):
        return ops.transpose(tensor.view(bsz, seq_len, self.num_heads, self.head_dim), 1, 2)

    def forward(
        self,
        hidden_states: Tensor,
        key_value_states: Optional[Tensor] = None,
        past_key_value: Optional[Tuple[Tensor]] = None,
        attention_mask: Optional[Tensor] = None,
        layer_head_mask: Optional[Tensor] = None,
        output_attentions: bool = False,
    ) -> Tuple[Tensor, Optional[Tensor], Optional[Tuple[Tensor]]]:
        """Input shape: Batch x Time x Channel"""

        # if key_value_states are provided this layer is used as a cross-attention layer
        # for the decoder
        is_cross_attention = key_value_states is not None

        bsz, tgt_len, _ = hidden_states.shape

        # get query proj
        query_states = self.q_proj(hidden_states) * self.scaling
        # get key, value proj
        # `past_key_value[0].shape[2] == key_value_states.shape[1]`
        # is checking that the `sequence_length` of the `past_key_value` is the same as
        # the provided `key_value_states` to support prefix tuning
        if (
            is_cross_attention
            and past_key_value is not None
            and past_key_value[0].shape[2] == key_value_states.shape[1]
        ):
            # reuse k,v, cross_attentions
            key_states = past_key_value[0]
            value_states = past_key_value[1]
        elif is_cross_attention:
            # cross_attentions
            key_states = self._shape(self.k_proj(key_value_states), -1, bsz)
            value_states = self._shape(self.v_proj(key_value_states), -1, bsz)
        elif past_key_value is not None:
            # reuse k, v, self_attention
            key_states = self._shape(self.k_proj(hidden_states), -1, bsz)
            value_states = self._shape(self.v_proj(hidden_states), -1, bsz)
            key_states = ops.cat([past_key_value[0], key_states], dim=2)
            value_states = ops.cat([past_key_value[1], value_states], dim=2)
        else:
            # self_attention
            key_states = self._shape(self.k_proj(hidden_states), -1, bsz)
            value_states = self._shape(self.v_proj(hidden_states), -1, bsz)

        if self.is_decoder:
            # if cross_attention save Tuple(Tensor, Tensor) of all cross attention key/value_states.
            # Further calls to cross_attention layer can then reuse all cross-attention
            # key/value_states (first "if" case)
            # if uni-directional self-attention (decoder) save Tuple(Tensor, Tensor) of
            # all previous decoder key/value_states. Further calls to uni-directional self-attention
            # can concat previous decoder key/value_states to current projected key/value_states (third "elif" case)
            # if encoder bi-directional self-attention `past_key_value` is always `None`
            past_key_value = (key_states, value_states)

        proj_shape = (bsz * self.num_heads, -1, self.head_dim)
        query_states = self._shape(query_states, tgt_len, bsz).view(*proj_shape)
        key_states = key_states.reshape(*proj_shape)
        value_states = value_states.reshape(*proj_shape)

        src_len = key_states.shape[1]
        attn_weights = ops.bmm(query_states, ops.transpose(key_states, 1, 2))

        if attn_weights.shape != (bsz * self.num_heads, tgt_len, src_len):
            raise ValueError(
                f"Attention weights should be of size {(bsz * self.num_heads, tgt_len, src_len)}, but is"
                f" {attn_weights.shape}"
            )

        if attention_mask is not None:
            if attention_mask.shape != (bsz, 1, tgt_len, src_len):
                raise ValueError(
                    f"Attention mask should be of size {(bsz, 1, tgt_len, src_len)}, but is {attention_mask.shape}"
                )
            attn_weights = attn_weights.view(bsz, self.num_heads, tgt_len, src_len) + attention_mask
            attn_weights = attn_weights.view(bsz * self.num_heads, tgt_len, src_len)

        attn_weights = nn.functional.softmax(attn_weights, dim=-1)

        if layer_head_mask is not None:
            if layer_head_mask.shape != (self.num_heads,):
                raise ValueError(
                    f"Head mask for a single layer should be of size {(self.num_heads,)}, but is"
                    f" {layer_head_mask.shape}"
                )
            attn_weights = layer_head_mask.view(1, -1, 1, 1) * attn_weights.view(bsz, self.num_heads, tgt_len, src_len)
            attn_weights = attn_weights.view(bsz * self.num_heads, tgt_len, src_len)

        if output_attentions:
            # this operation is a bit awkward, but it's required to
            # make sure that attn_weights keeps its gradient.
            # In order to do so, attn_weights have to be reshaped
            # twice and have to be reused in the following
            attn_weights_reshaped = attn_weights.view(bsz, self.num_heads, tgt_len, src_len)
            attn_weights = attn_weights_reshaped.view(bsz * self.num_heads, tgt_len, src_len)
        else:
            attn_weights_reshaped = None

        attn_probs = nn.functional.dropout(attn_weights, p=self.dropout, training=self.training)

        attn_output = ops.bmm(attn_probs, value_states)

        if attn_output.shape != (bsz * self.num_heads, tgt_len, self.head_dim):
            raise ValueError(
                f"`attn_output` should be of size {(bsz * self.num_heads, tgt_len, self.head_dim)}, but is"
                f" {attn_output.shape}"
            )

        attn_output = attn_output.view(bsz, self.num_heads, tgt_len, self.head_dim)
        attn_output = ops.transpose(attn_output, 1, 2)

        # Use the `embed_dim` from the config (stored in the class) rather than `hidden_state` because `attn_output` can be
        # partitioned across GPUs when using tensor-parallelism.
        attn_output = attn_output.reshape(bsz, tgt_len, self.embed_dim)

        attn_output = self.out_proj(attn_output)

        return attn_output, attn_weights_reshaped, past_key_value


class Florence2SdpaAttention(Florence2Attention):
    def forward(
        self,
        hidden_states: Tensor,
        key_value_states: Optional[Tensor] = None,
        past_key_value: Optional[Tuple[Tensor]] = None,
        attention_mask: Optional[Tensor] = None,
        layer_head_mask: Optional[Tensor] = None,
        output_attentions: bool = False,
    ) -> Tuple[Tensor, Optional[Tensor], Optional[Tuple[Tensor]]]:
        """Input shape: Batch x Time x Channel"""
        if output_attentions or layer_head_mask is not None:
            # TODO: Improve this warning with e.g. `model.config._attn_implementation = "manual"` once this is implemented.
            logger.warning(
                "Florence2Model is using Florence2SdpaAttention, but `torch.nn.functional."
                "scaled_dot_product_attention` does not support `output_attentions=True` or `layer_head_mask` not None."
                "Falling back to the manual attention implementation, but specifying the manual implementation will "
                "be required from Transformers version v5.0.0 onwards."
                ' This warning can be removed using the argument `attn_implementation="eager"` when loading the model.'
            )
            return super().forward(
                hidden_states,
                key_value_states=key_value_states,
                past_key_value=past_key_value,
                attention_mask=attention_mask,
                layer_head_mask=layer_head_mask,
                output_attentions=output_attentions,
            )

        # if key_value_states are provided this layer is used as a cross-attention layer
        # for the decoder
        is_cross_attention = key_value_states is not None

        bsz, tgt_len, _ = hidden_states.shape

        # get query proj
        query_states = self.q_proj(hidden_states)
        # get key, value proj
        # `past_key_value[0].shape[2] == key_value_states.shape[1]`
        # is checking that the `sequence_length` of the `past_key_value` is the same as
        # the provided `key_value_states` to support prefix tuning
        if (
            is_cross_attention
            and past_key_value is not None
            and past_key_value[0].shape[2] == key_value_states.shape[1]
        ):
            # reuse k,v, cross_attentions
            key_states = past_key_value[0]
            value_states = past_key_value[1]
        elif is_cross_attention:
            # cross_attentions
            key_states = self._shape(self.k_proj(key_value_states), -1, bsz)
            value_states = self._shape(self.v_proj(key_value_states), -1, bsz)
        elif past_key_value is not None:
            # reuse k, v, self_attention
            key_states = self._shape(self.k_proj(hidden_states), -1, bsz)
            value_states = self._shape(self.v_proj(hidden_states), -1, bsz)
            key_states = ops.cat([past_key_value[0], key_states], dim=2)
            value_states = ops.cat([past_key_value[1], value_states], dim=2)
        else:
            # self_attention
            key_states = self._shape(self.k_proj(hidden_states), -1, bsz)
            value_states = self._shape(self.v_proj(hidden_states), -1, bsz)

        if self.is_decoder:
            # if cross_attention save Tuple(Tensor, Tensor) of all cross attention key/value_states.
            # Further calls to cross_attention layer can then reuse all cross-attention
            # key/value_states (first "if" case)
            # if uni-directional self-attention (decoder) save Tuple(Tensor, Tensor) of
            # all previous decoder key/value_states. Further calls to uni-directional self-attention
            # can concat previous decoder key/value_states to current projected key/value_states (third "elif" case)
            # if encoder bi-directional self-attention `past_key_value` is always `None`
            past_key_value = (key_states, value_states)

        query_states = self._shape(query_states, tgt_len, bsz)

        # We dispatch to SDPA's Flash Attention or Efficient kernels via this `is_causal` if statement instead of an inline conditional assignment
        # in SDPA to support both torch.compile's dynamic shapes and full graph options. An inline conditional prevents dynamic shapes from compiling.
        # The tgt_len > 1 is necessary to match with AttentionMaskConverter.to_causal_4d that does not create a causal mask in case tgt_len == 1.
        is_causal = self.is_causal and attention_mask is not None and tgt_len > 1

        # NOTE: SDPA with memory-efficient backend is currently (torch==2.1.2) bugged when using non-contiguous inputs and a custom attn_mask,
        # but we are fine here as `_shape` do call ``. Reference: https://github.com/pytorch/pytorch/issues/112577
        attn_output = F.scaled_dot_product_attention(
            query_states,
            key_states,
            value_states,
            attn_mask=attention_mask,
            dropout_p=self.dropout if self.training else 0.0,
            is_causal=is_causal,
        )

        if attn_output.shape != (bsz, self.num_heads, tgt_len, self.head_dim):
            raise ValueError(
                f"`attn_output` should be of size {(bsz, self.num_heads, tgt_len, self.head_dim)}, but is"
                f" {attn_output.shape}"
            )

        attn_output = ops.transpose(attn_output, 1, 2)

        # Use the `embed_dim` from the config (stored in the class) rather than `hidden_state` because `attn_output` can be
        # partitioned across GPUs when using tensor-parallelism.
        attn_output = attn_output.reshape(bsz, tgt_len, self.embed_dim)

        attn_output = self.out_proj(attn_output)

        return attn_output, None, past_key_value


FLORENCE2_ATTENTION_CLASSES = {
    "eager": Florence2Attention,
    "sdpa": Florence2SdpaAttention,
}


class Florence2EncoderLayer(nn.Module):
    def __init__(self, config: Florence2LanguageConfig):
        super().__init__()
        self.embed_dim = config.d_model

        self.self_attn = FLORENCE2_ATTENTION_CLASSES[config._attn_implementation](
            embed_dim=self.embed_dim,
            num_heads=config.encoder_attention_heads,
            dropout=config.attention_dropout,
            config=config,
        )
        self.self_attn_layer_norm = nn.LayerNorm(self.embed_dim)
        self.dropout = config.dropout
        self.activation_fn = ACT2FN[config.activation_function]
        self.activation_dropout = config.activation_dropout
        self.fc1 = nn.Linear(self.embed_dim, config.encoder_ffn_dim)
        self.fc2 = nn.Linear(config.encoder_ffn_dim, self.embed_dim)
        self.final_layer_norm = nn.LayerNorm(self.embed_dim)

    def forward(
        self,
        hidden_states: Tensor,
        attention_mask: Tensor,
        layer_head_mask: Tensor,
        output_attentions: Optional[bool] = False,
    ) -> Tuple[Tensor, Optional[Tensor]]:
        """
        Args:
            hidden_states (`Tensor`): input to the layer of shape `(batch, seq_len, embed_dim)`
            attention_mask (`Tensor`): attention mask of size
                `(batch, 1, tgt_len, src_len)` where padding elements are indicated by very large negative values.
            layer_head_mask (`Tensor`): mask for attention heads in a given layer of size
                `(encoder_attention_heads,)`.
            output_attentions (`bool`, *optional*):
                Whether or not to return the attentions tensors of all attention layers. See `attentions` under
                returned tensors for more detail.
        """
        residual = hidden_states
        hidden_states, attn_weights, _ = self.self_attn(
            hidden_states=hidden_states,
            attention_mask=attention_mask,
            layer_head_mask=layer_head_mask,
            output_attentions=output_attentions,
        )
        hidden_states = nn.functional.dropout(hidden_states, p=self.dropout, training=self.training)
        hidden_states = residual + hidden_states
        hidden_states = self.self_attn_layer_norm(hidden_states)

        residual = hidden_states
        hidden_states = self.activation_fn(self.fc1(hidden_states))
        hidden_states = nn.functional.dropout(hidden_states, p=self.activation_dropout, training=self.training)
        hidden_states = self.fc2(hidden_states)
        hidden_states = nn.functional.dropout(hidden_states, p=self.dropout, training=self.training)
        hidden_states = residual + hidden_states
        hidden_states = self.final_layer_norm(hidden_states)

        if hidden_states.dtype == mindspore.float16 and (
            ops.isinf(hidden_states).any() or ops.isnan(hidden_states).any()
        ):
            clamp_value = ops.finfo(hidden_states.dtype).max - 1000
            hidden_states = ops.clamp(hidden_states, min=-clamp_value, max=clamp_value)

        outputs = (hidden_states,)

        if output_attentions:
            outputs += (attn_weights,)

        return outputs


class Florence2DecoderLayer(nn.Module):
    def __init__(self, config: Florence2LanguageConfig):
        super().__init__()
        self.embed_dim = config.d_model

        self.self_attn = FLORENCE2_ATTENTION_CLASSES[config._attn_implementation](
            embed_dim=self.embed_dim,
            num_heads=config.decoder_attention_heads,
            dropout=config.attention_dropout,
            is_decoder=True,
            is_causal=True,
            config=config,
        )
        self.dropout = config.dropout
        self.activation_fn = ACT2FN[config.activation_function]
        self.activation_dropout = config.activation_dropout

        self.self_attn_layer_norm = nn.LayerNorm(self.embed_dim)
        self.encoder_attn = FLORENCE2_ATTENTION_CLASSES[config._attn_implementation](
            self.embed_dim,
            config.decoder_attention_heads,
            dropout=config.attention_dropout,
            is_decoder=True,
            config=config,
        )
        self.encoder_attn_layer_norm = nn.LayerNorm(self.embed_dim)
        self.fc1 = nn.Linear(self.embed_dim, config.decoder_ffn_dim)
        self.fc2 = nn.Linear(config.decoder_ffn_dim, self.embed_dim)
        self.final_layer_norm = nn.LayerNorm(self.embed_dim)

    def forward(
        self,
        hidden_states: Tensor,
        attention_mask: Optional[Tensor] = None,
        encoder_hidden_states: Optional[Tensor] = None,
        encoder_attention_mask: Optional[Tensor] = None,
        layer_head_mask: Optional[Tensor] = None,
        cross_attn_layer_head_mask: Optional[Tensor] = None,
        past_key_value: Optional[Tuple[Tensor]] = None,
        output_attentions: Optional[bool] = False,
        use_cache: Optional[bool] = True,
    ) -> Tuple[Tensor, Optional[Tuple[Tensor, Tensor]]]:
        """
        Args:
            hidden_states (`Tensor`): input to the layer of shape `(batch, seq_len, embed_dim)`
            attention_mask (`Tensor`): attention mask of size
                `(batch, 1, tgt_len, src_len)` where padding elements are indicated by very large negative values.
            encoder_hidden_states (`Tensor`):
                cross attention input to the layer of shape `(batch, seq_len, embed_dim)`
            encoder_attention_mask (`Tensor`): encoder attention mask of size
                `(batch, 1, tgt_len, src_len)` where padding elements are indicated by very large negative values.
            layer_head_mask (`Tensor`): mask for attention heads in a given layer of size
                `(encoder_attention_heads,)`.
            cross_attn_layer_head_mask (`Tensor`): mask for cross-attention heads in a given layer of
                size `(decoder_attention_heads,)`.
            past_key_value (`Tuple(Tensor)`): cached past key and value projection states
            output_attentions (`bool`, *optional*):
                Whether or not to return the attentions tensors of all attention layers. See `attentions` under
                returned tensors for more detail.
        """
        residual = hidden_states

        # Self Attention
        # decoder uni-directional self-attention cached key/values tuple is at positions 1,2
        self_attn_past_key_value = past_key_value[:2] if past_key_value is not None else None
        # add present self-attn cache to positions 1,2 of present_key_value tuple
        hidden_states, self_attn_weights, present_key_value = self.self_attn(
            hidden_states=hidden_states,
            past_key_value=self_attn_past_key_value,
            attention_mask=attention_mask,
            layer_head_mask=layer_head_mask,
            output_attentions=output_attentions,
        )
        hidden_states = nn.functional.dropout(hidden_states, p=self.dropout, training=self.training)
        hidden_states = residual + hidden_states
        hidden_states = self.self_attn_layer_norm(hidden_states)

        # Cross-Attention Block
        cross_attn_present_key_value = None
        cross_attn_weights = None
        if encoder_hidden_states is not None:
            residual = hidden_states

            # cross_attn cached key/values tuple is at positions 3,4 of present_key_value tuple
            cross_attn_past_key_value = past_key_value[-2:] if past_key_value is not None else None
            hidden_states, cross_attn_weights, cross_attn_present_key_value = self.encoder_attn(
                hidden_states=hidden_states,
                key_value_states=encoder_hidden_states,
                attention_mask=encoder_attention_mask,
                layer_head_mask=cross_attn_layer_head_mask,
                past_key_value=cross_attn_past_key_value,
                output_attentions=output_attentions,
            )
            hidden_states = nn.functional.dropout(hidden_states, p=self.dropout, training=self.training)
            hidden_states = residual + hidden_states
            hidden_states = self.encoder_attn_layer_norm(hidden_states)

            # add cross-attn to positions 3,4 of present_key_value tuple
            present_key_value = present_key_value + cross_attn_present_key_value

        # Fully Connected
        residual = hidden_states
        hidden_states = self.activation_fn(self.fc1(hidden_states))
        hidden_states = nn.functional.dropout(hidden_states, p=self.activation_dropout, training=self.training)
        hidden_states = self.fc2(hidden_states)
        hidden_states = nn.functional.dropout(hidden_states, p=self.dropout, training=self.training)
        hidden_states = residual + hidden_states
        hidden_states = self.final_layer_norm(hidden_states)

        outputs = (hidden_states,)

        if output_attentions:
            outputs += (self_attn_weights, cross_attn_weights)

        if use_cache:
            outputs += (present_key_value,)

        return outputs


class Florence2LanguagePreTrainedModel(PreTrainedModel):
    config_class = Florence2LanguageConfig
    base_model_prefix = "model"
    supports_gradient_checkpointing = True
    _keys_to_ignore_on_load_unexpected = ["encoder.version", "decoder.version"]
    _no_split_modules = [r"Florence2EncoderLayer", r"Florence2DecoderLayer"]
    _skip_keys_device_placement = "past_key_values"
    _supports_sdpa = True

    def _init_weights(self, module):
        std = self.config.init_std
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=std)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=std)
            if module.padding_idx is not None:
                # module.weight.data[module.padding_idx].zero_()
                nn.init.zeros_(module.weight[module.padding_idx])

    @property
    def dummy_inputs(self):
        pad_token = self.config.pad_token_id
        input_ids = mindspore.tensor([[0, 6, 10, 4, 2], [0, 8, 12, 2, pad_token]])
        dummy_inputs = {
            "attention_mask": input_ids.ne(pad_token),
            "input_ids": input_ids,
        }
        return dummy_inputs


class Florence2Encoder(Florence2LanguagePreTrainedModel):
    """
    Transformer encoder consisting of *config.encoder_layers* self attention layers. Each layer is a
    [`Florence2EncoderLayer`].

    Args:
        config: Florence2LanguageConfig
        embed_tokens (nn.Embedding): output embedding
    """

    def __init__(self, config: Florence2LanguageConfig, embed_tokens: Optional[nn.Embedding] = None):
        super().__init__(config)

        self.dropout = config.dropout
        self.layerdrop = config.encoder_layerdrop

        embed_dim = config.d_model
        self.padding_idx = config.pad_token_id
        self.max_source_positions = config.max_position_embeddings
        embed_scale = math.sqrt(embed_dim) if config.scale_embedding else 1.0

        self.embed_tokens = Florence2ScaledWordEmbedding(
            config.vocab_size, embed_dim, self.padding_idx, embed_scale=embed_scale
        )

        if embed_tokens is not None:
            self.embed_tokens.weight = embed_tokens.weight

        self.embed_positions = Florence2LearnedPositionalEmbedding(
            config.max_position_embeddings,
            embed_dim,
        )
        self.layers = nn.ModuleList([Florence2EncoderLayer(config) for _ in range(config.encoder_layers)])
        self._use_sdpa = config._attn_implementation == "sdpa"
        self.layernorm_embedding = nn.LayerNorm(embed_dim)

        self.gradient_checkpointing = False
        # Initialize weights and apply final processing
        self.post_init()

    def get_input_embeddings(self):
        return self.embed_tokens

    def set_input_embeddings(self, value):
        self.embed_tokens = value

    def forward(
        self,
        input_ids: Tensor = None,
        attention_mask: Optional[Tensor] = None,
        head_mask: Optional[Tensor] = None,
        inputs_embeds: Optional[Tensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple, BaseModelOutput]:
        r"""
        Args:
            input_ids (`Tensor` of shape `(batch_size, sequence_length)`):
                Indices of input sequence tokens in the vocabulary. Padding will be ignored by default should you
                provide it.

                Indices can be obtained using [`AutoTokenizer`]. See [`PreTrainedTokenizer.encode`] and
                [`PreTrainedTokenizer.__call__`] for details.

                [What are input IDs?](../glossary#input-ids)
            attention_mask (`Tensor` of shape `(batch_size, sequence_length)`, *optional*):
                Mask to avoid performing attention on padding token indices. Mask values selected in `[0, 1]`:

                - 1 for tokens that are **not masked**,
                - 0 for tokens that are **masked**.

                [What are attention masks?](../glossary#attention-mask)
            head_mask (`Tensor` of shape `(encoder_layers, encoder_attention_heads)`, *optional*):
                Mask to nullify selected heads of the attention modules. Mask values selected in `[0, 1]`:

                - 1 indicates the head is **not masked**,
                - 0 indicates the head is **masked**.

            inputs_embeds (`Tensor` of shape `(batch_size, sequence_length, hidden_size)`, *optional*):
                Optionally, instead of passing `input_ids` you can choose to directly pass an embedded representation.
                This is useful if you want more control over how to convert `input_ids` indices into associated vectors
                than the model's internal embedding lookup matrix.
            output_attentions (`bool`, *optional*):
                Whether or not to return the attentions tensors of all attention layers. See `attentions` under
                returned tensors for more detail.
            output_hidden_states (`bool`, *optional*):
                Whether or not to return the hidden states of all layers. See `hidden_states` under returned tensors
                for more detail.
            return_dict (`bool`, *optional*):
                Whether or not to return a [`~utils.ModelOutput`] instead of a plain tuple.
        """
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        # retrieve input_ids and inputs_embeds
        if input_ids is not None and inputs_embeds is not None:
            raise ValueError("You cannot specify both input_ids and inputs_embeds at the same time")
        elif input_ids is not None:
            input = input_ids
            input_ids = input_ids.view(-1, input_ids.shape[-1])
        elif inputs_embeds is not None:
            input = inputs_embeds[:, :, -1]
        else:
            raise ValueError("You have to specify either input_ids or inputs_embeds")

        if inputs_embeds is None:
            inputs_embeds = self.embed_tokens(input_ids)

        embed_pos = self.embed_positions(input)
        embed_pos = embed_pos.to(inputs_embeds.dtype)

        hidden_states = inputs_embeds + embed_pos
        hidden_states = self.layernorm_embedding(hidden_states)
        hidden_states = nn.functional.dropout(hidden_states, p=self.dropout, training=self.training)

        # expand attention_mask
        if attention_mask is not None:

            # [bsz, seq_len] -> [bsz, 1, tgt_seq_len, src_seq_len]
            attention_mask = _prepare_4d_attention_mask(attention_mask, inputs_embeds.dtype)

        encoder_states = () if output_hidden_states else None
        all_attentions = () if output_attentions else None

        # check if head_mask has a correct number of layers specified if desired
        if head_mask is not None:
            if head_mask.shape[0] != (len(self.layers)):
                raise ValueError(
                    f"The head_mask should be specified for {len(self.layers)} layers, but it is for"
                    f" {head_mask.shape[0]}."
                )

        for idx, encoder_layer in enumerate(self.layers):
            if output_hidden_states:
                encoder_states = encoder_states + (hidden_states,)
            # add LayerDrop (see https://arxiv.org/abs/1909.11556 for description)
            to_drop = False
            if self.training:
                dropout_probability = ops.rand([])
                if dropout_probability < self.layerdrop:  # skip the layer
                    to_drop = True

            if to_drop:
                layer_outputs = (None, None)
            else:
                if self.gradient_checkpointing and self.training:
                    layer_outputs = self._gradient_checkpointing_func(
                        encoder_layer.__call__,
                        hidden_states,
                        attention_mask,
                        (head_mask[idx] if head_mask is not None else None),
                        output_attentions,
                    )
                else:
                    layer_outputs = encoder_layer(
                        hidden_states,
                        attention_mask,
                        layer_head_mask=(head_mask[idx] if head_mask is not None else None),
                        output_attentions=output_attentions,
                    )

                hidden_states = layer_outputs[0]

            if output_attentions:
                all_attentions = all_attentions + (layer_outputs[1],)

        if output_hidden_states:
            encoder_states = encoder_states + (hidden_states,)

        if not return_dict:
            return tuple(v for v in [hidden_states, encoder_states, all_attentions] if v is not None)
        return BaseModelOutput(
            last_hidden_state=hidden_states, hidden_states=encoder_states, attentions=all_attentions
        )


class Florence2Decoder(Florence2LanguagePreTrainedModel):
    """
    Transformer decoder consisting of *config.decoder_layers* layers. Each layer is a [`Florence2DecoderLayer`]

    Args:
        config: Florence2LanguageConfig
        embed_tokens (nn.Embedding): output embedding
    """

    def __init__(self, config: Florence2LanguageConfig, embed_tokens: Optional[nn.Embedding] = None):
        super().__init__(config)
        self.dropout = config.dropout
        self.layerdrop = config.decoder_layerdrop
        self.padding_idx = config.pad_token_id
        self.max_target_positions = config.max_position_embeddings
        embed_scale = math.sqrt(config.d_model) if config.scale_embedding else 1.0

        self.embed_tokens = Florence2ScaledWordEmbedding(
            config.vocab_size, config.d_model, self.padding_idx, embed_scale=embed_scale
        )

        if embed_tokens is not None:
            self.embed_tokens.weight = embed_tokens.weight

        self.embed_positions = Florence2LearnedPositionalEmbedding(
            config.max_position_embeddings,
            config.d_model,
        )
        self.layers = nn.ModuleList([Florence2DecoderLayer(config) for _ in range(config.decoder_layers)])
        self._use_sdpa = config._attn_implementation == "sdpa"

        self.layernorm_embedding = nn.LayerNorm(config.d_model)

        self.gradient_checkpointing = False
        # Initialize weights and apply final processing
        self.post_init()

    def get_input_embeddings(self):
        return self.embed_tokens

    def set_input_embeddings(self, value):
        self.embed_tokens = value

    def forward(
        self,
        input_ids: Tensor = None,
        attention_mask: Optional[Tensor] = None,
        encoder_hidden_states: Optional[Tensor] = None,
        encoder_attention_mask: Optional[Tensor] = None,
        head_mask: Optional[Tensor] = None,
        cross_attn_head_mask: Optional[Tensor] = None,
        past_key_values: Optional[List[Tensor]] = None,
        inputs_embeds: Optional[Tensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple, BaseModelOutputWithPastAndCrossAttentions]:
        r"""
        Args:
            input_ids (`Tensor` of shape `(batch_size, sequence_length)`):
                Indices of input sequence tokens in the vocabulary. Padding will be ignored by default should you
                provide it.

                Indices can be obtained using [`AutoTokenizer`]. See [`PreTrainedTokenizer.encode`] and
                [`PreTrainedTokenizer.__call__`] for details.

                [What are input IDs?](../glossary#input-ids)
            attention_mask (`Tensor` of shape `(batch_size, sequence_length)`, *optional*):
                Mask to avoid performing attention on padding token indices. Mask values selected in `[0, 1]`:

                - 1 for tokens that are **not masked**,
                - 0 for tokens that are **masked**.

                [What are attention masks?](../glossary#attention-mask)
            encoder_hidden_states (`Tensor` of shape `(batch_size, encoder_sequence_length, hidden_size)`, *optional*):
                Sequence of hidden-states at the output of the last layer of the encoder. Used in the cross-attention
                of the decoder.
            encoder_attention_mask (`Tensor` of shape `(batch_size, encoder_sequence_length)`, *optional*):
                Mask to avoid performing cross-attention on padding tokens indices of encoder input_ids. Mask values
                selected in `[0, 1]`:

                - 1 for tokens that are **not masked**,
                - 0 for tokens that are **masked**.

                [What are attention masks?](../glossary#attention-mask)
            head_mask (`Tensor` of shape `(decoder_layers, decoder_attention_heads)`, *optional*):
                Mask to nullify selected heads of the attention modules. Mask values selected in `[0, 1]`:

                - 1 indicates the head is **not masked**,
                - 0 indicates the head is **masked**.

            cross_attn_head_mask (`Tensor` of shape `(decoder_layers, decoder_attention_heads)`, *optional*):
                Mask to nullify selected heads of the cross-attention modules in the decoder to avoid performing
                cross-attention on hidden heads. Mask values selected in `[0, 1]`:

                - 1 indicates the head is **not masked**,
                - 0 indicates the head is **masked**.

            past_key_values (`tuple(tuple(Tensor))`, *optional*, returned when `use_cache=True` is passed or when `config.use_cache=True`):
                Tuple of `tuple(Tensor)` of length `config.n_layers`, with each tuple having 2 tensors of
                shape `(batch_size, num_heads, sequence_length, embed_size_per_head)`) and 2 additional tensors of
                shape `(batch_size, num_heads, encoder_sequence_length, embed_size_per_head)`.

                Contains pre-computed hidden-states (key and values in the self-attention blocks and in the
                cross-attention blocks) that can be used (see `past_key_values` input) to speed up sequential decoding.

                If `past_key_values` are used, the user can optionally input only the last `decoder_input_ids` (those
                that don't have their past key value states given to this model) of shape `(batch_size, 1)` instead of
                all `decoder_input_ids` of shape `(batch_size, sequence_length)`.
            inputs_embeds (`Tensor` of shape `(batch_size, sequence_length, hidden_size)`, *optional*):
                Optionally, instead of passing `input_ids` you can choose to directly pass an embedded representation.
                This is useful if you want more control over how to convert `input_ids` indices into associated vectors
                than the model's internal embedding lookup matrix.
            output_attentions (`bool`, *optional*):
                Whether or not to return the attentions tensors of all attention layers. See `attentions` under
                returned tensors for more detail.
            output_hidden_states (`bool`, *optional*):
                Whether or not to return the hidden states of all layers. See `hidden_states` under returned tensors
                for more detail.
            return_dict (`bool`, *optional*):
                Whether or not to return a [`~utils.ModelOutput`] instead of a plain tuple.
        """
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        use_cache = use_cache if use_cache is not None else self.config.use_cache
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        # retrieve input_ids and inputs_embeds
        if input_ids is not None and inputs_embeds is not None:
            raise ValueError("You cannot specify both decoder_input_ids and decoder_inputs_embeds at the same time")
        elif input_ids is not None:
            input = input_ids
            input_shape = input.shape
            input_ids = input_ids.view(-1, input_shape[-1])
        elif inputs_embeds is not None:
            input_shape = inputs_embeds.shape[:-1]
            input = inputs_embeds[:, :, -1]
        else:
            raise ValueError("You have to specify either decoder_input_ids or decoder_inputs_embeds")

        # past_key_values_length
        past_key_values_length = past_key_values[0][0].shape[2] if past_key_values is not None else 0

        if inputs_embeds is None:
            inputs_embeds = self.embed_tokens(input)

        # 4d mask is passed through the layers
        attention_mask = _prepare_4d_causal_attention_mask(
            attention_mask, input_shape, inputs_embeds, past_key_values_length
        )

        # expand encoder attention mask
        if encoder_hidden_states is not None and encoder_attention_mask is not None:
            # [bsz, seq_len] -> [bsz, 1, tgt_seq_len, src_seq_len]
            encoder_attention_mask = _prepare_4d_attention_mask(
                encoder_attention_mask, inputs_embeds.dtype, tgt_len=input_shape[-1]
            )

        # embed positions
        positions = self.embed_positions(input, past_key_values_length)
        positions = positions.to(inputs_embeds.dtype)

        hidden_states = inputs_embeds + positions
        hidden_states = self.layernorm_embedding(hidden_states)

        hidden_states = nn.functional.dropout(hidden_states, p=self.dropout, training=self.training)

        if self.gradient_checkpointing and self.training:
            if use_cache:
                logger.warning_once(
                    "`use_cache=True` is incompatible with gradient checkpointing. Setting `use_cache=False`..."
                )
                use_cache = False

        # decoder layers
        all_hidden_states = () if output_hidden_states else None
        all_self_attns = () if output_attentions else None
        all_cross_attentions = () if (output_attentions and encoder_hidden_states is not None) else None
        next_decoder_cache = () if use_cache else None

        # check if head_mask/cross_attn_head_mask has a correct number of layers specified if desired
        for attn_mask, mask_name in zip([head_mask, cross_attn_head_mask], ["head_mask", "cross_attn_head_mask"]):
            if attn_mask is not None:
                if attn_mask.shape[0] != (len(self.layers)):
                    raise ValueError(
                        f"The `{mask_name}` should be specified for {len(self.layers)} layers, but it is for"
                        f" {head_mask.shape[0]}."
                    )

        for idx, decoder_layer in enumerate(self.layers):
            # add LayerDrop (see https://arxiv.org/abs/1909.11556 for description)
            if output_hidden_states:
                all_hidden_states += (hidden_states,)
            if self.training:
                dropout_probability = ops.rand([])
                if dropout_probability < self.layerdrop:
                    continue

            past_key_value = past_key_values[idx] if past_key_values is not None else None

            if self.gradient_checkpointing and self.training:
                layer_outputs = self._gradient_checkpointing_func(
                    decoder_layer.__call__,
                    hidden_states,
                    attention_mask,
                    encoder_hidden_states,
                    encoder_attention_mask,
                    head_mask[idx] if head_mask is not None else None,
                    cross_attn_head_mask[idx] if cross_attn_head_mask is not None else None,
                    None,
                    output_attentions,
                    use_cache,
                )
            else:
                layer_outputs = decoder_layer(
                    hidden_states,
                    attention_mask=attention_mask,
                    encoder_hidden_states=encoder_hidden_states,
                    encoder_attention_mask=encoder_attention_mask,
                    layer_head_mask=(head_mask[idx] if head_mask is not None else None),
                    cross_attn_layer_head_mask=(
                        cross_attn_head_mask[idx] if cross_attn_head_mask is not None else None
                    ),
                    past_key_value=past_key_value,
                    output_attentions=output_attentions,
                    use_cache=use_cache,
                )
            hidden_states = layer_outputs[0]

            if use_cache:
                next_decoder_cache += (layer_outputs[3 if output_attentions else 1],)

            if output_attentions:
                all_self_attns += (layer_outputs[1],)

                if encoder_hidden_states is not None:
                    all_cross_attentions += (layer_outputs[2],)

        # add hidden states from the last decoder layer
        if output_hidden_states:
            all_hidden_states += (hidden_states,)

        next_cache = next_decoder_cache if use_cache else None
        if not return_dict:
            return tuple(
                v
                for v in [hidden_states, next_cache, all_hidden_states, all_self_attns, all_cross_attentions]
                if v is not None
            )
        return BaseModelOutputWithPastAndCrossAttentions(
            last_hidden_state=hidden_states,
            past_key_values=next_cache,
            hidden_states=all_hidden_states,
            attentions=all_self_attns,
            cross_attentions=all_cross_attentions,
        )


class Florence2LanguageModel(Florence2LanguagePreTrainedModel):
    _tied_weights_keys = ["encoder.embed_tokens.weight", "decoder.embed_tokens.weight"]

    def __init__(self, config: Florence2LanguageConfig):
        super().__init__(config)

        padding_idx, vocab_size = config.pad_token_id, config.vocab_size
        self.shared = nn.Embedding(vocab_size, config.d_model, padding_idx)

        self.encoder = Florence2Encoder(config, self.shared)
        self.decoder = Florence2Decoder(config, self.shared)

        # Initialize weights and apply final processing
        self.post_init()

    def _tie_weights(self):
        if self.config.tie_word_embeddings:
            self._tie_or_clone_weights(self.encoder.embed_tokens, self.shared)
            self._tie_or_clone_weights(self.decoder.embed_tokens, self.shared)

    def get_input_embeddings(self):
        return self.shared

    def set_input_embeddings(self, value):
        self.shared = value
        self.encoder.embed_tokens = self.shared
        self.decoder.embed_tokens = self.shared

    def get_encoder(self):
        return self.encoder

    def get_decoder(self):
        return self.decoder

    def forward(
        self,
        input_ids: Tensor = None,
        attention_mask: Optional[Tensor] = None,
        decoder_input_ids: Optional[Tensor] = None,
        decoder_attention_mask: Optional[Tensor] = None,
        head_mask: Optional[Tensor] = None,
        decoder_head_mask: Optional[Tensor] = None,
        cross_attn_head_mask: Optional[Tensor] = None,
        encoder_outputs: Optional[List[Tensor]] = None,
        past_key_values: Optional[List[Tensor]] = None,
        inputs_embeds: Optional[Tensor] = None,
        decoder_inputs_embeds: Optional[Tensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple, Seq2SeqModelOutput]:
        # different to other models, Florence2 automatically creates decoder_input_ids from
        # input_ids if no decoder_input_ids are provided
        if decoder_input_ids is None and decoder_inputs_embeds is None:
            if input_ids is None:
                raise ValueError(
                    "If no `decoder_input_ids` or `decoder_inputs_embeds` are "
                    "passed, `input_ids` cannot be `None`. Please pass either "
                    "`input_ids` or `decoder_input_ids` or `decoder_inputs_embeds`."
                )

            decoder_input_ids = shift_tokens_right(
                input_ids, self.config.pad_token_id, self.config.decoder_start_token_id
            )

        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        use_cache = use_cache if use_cache is not None else self.config.use_cache
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        if encoder_outputs is None:
            encoder_outputs = self.encoder(
                input_ids=input_ids,
                attention_mask=attention_mask,
                head_mask=head_mask,
                inputs_embeds=inputs_embeds,
                output_attentions=output_attentions,
                output_hidden_states=output_hidden_states,
                return_dict=return_dict,
            )
        # If the user passed a tuple for encoder_outputs, we wrap it in a BaseModelOutput when return_dict=True
        elif return_dict and not isinstance(encoder_outputs, BaseModelOutput):
            encoder_outputs = BaseModelOutput(
                last_hidden_state=encoder_outputs[0],
                hidden_states=encoder_outputs[1] if len(encoder_outputs) > 1 else None,
                attentions=encoder_outputs[2] if len(encoder_outputs) > 2 else None,
            )

        # decoder outputs consists of (dec_features, past_key_value, dec_hidden, dec_attn)
        decoder_outputs = self.decoder(
            input_ids=decoder_input_ids,
            attention_mask=decoder_attention_mask,
            encoder_hidden_states=encoder_outputs[0],
            encoder_attention_mask=attention_mask,
            head_mask=decoder_head_mask,
            cross_attn_head_mask=cross_attn_head_mask,
            past_key_values=past_key_values,
            inputs_embeds=decoder_inputs_embeds,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        if not return_dict:
            return decoder_outputs + encoder_outputs

        return Seq2SeqModelOutput(
            last_hidden_state=decoder_outputs.last_hidden_state,
            past_key_values=decoder_outputs.past_key_values,
            decoder_hidden_states=decoder_outputs.hidden_states,
            decoder_attentions=decoder_outputs.attentions,
            cross_attentions=decoder_outputs.cross_attentions,
            encoder_last_hidden_state=encoder_outputs.last_hidden_state,
            encoder_hidden_states=encoder_outputs.hidden_states,
            encoder_attentions=encoder_outputs.attentions,
        )


class Florence2LanguageForConditionalGeneration(Florence2LanguagePreTrainedModel):
    base_model_prefix = "model"
    _tied_weights_keys = ["encoder.embed_tokens.weight", "decoder.embed_tokens.weight", "lm_head.weight"]
    _keys_to_ignore_on_load_missing = ["final_logits_bias"]

    def __init__(self, config: Florence2LanguageConfig):
        super().__init__(config)
        self.model = Florence2LanguageModel(config)
        self.register_buffer("final_logits_bias", ops.zeros((1, self.model.shared.num_embeddings)))
        self.lm_head = nn.Linear(config.d_model, self.model.shared.num_embeddings, bias=False)

        # Initialize weights and apply final processing
        self.post_init()

    def get_encoder(self):
        return self.model.get_encoder()

    def get_decoder(self):
        return self.model.get_decoder()

    def resize_token_embeddings(self, new_num_tokens: int, pad_to_multiple_of: Optional[int] = None) -> nn.Embedding:
        new_embeddings = super().resize_token_embeddings(new_num_tokens, pad_to_multiple_of)
        self._resize_final_logits_bias(new_embeddings.weight.shape[0])
        return new_embeddings

    def _resize_final_logits_bias(self, new_num_tokens: int) -> None:
        old_num_tokens = self.final_logits_bias.shape[-1]
        if new_num_tokens <= old_num_tokens:
            new_bias = self.final_logits_bias[:, :new_num_tokens]
        else:
            extra_bias = ops.zeros((1, new_num_tokens - old_num_tokens))
            new_bias = ops.cat([self.final_logits_bias, extra_bias], dim=1)
        self.register_buffer("final_logits_bias", new_bias)

    def get_output_embeddings(self):
        return self.lm_head

    def set_output_embeddings(self, new_embeddings):
        self.lm_head = new_embeddings

    def forward(
        self,
        input_ids: Tensor = None,
        attention_mask: Optional[Tensor] = None,
        decoder_input_ids: Optional[Tensor] = None,
        decoder_attention_mask: Optional[Tensor] = None,
        head_mask: Optional[Tensor] = None,
        decoder_head_mask: Optional[Tensor] = None,
        cross_attn_head_mask: Optional[Tensor] = None,
        encoder_outputs: Optional[List[Tensor]] = None,
        past_key_values: Optional[List[Tensor]] = None,
        inputs_embeds: Optional[Tensor] = None,
        decoder_inputs_embeds: Optional[Tensor] = None,
        labels: Optional[Tensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple, Seq2SeqLMOutput]:
        r"""
        labels (`Tensor` of shape `(batch_size, sequence_length)`, *optional*):
            Labels for computing the masked language modeling loss. Indices should either be in `[0, ...,
            config.vocab_size]` or -100 (see `input_ids` docstring). Tokens with indices set to `-100` are ignored
            (masked), the loss is only computed for the tokens with labels in `[0, ..., config.vocab_size]`.

        Returns:
        """
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        if labels is not None:
            if use_cache:
                logger.warning("The `use_cache` argument is changed to `False` since `labels` is provided.")
            use_cache = False
            if decoder_input_ids is None and decoder_inputs_embeds is None:
                decoder_input_ids = shift_tokens_right(
                    labels, self.config.pad_token_id, self.config.decoder_start_token_id
                )

        outputs = self.model(
            input_ids,
            attention_mask=attention_mask,
            decoder_input_ids=decoder_input_ids,
            encoder_outputs=encoder_outputs,
            decoder_attention_mask=decoder_attention_mask,
            head_mask=head_mask,
            decoder_head_mask=decoder_head_mask,
            cross_attn_head_mask=cross_attn_head_mask,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            decoder_inputs_embeds=decoder_inputs_embeds,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        lm_logits = self.lm_head(outputs[0])
        lm_logits = lm_logits + self.final_logits_bias.to(lm_logits.dtype)

        masked_lm_loss = None
        if labels is not None:
            labels = labels.to(lm_logits.dtype)
            loss_fct = CrossEntropyLoss()
            masked_lm_loss = loss_fct(lm_logits.view(-1, self.config.vocab_size), labels.view(-1))

        if not return_dict:
            output = (lm_logits,) + outputs[1:]
            return ((masked_lm_loss,) + output) if masked_lm_loss is not None else output

        return Seq2SeqLMOutput(
            loss=masked_lm_loss,
            logits=lm_logits,
            past_key_values=outputs.past_key_values,
            decoder_hidden_states=outputs.decoder_hidden_states,
            decoder_attentions=outputs.decoder_attentions,
            cross_attentions=outputs.cross_attentions,
            encoder_last_hidden_state=outputs.encoder_last_hidden_state,
            encoder_hidden_states=outputs.encoder_hidden_states,
            encoder_attentions=outputs.encoder_attentions,
        )

    def prepare_inputs_for_generation(
        self,
        decoder_input_ids,
        past_key_values=None,
        attention_mask=None,
        decoder_attention_mask=None,
        head_mask=None,
        decoder_head_mask=None,
        cross_attn_head_mask=None,
        use_cache=None,
        encoder_outputs=None,
        **kwargs,
    ):
        # cut decoder_input_ids if past_key_values is used
        if past_key_values is not None:
            past_length = past_key_values[0][0].shape[2]

            # Some generation methods already pass only the last input ID
            if decoder_input_ids.shape[1] > past_length:
                remove_prefix_length = past_length
            else:
                # Default to old behavior: keep only final ID
                remove_prefix_length = decoder_input_ids.shape[1] - 1

            decoder_input_ids = decoder_input_ids[:, remove_prefix_length:]

        return {
            "input_ids": None,  # encoder_outputs is defined. input_ids not needed
            "encoder_outputs": encoder_outputs,
            "past_key_values": past_key_values,
            "decoder_input_ids": decoder_input_ids,
            "attention_mask": attention_mask,
            "decoder_attention_mask": decoder_attention_mask,
            "head_mask": head_mask,
            "decoder_head_mask": decoder_head_mask,
            "cross_attn_head_mask": cross_attn_head_mask,
            "use_cache": use_cache,  # change this to avoid caching (presumably for debugging)
        }

    def prepare_decoder_input_ids_from_labels(self, labels: Tensor):
        return shift_tokens_right(labels, self.config.pad_token_id, self.config.decoder_start_token_id)

    @staticmethod
    def _reorder_cache(past_key_values, beam_idx):
        reordered_past = ()
        for layer_past in past_key_values:
            # cached cross_attention states don't have to be reordered -> they are always the same
            reordered_past += (
                tuple(past_state.index_select(0, beam_idx) for past_state in layer_past[:2])
                + layer_past[2:],
            )
        return reordered_past

@dataclass
class Florence2Seq2SeqLMOutput(ModelOutput):
    """
    Base class for Florence-2 model's outputs that also contains : pre-computed hidden states that can speed up sequential
    decoding.

    Args:
        loss (`Tensor` of shape `(1,)`, *optional*, returned when `labels` is provided):
            Language modeling loss.
        logits (`Tensor` of shape `(batch_size, sequence_length, config.vocab_size)`):
            Prediction scores of the language modeling head (scores for each vocabulary token before SoftMax).
        last_hidden_state (`Tensor` of shape `(batch_size, sequence_length, hidden_size)`):
            Sequence of hidden-states at the output of the last layer of the decoder of the model.

            If `past_key_values` is used only the last hidden-state of the sequences of shape `(batch_size, 1,
            hidden_size)` is output.
        past_key_values (`tuple(tuple(Tensor))`, *optional*, returned when `use_cache=True` is passed or when `config.use_cache=True`):
            Tuple of `tuple(Tensor)` of length `config.n_layers`, with each tuple having 2 tensors of shape
            `(batch_size, num_heads, sequence_length, embed_size_per_head)`) and 2 additional tensors of shape
            `(batch_size, num_heads, encoder_sequence_length, embed_size_per_head)`.

            Contains pre-computed hidden-states (key and values in the self-attention blocks and in the cross-attention
            blocks) that can be used (see `past_key_values` input) to speed up sequential decoding.
        decoder_hidden_states (`tuple(Tensor)`, *optional*, returned when `output_hidden_states=True` is passed or when `config.output_hidden_states=True`):
            Tuple of `Tensor` (one for the output of the embeddings, if the model has an embedding layer, +
            one for the output of each layer) of shape `(batch_size, sequence_length, hidden_size)`.

            Hidden-states of the decoder at the output of each layer plus the optional initial embedding outputs.
        decoder_attentions (`tuple(Tensor)`, *optional*, returned when `output_attentions=True` is passed or when `config.output_attentions=True`):
            Tuple of `Tensor` (one for each layer) of shape `(batch_size, num_heads, sequence_length,
            sequence_length)`.

            Attentions weights of the decoder, after the attention softmax, used to compute the weighted average in the
            self-attention heads.
        cross_attentions (`tuple(Tensor)`, *optional*, returned when `output_attentions=True` is passed or when `config.output_attentions=True`):
            Tuple of `Tensor` (one for each layer) of shape `(batch_size, num_heads, sequence_length,
            sequence_length)`.

            Attentions weights of the decoder's cross-attention layer, after the attention softmax, used to compute the
            weighted average in the cross-attention heads.
        encoder_last_hidden_state (`Tensor` of shape `(batch_size, sequence_length, hidden_size)`, *optional*):
            Sequence of hidden-states at the output of the last layer of the encoder of the model.
        encoder_hidden_states (`tuple(Tensor)`, *optional*, returned when `output_hidden_states=True` is passed or when `config.output_hidden_states=True`):
            Tuple of `Tensor` (one for the output of the embeddings, if the model has an embedding layer, +
            one for the output of each layer) of shape `(batch_size, sequence_length, hidden_size)`.

            Hidden-states of the encoder at the output of each layer plus the optional initial embedding outputs.
        encoder_attentions (`tuple(Tensor)`, *optional*, returned when `output_attentions=True` is passed or when `config.output_attentions=True`):
            Tuple of `Tensor` (one for each layer) of shape `(batch_size, num_heads, sequence_length,
            sequence_length)`.

            Attentions weights of the encoder, after the attention softmax, used to compute the weighted average in the
            self-attention heads.
        image_hidden_states (`tuple(Tensor)`, *optional*):
            Tuple of `Tensor` (one for the output of the image embeddings, `(batch_size,
            num_image_tokens, hidden_size)`.

            image_hidden_states of the model produced by the vision encoder
    """
    loss: Optional[Tensor] = None
    logits: Tensor = None
    last_hidden_state: Tensor = None
    past_key_values: Optional[Tuple[Tuple[Tensor]]] = None
    decoder_hidden_states: Optional[Tuple[Tensor, ...]] = None
    decoder_attentions: Optional[Tuple[Tensor, ...]] = None
    cross_attentions: Optional[Tuple[Tensor, ...]] = None
    encoder_last_hidden_state: Optional[Tensor] = None
    encoder_hidden_states: Optional[Tuple[Tensor, ...]] = None
    encoder_attentions: Optional[Tuple[Tensor, ...]] = None
    image_hidden_states: Optional[Tuple[Tensor, ...]] = None


FLORENCE2_START_DOCSTRING = r"""
    This model inherits from [`PreTrainedModel`]. Check the superclass documentation for the generic methods the
    library implements for all its model (such as downloading or saving, resizing the input embeddings, pruning heads
    etc.)

    This model is also a PyTorch [torch.nn.Module](https://pytorch.org/docs/stable/nn.html#torch.nn.Module) subclass.
    Use it as a regular PyTorch Module and refer to the PyTorch documentation for all matter related to general usage
    and behavior.

    Parameters:
        config ([`Florence2Config`] or [`Florence2VisionConfig`]):
            Model configuration class with all the parameters of the model. Initializing with a config file does not
            load the weights associated with the model, only the configuration. Check out the
            [`~PreTrainedModel.from_pretrained`] method to load the model weights.
"""


class Florence2PreTrainedModel(PreTrainedModel):
    config_class = Florence2Config
    base_model_prefix = "model"
    supports_gradient_checkpointing = True
    _skip_keys_device_placement = "past_key_values"

    @property
    def _supports_sdpa(self):
        """
        Retrieve language_model's attribute to check whether the model supports
        SDPA or not.
        """
        return self.language_model._supports_sdpa


FLORENCE2_INPUTS_DOCSTRING = r"""
    Args:
        input_ids (`Tensor` of shape `(batch_size, sequence_length)`):
            Indices of input sequence tokens in the vocabulary. Padding will be ignored by default should you provide
            it.

            Indices can be obtained using [`AutoTokenizer`]. See [`PreTrainedTokenizer.encode`] and
            [`PreTrainedTokenizer.__call__`] for details.

            [What are input IDs?](../glossary#input-ids)
        pixel_values (`Tensor` of shape `(batch_size, num_channels, image_size, image_size)):
            The tensors corresponding to the input images. Pixel values can be obtained using
            [`AutoImageProcessor`]. See [`CLIPImageProcessor.__call__`] for details ([]`Florence2Processor`] uses
            [`CLIPImageProcessor`] for processing images).
        attention_mask (`Tensor` of shape `(batch_size, sequence_length)`, *optional*):
            Mask to avoid performing attention on padding token indices. Mask values selected in `[0, 1]`:

            - 1 for tokens that are **not masked**,
            - 0 for tokens that are **masked**.

            [What are attention masks?](../glossary#attention-mask)

            Indices can be obtained using [`AutoTokenizer`]. See [`PreTrainedTokenizer.encode`] and
            [`PreTrainedTokenizer.__call__`] for details.

            If `past_key_values` is used, optionally only the last `decoder_input_ids` have to be input (see
            `past_key_values`).

            If you want to change padding behavior, you should read [`modeling_opt._prepare_decoder_attention_mask`]
            and modify to your needs. See diagram 1 in [the paper](https://arxiv.org/abs/1910.13461) for more
            information on the default strategy.

            - 1 indicates the head is **not masked**,
            - 0 indicates the head is **masked**.
        position_ids (`Tensor` of shape `(batch_size, sequence_length)`, *optional*):
            Indices of positions of each input sequence tokens in the position embeddings. Selected in the range `[0,
            config.n_positions - 1]`. [What are position IDs?](../glossary#position-ids)
        past_key_values (`tuple(tuple(Tensor))`, *optional*, returned when `use_cache=True` is passed or when `config.use_cache=True`):
            Tuple of `tuple(Tensor)` of length `config.n_layers`, with each tuple having 2 tensors of shape
            `(batch_size, num_heads, sequence_length, embed_size_per_head)`) and 2 additional tensors of shape
            `(batch_size, num_heads, encoder_sequence_length, embed_size_per_head)`.

            Contains pre-computed hidden-states (key and values in the self-attention blocks and in the cross-attention
            blocks) that can be used (see `past_key_values` input) to speed up sequential decoding.

            If `past_key_values` are used, the user can optionally input only the last `decoder_input_ids` (those that
            don't have their past key value states given to this model) of shape `(batch_size, 1)` instead of all
            `decoder_input_ids` of shape `(batch_size, sequence_length)`.
        inputs_embeds (`Tensor` of shape `(batch_size, sequence_length, hidden_size)`, *optional*):
            Optionally, instead of passing `input_ids` you can choose to directly pass an embedded representation. This
            is useful if you want more control over how to convert `input_ids` indices into associated vectors than the
            model's internal embedding lookup matrix.
        use_cache (`bool`, *optional*):
            If set to `True`, `past_key_values` key value states are returned and can be used to speed up decoding (see
            `past_key_values`).
        output_attentions (`bool`, *optional*):
            Whether or not to return the attentions tensors of all attention layers. See `attentions` under returned
            tensors for more detail.
        output_hidden_states (`bool`, *optional*):
            Whether or not to return the hidden states of all layers. See `hidden_states` under returned tensors for
            more detail.
        return_dict (`bool`, *optional*):
            Whether or not to return a [`~utils.ModelOutput`] instead of a plain tuple.
"""


class Florence2VisionModel(Florence2PreTrainedModel):
    def __init__(self, config: Florence2VisionConfig):
        super().__init__(config)
        assert config.model_type == 'davit', 'only DaViT is supported for now'
        self.vision_tower = DaViT.from_config(config=config)

        self.post_init()

    def forward(self, pixel_values):
        print("Florence2VisionModel")
        if len(pixel_values.shape) == 4:
            x = self.vision_tower.forward_features_unpool(pixel_values)
        else:
            raise ValueError(f'invalid image shape {pixel_values.shape}')
        return x


class Florence2VisionModelWithProjection(Florence2PreTrainedModel):
    def __init__(self, config: Florence2VisionConfig):
        super().__init__(config)
        assert config.model_type == 'davit', 'only DaViT is supported for now'
        self.vision_tower = DaViT.from_config(config=config)

        self._build_image_projection_layers(config)

        self.post_init()

    def _build_image_projection_layers(self, config):
        image_dim_out = config.dim_embed[-1]
        dim_projection = config.projection_dim
        self.image_projection = nn.Parameter(
            ops.empty(image_dim_out, dim_projection)
        )
        self.image_proj_norm = nn.LayerNorm(dim_projection)
        image_pos_embed_config = config.image_pos_embed
        if image_pos_embed_config['type'] == 'learned_abs_2d':
            self.image_pos_embed = LearnedAbsolutePositionEmbedding2D(
                embedding_dim=image_dim_out,
                num_pos=image_pos_embed_config['max_pos_embeddings']
            )
        else:
            raise NotImplementedError('Not implemented yet')

        self.image_feature_source = config.image_feature_source

        # temporal embedding
        visual_temporal_embedding_config = config.visual_temporal_embedding
        if visual_temporal_embedding_config['type'] == 'COSINE':
            self.visual_temporal_embed = PositionalEmbeddingCosine1D(
                embed_dim=image_dim_out,
                max_seq_len=visual_temporal_embedding_config['max_temporal_embeddings']
            )
        else:
            raise NotImplementedError('Not implemented yet')

    def forward(self, pixel_values):
        print("Florence2VisionModelWithProjection")
        if len(pixel_values.shape) == 4:
            batch_size, C, H, W = pixel_values.shape
            T = 1
            x = self.vision_tower.forward_features_unpool(pixel_values)
        else:
            raise ValueError(f'invalid image shape {pixel_values.shape}')

        if self.image_pos_embed is not None:
            x = x.view(batch_size * T, -1, x.shape[-1])
            num_tokens = x.shape[-2]
            h, w = int(num_tokens ** 0.5), int(num_tokens ** 0.5)
            assert h * w == num_tokens, 'only support square feature maps for now'
            x = x.view(batch_size * T, h, w, x.shape[-1])
            pos_embed = self.image_pos_embed(x)
            x = x + pos_embed
            x = x.view(batch_size, T * h*w, x.shape[-1])

        if self.visual_temporal_embed is not None:
            visual_temporal_embed = self.visual_temporal_embed(x.view(batch_size, T, -1, x.shape[-1])[:, :, 0])
            x = x.view(batch_size, T, -1, x.shape[-1]) + visual_temporal_embed.view(1, T, 1, x.shape[-1])

        x_feat_dict = {}

        spatial_avg_pool_x = x.view(batch_size, T, -1, x.shape[-1]).mean(axis=2)
        x_feat_dict['spatial_avg_pool'] = spatial_avg_pool_x

        temporal_avg_pool_x = x.view(batch_size, T, -1, x.shape[-1]).mean(axis=1)
        x_feat_dict['temporal_avg_pool'] = temporal_avg_pool_x

        x = x.view(batch_size, T, -1, x.shape[-1])[:, -1]
        x_feat_dict['last_frame'] = x

        new_x = []
        for _image_feature_source in self.image_feature_source:
            if _image_feature_source not in x_feat_dict:
                raise ValueError('invalid image feature source: {}'.format(_image_feature_source))
            new_x.append(x_feat_dict[_image_feature_source])

        x = ops.cat(new_x, dim=1)

        x = x @ self.image_projection
        x = self.image_proj_norm(x)

        return x


class Florence2ForConditionalGeneration(Florence2PreTrainedModel):
    def __init__(self, config: Florence2Config):
        super().__init__(config)
        assert config.vision_config.model_type == 'davit', 'only DaViT is supported for now'
        self.vision_tower = DaViT.from_config(config=config.vision_config)
        # remove unused layers
        del self.vision_tower.head
        del self.vision_tower.norms

        self.vocab_size = config.vocab_size
        self._attn_implementation = config._attn_implementation
        self._build_image_projection_layers(config)

        language_model = Florence2LanguageForConditionalGeneration(config=config.text_config)

        if language_model._tied_weights_keys is not None:
            self._tied_weights_keys = [f"language_model.{k}" for k in language_model._tied_weights_keys]
        self.language_model = language_model

        self.pad_token_id = self.config.pad_token_id if self.config.pad_token_id is not None else -1
        self.post_init()

    def _build_image_projection_layers(self, config):
        image_dim_out = config.vision_config.dim_embed[-1]
        dim_projection = config.vision_config.projection_dim
        self.image_projection = nn.Parameter(
            ops.empty(image_dim_out, dim_projection)
        )
        self.image_proj_norm = nn.LayerNorm(dim_projection)
        image_pos_embed_config = config.vision_config.image_pos_embed
        if image_pos_embed_config['type'] == 'learned_abs_2d':
            self.image_pos_embed = LearnedAbsolutePositionEmbedding2D(
                embedding_dim=image_dim_out,
                num_pos=image_pos_embed_config['max_pos_embeddings']
            )
        else:
            raise NotImplementedError('Not implemented yet')

        self.image_feature_source = config.vision_config.image_feature_source

        # temporal embedding
        visual_temporal_embedding_config = config.vision_config.visual_temporal_embedding
        if visual_temporal_embedding_config['type'] == 'COSINE':
            self.visual_temporal_embed = PositionalEmbeddingCosine1D(
                embed_dim=image_dim_out,
                max_seq_len=visual_temporal_embedding_config['max_temporal_embeddings']
            )
        else:
            raise NotImplementedError('Not implemented yet')

    def get_encoder(self):
        return self.language_model.get_encoder()

    def get_decoder(self):
        return self.language_model.get_decoder()

    def get_input_embeddings(self):
        return self.language_model.get_input_embeddings()

    def resize_token_embeddings(self, new_num_tokens: Optional[int] = None, pad_to_multiple_of=None) -> nn.Embedding:
        model_embeds = self.language_model.resize_token_embeddings(new_num_tokens, pad_to_multiple_of)
        # update vocab size
        self.config.text_config.vocab_size = model_embeds.num_embeddings
        self.config.vocab_size = model_embeds.num_embeddings
        self.vocab_size = model_embeds.num_embeddings
        return model_embeds

    def _encode_image(self, pixel_values):
        if len(pixel_values.shape) == 4:
            batch_size, C, H, W = pixel_values.shape
            T = 1
            x = self.vision_tower.forward_features_unpool(pixel_values)
        else:
            raise ValueError(f'invalid image shape {pixel_values.shape}')

        if self.image_pos_embed is not None:
            x = x.view(batch_size * T, -1, x.shape[-1])
            num_tokens = x.shape[-2]
            h, w = int(num_tokens ** 0.5), int(num_tokens ** 0.5)
            assert h * w == num_tokens, 'only support square feature maps for now'
            x = x.view(batch_size * T, h, w, x.shape[-1])
            pos_embed = self.image_pos_embed(x)
            x = x + pos_embed
            x = x.view(batch_size, T * h*w, x.shape[-1])

        if self.visual_temporal_embed is not None:
            visual_temporal_embed = self.visual_temporal_embed(x.view(batch_size, T, -1, x.shape[-1])[:, :, 0])
            x = x.view(batch_size, T, -1, x.shape[-1]) + visual_temporal_embed.view(1, T, 1, x.shape[-1])

        x_feat_dict = {}

        spatial_avg_pool_x = x.view(batch_size, T, -1, x.shape[-1]).mean(axis=2)
        x_feat_dict['spatial_avg_pool'] = spatial_avg_pool_x

        temporal_avg_pool_x = x.view(batch_size, T, -1, x.shape[-1]).mean(axis=1)
        x_feat_dict['temporal_avg_pool'] = temporal_avg_pool_x

        x = x.view(batch_size, T, -1, x.shape[-1])[:, -1]
        x_feat_dict['last_frame'] = x

        new_x = []
        for _image_feature_source in self.image_feature_source:
            if _image_feature_source not in x_feat_dict:
                raise ValueError('invalid image feature source: {}'.format(_image_feature_source))
            new_x.append(x_feat_dict[_image_feature_source])

        x = ops.cat(new_x, dim=1)

        x = x @ self.image_projection
        x = self.image_proj_norm(x)

        return x

    def _merge_input_ids_with_image_features(
        self, image_features, inputs_embeds
    ):
        batch_size, image_token_length = image_features.shape[:-1]
        image_attention_mask = ops.ones(batch_size, image_token_length)

        # task_prefix_embeds: [batch_size, padded_context_length, hidden_size]
        # task_prefix_attention_mask: [batch_size, context_length]
        if inputs_embeds is None:
            return image_features, image_attention_mask

        task_prefix_embeds = inputs_embeds
        task_prefix_attention_mask = ops.ones(batch_size, task_prefix_embeds.shape[1])

        if len(task_prefix_attention_mask.shape) == 3:
            task_prefix_attention_mask = task_prefix_attention_mask[:, 0]

        # concat [image embeds, task prefix embeds]
        inputs_embeds = ops.cat([image_features, task_prefix_embeds], dim=1)
        attention_mask = ops.cat([image_attention_mask, task_prefix_attention_mask], dim=1)

        return inputs_embeds, attention_mask

    def forward(
        self,
        input_ids: Tensor = None,
        pixel_values: Tensor = None,
        attention_mask: Optional[Tensor] = None,
        decoder_input_ids: Optional[Tensor] = None,
        decoder_attention_mask: Optional[Tensor] = None,
        head_mask: Optional[Tensor] = None,
        decoder_head_mask: Optional[Tensor] = None,
        cross_attn_head_mask: Optional[Tensor] = None,
        encoder_outputs: Optional[List[Tensor]] = None,
        past_key_values: Optional[List[Tensor]] = None,
        inputs_embeds: Optional[Tensor] = None,
        decoder_inputs_embeds: Optional[Tensor] = None,
        labels: Optional[Tensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple, Florence2Seq2SeqLMOutput]:
        r"""
        Args:
            labels (`Tensor` of shape `(batch_size, sequence_length)`, *optional*):
                Labels for computing the masked language modeling loss. Indices should either be in `[0, ...,
                config.vocab_size]` or -100 (see `input_ids` docstring). Tokens with indices set to `-100` are ignored
                (masked), the loss is only computed for the tokens with labels in `[0, ..., config.vocab_size]`.

        Returns:

        Example:

        ```python

        "A green car parked in front of a yellow building."
        ```"""
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        image_features = None
        if inputs_embeds is None:
            # 1. Extra the input embeddings
            if input_ids is not None:
                inputs_embeds = self.get_input_embeddings()(input_ids)
            # 2. Merge text and images
            if pixel_values is not None:
                # (batch_size, num_image_tokens, hidden_size)
                image_features = self._encode_image(pixel_values)
                inputs_embeds, attention_mask = self._merge_input_ids_with_image_features(image_features, inputs_embeds)

        if inputs_embeds is not None:
            attention_mask = attention_mask.to(inputs_embeds.dtype)
        outputs = self.language_model(
            attention_mask=attention_mask,
            labels=labels,
            inputs_embeds=inputs_embeds,
            decoder_input_ids=decoder_input_ids,
            encoder_outputs=encoder_outputs,
            decoder_attention_mask=decoder_attention_mask,
            head_mask=head_mask,
            decoder_head_mask=decoder_head_mask,
            cross_attn_head_mask=cross_attn_head_mask,
            past_key_values=past_key_values,
            decoder_inputs_embeds=decoder_inputs_embeds,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        logits = outputs.logits
        logits = logits.float()
        loss = outputs.loss
        if not return_dict:
            output = (logits,) + outputs[1:]
            return (loss,) + output if loss is not None else output

        return Florence2Seq2SeqLMOutput(
            loss=loss,
            logits=logits,
            past_key_values=outputs.past_key_values,
            decoder_hidden_states=outputs.decoder_hidden_states,
            decoder_attentions=outputs.decoder_attentions,
            cross_attentions=outputs.cross_attentions,
            encoder_last_hidden_state=outputs.encoder_last_hidden_state,
            encoder_hidden_states=outputs.encoder_hidden_states,
            encoder_attentions=outputs.encoder_attentions,
            image_hidden_states=image_features
        )

    def generate(self, input_ids, inputs_embeds=None, pixel_values=None, **kwargs):
        if inputs_embeds is None:
            # 1. Extra the input embeddings
            if input_ids is not None:
                inputs_embeds = self.get_input_embeddings()(input_ids)
            # 2. Merge text and images
            if pixel_values is not None:
                image_features = self._encode_image(pixel_values)
                inputs_embeds, attention_mask = self._merge_input_ids_with_image_features(image_features, inputs_embeds)

        return self.language_model.generate(
            input_ids=None,
            inputs_embeds=inputs_embeds,
            **kwargs
        )

    def prepare_inputs_for_generation(
        self,
        decoder_input_ids,
        past_key_values=None,
        attention_mask=None,
        pixel_values=None,
        decoder_attention_mask=None,
        head_mask=None,
        decoder_head_mask=None,
        cross_attn_head_mask=None,
        use_cache=None,
        encoder_outputs=None,
        **kwargs,
    ):
        # cut decoder_input_ids if past_key_values is used
        if past_key_values is not None:
            past_length = past_key_values[0][0].shape[2]

            # Some generation methods already pass only the last input ID
            if decoder_input_ids.shape[1] > past_length:
                remove_prefix_length = past_length
            else:
                # Default to old behavior: keep only final ID
                remove_prefix_length = decoder_input_ids.shape[1] - 1

            decoder_input_ids = decoder_input_ids[:, remove_prefix_length:]

        return {
            "input_ids": None,  # encoder_outputs is defined. input_ids not needed
            "encoder_outputs": encoder_outputs,
            "past_key_values": past_key_values,
            "decoder_input_ids": decoder_input_ids,
            "attention_mask": attention_mask,
            "pixel_values": pixel_values,
            "decoder_attention_mask": decoder_attention_mask,
            "head_mask": head_mask,
            "decoder_head_mask": decoder_head_mask,
            "cross_attn_head_mask": cross_attn_head_mask,
            "use_cache": use_cache,  # change this to avoid caching (presumably for debugging)
        }

    def prepare_decoder_input_ids_from_labels(self, labels: Tensor):
        return self.language_model.shift_tokens_right(labels)

    def _reorder_cache(self, *args, **kwargs):
        return self.language_model._reorder_cache(*args, **kwargs)


__all__ = ['Florence2ForConditionalGeneration',
           'Florence2LanguageForConditionalGeneration',
           'Florence2PreTrainedModel',
           'Florence2VisionModel',
           'Florence2LanguageModel',
           'Florence2VisionModelWithProjection',
           'DaViT']
