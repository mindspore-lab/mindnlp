import math
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import mindspore as ms
import mindspore.ops as ops
from mindspore import Tensor, Parameter

import mindspore.nn as nn
from mindspore.nn import BCEWithLogitsLoss, CrossEntropyLoss, MSELoss

# from transformers import top_k_top_p_filtering
from mindnlp.transformers.modeling_outputs import ModelOutput, SequenceClassifierOutputWithPast
from mindnlp.transformers.modeling_utils import PreTrainedModel
from mindnlp.utils import logging
from mindnlp.transformers.generation import TopKLogitsWarper, TopPLogitsWarper
from mindspore.common.initializer import initializer, XavierUniform, Initializer,Normal


try:
    from apex.normalization import FusedLayerNorm as LayerNorm
except ModuleNotFoundError:
    from mindspore.nn import LayerNorm
from mindspore import dtype as mstype

from retnet.configuration_retnet import RetNetConfig

logger = logging.get_logger(__name__)


def drop_path(input: Tensor, drop_prob: float = 0.0, training: bool = False) -> Tensor:
    """
    Drop paths (Stochastic Depth) per sample (when applied in main path of residual blocks).

    Comment by Ross Wightman: This is the same as the DropConnect impl I created for EfficientNet, etc networks,
    however, the original name is misleading as 'Drop Connect' is a different form of dropout in a separate paper...
    See discussion: https://github.com/tensorflow/tpu/issues/494#issuecomment-532968956 ... I've opted for changing the
    layer and argument names to 'drop path' rather than mix DropConnect as a layer name and use 'survival rate' as the
    argument.
    """
    if drop_prob == 0.0 or not training:
        return input
    keep_prob = 1 - drop_prob
    shape = (input.shape[0],) + (1,) * (input.ndim - 1)  # work with diff dim tensors, not just 2D ConvNets
    random_tensor = keep_prob + ops.rand(shape, dtype=input.dtype)
    random_tensor.floor()  # binarize
    output = input.div(keep_prob) * random_tensor
    return output

def top_k_top_p_filtering(
    logits: Tensor,
    top_k: int = 0,
    top_p: float = 1.0,
    filter_value: float = -float("Inf"),
    min_tokens_to_keep: int = 1,
) -> Tensor:
    """
    Filter a distribution of logits using top-k and/or nucleus (top-p) filtering.

    Args:
        logits: logits distribution shape (batch size, vocabulary size)
        top_k (`int`, *optional*, defaults to 0):
            If > 0, only keep the top k tokens with highest probability (top-k filtering)
        top_p (`float`, *optional*, defaults to 1.0):
            If < 1.0, only keep the top tokens 
            with cumulative probability >= top_p (nucleus filtering). Nucleus
            filtering is described in Holtzman et al. (http://arxiv.org/abs/1904.09751)
        min_tokens_to_keep (`int`, *optional*, defaults to 1):
            Minimumber of tokens we keep per batch example in the output.

    From: https://gist.github.com/thomwolf/1a5a29f6962089e871b94cbd09daf317
    """

    if top_k > 0:
        logits = TopKLogitsWarper(top_k=top_k,
                                  filter_value=filter_value,
                                  min_tokens_to_keep=min_tokens_to_keep)(None, logits)

    if 0 <= top_p <= 1.0:
        logits = TopPLogitsWarper(top_p=top_p,
                                  filter_value=filter_value,
                                  min_tokens_to_keep=min_tokens_to_keep)(None, logits)

    return logits

# helper functions
def split_heads(tensors, bsz, seqlen, num_heads):
    assert isinstance(tensors, (tuple, list))
    reshaped_tensors = []
    for x in tensors:
        # Reshape the tensor
        new_shape = (bsz, seqlen, num_heads, -1)
        x = ops.reshape(x, new_shape)
        # Transpose the tensor
        x = x.swapaxes(1,2)
        reshaped_tensors.append(x)
    return reshaped_tensors

def rotate_every_two(x):
    x1 = x[:, :, :, ::2]
    x2 = x[:, :, :, 1::2]
    x = ops.stack((-x2, x1), -1)
    return x.flatten(start_dim=-2)  # in einsum notation: rearrange(x, '... d j -> ... (d j)')\


def theta_shift(x, sin, cos):
    return (x * cos) + (rotate_every_two(x) * sin)

def get_activation_fn(activation):
    if activation == "relu":
        return nn.ReLU()
    elif activation == "gelu":
        return nn.GELU(False)
    elif activation == "swish":
        return nn.SiLU()
    else:
        raise NotImplementedError

def initializer_decorator(generator: Initializer):
    """
    A decorator function that uses a given generator to initialize a tensor

    Args:
        generator (Initializer): The initializer for generating the tensor

    Returns:
        A function `func(param: Parameter, *args, **kwargs)` that initializes
        a tensor matching the shape and dtype of `param`
    """
    def func(param: Parameter, *args, **kwargs):
        return initializer(generator(*args, **kwargs), param.shape, param.dtype)
    return func

init_xavier_uniform = initializer_decorator(XavierUniform)
init_norm = initializer_decorator(Normal)

class RMSNorm(nn.Cell):
    def __init__(self, dim: int, eps: float = 1e-6, elementwise_affine=True):
        """
        Initialize the RMSNorm normalization layer.

        Args:
            dim (int): The dimension of the input tensor.
            eps (float, optional): A small value added to the denominator for numerical stability. Default is 1e-6.

        Attributes:
            eps (float): A small value added to the denominator for numerical stability.
            weight (nn.Parameter): Learnable scaling parameter.

        """
        super().__init__()
        self.normalized_shape = dim
        self.eps = eps
        self.elementwise_affine = elementwise_affine
        if self.elementwise_affine:
            self.weight = Parameter(Tensor(ms.numpy.ones(dim), ms.float32))
        else:
            self.weight = Parameter(Tensor(ms.numpy.zeros(dim), ms.float32))

    def _norm(self, x):
        """
        Apply the RMSNorm normalization to the input tensor.

        Args:
            x (mindspore.Tensor): The input tensor.

        Returns:
            mindspore.Tensor: The normalized tensor.

        """
        return x * ops.rsqrt(x.pow(2).mean(-1, keep_dims=True) + self.eps)

    def construct(self, x):
        """
        Forward pass through the RMSNorm layer.

        Args:
            x (mindspore.Tensor): The input tensor.

        Returns:
            mindspore.Tensor: The output tensor after applying RMSNorm.

        """
        output = self._norm(x.float()).astype(x.dtype)
        return output * self.weight

class RetNetRelPos(nn.Cell):
    def __init__(self, config: RetNetConfig):
        super().__init__()
        self.config = config
        num_heads = config.decoder_retention_heads

        self.angle = 1.0 / (
            10000 ** ops.linspace(0, 1, config.decoder_embed_dim // num_heads // 2)
        )
        self.angle = ms.numpy.tile(ops.ExpandDims()(self.angle,-1),(1, 2)).flatten()
        # decay (gamma)
        if config.use_lm_decay:
            # NOTE: alternative way described in the paper
            s = ops.log(Tensor(1 / 32, dtype=ms.float32))
            e = ops.log(Tensor(1 / 512,  dtype=ms.float32))
            self.decay = ops.log(1 - ops.exp(ops.linspace(s, e, num_heads)))  # [h,]
        else:
            self.decay = ops.log(
                1 - 2 ** (-5 - ms.numpy.arange(num_heads, dtype=ms.float32))
            )
        # self.register_buffer("angle", angle)
        # self.register_buffer("decay", decay)
        self.recurrent_chunk_size = config.recurrent_chunk_size

    def construct(
        self,
        slen,
        forward_impl="parallel",
        recurrent_chunk_size=None,
        retention_mask=None,
        get_decay_scale=True,
    ):
        if forward_impl == "recurrent":
            sin = ops.sin(self.angle * (slen - 1))
            cos = ops.cos(self.angle * (slen - 1))
            retention_rel_pos = ((sin, cos), self.decay.view(1, -1, 1, 1).exp())
        elif forward_impl == "chunkwise":
            if recurrent_chunk_size is None:
                recurrent_chunk_size = self.recurrent_chunk_size
            index = ms.numpy.arange(slen)
            sin = ops.sin(index[:, None] * self.angle[None, :])
            cos = ops.cos(index[:, None] * self.angle[None, :])

            block_index = ms.numpy.arange(recurrent_chunk_size)
            mask = ops.tril(
                ops.ones(recurrent_chunk_size, recurrent_chunk_size)
            )
            mask = ops.masked_fill(
                block_index[:, None] - block_index[None, :], ~mask.bool(), float("inf")
            )
            mask = ops.exp(mask * self.decay[:, None, None])
            mask = ops.nan_to_num(mask)
            mask = ops.ExpandDims()(mask,0)  # [1, h, t, t]
            # TODO: need to handle retention_mask
            # scaling
            
            value_inner_decay = mask[:, :, -1] / mask[:, :, -1].sum(
                axis=-1, keepdims=True
            )
            value_inner_decay = ops.ExpandDims()(value_inner_decay, -1)
            
            scale = mask.sum(axis=-1, keepdims=True).sqrt()
            inner_mask = mask / scale

            cross_decay = ops.exp(self.decay * recurrent_chunk_size)
            query_inner_decay = ops.exp(self.decay[:, None] * (block_index + 1))
            cross_decay = cross_decay[None, :, None, None]
            query_inner_decay = query_inner_decay[None, :, :, None] / (
                scale / mask[:, :, -1].sum(axis=-1)[:, :, None, None]
            )
            # decay_scale (used for kv cache)
            if get_decay_scale:
                decay_scale = self.compute_decay_scale(slen, retention_mask)
            else:
                decay_scale = None
            retention_rel_pos = (
                (sin, cos),
                (
                    inner_mask,
                    cross_decay,
                    query_inner_decay,
                    value_inner_decay,
                    decay_scale,
                ),
            )
        else:  # parallel
            index = ms.numpy.arange(slen)
            sin = ops.sin(index[:, None] * self.angle[None, :])
            cos = ops.cos(index[:, None] * self.angle[None, :])
            mask = ops.tril(ops.ones(slen, slen))
            mask = ops.masked_fill(
                index[:, None] - index[None, :], ~mask.bool(), float("inf")
            )
            mask = ops.exp(mask * self.decay[:, None, None])
            mask = ops.nan_to_num(mask)
            mask = ops.ExpandDims()(mask, 0)  # [1, h, t, t]
            if retention_mask is not None:
                # this is required for left padding
                mask = mask * retention_mask.float().view(-1, 1, 1, slen)

            # scaling
            
            mask = mask / mask.sum(axis=-1, keepdims=True).sqrt()
            mask = ops.nan_to_num(mask, nan=0.0)
            # decay_scale (used for kv cache)
            if get_decay_scale:
                decay_scale = self.compute_decay_scale(slen, retention_mask)
            else:
                decay_scale = None
            # mask processing for intra decay
            if retention_mask is not None:
                max_non_zero = (
                    ops.nonzero(ops.CumSum()(retention_mask, -1).max(-1))
                )  # [b,]
                
                intra_decay = mask[range(mask.shape[0]), :, max_non_zero]
            else:
                intra_decay = mask[:, :, -1]

            retention_rel_pos = ((sin, cos), (mask, intra_decay, decay_scale))

        return retention_rel_pos

    def compute_decay_scale(self, slen, retention_mask=None):
        exponent = ms.numpy.arange(slen).float()
        decay_scale = self.decay.exp().view(-1, 1) ** exponent.view(1, -1)  # [h, t]
        if retention_mask is not None:
            
            seqlen = retention_mask.sum(axis=-1)  # [b,]
            bsz = seqlen.shape[0]
            decay_scale = ms.numpy.tile(ops.ExpandDims()(decay_scale, 0), (bsz, 1, 1))  # [b, h, t]
            for i, pos in enumerate(seqlen):
                # the formula for decay_scale is `sum(gamma^i) for i in [0, slen).`
                # Since the retention_mask is 0 for padding, we can set the decay_scale
                # to 0 for the padding positions.
                decay_scale[i, :, pos.item() :] = 0
        else:
            bsz = 1
        
        decay_scale = decay_scale.sum(axis = -1).view(bsz, -1, 1, 1)  # [b, h, 1, 1]
        return decay_scale


class MultiScaleRetention(nn.Cell):
    def __init__(
        self,
        config: RetNetConfig,
        gate_fn="swish",
        use_bias=False,
        tensor_parallel=False,
    ):
        super().__init__()
        self.config = config
        self.embed_dim = config.decoder_embed_dim
        self.value_dim = config.decoder_value_embed_dim
        self.num_heads = config.decoder_retention_heads
        self.head_dim = self.value_dim // self.num_heads
        self.key_dim = self.embed_dim // self.num_heads
        self.scaling = self.key_dim**-0.5

        self.gate_fn = get_activation_fn(activation=str(gate_fn))

        self.q_proj = nn.Dense(self.embed_dim, self.embed_dim, has_bias=use_bias)
        self.k_proj = nn.Dense(self.embed_dim, self.embed_dim, has_bias=use_bias)
        self.v_proj = nn.Dense(self.embed_dim, self.value_dim, has_bias=use_bias)
        self.g_proj = nn.Dense(self.embed_dim, self.value_dim, has_bias=use_bias)

        self.out_proj = nn.Dense(self.value_dim, self.embed_dim, has_bias=use_bias)

        self.group_norm = RMSNorm(
            self.head_dim, eps=config.layernorm_eps, elementwise_affine=False
        )
        self.reset_parameters()

        if tensor_parallel:
            self.decay_proj = nn.Dense(self.num_heads, self.num_heads, has_bias=False)
        else:
            self.decay_proj = None

    def reset_parameters(self):
        gain = 2**-2.5
        
        self.k_proj.weight.set_data(init_xavier_uniform(self.k_proj.weight, gain))
        self.v_proj.weight.set_data(init_xavier_uniform(self.v_proj.weight, gain))
        self.q_proj.weight.set_data(init_xavier_uniform(self.q_proj.weight, gain))
        self.g_proj.weight.set_data(init_xavier_uniform(self.g_proj.weight, gain))
        self.out_proj.weight.set_data(init_xavier_uniform(self.out_proj.weight, gain))


    def parallel_retention(self, q, k, v, decay_mask):
        """
        q,  # bsz * num_head * len * qk_dim
        k,  # bsz * num_head * len * qk_dim
        v,  # bsz * num_head * len * v_dim
        decay_mask,  # (1 or bsz) * num_head * len * len
        """
        decay_mask, intra_decay, scale = decay_mask
        
        # just return retention_rel_pos projected
        # TODO: for shardformer
        if self.decay_proj is not None:
            decay_mask = self.decay_proj(decay_mask.swapaxes(-1,-3)).swapaxes(-3,-1)

        # [b, h, t, t]
        retention = q @ k.swapaxes(-1,-2)  # (scaled dot-product)
        retention = retention * decay_mask

        # invariant after normalization
        no_grad_retention = ops.stop_gradient(retention)
        
        retention = retention / no_grad_retention.abs().sum(
            axis=-1, keepdims=True
        )
        
        retention = ops.clip_by_value(retention, 1, 5e4)

        output = retention @ v.astype(ms.float32)  # [b, h, t, v_dim / h]
        # print(output.shape)
        output = output.swapaxes(1,2)  # [b, t, h, v_dim / h]
        # print(output.shape)
        if self.training:  # skip cache
            return output, None, retention

        if self.decay_proj is not None:
            intra_decay = self.decay_proj(intra_decay.swapaxes(-1,-2)).swapaxes(-2,-1)

        # kv cache: [b, h, t, v_dim, qk_dim]
        current_kv = ops.ExpandDims()(k, -2) * ops.ExpandDims()(v, -1)
        
        intra_decay = intra_decay[:, :, :, None, None]  # [b, h, t, 1, 1]
        
        
        current_kv = (current_kv.astype(ms.float32) * intra_decay.astype(ms.float32)).sum(axis = 2)  # [b, h, v_dim, qk_dim]

        cache = {"prev_key_value": current_kv, "scale": scale}
        return output, cache, retention

    def recurrent_retention(
        self, q, k, v, decay, past_key_value=None, retention_mask=None
    ):
        """
        q, k, v, # bsz * num_head * 1 * qkv_dim
        past_key_value:
            - "prev_key_value"  # bsz * num_head * v_dim * qk_dim
            - "scale"  # (1 or bsz) * num_head * 1 * 1
        decay # (1 or bsz) * num_head * 1 * 1
        retention_mask # bsz * 1
        """
        if retention_mask is not None:
            retention_mask = retention_mask.float().view(-1, 1, 1, 1)
        else:
            retention_mask = ops.ones(k.shape[0], 1, 1, 1)
        # (b, h, v_dim, qk_dim)
        current_kv = k * v.swapaxes(-1,-2) * retention_mask

        if past_key_value is not None and "prev_key_value" in past_key_value:
            prev_kv = past_key_value["prev_key_value"]
            prev_scale = past_key_value["scale"]
            scale = ops.where(retention_mask == 0, prev_scale, prev_scale * decay + 1)
            # connect prev_kv and current_kv
            # how much to decay prev_kv
            decay_amount = prev_scale.sqrt() * decay / scale.sqrt()
            decay_amount = ops.where(retention_mask == 0, 1, decay_amount)
            prev_kv = prev_kv * decay_amount  # decay prev_kv
            current_kv = current_kv / scale.sqrt()  # scale current_kv
            current_kv = ops.nan_to_num(
                current_kv, nan=0.0
            )  # remove nan, scale might be 0

            current_kv = prev_kv + current_kv
        else:
            scale = ops.ones_like(decay)
            # when retention_mask is 0 at the beginning, setting scale to 1 will
            # make the first retention to use the padding incorrectly. Hence,
            # setting it to 0 here. This is a little ugly, so we might want to
            # change this later. TODO: improve
            scale = ops.where(retention_mask == 0, ops.zeros_like(decay), scale)

        
        output = ops.ExpandDims()(ops.ReduceSum()(q * current_kv, axis=3), 1)  # (b, 1, h, d_v)

        cache = {"prev_key_value": current_kv, "scale": scale}
        return output, cache

    def chunkwise_retention(self, q, k, v, decay_mask):
        """
        q, k, v,  # bsz * num_head * seqlen * qkv_dim
        past_key_value:
            - "prev_key_value"  # bsz * num_head * v_dim * qk_dim
            - "scale"  # (1 or bsz) * num_head * 1 * 1
        decay_mask,  # 1 * num_head * chunk_size * chunk_size
        cross_decay,  # 1 * num_head * 1 * 1
        inner_decay,  # 1 * num_head * chunk_size * 1
        """
        # TODO: not working properly
        (
            decay_mask,
            cross_decay,
            query_inner_decay,
            value_inner_decay,
            decay_scale,
        ) = decay_mask
        bsz, _, tgt_len, _ = v.shape[0],v.shape[1],v.shape[2],v.shape[3]
        chunk_len = decay_mask.shape[-1]
        assert tgt_len % chunk_len == 0
        num_chunks = tgt_len // chunk_len

        # [b, n_c, h, t_c, qkv_dim]
        q = q.view(bsz, self.num_heads, num_chunks, chunk_len, self.key_dim).swapaxes(
            1, 2
        )
        k = k.view(bsz, self.num_heads, num_chunks, chunk_len, self.key_dim).swapaxes(
            1, 2
        )
        v = v.view(bsz, self.num_heads, num_chunks, chunk_len, self.head_dim).swapaxes(
            1, 2
        )

        k_t = k.swapaxes(-1, -2)

        qk_mat = q @ k_t  # [b, n_c, h, t_c, t_c]
        qk_mat = qk_mat * ops.ExpandDims()(decay_mask, 1)
        
        inner_scale = ops.clip_by_value(ops.stop_gradient(qk_mat).abs().sum(axis=-1, keepdims=True), 1, float('inf'))
        qk_mat = qk_mat / inner_scale
        # [b, n_c, h, t_c, v_dim]
        inner_output = ops.matmul(qk_mat, v)

        # reduce kv in one chunk
        # [b, n_c, h, qk_dim, v_dim]
        kv = k_t @ (v * value_inner_decay)
        # kv = kv.view(bsz, num_chunks, self.num_heads, self.key_dim, self.head_dim)

        kv_recurrent = []
        cross_scale = []
        kv_state = ops.zeros(bsz, self.num_heads, self.key_dim, self.head_dim)
        kv_scale = ops.ones(bsz, self.num_heads, 1, 1)

        # accumulate kv by loop
        for i in range(num_chunks):
            kv_recurrent.append(kv_state / kv_scale)
            cross_scale.append(kv_scale)
            kv_state = kv_state * cross_decay + kv[:, i]
            
            kv_scale = (  
                ops.clip_by_value(ops.stop_gradient(kv_state)
                .abs()
                .sum(axis=-2, keepdims=True)
                .max(axis=-1, keepdims=True)
                , 1, float('inf'))
            )

        kv_recurrent = ops.stack(kv_recurrent, 1)
        cross_scale = ops.stack(cross_scale, 1)

        all_scale = ops.maximum(inner_scale, cross_scale)
        align_inner_scale = all_scale / inner_scale
        align_cross_scale = all_scale / cross_scale

        cross_output = (ops.ExpandDims()(q * query_inner_decay, 1)) @ kv_recurrent
        output = inner_output / align_inner_scale + cross_output / align_cross_scale
        output = output.swapaxes(2, 3)  # [b, n_c, t_c, h, v_dim]

        cache = {"prev_key_value": kv_state.swapaxes(-2, -1), "scale": decay_scale}
        return output, cache

    def construct(
        self,
        hidden_states: Tensor,
        rel_pos: Tuple[Tuple[Tensor]],
        retention_mask: Optional[Tensor] = None,
        past_key_value: Optional[Tuple[Tensor]] = None,
        forward_impl: str = "parallel",
        output_retentions: Optional[bool] = False,
    ) -> Tuple[Tensor, Tensor, Optional[Tensor]]:
        B, T, H = hidden_states.shape[0], hidden_states.shape[1], hidden_states.shape[2]
        (sin, cos), decay_mask = rel_pos
        # projections
        q = self.q_proj(hidden_states)
        k = self.k_proj(hidden_states)
        v = self.v_proj(hidden_states)
        g = self.g_proj(hidden_states)
        # multi-head
        q, k, v = split_heads((q, k, v), B, T, self.num_heads)
        k *= self.scaling  # for scaled dot product
        # rotate
        # NOTE: theta_shift has bug with mps device.
        qr = theta_shift(q, sin, cos)
        kr = theta_shift(k, sin, cos)

        # retention
        retention_weights = None
        if forward_impl == "parallel":
            retention_out, curr_kv, retention_weights = self.parallel_retention(
                qr, kr, v, decay_mask
            )
        elif forward_impl == "recurrent":
            retention_out, curr_kv = self.recurrent_retention(
                qr,
                kr,
                v,
                decay_mask,
                past_key_value=past_key_value,
                retention_mask=retention_mask,
            )
        elif forward_impl == "chunkwise":
            retention_out, curr_kv = self.chunkwise_retention(qr, kr, v, decay_mask)
        else:
            raise ValueError(f"forward_impl {forward_impl} not supported.")

        # concaat heads
        normed = self.group_norm(retention_out).reshape(B, T, self.value_dim)
        # out gate & proj
        out = self.gate_fn(g) * normed
        out = self.out_proj(out)

        outputs = (out, curr_kv)
        if output_retentions:
            print(outputs.shape, retention_weights.shape)
            outputs += retention_weights if forward_impl == "parallel" else (None,)
        return outputs


class FeedForwardNetwork(nn.Cell):
    def __init__(
        self,
        embed_dim,
        ffn_dim,
        activation_fn,
        dropout,
        activation_dropout,
        layernorm_eps,
        subln=False,
        use_rms_norm=False,
    ):
        super().__init__()
        self.embed_dim = embed_dim
        self.activation_fn = get_activation_fn(activation=str(activation_fn))
        self.activation_dropout_module = nn.Dropout(activation_dropout)
        self.dropout_module = nn.Dropout(dropout)
        self.ffn_dim = ffn_dim
        self.fc1 = nn.Dense(self.embed_dim, self.ffn_dim)
        self.fc2 = nn.Dense(self.ffn_dim, self.embed_dim)
        self.use_rms_norm = use_rms_norm
        self.layernorm_eps = layernorm_eps
        if subln:
            if self.use_rms_norm:
                self.ffn_layernorm = RMSNorm(self.embed_dim, eps=self.layernorm_eps)
            else:
                self.ffn_layernorm = LayerNorm(self.embed_dim, eps=self.layernorm_eps)
        else:
            self.ffn_layernorm = None

    def reset_parameters(self):
        self.fc1 = nn.Dense(self.embed_dim, self.ffn_dim)
        self.fc2 = nn.Dense(self.ffn_dim, self.embed_dim)
        if self.ffn_layernorm is not None:
            if self.use_rms_norm:
                self.ffn_layernorm = RMSNorm(self.embed_dim, eps=self.layernorm_eps)
            else:
                #这里的layernorm的输入有问题，应该输入tuple的，但是源代码也错了
                self.ffn_layernorm = LayerNorm(self.embed_dim, eps=self.layernorm_eps)
        

    def construct(self, x):
        x_shape = x.shape
        x = x.reshape(-1, x.shape[-1])
        x = self.fc1(x)
        x = self.activation_fn(x.float()).type_as(x)
        x = self.activation_dropout_module(x)
        if self.ffn_layernorm is not None:
            x = self.ffn_layernorm(x)
        x = self.fc2(x)
        x = x.view(x_shape)
        x = self.dropout_module(x)
        return x


class GLU(nn.Cell):
    def __init__(
        self,
        embed_dim,
        ffn_dim,
        activation_fn,
        dropout,
        activation_dropout,
    ):
        super().__init__()
        self.embed_dim = embed_dim
        self.activation_fn = get_activation_fn(activation=str(activation_fn))
        self.activation_dropout_module = nn.Dropout(activation_dropout)
        self.dropout_module = nn.Dropout(dropout)
        self.ffn_dim = ffn_dim
        self.fc1 = nn.Dense(self.embed_dim, self.ffn_dim, has_bias=False)
        self.fc2 = nn.Dense(self.ffn_dim, self.embed_dim, has_bias=False)
        self.gate = nn.Dense(self.embed_dim, self.ffn_dim, has_bias=False)

    def reset_parameters(self):
        self.fc1 = nn.Dense(self.embed_dim, self.ffn_dim, has_bias=False)
        self.fc2 = nn.Dense(self.ffn_dim, self.embed_dim, has_bias=False)
        self.gate = nn.Dense(self.embed_dim, self.ffn_dim, has_bias=False)

    def construct(self, x):
        x_shape = x.shape
        x = x.reshape(-1, x.shape[-1])
        g = self.gate(x)
        x = self.fc1(x)
        x = self.activation_fn(x.float()).type_as(x) * g
        x = self.activation_dropout_module(x)
        x = self.fc2(x)
        x = x.view(x_shape)
        x = self.dropout_module(x)
        return x


class DropPath(nn.Cell):
    """Drop paths (Stochastic Depth) per sample  (when applied in main path of residual blocks)."""

    def __init__(self, drop_prob=None):
        super(DropPath, self).__init__()
        self.drop_prob = drop_prob

    def construct(self, x):
        return drop_path(x, self.drop_prob, self.training)

    def extra_repr(self):
        return "p={}".format(self.drop_prob)


class RetNetDecoderLayer(nn.Cell):
    def __init__(self, config: RetNetConfig, depth: int, tensor_parallel: bool = False):
        super().__init__()
        self.config = config
        self.embed_dim = config.decoder_embed_dim
        self.dropout_module = nn.Dropout(config.dropout)

        if config.drop_path_rate > 0:
            drop_path_prob = np.linspace(
                0, config.drop_path_rate, config.decoder_layers
            )[depth]
            self.drop_path = DropPath(drop_path_prob)
        else:
            self.drop_path = None

        self.retention = MultiScaleRetention(
            config, use_bias=False, tensor_parallel=tensor_parallel
        )

        self.normalize_before = config.decoder_normalize_before

        self.retention_layer_norm = RMSNorm(self.embed_dim, eps=config.layernorm_eps)

        self.ffn_dim = config.decoder_ffn_embed_dim

        self.ffn = self.build_ffn()

        self.final_layer_norm = RMSNorm(self.embed_dim, eps=config.layernorm_eps)

        if config.deepnorm:
            self.alpha = math.pow(2.0 * config.decoder_layers, 0.25)
        else:
            self.alpha = 1.0

    def build_ffn(self):
        if self.config.use_glu:
            return GLU(
                self.embed_dim,
                self.ffn_dim,
                self.config.activation_fn,
                self.config.dropout,
                self.config.activation_dropout,
            )
        else:
            return FeedForwardNetwork(
                self.embed_dim,
                self.ffn_dim,
                self.config.activation_fn,
                self.config.dropout,
                self.config.activation_dropout,
                self.config.layernorm_eps,
                self.config.subln,
                self.config.use_ffn_rms_norm,
            )

    def residual_connection(self, x, residual):
        return residual * self.alpha + x

    def construct(
        self,
        hidden_states: Tensor,
        retention_rel_pos: Tuple[Tuple[Tensor]],
        retention_mask: Optional[Tensor] = None,
        forward_impl: str = "parallel",
        past_key_value: Optional[Tuple[Tensor]] = None,
        output_retentions: Optional[bool] = False,
    ) -> Tuple[Tensor, Tensor, Optional[Tensor]]:
        residual = hidden_states
        if self.normalize_before:
            hidden_states = self.retention_layer_norm(hidden_states)

        msr_outs = self.retention(
            hidden_states,
            retention_rel_pos,
            retention_mask=retention_mask,
            past_key_value=past_key_value,
            forward_impl=forward_impl,
            output_retentions=output_retentions,
        )
        hidden_states = msr_outs[0]
        curr_kv = msr_outs[1]

        hidden_states = self.dropout_module(hidden_states)

        if self.drop_path is not None:
            hidden_states = self.drop_path(hidden_states)

        hidden_states = self.residual_connection(hidden_states, residual)
        if not self.normalize_before:
            hidden_states = self.retention_layer_norm(hidden_states)

        residual = hidden_states
        if self.normalize_before:
            hidden_states = self.final_layer_norm(hidden_states)

        hidden_states = self.ffn(hidden_states)

        if self.drop_path is not None:
            hidden_states = self.drop_path(hidden_states)

        hidden_states = self.residual_connection(hidden_states, residual)
        if not self.normalize_before:
            hidden_states = self.final_layer_norm(hidden_states)

        outputs = (hidden_states, curr_kv)

        if output_retentions:
            outputs += msr_outs[2]
        return outputs


class RetNetPreTrainedModel(PreTrainedModel):
    # copied from LlamaPretrainedModel
    config_class = RetNetConfig
    base_model_prefix = "model"
    supports_gradient_checkpointing = True
    _no_split_modules = ["RetNetDecoderLayer"]
    _keys_to_ignore_on_load_unexpected = [r"decoder\.version"]

    def _init_weights(self, module):
        """
        Following original retnet, weights are already initialized in their own
        ways within their own init.
        """
        pass
        # below is copied from LlamaPretrainedModel
        # std = self.config.initializer_range
        # if isinstance(module, nn.Linear):
        #     module.weight.data.normal_(mean=0.0, std=std)
        #     if module.bias is not None:
        #         module.bias.data.zero_()
        # elif isinstance(module, nn.Embedding):
        #     module.weight.data.normal_(mean=0.0, std=std)
        #     if module.padding_idx is not None:
        #         module.weight.data[module.padding_idx].zero_()


@dataclass
class RetNetOutputWithPast(ModelOutput):
    """
    class for RetNet model's outputs that may also contain a past key/values (to speed up sequential decoding).

    config:
        last_hidden_state (`torch.FloatTensor` of shape `(batch_size, sequence_length, decoder_embed_dim)`):
            Sequence of hidden-states at the output of the last layer of the model.

            If `past_key_values` is used only the last hidden-state of the sequences of shape `(batch_size, 1,
            decoder_embed_dim)` is output.
        past_key_values (`List(Dict(str, torch.FloatTensor))`, *optional*, returned when `use_cache=True` is passed or when `config.use_cache=True`):
            - "prev_key_value": shape=(bsz * num_head * v_dim * qk_dim)
            - "scale": shape=((1 or bsz) * num_head * 1 * 1)

            Contains pre-computed hidden-states (key and values in the multi-scale retention blocks)
            that can be used (see `past_key_values` input) to speed up sequential decoding.
        hidden_states (`tuple(torch.FloatTensor)`, *optional*, returned when `output_hidden_states=True` is passed or when `config.output_hidden_states=True`):
            Tuple of `torch.FloatTensor` (one for the output of the embeddings, if the model has an embedding layer, +
            one for the output of each layer) of shape `(batch_size, sequence_length, decoder_embed_dim)`.

            Hidden-states of the model at the output of each layer plus the optional initial embedding outputs.
        retentions (`tuple(torch.FloatTensor)`, *optional*, returned when `output_retentions=True` is passed or when `config.output_retentions=True`):
            Tuple of `torch.FloatTensor` (one for each layer) of shape `(batch_size, num_heads, sequence_length,
            sequence_length)`.

            Retentions weights, used for visualization.

        attentions (`tuple(torch.FloatTensor)`, *optional*, for backward compatibility. Same as retentions.
    """

    last_hidden_state: Tensor = None
    past_key_values: Optional[List[Dict[str, Tensor]]] = None
    hidden_states: Optional[Tuple[Tensor]] = None
    retentions: Optional[Tuple[Tensor]] = None
    attentions: Optional[Tuple[Tensor]] = None

#############################################################################################################################################

class RetNetModel(RetNetPreTrainedModel):
    def __init__(
        self,
        config: RetNetConfig,
        embed_tokens: nn.Embedding = None,
        tensor_parallel: bool = False,
    ):
        super().__init__(config)
        self.config = config

        self.dropout_module = nn.Dropout(config.dropout)

        self.embed_dim = config.decoder_embed_dim
        self.embed_scale = (
            1.0 if config.no_scale_embedding else math.sqrt(self.embed_dim)
        )

        if embed_tokens is None:
            embed_tokens = nn.Embedding(
                config.vocab_size, config.decoder_embed_dim, config.pad_token_id
            )
        self.embed_tokens = embed_tokens

        if config.layernorm_embedding:
            self.layernorm_embedding = RMSNorm(self.embed_dim, eps=config.layernorm_eps)
        else:
            self.layernorm_embedding = None

        self.layers = nn.CellList([])

        for i in range(config.decoder_layers):
            self.layers.append(
                RetNetDecoderLayer(config, depth=i, tensor_parallel=tensor_parallel)
            )

        self.decoder_layers = len(self.layers)

        if config.decoder_normalize_before:
            self.layer_norm = RMSNorm(self.embed_dim, eps=config.layernorm_eps)
        else:
            self.layer_norm = None

        self.retnet_rel_pos = RetNetRelPos(config)
        self.recurrent_chunk_size = config.recurrent_chunk_size

        if config.deepnorm:
            init_scale = math.pow(8.0 * config.decoder_layers, 0.25)
            for name, p in self.named_parameters():
                if (
                    "fc1" in name
                    or "fc2" in name
                    or "out_proj" in name
                    or "v_proj" in name
                ):
                    p.data.div_(init_scale)

        if config.subln and not config.use_glu:
            init_scale = math.sqrt(math.log(config.decoder_layers * 2))
            for name, p in self.named_parameters():
                if (
                    "fc1" in name
                    or "fc2" in name
                    or "out_proj" in name
                    or "v_proj" in name
                ):
                    p.data.mul_(init_scale)

        self.gradient_checkpointing = False
        self.post_init()

    def get_input_embeddings(self):
        return self.embed_tokens

    def set_input_embeddings(self, value):
        self.embed_tokens = value

    def forward_embedding(
        self,
        input_ids,
        forward_impl,
        inputs_embeds=None,
        past_key_values=None,
    ):
        # if past_key_values is not None:
        if forward_impl == "recurrent":
            input_ids = input_ids[:, -1:]

        if inputs_embeds is None:
            inputs_embeds = self.embed_tokens(input_ids)

        embed = self.embed_scale * inputs_embeds

        if self.layernorm_embedding is not None:
            embed = self.layernorm_embedding(embed)

        embed = self.dropout_module(embed)

        return embed

    def construct(
        self,
        input_ids: mstype.int64 = None,
        retention_mask: Optional[Tensor] = None,
        attention_mask: Optional[Tensor] = None,
        past_key_values: Optional[List[Dict[str, Tensor]]] = None,
        inputs_embeds: Optional[Tensor] = None,
        output_retentions: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        use_cache: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        forward_impl: Optional[str] = "parallel",
        recurrent_chunk_size: Optional[int] = None,
        retention_rel_pos: Optional[Tuple[Tensor]] = None,
    ) -> Union[Tuple, RetNetOutputWithPast]:
        if output_retentions is None and output_attentions is not None:
            output_retentions = output_attentions
        output_retentions = (
            output_retentions
            if output_retentions is not None
            else self.config.output_retentions
        )
        output_hidden_states = (
            output_hidden_states
            if output_hidden_states is not None
            else self.config.output_hidden_states
        )
        use_cache = use_cache if use_cache is not None else self.config.use_cache

        return_dict = (
            return_dict if return_dict is not None else self.config.use_return_dict
        )

        # retrieve input_ids and inputs_embeds
        if input_ids is not None and inputs_embeds is not None:
            raise ValueError(
                "You cannot specify both input_ids and inputs_embeds at the same time"
            )
        elif input_ids is not None:
            batch_size, seq_length = input_ids.shape[0], input_ids.shape[1]
        elif inputs_embeds is not None:
            batch_size, seq_length, _ = inputs_embeds.shape[0], inputs_embeds.shape[1]
        else:
            raise ValueError("You have to specify either input_ids or inputs_embeds")

        # embed tokens
        if inputs_embeds is None:
            inputs_embeds = self.forward_embedding(
                input_ids, forward_impl, inputs_embeds, past_key_values
            )

        if retention_mask is None and attention_mask is not None:
            retention_mask = attention_mask
        if retention_mask is not None and forward_impl == "recurrent":
            retention_mask = retention_mask[:, -1:]

        hidden_states = inputs_embeds

        # handling chunking here
        if recurrent_chunk_size is None:
            recurrent_chunk_size = self.recurrent_chunk_size
        need_pad_for_chunkwise = (
            forward_impl == "chunkwise" and seq_length % recurrent_chunk_size != 0
        )
        if need_pad_for_chunkwise:
            padding_len = recurrent_chunk_size - seq_length % recurrent_chunk_size
            slen = seq_length + padding_len
            hidden_states = ops.pad(hidden_states, (0, 0, 0, padding_len))
        else:
            slen = seq_length
        # relative position
        if retention_rel_pos is None:
            retention_rel_pos = self.retnet_rel_pos(
                slen,
                forward_impl=forward_impl,
                recurrent_chunk_size=recurrent_chunk_size,
                retention_mask=retention_mask,
                get_decay_scale=not self.training,
            )

        # start running through the decoder layers
        all_hidden_states = () if output_hidden_states else None
        all_retentions = () if output_retentions else None
        # layers * [bsz, num_head, qk_dim, decoder_embed_dim]
        next_decoder_cache = () if use_cache else None

        for idx, layer in enumerate(self.layers):
            if output_hidden_states:
                all_hidden_states += (hidden_states,)
            past_key_value = (
                past_key_values[idx] if past_key_values is not None else None
            )

            # if self.gradient_checkpointing and self.training:

            #     def create_custom_forward(module):
            #         def custom_forward(*inputs):
            #             return module(*inputs, output_retentions)

            #         return custom_forward

            #     layer_outputs = torch.utils.checkpoint.checkpoint(
            #         create_custom_forward(layer),
            #         hidden_states,
            #         retention_rel_pos,
            #         retention_mask,
            #         forward_impl,
            #         past_key_value,
            #     )
            # else:
            #     layer_outputs = layer(
            #         hidden_states,
            #         retention_rel_pos,
            #         retention_mask=retention_mask,
            #         forward_impl=forward_impl,
            #         past_key_value=past_key_value,
            #         output_retentions=output_retentions,
            #     )
            
            layer_outputs = layer(
                    hidden_states,
                    retention_rel_pos,
                    retention_mask=retention_mask,
                    forward_impl=forward_impl,
                    past_key_value=past_key_value,
                    output_retentions=output_retentions,
                )

            hidden_states = layer_outputs[0]

            if use_cache:
                next_decoder_cache += (layer_outputs[1],)

            if output_retentions:
                all_retentions += (layer_outputs[2],)

        next_cache = next_decoder_cache if use_cache else None

        if need_pad_for_chunkwise:
            hidden_states = hidden_states[:, :seq_length, :]

        if self.layer_norm is not None:
            hidden_states = self.layer_norm(hidden_states)

        # add hidden states from the last decoder layer
        if output_hidden_states:
            all_hidden_states += (hidden_states,)

        if not return_dict:
            return tuple(
                v
                for v in [hidden_states, next_cache, all_hidden_states, all_retentions]
                if v is not None
            )
        return RetNetOutputWithPast(
            last_hidden_state=hidden_states,
            past_key_values=next_cache,
            hidden_states=all_hidden_states,
            retentions=all_retentions,
            attentions=all_retentions,
        )


@dataclass
class RetNetCausalLMOutputWithPast(ModelOutput):
    """
    class for RetNet causal language model (or autoregressive) outputs.

    config:
        loss (`torch.FloatTensor` of shape `(1,)`, *optional*, returned when `labels` is provided):
            Language modeling loss (for next-token prediction).
        logits (`torch.FloatTensor` of shape `(batch_size, sequence_length, config.vocab_size)`):
            Prediction scores of the language modeling head (scores for each vocabulary token before SoftMax).
        past_key_values (`List(Dict(str, torch.FloatTensor))`, *optional*, returned when `use_cache=True` is passed or when `config.use_cache=True`):
            - "prev_key_value": shape=(bsz * num_head * v_dim * qk_dim)
            - "scale": shape=((1 or bsz) * num_head * 1 * 1)

            Contains pre-computed hidden-states (key and values in the multi-scale retention blocks)
            that can be used (see `past_key_values` input) to speed up sequential decoding.
        hidden_states (`tuple(torch.FloatTensor)`, *optional*, returned when `output_hidden_states=True` is passed or when `config.output_hidden_states=True`):
            Tuple of `torch.FloatTensor` (one for the output of the embeddings, if the model has an embedding layer, +
            one for the output of each layer) of shape `(batch_size, sequence_length, decoder_embed_dim)`.

            Hidden-states of the model at the output of each layer plus the optional initial embedding outputs.
        retentions (`tuple(torch.FloatTensor)`, *optional*, returned when `output_retentions=True` is passed or when `config.output_retentions=True`):
            Tuple of `torch.FloatTensor` (one for each layer) of shape `(batch_size, num_heads, sequence_length,
            sequence_length)`.

            Retentions weights, used for visualization.

        attentions (`tuple(torch.FloatTensor)`, *optional*, for backward compatibility. Same as retentions.
    """

    loss: Optional[Tensor] = None
    logits: Tensor = None
    past_key_values: Optional[List[Dict[str, Tensor]]] = None
    hidden_states: Optional[Tuple[Tensor]] = None
    retentions: Optional[Tuple[Tensor]] = None
    attentions: Optional[Tuple[Tensor]] = None


class RetNetForCausalLM(RetNetPreTrainedModel):
    def __init__(
        self,
        config: RetNetConfig,
        embed_tokens: nn.Embedding = None,
        tensor_parallel: bool = False,
    ) -> None:
        super().__init__(config)
        self.model = RetNetModel(
            config, embed_tokens=embed_tokens, tensor_parallel=tensor_parallel
        )
        self.lm_head = nn.Dense(
            config.decoder_embed_dim, config.vocab_size, has_bias=False
        )

        # init here
        # self.lm_head.weight = initializer_decorator(Normal(mean=0, sigma=config.decoder_embed_dim**-0.5))
        
        self.lm_head.weight.set_data(init_norm(self.lm_head.weight, mean=0, sigma=config.decoder_embed_dim**-0.5))

        self.post_init()

    def get_input_embeddings(self):
        return self.model.embed_tokens

    def set_input_embeddings(self, value):
        self.model.embed_tokens = value

    def get_output_embeddings(self):
        return self.lm_head

    def set_output_embeddings(self, new_embeddings):
        self.lm_head = new_embeddings

    def set_decoder(self, decoder):
        self.model = decoder

    def get_decoder(self):
        return self.model

    def construct(
        self,
        input_ids: Tensor = None,
        retention_mask: Optional[Tensor] = None,
        attention_mask: Optional[Tensor] = None,
        past_key_values: Optional[List[Tensor]] = None,
        inputs_embeds: Optional[Tensor] = None,
        labels: Optional[Tensor] = None,
        use_cache: Optional[bool] = None,
        output_retentions: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        forward_impl: Optional[str] = None,
        recurrent_chunk_size: Optional[int] = None,
        retention_rel_pos: Optional[Tuple[Tensor]] = None,
    ) -> Union[Tuple, RetNetCausalLMOutputWithPast]:
        if output_retentions is None and output_attentions is not None:
            output_retentions = output_attentions
        output_retentions = (
            output_retentions
            if output_retentions is not None
            else self.config.output_retentions
        )
        output_hidden_states = (
            output_hidden_states
            if output_hidden_states is not None
            else self.config.output_hidden_states
        )
        return_dict = (
            return_dict if return_dict is not None else self.config.use_return_dict
        )
        forward_impl = (
            forward_impl if forward_impl is not None else self.config.forward_impl
        )
        recurrent_chunk_size = (
            recurrent_chunk_size
            if recurrent_chunk_size is not None
            else self.config.recurrent_chunk_size
        )

        if retention_mask is None and attention_mask is not None:
            retention_mask = attention_mask

        outputs = self.model(
            input_ids,
            retention_mask=retention_mask,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            output_retentions=output_retentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
            forward_impl=forward_impl,
            use_cache=use_cache,
            recurrent_chunk_size=recurrent_chunk_size,
            retention_rel_pos=retention_rel_pos,
        )

        hidden_states = outputs[0]
        logits = self.lm_head(hidden_states)

        loss = None
        if labels is not None:
            # Shift so that tokens < n predict n
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            # Flatten the tokens
            loss_fct = nn.CrossEntropyLoss()
            shift_logits = shift_logits.view(-1, self.config.vocab_size)
            shift_labels = shift_labels.view(-1)
            # Enable model parallelism
            loss = loss_fct(shift_logits, shift_labels)

            if self.config.z_loss_coeff > 0:
                # z_loss from PaLM paper
                # z_loss = 1e-4 * log(log(z)), where z = sum(exp(logits))
                z_loss = ops.logsumexp(shift_logits, dim=-1).log().mean()
                loss += self.config.z_loss_coeff * z_loss

        if not return_dict:
            output = (logits,) + outputs[1:]
            return (loss,) + output if loss is not None else output

        return RetNetCausalLMOutputWithPast(
            loss=loss,
            logits=logits,
            past_key_values=outputs.past_key_values,
            hidden_states=outputs.hidden_states,
            retentions=outputs.retentions,
            attentions=outputs.retentions,
        )

    def _crop_past_key_values(model, past_key_values, maximum_length):
        """Since retnet's kv do not have length, no need to crop. Just return"""
        return past_key_values

    def prepare_inputs_for_generation(
        self,
        input_ids,
        past_key_values=None,
        attention_mask=None,
        inputs_embeds=None,
        **kwargs,
    ):
        # if `inputs_embeds` are passed, we only want to use them in the 1st generation step
        if inputs_embeds is not None and past_key_values is None:
            model_inputs = {"inputs_embeds": inputs_embeds}
        else:
            model_inputs = {"input_ids": input_ids}

        forward_impl = kwargs.get("forward_impl", "parallel")
        if past_key_values is not None:
            forward_impl = "recurrent"

        model_inputs.update(
            {
                "past_key_values": past_key_values,
                "use_cache": kwargs.get("use_cache"),
                "attention_mask": attention_mask,
                "forward_impl": forward_impl,
            }
        )
        return model_inputs

    @staticmethod
    def _reorder_cache(past_key_values, beam_idx):
        reordered_past = ()
        for layer_past in past_key_values:  # dict
            layer_past_kv = layer_past["prev_key_value"]  # [b, h, v_dim / h, qk_dim]
            layer_past_scale = layer_past["scale"]  # [b, h, 1, 1]
            if layer_past_scale.shape[0] > 1:
                # this means that retention_mask is not None, so the scale for
                # each batch is different. We need to select the correct scale then.
                # NOTE: during huggingface generate, it will generate attention_mask
                # if it is None, so this linke will always be true. Still, having
                # this line here for safety.
                layer_past_scale = layer_past_scale.index_select(0, beam_idx)
            reordered_past += (
                {
                    "prev_key_value": layer_past_kv.index_select(0, beam_idx),
                    "scale": layer_past_scale,
                },
            )
        return reordered_past

    def sample_token(self, logit, do_sample=False, top_k=1, top_p=1.0, temperature=1.0):
        if not do_sample:
            
            return ops.argmax(logit, axis=-1, keepdim=True)
        filtered = top_k_top_p_filtering(logit / temperature, top_k=top_k, top_p=top_p)
        
        return ops.multinomial(nn.Softmax()(filtered, axis=-1), num_samples=1)

    # @torch.inference_mode()
    def custom_generate(
        self,
        input_ids: Tensor = None,
        retention_mask: Optional[Tensor] = None,
        attention_mask: Optional[Tensor] = None,
        parallel_compute_prompt=True,
        max_new_tokens=20,
        bos_token_id=0,
        eos_token_id=0,
        do_sample=False,
        top_k=0,
        top_p=1.0,
        temperature=1.0,
        early_stopping=True,
    ):
        ms.context.set_context(mode=ms.context.GRAPH_MODE)
        if retention_mask is None and attention_mask is not None:
            retention_mask = attention_mask
        outputs = None
        if input_ids is not None:
            if input_ids.shape[1] == 1:
                past_key_values = None
            elif parallel_compute_prompt:
                ret_mask = (
                    retention_mask[:, :-1] if retention_mask is not None else None
                )
                outputs = self(
                    input_ids[:, :-1],
                    retention_mask=ret_mask,
                    forward_impl="parallel",
                    return_dict=True,
                    use_cache=True,
                )
                past_key_values = outputs.past_key_values
            else:
                past_key_values = None
                for p_i in range(input_ids.shape[1] - 1):
                    ret_mask = (
                        retention_mask[:, : p_i + 1]
                        if retention_mask is not None
                        else None
                    )
                    outputs = self(
                        input_ids[:, : p_i + 1],
                        retention_mask=ret_mask,
                        forward_impl="recurrent",
                        past_key_values=past_key_values,
                        return_dict=True,
                        use_cache=True,
                    )
                    past_key_values = outputs.past_key_values

            generated = input_ids
        else:
            generated = Tensor([[bos_token_id]])
            past_key_values = None

        for i in range(max_new_tokens):
            outputs = self(
                generated,
                retention_mask=retention_mask,
                forward_impl="recurrent",
                past_key_values=past_key_values,
                use_cache=True,
                return_dict=True,
            )
            logit = outputs.logits[:, -1, :]  # [batch_size, vocab_size]
            past_key_values = outputs.past_key_values
            token = self.sample_token(
                logit,
                do_sample=do_sample,
                top_k=top_k,
                top_p=top_p,
                temperature=temperature,
            )
            
            generated = ops.Concat()([generated, token], axis=-1)
            if retention_mask is not None:
                retention_mask = ops.Concat()(
                    
                    [retention_mask, ops.ones_like(token)], axis=-1
                )
            if early_stopping and (token == eos_token_id).all():
                break
        ms.context.set_context(mode=ms.context.PYNATIVE_MODE)
        return generated


class RetNetForSequenceClassification(RetNetPreTrainedModel):
    def __init__(self, config, tensor_parallel=False):
        super().__init__(config)
        self.num_labels = config.num_labels
        self.model = RetNetModel(config, tensor_parallel=tensor_parallel)
        self.score = nn.Dense(config.decoder_embed_dim, self.num_labels, has_bias=False)

        # Initialize weights and apply final processing
        self.post_init()

    def get_input_embeddings(self):
        return self.model.embed_tokens

    def set_input_embeddings(self, value):
        self.model.embed_tokens = value

    def construct(
        self,
        input_ids: Tensor = None,
        retention_mask: Optional[Tensor] = None,
        attention_mask: Optional[Tensor] = None,
        past_key_values: Optional[List[Tensor]] = None,
        inputs_embeds: Optional[Tensor] = None,
        labels: Optional[Tensor] = None,
        use_cache: Optional[bool] = None,
        output_retentions: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        forward_impl: Optional[str] = None,
        recurrent_chunk_size: Optional[int] = None,
        retention_rel_pos: Optional[Tuple[Tensor]] = None,
    ) -> Union[Tuple, SequenceClassifierOutputWithPast]:
        if output_retentions is None and output_attentions is not None:
            output_retentions = output_attentions
        output_retentions = (
            output_retentions
            if output_retentions is not None
            else self.config.output_retentions
        )
        output_hidden_states = (
            output_hidden_states
            if output_hidden_states is not None
            else self.config.output_hidden_states
        )
        return_dict = (
            return_dict if return_dict is not None else self.config.use_return_dict
        )
        forward_impl = (
            forward_impl if forward_impl is not None else self.config.forward_impl
        )
        recurrent_chunk_size = (
            recurrent_chunk_size
            if recurrent_chunk_size is not None
            else self.config.recurrent_chunk_size
        )

        if retention_mask is None and attention_mask is not None:
            retention_mask = attention_mask

        outputs = self.model(
            input_ids,
            retention_mask=retention_mask,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            output_retentions=output_retentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
            forward_impl=forward_impl,
            use_cache=use_cache,
            recurrent_chunk_size=recurrent_chunk_size,
            retention_rel_pos=retention_rel_pos,
        )

        hidden_states = outputs[0]
        logits = self.score(hidden_states)

        if input_ids is not None:
            batch_size = input_ids.shape[0]
        else:
            batch_size = inputs_embeds.shape[0]

        if self.config.pad_token_id is None and batch_size != 1:
            raise ValueError(
                "Cannot handle batch sizes > 1 if no padding token is defined."
            )
        if self.config.pad_token_id is None:
            sequence_lengths = -1
        else:
            if input_ids is not None:
                sequence_lengths = (
                    ops.equal(input_ids, self.config.pad_token_id).long().argmax(-1) - 1
                )
            else:
                sequence_lengths = -1

        pooled_logits = logits[
            ms.numpy.arange(batch_size), sequence_lengths
        ]

        loss = None
        if labels is not None:
            if self.config.problem_type is None:
                if self.num_labels == 1:
                    self.config.problem_type = "regression"
                elif self.num_labels > 1 and (
                    labels.dtype == ms.int64 or labels.dtype == ms.int32
                ):
                    self.config.problem_type = "single_label_classification"
                else:
                    self.config.problem_type = "multi_label_classification"

            if self.config.problem_type == "regression":
                loss_fct = MSELoss()
                if self.num_labels == 1:
                    loss = loss_fct(pooled_logits.squeeze(), labels.squeeze())
                else:
                    loss = loss_fct(pooled_logits, labels)
            elif self.config.problem_type == "single_label_classification":
                loss_fct = CrossEntropyLoss()
                loss = loss_fct(
                    pooled_logits.view(-1, self.num_labels), labels.view(-1)
                )
            elif self.config.problem_type == "multi_label_classification":
                loss_fct = BCEWithLogitsLoss()
                loss = loss_fct(pooled_logits, labels)
        if not return_dict:
            output = (pooled_logits,) + outputs[1:]
            return ((loss,) + output) if loss is not None else output

        return SequenceClassifierOutputWithPast(
            loss=loss,
            logits=pooled_logits,
            past_key_values=outputs.past_key_values,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )
