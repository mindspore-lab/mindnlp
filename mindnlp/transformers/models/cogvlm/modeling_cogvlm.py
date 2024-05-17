"""largely copy from llama and adapt for cogvlm"""
import warnings
from typing import TYPE_CHECKING, Optional, Tuple, List, Union, Literal, Dict, Any

import math
import numpy as np
import mindspore
from mindspore import ops, nn, Tensor
from mindspore.common.initializer import initializer, Normal
from mindspore.dataset import transforms,vision
from mindnlp.modules.functional import finfo
from ...modeling_utils import PreTrainedModel
from ...tokenization_utils import PreTrainedTokenizer
from ...activations import ACT2FN
from ...modeling_outputs import BaseModelOutputWithPast,CausalLMOutputWithPast

from .configuration_cogvlm import CogVLMConfig
from .visual import EVA2CLIPModel
if TYPE_CHECKING:
    from mindnlp.utils import ModelOutput

LANGUAGE_TOKEN_TYPE = 0
VISION_TOKEN_TYPE = 1


# Copied from transformers.models.bart.modeling_bart._make_causal_mask
def _make_causal_mask(
        input_ids_shape, dtype: mindspore.dtype, past_key_values_length: int = 0
):
    """
    Make causal mask used for bi-directional self-attention.
    """
    bsz, tgt_len = input_ids_shape
    mask = ops.full((tgt_len, tgt_len),finfo(dtype=dtype,attr='min'),dtype=dtype)
    mask_cond = ops.arange(mask.shape[-1])
    mask = mask.masked_fill(mask_cond < (mask_cond + 1).view(mask.shape[-1], 1), 0)
    mask = mask.to(dtype)

    if past_key_values_length > 0:
        mask = ops.cat([ops.zeros((tgt_len, past_key_values_length), dtype=dtype), mask], axis=-1)
    return mask[None, None, :, :].expand(bsz, 1, tgt_len, tgt_len + past_key_values_length)



# Copied from transformers.models.bart.modeling_bart._expand_mask
def _expand_mask(mask: mindspore.Tensor, dtype: mindspore.dtype, tgt_len: Optional[int] = None):
    """
    Expands attention_mask from `[bsz, seq_len]` to `[bsz, 1, tgt_seq_len, src_seq_len]`.
    """
    bsz, src_len = mask.shape
    tgt_len = tgt_len if tgt_len is not None else src_len
    expanded_mask = mask[:, None, None, :].expand(bsz, 1, tgt_len, src_len).to(dtype)
    inverted_mask = 1.0 - expanded_mask
    return inverted_mask.masked_fill(inverted_mask.to(mindspore.bool_), finfo(dtype=dtype,attr='min'))


class RMSNorm(nn.Cell):
    def __init__(self, hidden_size, eps=1e-6):
        super().__init__()
        self.weight = mindspore.Parameter(ops.ones(hidden_size))
        self.variance_epsilon = eps

    def construct(self, hidden_states):
        input_dtype = hidden_states.dtype
        hidden_states = hidden_states.to(mindspore.float32)
        variance = hidden_states.pow(2).mean(-1, keep_dims=True)
        hidden_states = hidden_states * ops.rsqrt(variance + self.variance_epsilon)
        return (self.weight * hidden_states).to(input_dtype)


class MLP(nn.Cell):
    def __init__(self, config):
        super().__init__()
        self.hidden_size = config.hidden_size
        self.intermediate_size = config.intermediate_size
        self.gate_proj = nn.Dense(self.hidden_size, self.intermediate_size, has_bias=False)
        self.up_proj = nn.Dense(self.hidden_size, self.intermediate_size, has_bias=False)
        self.down_proj = nn.Dense(self.intermediate_size, self.hidden_size, has_bias=False)
        self.act_fn = ACT2FN[config.hidden_act]

    def construct(self, x):
        down_proj = self.down_proj(self.act_fn(self.gate_proj(x)) * self.up_proj(x))
        return down_proj


def get_expert_mask(token_type_ids):
    vision_token_mask = ops.zeros_like(token_type_ids, dtype=mindspore.bool_)
    vision_token_mask[:, :-1] = (token_type_ids[:, :-1] == VISION_TOKEN_TYPE) & (token_type_ids[:, 1:] == VISION_TOKEN_TYPE)
    language_token_mask = ~vision_token_mask
    return vision_token_mask, language_token_mask


class VisionExpertMLP(nn.Cell):
    def __init__(self, config):
        super().__init__()
        self.language_mlp = MLP(config)
        self.vision_mlp = MLP(config)

    def construct(self, hidden_states: "mindspore.Tensor(B, L, D)", token_type_ids: "mindspore.Tensor(B, L)"):
        output = ops.zeros(hidden_states.shape, dtype=hidden_states.dtype)
        vision_token_mask, language_token_mask = get_expert_mask(token_type_ids)
        output[vision_token_mask] = self.vision_mlp(hidden_states[vision_token_mask])
        output[language_token_mask] = self.language_mlp(hidden_states[language_token_mask])
        return output


def scaled_dot_product_attention(query, key, value, attn_mask=None, dropout_p=0.0, is_causal=False, scale=None) -> mindspore.Tensor:
    # Efficient implementation equivalent to the following:
    L, S = query.shape[-2], key.shape[-2]
    scale_factor = 1 / math.sqrt(query.shape[-1]) if scale is None else scale
    attn_bias = ops.zeros((L, S), dtype=query.dtype)
    if is_causal:
        assert attn_mask is None
        temp_mask = ops.ones((L, S), dtype=mindspore.bool_).tril(diagonal=0)
        attn_bias = attn_bias.masked_fill(temp_mask.logical_not(), finfo(dtype=query.dtype,attr='min'))
        attn_bias.to(query.dtype)

    if attn_mask is not None:
        if attn_mask.dtype == mindspore.bool_:
            attn_bias=  attn_bias.masked_fill_(attn_mask.logical_not(), finfo(dtype=query.dtype,attr='min'))
        else:
            attn_bias += attn_mask

    attn_weight = query @ key.swapaxes(-2, -1) * scale_factor
    attn_weight += attn_bias
    attn_weight = ops.softmax(attn_weight, axis=-1)
    attn_weight = ops.dropout(attn_weight, dropout_p)
    return attn_weight @ value


def attention_fn(
        query_layer: "mindspore.Tensor(B, H, L, HD)",
        key_layer: "mindspore.Tensor(B, H, L, HD)",
        value_layer: "mindspore.Tensor(B, H, L, HD)",
        attention_mask: "mindspore.Tensor(B, H, L, HD)",
        *,
        scaling_attention_score: bool = True,
        attention_dropout: nn.Cell = None):
    if scaling_attention_score:
        query_layer = query_layer / math.sqrt(query_layer.shape[-1])
    attention_scores = ops.matmul(query_layer, key_layer.swapaxes(-1, -2))
    attention_scores = attention_scores + attention_mask
    attention_scores = ops.softmax(attention_scores, axis=-1, dtype=mindspore.float32).to(query_layer.dtype)
    if attention_dropout is not None:
        attention_scores = attention_dropout(attention_scores)
    context_layer = ops.matmul(attention_scores, value_layer)
    return context_layer


class RotaryEmbedding(mindspore.nn.Cell):
    def __init__(self, dim, max_position_embeddings=2048, base=10000):
        super().__init__()

        self.dim = dim
        self.max_position_embeddings = max_position_embeddings
        self.base = base
        self.inv_freq = self._compute_inv_freq()
        self.max_seq_len_cached = 0

    def _compute_inv_freq(self):
        return 1.0 / (
                self.base
                ** (ops.arange(0, self.dim, 2).to(mindspore.float32) / self.dim)
        )

    def _set_cos_sin_cache(self, seq_len, dtype):
        self.max_seq_len_cached = seq_len
        t = ops.arange(self.max_seq_len_cached, dtype=self.inv_freq.dtype)
        freqs = ops.einsum("i,j->ij", t, self.inv_freq)
        # Different from paper, but it uses a different permutation in order to obtain the same calculation
        emb = ops.cat((freqs, freqs), axis=-1)
        self.cos_cached = emb.cos()[:, None, :].to(dtype)
        self.sin_cached = emb.sin()[:, None, :].to(dtype)


    def construct(self, x, seq_len):
        # x: [bs, num_attention_heads, seq_len, head_size]
        if seq_len > self.max_seq_len_cached:
            self._set_cos_sin_cache(seq_len=seq_len, dtype=x.dtype)

        return (
            self.cos_cached[:seq_len, ...].to(dtype=x.dtype),
            self.sin_cached[:seq_len, ...].to(dtype=x.dtype),
        )


def rotate_half(x):
    x1, x2 = x[..., :x.shape[-1] // 2], x[..., x.shape[-1] // 2:]
    return ops.cat((-x2, x1), axis=x1.ndim - 1)


def apply_rotary_pos_emb_index_bhs(q, k, cos, sin, position_id,unsqueeze_dim=1):

    cos = cos.squeeze()
    sin = sin.squeeze()
    cos = cos[position_id].unsqueeze(unsqueeze_dim)
    sin = sin[position_id].unsqueeze(unsqueeze_dim)
    q, k = (q * cos) + (rotate_half(q) * sin), (k * cos) + (rotate_half(k) * sin)
    return q, k


class VisionExpertAttention(nn.Cell):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.hidden_size = config.hidden_size
        self.num_heads = config.num_attention_heads
        self.head_dim = self.hidden_size // self.num_heads
        self.max_position_embeddings = config.max_position_embeddings

        self.rotary_emb = RotaryEmbedding(self.head_dim)
        self.vision_expert_query_key_value = nn.Dense(self.hidden_size, self.hidden_size * 3, has_bias=False)
        self.vision_expert_dense = nn.Dense(self.hidden_size, self.hidden_size, has_bias=False)
        self.language_expert_query_key_value = nn.Dense(self.hidden_size, self.hidden_size * 3, has_bias=False)
        self.language_expert_dense = nn.Dense(self.hidden_size, self.hidden_size, has_bias=False)

    def _swapaxes_for_scores(self, tensor):
        """Transpose a 3D tensor [B, L, H*HD] into a 4D tensor with size [B H L HD]."""
        new_tensor_shape = tensor.shape[:-1] + (self.num_heads, self.head_dim)
        tensor = tensor.view(*new_tensor_shape)
        return tensor.permute(0, 2, 1, 3)

    def construct(
            self,
            hidden_states: mindspore.Tensor,
            token_type_ids: mindspore.Tensor,
            position_ids: mindspore.Tensor,
            attention_mask: Optional[mindspore.Tensor] = None,
            past_key_value: Optional[Tuple[mindspore.Tensor]] = None,
            output_attentions: bool = False,
            use_cache: bool = False,
    ) -> Tuple[mindspore.Tensor, Optional[mindspore.Tensor], Optional[Tuple[mindspore.Tensor]]]:
        bsz, q_len, _ = hidden_states.shape
        vision_token_mask, language_token_mask = get_expert_mask(token_type_ids)

        shape = list(hidden_states.shape)
        shape[-1] = shape[-1] * 3
        shape = tuple(shape)

        mixed_raw_layer = ops.zeros(shape,dtype=hidden_states.dtype)
        mixed_raw_layer[vision_token_mask] = self.vision_expert_query_key_value(hidden_states[vision_token_mask])
        mixed_raw_layer[language_token_mask] = self.language_expert_query_key_value(hidden_states[language_token_mask])
        query_states, key_states, value_states = ops.split(mixed_raw_layer, self.hidden_size, axis=-1)

        query_states = self._swapaxes_for_scores(query_states)  # B, H, L, HD
        key_states = self._swapaxes_for_scores(key_states)  # B, H, L, HD
        value_states = self._swapaxes_for_scores(value_states)  # B, H, L, HD

        kv_seq_len = key_states.shape[-2]
        if past_key_value is not None:
            kv_seq_len += past_key_value[0].shape[-2]
        cos, sin = self.rotary_emb(value_states, seq_len=position_ids.max() + 1)

        tmp = [i.asnumpy() for i in [mixed_raw_layer,query_states, key_states, value_states,cos, sin]]

        query_states, key_states = apply_rotary_pos_emb_index_bhs(query_states, key_states, cos, sin, position_ids)
        if past_key_value is not None:
            key_states = ops.cat([past_key_value[0], key_states], axis=2)
            value_states = ops.cat([past_key_value[1], value_states], axis=2)
        past_key_value = (key_states, value_states) if use_cache else None
        context_layer = attention_fn(
            query_layer=query_states, key_layer=key_states, value_layer=value_states, attention_mask=attention_mask,
            scaling_attention_score=True, attention_dropout=None)

        if context_layer.shape != (bsz, self.num_heads, q_len, self.head_dim):
            raise ValueError(
                f"`attn_output` should be of size {(bsz, self.num_heads, q_len, self.head_dim)}, but is"
                f" {context_layer.shape}"
            )

        context_layer = context_layer.swapaxes(1, 2).reshape(bsz, q_len, self.hidden_size)
        attn_output = ops.zeros(context_layer.shape, dtype=hidden_states.dtype)
        attn_output[vision_token_mask] = self.vision_expert_dense(context_layer[vision_token_mask])
        attn_output[language_token_mask] = self.language_expert_dense(context_layer[language_token_mask])
        if output_attentions:
            warnings.warn("output_attentions is not implemented.")

        return attn_output, None, past_key_value


class CogVLMDecoderLayer(nn.Cell):
    def __init__(self, config):
        super().__init__()
        self.hidden_size = config.hidden_size
        self.self_attn = VisionExpertAttention(config=config)
        self.mlp = VisionExpertMLP(config)
        self.input_layernorm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)

    def construct(
            self,
            hidden_states: mindspore.Tensor,
            token_type_ids: mindspore.Tensor,
            position_ids: mindspore.Tensor,
            attention_mask: Optional[mindspore.Tensor] = None,
            past_key_value: Optional[Tuple[mindspore.Tensor]] = None,
            output_attentions: Optional[bool] = False,
            use_cache: Optional[bool] = False,
    ) -> Tuple[mindspore.Tensor, Optional[Tuple[mindspore.Tensor, mindspore.Tensor]]]:
        residual = hidden_states

        hidden_states = self.input_layernorm(hidden_states)

        # Self Attention
        hidden_states, self_attn_weights, present_key_value = self.self_attn(
            hidden_states=hidden_states,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            attention_mask=attention_mask,
            past_key_value=past_key_value,
            output_attentions=output_attentions,
            use_cache=use_cache,
        )
        hidden_states = residual + hidden_states
        # Fully Connected
        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states = self.mlp(hidden_states, token_type_ids=token_type_ids)
        hidden_states = residual + hidden_states
        outputs = (hidden_states,)

        if output_attentions:
            outputs += (self_attn_weights,)

        if use_cache:
            outputs += (present_key_value,)

        return outputs  # type: ignore


class CogVLMPreTrainedModel(PreTrainedModel):
    config_class = CogVLMConfig
    base_model_prefix = "model"
    supports_gradient_checkpointing = False
    _no_split_modules = ["CogVLMDecoderLayer", "TransformerLayer"]
    _skip_keys_device_placement = "past_key_values"

    def _init_weights(self, cell):
        if isinstance(cell, nn.Dense):
            # Slightly different from the TF version which uses truncated_normal for initialization
            # cf https://github.com/pytorch/pytorch/pull/5617
            cell.weight.set_data(initializer(Normal(mean=0,sigma=self.config.initializer_range),
                                                    cell.weight.shape, cell.weight.dtype))
            if cell.has_bias:
                cell.bias.set_data(initializer('zeros', cell.bias.shape, cell.bias.dtype))
        elif isinstance(cell, nn.Embedding):
            weight = np.random.normal(0.0, self.config.initializer_range, cell.weight.shape)
            if cell.padding_idx:
                weight[cell.padding_idx] = 0

            cell.weight.set_data(Tensor(weight, cell.weight.dtype))


def is_empty(images_list: Optional[List[List[mindspore.Tensor]]]):
    if images_list is None or len(images_list) == 0:
        return True
    for image_list in images_list:
        if len(image_list):
            return False
    return True



def build_position_ids(x:"mindspore.Tensor(B, L)", attention_mask: Optional["mindspore.Tensor(B, L)"] = None) -> "mindspore.Tensor(B, L)":

    if attention_mask is not None:
        tmp = x.copy()
        bool_a = attention_mask.bool()
        bool_a = ~bool_a
        tmp[bool_a] = -1
    else:
        tmp = x.copy()
    is_boi_eoi = ops.zeros_like(x, dtype=mindspore.bool_)
    is_boi_eoi[:, 1:] |= (tmp[:, 1:] == VISION_TOKEN_TYPE) & (tmp[:, :-1] == LANGUAGE_TOKEN_TYPE)
    is_boi_eoi[:, 0] |= (tmp[:, 0] == VISION_TOKEN_TYPE)
    is_boi_eoi[:, :-1] |= (tmp[:, :-1] == VISION_TOKEN_TYPE) & (tmp[:, 1:] == LANGUAGE_TOKEN_TYPE)
    is_boi_eoi[:, -1] |= (tmp[:, -1] == VISION_TOKEN_TYPE)
    tmp[is_boi_eoi] = LANGUAGE_TOKEN_TYPE

    # final position ids
    y = ops.zeros_like(x, dtype=mindspore.int64)
    y[:, 1:] = (tmp[:, 1:] == LANGUAGE_TOKEN_TYPE) | ((tmp[:, 1:] == VISION_TOKEN_TYPE) & (tmp[:, :-1] == LANGUAGE_TOKEN_TYPE))
    y = y.cumsum(axis=-1)
    return y

def my_index_put(index_tensor,update_data,original_data):
    """
    as mindspore in GPU does not support this operation: tensor.index_put, I simply implement this function instead.
    """
    left,right = -1,-1
    for j,i in enumerate(index_tensor[0]):
        if i:
            if left == -1:
                left = j
            else:
                right = j
    original_data[0][left:right+1] = update_data
    return original_data

class CogVLMModel(CogVLMPreTrainedModel):
    def __init__(self, config):
        super().__init__(config)
        self.padding_idx = config.pad_token_id
        self.vocab_size = config.vocab_size

        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size, self.padding_idx,dtype=mindspore.float32)
        self.layers = nn.CellList([CogVLMDecoderLayer(config) for _ in range(config.num_hidden_layers)])
        self.norm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)

        self.vision = EVA2CLIPModel(config)

        self.gradient_checkpointing = False
        # Initialize weights and apply final processing
        self.post_init()

    def encode_images(self, images: List[List[mindspore.Tensor]]) -> mindspore.Tensor:
        images_list, images = images, []

        images = []
        for image_list in images_list:
            for image in image_list:
                images.append(image)

        images = ops.stack(images)
        images_features = self.vision(images)
        return images_features

    def construct(
            self,
            input_ids: mindspore.Tensor = None,
            images: List[List[mindspore.Tensor]] = None,
            token_type_ids: Optional[mindspore.Tensor] = None,
            attention_mask: Optional[mindspore.Tensor] = None,
            position_ids: Optional[mindspore.Tensor] = None,
            past_key_values: Optional[List[mindspore.Tensor]] = None,
            inputs_embeds: Optional[mindspore.Tensor] = None,
            use_cache: Optional[bool] = None,
            output_attentions: Optional[bool] = None,
            output_hidden_states: Optional[bool] = None,
            return_dict: Optional[bool] = None,
    ) -> Union[Tuple, BaseModelOutputWithPast]:
        """take care of image_encode, token_type_ids, position_ids and (attention_mask = None is fine)"""
        if past_key_values is not None:
            pass  # generate mode with past_key_values. the image features are already mapped
        else:
            # not allow for inputs_embeds, because we want to process image feature

            assert input_ids is not None and inputs_embeds is None, f"{input_ids} {inputs_embeds}"
            if not is_empty(images):  # multi-modality

                assert token_type_ids is not None, "multi-modality requires `token_type_ids`!"

                assert len(input_ids) == len(images), f"{len(input_ids)} {len(images)}"
                inputs_embeds = self.embed_tokens(input_ids)
                images_features = self.encode_images(images)
                images_features = mindspore.Tensor(images_features)
                images_features = images_features.squeeze(0)
                #inputs_embeds = inputs_embeds.index_put([token_type_ids == VISION_TOKEN_TYPE], images_features)
                inputs_embeds = my_index_put(token_type_ids == VISION_TOKEN_TYPE, images_features, inputs_embeds)

            else:  # single-modality
                if token_type_ids is None:
                    token_type_ids = ops.ones_like(input_ids, dtype=mindspore.int64) * LANGUAGE_TOKEN_TYPE
                assert not (token_type_ids == VISION_TOKEN_TYPE).any(), f"{(token_type_ids == VISION_TOKEN_TYPE).sum()}"
                inputs_embeds = self.embed_tokens(input_ids)
            if position_ids is None:
                position_ids = build_position_ids(token_type_ids, attention_mask)

            input_ids = None
        return self.llm_forward(
            input_ids=input_ids,
            token_type_ids=token_type_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

    def llm_forward(
            self,
            input_ids: mindspore.Tensor = None,
            token_type_ids: mindspore.Tensor = None,
            attention_mask: Optional[mindspore.Tensor] = None,
            position_ids: Optional[mindspore.Tensor] = None,
            past_key_values: Optional[List[mindspore.Tensor]] = None,
            inputs_embeds: Optional[mindspore.Tensor] = None,
            use_cache: Optional[bool] = None,
            output_attentions: Optional[bool] = None,
            output_hidden_states: Optional[bool] = None,
            return_dict: Optional[bool] = None,
    ) -> Union[Tuple, BaseModelOutputWithPast]:
        """largely copy from llama forward and adapt for cogvlm with `token_type_ids`"""
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
            batch_size, seq_length = input_ids.shape
        elif inputs_embeds is not None:
            batch_size, seq_length, _ = inputs_embeds.shape
        else:
            raise ValueError("You have to specify either decoder_input_ids or decoder_inputs_embeds")

        seq_length_with_past = seq_length
        past_key_values_length = 0

        if past_key_values is not None:
            past_key_values_length = past_key_values[0][0].shape[2]
            seq_length_with_past = seq_length_with_past + past_key_values_length

        if position_ids is None:
            position_ids = ops.arange(
                past_key_values_length, seq_length + past_key_values_length, dtype=mindspore.int64
            )
            position_ids = position_ids.unsqueeze(0).view(-1, seq_length)
        else:
            position_ids = position_ids.view(-1, seq_length).long()

        if inputs_embeds is None:
            inputs_embeds = self.embed_tokens(input_ids)
        # embed positions
        if attention_mask is None:
            attention_mask = ops.ones(
                (batch_size, seq_length_with_past), dtype=mindspore.bool_
            )
        attention_mask = self._prepare_decoder_attention_mask(
            attention_mask, (batch_size, seq_length), inputs_embeds, past_key_values_length
        )

        hidden_states = inputs_embeds

        # decoder layers
        all_hidden_states = () if output_hidden_states else None
        all_self_attns = () if output_attentions else None
        next_decoder_cache = () if use_cache else None
        for idx, decoder_layer in enumerate(self.layers):
            if output_hidden_states:
                all_hidden_states += (hidden_states,)

            past_key_value = past_key_values[idx] if past_key_values is not None else None
            layer_outputs = decoder_layer(
                hidden_states,
                token_type_ids=token_type_ids,
                attention_mask=attention_mask,
                position_ids=position_ids,
                past_key_value=past_key_value,
                output_attentions=output_attentions,
                use_cache=use_cache,
            )
            hidden_states = layer_outputs[0]

            if use_cache:
                next_decoder_cache += (layer_outputs[2 if output_attentions else 1],)

            if output_attentions:
                all_self_attns += (layer_outputs[1],)
        hidden_states = self.norm(hidden_states)
        next_cache = next_decoder_cache if use_cache else None
        if not return_dict:
            return tuple(v for v in [hidden_states, next_cache, all_hidden_states, all_self_attns] if v is not None)
        out = BaseModelOutputWithPast(
            last_hidden_state=hidden_states,
            past_key_values=next_cache,
            hidden_states=all_hidden_states,
            attentions=all_self_attns,
        )[0]
        return BaseModelOutputWithPast(
            last_hidden_state=hidden_states,
            past_key_values=next_cache,
            hidden_states=all_hidden_states,
            attentions=all_self_attns,
        )

    def get_input_embeddings(self):
        return self.embed_tokens

    def set_input_embeddings(self, value):
        self.embed_tokens = value

    # noinspection PyMethodMayBeStatic
    # Copied from transformers.models.bart.modeling_bart.BartDecoder._prepare_decoder_attention_mask
    def _prepare_decoder_attention_mask(self, attention_mask, input_shape, inputs_embeds, past_key_values_length):
        # create causal mask
        # [bsz, seq_len] -> [bsz, 1, tgt_seq_len, src_seq_len]
        combined_attention_mask = None
        if input_shape[-1] > 1:
            combined_attention_mask = _make_causal_mask(
                input_shape,
                inputs_embeds.dtype,
                past_key_values_length=past_key_values_length,
            )

        if attention_mask is not None:
            # [bsz, seq_len] -> [bsz, 1, tgt_seq_len, src_seq_len]
            expanded_attn_mask = _expand_mask(attention_mask, inputs_embeds.dtype, tgt_len=input_shape[-1])
            combined_attention_mask = (
                expanded_attn_mask if combined_attention_mask is None else expanded_attn_mask + combined_attention_mask
            )

        return combined_attention_mask


def _history_to_prompt(signal_type, history, query):
    if signal_type == 'base':
        return query
    elif signal_type == 'vqa':
        answer_format = 'Short answer:'
    elif signal_type == 'chat':
        answer_format = 'Answer:'
    else:
        assert False, f"Unknown signal type {signal_type}"

    prompt = ''
    for i, (old_query, response) in enumerate(history):
        prompt += 'Question: ' + old_query + " {} ".format(answer_format) + response + "\n"
    prompt += 'Question: {} {}'.format(query, answer_format)
    return prompt


class CogVLMForCausalLM(CogVLMPreTrainedModel):
    _auto_class = "AutoModelForCausalLM"

    def __init__(self, config):
        super().__init__(config)
        self.model = CogVLMModel(config)
        self.vocab_size = config.vocab_size
        self.lm_head = nn.Dense(config.hidden_size, config.vocab_size, has_bias=False)

        # Initialize weights and apply final processing
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
            input_ids: mindspore.Tensor = None,
            images: List[List[mindspore.Tensor]] = None,
            token_type_ids: Optional[mindspore.Tensor] = None,
            attention_mask: Optional[mindspore.Tensor] = None,
            position_ids: Optional[mindspore.Tensor] = None,
            past_key_values: Optional[List[mindspore.Tensor]] = None,
            inputs_embeds: Optional[mindspore.Tensor] = None,
            use_cache: Optional[bool] = None,
            output_attentions: Optional[bool] = None,
            output_hidden_states: Optional[bool] = None,
            return_dict: Optional[bool] = None,
            labels: Optional[mindspore.Tensor] = None,
    ) -> Union[Tuple, CausalLMOutputWithPast]:
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict
        outputs = self.model(
            input_ids=input_ids,
            images=images,
            token_type_ids=token_type_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        hidden_states = outputs[0]
        logits = self.lm_head(hidden_states)
        logits = logits.float()

        loss = None
        if labels is not None:
            # Shift so that tokens < n predict n
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            # Flatten the tokens
            loss_fct = ops.cross_entropy#CrossEntropyLoss()
            shift_logits = shift_logits.view(-1, self.config.vocab_size)
            shift_labels = shift_labels.view(-1)
            # Enable model parallelism
            loss = loss_fct(shift_logits, shift_labels)

        if not return_dict:
            output = (logits,) + outputs[1:]
            return (loss,) + output if loss is not None else output

        return CausalLMOutputWithPast(
            loss=loss,
            logits=logits,
            past_key_values=outputs.past_key_values,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )

    def _prepare_attention_mask_for_generation(
            self,
            inputs: mindspore.Tensor,
            pad_token_id: Optional[int],
            eos_token_id: Optional[Union[int, List[int]]],
    ) -> mindspore.Tensor:
        return ops.ones(inputs.shape[:2], dtype=mindspore.int64)  # type: ignore

    def prepare_inputs_for_generation(
            self, input_ids, token_type_ids, images=None, past_key_values=None, attention_mask=None, inputs_embeds=None, **kwargs
    ):
        # build position_ids if needed
        position_ids = kwargs.get("position_ids", None)
        if position_ids is None:
            position_ids = build_position_ids(token_type_ids, attention_mask)

        if past_key_values:
            input_ids = input_ids[:, -1:]
            token_type_ids = token_type_ids[:, -1:]
            position_ids = position_ids[:, -1:]

        # if `inputs_embeds` are passed, we only want to use them in the 1st generation step
        if inputs_embeds is not None and past_key_values is None:
            model_inputs = {"inputs_embeds": inputs_embeds}
        else:
            model_inputs = {"input_ids": input_ids}

        model_inputs.update(
            {
                "token_type_ids": token_type_ids,
                "images": images,
                "position_ids": position_ids,
                "past_key_values": past_key_values,
                "use_cache": kwargs.get("use_cache"),
                "attention_mask": attention_mask,
            }
        )
        return model_inputs

    def _update_model_kwargs_for_generation(
            self,
            outputs: "ModelOutput",
            model_kwargs: Dict[str, Any],
            is_encoder_decoder: bool = False,
            standardize_cache_format: bool = False,
    ) -> Dict[str, Any]:
        # update past_key_values
        model_kwargs["past_key_values"] = self._extract_past_from_model_output(
            outputs, standardize_cache_format=standardize_cache_format
        )
        if getattr(outputs, "state", None) is not None:
            model_kwargs["state"] = outputs.state

        # update token_type_ids with last value
        if "token_type_ids" in model_kwargs:
            token_type_ids = model_kwargs["token_type_ids"]
            new_token_type_ids = ops.ones((token_type_ids.shape[0], 1), dtype=token_type_ids.dtype,
                                            ) * LANGUAGE_TOKEN_TYPE
            model_kwargs["token_type_ids"] = ops.cat([token_type_ids, new_token_type_ids], axis=-1)

        if not is_encoder_decoder:
            # update attention mask
            if "attention_mask" in model_kwargs:
                attention_mask = model_kwargs["attention_mask"]
                model_kwargs["attention_mask"] = ops.cat(
                    [attention_mask, attention_mask.new_ones((attention_mask.shape[0], 1))], axis=-1
                )
        else:
            # update decoder attention mask
            if "decoder_attention_mask" in model_kwargs:
                decoder_attention_mask = model_kwargs["decoder_attention_mask"]
                model_kwargs["decoder_attention_mask"] = ops.cat(
                    [decoder_attention_mask, decoder_attention_mask.new_ones((decoder_attention_mask.shape[0], 1))],
                    axis=-1,
                )

        return model_kwargs

    def _reorder_cache(self, past_key_values, beam_idx):
        reordered_past = ()
        for layer_past in past_key_values:
            reordered_past += (
                tuple(past_state.index_select(0, beam_idx) for past_state in layer_past),
            )
        return reordered_past

    def build_conversation_input_ids(
            self,
            tokenizer: "PreTrainedTokenizer",
            *,
            query: str,
            history: Optional[List[Tuple[str, str]]] = None,
            images: Optional[List["PIL.Image"]] = None,
            template_version: Optional[Literal["base", "chat", "vqa"]] = None,
    ):
        image_size: int = self.config.vision_config['image_size']
        patch_size: int = self.config.vision_config['patch_size']
        template_version = template_version or self.config.template_version
        assert images is None or len(images) <= 1, "not support multi images by now."
        history = history or []
        text = _history_to_prompt(template_version, history, query)
        input_ids = [tokenizer.bos_token_id]
        token_type_ids = [LANGUAGE_TOKEN_TYPE]
        if images is not None and len(images) == 1:
            # vision
            transform = transforms.Compose(
                [
                    vision.Resize(
                        (image_size, image_size), interpolation= vision.Inter.BICUBIC),
                    vision.ToTensor(),
                    vision.Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711),is_hwc=False),
                ]
            )
            images = transform(images[0])
            # language
            vision_token_num = (image_size // patch_size) * (image_size // patch_size) + 2
            input_ids += [tokenizer.pad_token_id] * vision_token_num
            token_type_ids += [VISION_TOKEN_TYPE] * vision_token_num
        text_ids = tokenizer.encode(text, add_special_tokens=False)
        input_ids += text_ids
        token_type_ids += [LANGUAGE_TOKEN_TYPE] * len(text_ids)
        attention_mask = [1] * len(input_ids)
        return {
            'input_ids': mindspore.tensor(input_ids, dtype=mindspore.int64),
            'token_type_ids': mindspore.tensor(token_type_ids, dtype=mindspore.int64),
            'attention_mask': mindspore.tensor(attention_mask, dtype=mindspore.int64),
            'images': images,
        }

__all__ = [
    "MLP",
    "RMSNorm",
    'VisionExpertMLP',
    'RotaryEmbedding',
    'VisionExpertAttention',
    'CogVLMDecoderLayer',
    'CogVLMPreTrainedModel',
    'CogVLMModel',
    'CogVLMForCausalLM',
]
