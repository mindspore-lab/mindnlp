# Copyright 2023 Huawei Technologies Co., Ltd
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ============================================================================
# pylint: disable=C0103
"""
Moss model
"""
from typing import Optional, Tuple, Union
import numpy as np
import mindspore
from mindspore import nn, ops, Tensor
from mindspore.nn import CrossEntropyLoss
from mindspore.common.initializer import initializer, Normal, Zero, One
from mindnlp.transformers.activations import ACT2FN
from .moss_configuration import MossConfig
from ...modeling_utils import PreTrainedModel
from ...modeling_outputs import BaseModelOutputWithPast, CausalLMOutputWithPast

_CHECKPOINT_FOR_DOC = "fnlp/moss-moon-003-base"
_CONFIG_FOR_DOC = "MossConfig"

MOSS_PRETRAINED_MODEL_ARCHIVE_LIST = [
    "fnlp/moss-moon-003-base",
    "fnlp/moss-moon-003-sft",
    "fnlp/moss-moon-003-sft-plugin",
    "fnlp/moss-moon-003-sft-int4",
    "fnlp/moss-moon-003-sft-plugin-int4",
    "fnlp/moss-moon-003-sft-int8",
    "fnlp/moss-moon-003-sft-plugin-int8",
]


def create_sinusoidal_positions(num_pos: int, dim: int) -> Tensor:
    """
    Copied from transformers.models.gptj.modeling_gptj.create_sinusoidal_positions
    """
    inv_freq = 1.0 / (10000 ** (ops.arange(0, dim, 2) * 1.0 / dim))
    sinusoid_inp = ops.einsum(
        "i , j -> i j", ops.arange(num_pos, dtype=mindspore.float32), inv_freq).float()
    res = ops.cat((ops.sin(sinusoid_inp), ops.cos(sinusoid_inp)), axis=1)
    return res


def rotate_every_two(input_tensor: Tensor) -> Tensor:
    """
    Copied from transformers.models.gptj.modeling_gptj.rotate_every_two
    """
    tensor1 = input_tensor[:, :, :, ::2]
    tensor2 = input_tensor[:, :, :, 1::2]
    out_tensor = ops.stack((-tensor2, tensor1), axis=-1)
    return ops.flatten(out_tensor, start_dim=-2)


def apply_rotary_pos_emb(tensor: Tensor, sin: Tensor, cos: Tensor) -> Tensor:
    """
    Copied from transformers.models.gptj.modeling_gptj.apply_rotary_pos_emb
    """
    sin = ops.repeat_elements(sin[:, :, None, :], 2, 3)
    cos = ops.repeat_elements(cos[:, :, None, :], 2, 3)
    return (tensor * cos) + (rotate_every_two(tensor) * sin)


class MossAttention(nn.Cell):
    """
    Moss attention layer
    """

    def __init__(self, config):
        super().__init__()

        max_positions = config.max_position_embeddings
        # self.register_buffer(
        #     "causal_mask",
        #     ops.tril(ops.ones(max_positions, max_positions)).view(
        #         1, 1, max_positions, max_positions
        #     ),
        # )
        self.causal_mask = ops.tril(ops.ones((max_positions, max_positions), dtype=mindspore.bool_)).view(
            1, 1, max_positions, max_positions)

        self.attn_dropout = nn.Dropout(p=config.attn_pdrop)
        self.resid_dropout = nn.Dropout(p=config.resid_pdrop)

        self.embed_dim = config.hidden_size
        self.num_attention_heads = config.num_attention_heads
        self.head_dim = self.embed_dim // self.num_attention_heads
        if self.head_dim * self.num_attention_heads != self.embed_dim:
            raise ValueError(
                f"embed_dim must be divisible by num_attention_heads (got `embed_dim`: {self.embed_dim} and"
                f" `num_attention_heads`: {self.num_attention_heads})."
            )
        self.scale_attn = ops.sqrt(Tensor(self.head_dim, dtype=mindspore.float32)).to(
            mindspore.float32
        )
        self.qkv_proj = nn.Dense(
            self.embed_dim, self.embed_dim * 3, has_bias=False)

        self.out_proj = nn.Dense(
            self.embed_dim, self.embed_dim, has_bias=False)
        self.rotary_dim = config.rotary_dim
        pos_embd_dim = self.rotary_dim or self.embed_dim
        self.embed_positions = create_sinusoidal_positions(
            max_positions, pos_embd_dim)

    def _split_heads(self, input_tensor, n_head, dim_head, mp_num):
        reshaped = input_tensor.reshape(input_tensor.shape[:-1] + (n_head // mp_num, dim_head))
        reshaped = reshaped.reshape(input_tensor.shape[:-2] + (-1,) + reshaped.shape[-1:])
        return reshaped

    def _merge_heads(self, tensor, num_attention_heads, attn_head_size):
        """
        Merges attn_head_size dim and num_attn_heads dim into n_ctx
        """
        if len(tensor.shape) == 5:
            # tensor = ops.permute(tensor,(0, 1, 3, 2, 4))
            tensor = ops.permute(tensor, (0, 1, 3, 2, 4))
        elif len(tensor.shape) == 4:
            # tensor = ops.permute(tensor, (0, 2, 1, 3))
            tensor = ops.permute(tensor, (0, 2, 1, 3))
        else:
            raise ValueError(
                f"Input tensor rank should be one of [4, 5], but is: {len(tensor.shape)}")
        new_shape = tensor.shape[:-2] + (num_attention_heads * attn_head_size,)
        return tensor.view(new_shape)

    def _attn(
            self,
            query,
            key,
            value,
            attention_mask=None,
            head_mask=None,
    ):
        # compute causal mask from causal mask buffer
        query_length, key_length = query.shape[-2], key.shape[-2]
        causal_mask = self.causal_mask[:, :, key_length -
                                             query_length: key_length, :key_length]

        # Keep the attention weights computation in fp32 to avoid overflow issues
        query = query.to(mindspore.float32)
        key = key.to(mindspore.float32)

        attn_weights = ops.matmul(query, ops.swapaxes(key, -1, -2))

        attn_weights = attn_weights / self.scale_attn
        mask_value = np.finfo(
            mindspore.dtype_to_nptype(attn_weights.dtype)).min
        # Need to be a tensor, otherwise we get error: `RuntimeError: expected scalar type float but found double`.
        # Need to be on the same device, otherwise `RuntimeError: ..., x and y to be on the same device`
        mask_value = Tensor(mask_value, dtype=attn_weights.dtype)
        attn_weights = ops.where(causal_mask, attn_weights, mask_value)

        if attention_mask is not None:
            # Apply the attention mask
            attn_weights = attn_weights + attention_mask

        attn_weights = nn.Softmax(axis=-1)(attn_weights)
        attn_weights = attn_weights.to(value.dtype)
        attn_weights = self.attn_dropout(attn_weights)

        # Mask heads if we want to
        if head_mask is not None:
            attn_weights = attn_weights * head_mask

        attn_output = ops.matmul(attn_weights, value)

        return attn_output, attn_weights

    def construct(
            self,
            hidden_states: Optional[Tensor],
            layer_past: Optional[Tuple[Tensor]] = None,
            attention_mask: Optional[Tensor] = None,
            position_ids: Optional[Tensor] = None,
            head_mask: Optional[Tensor] = None,
            use_cache: Optional[bool] = False,
            output_attentions: Optional[bool] = False,
    ) -> Union[
        Tuple[Tensor, Tuple[Tensor]],
        Optional[Tuple[Tensor, Tuple[Tensor],
        Tuple[Tensor, ...]]],
    ]:
        qkv = self.qkv_proj(hidden_states)
        # TODO(enijkamp): factor out number of logical TPU-v4 cores or make forward pass agnostic
        mp_num = 4
        qkv_split = qkv.reshape(qkv.shape[:-1] + (mp_num, -1))

        local_dim = self.head_dim * self.num_attention_heads // mp_num
        query, value, key = ops.split(qkv_split, local_dim, axis=-1)

        query = self._split_heads(
            query, self.num_attention_heads, self.head_dim, mp_num=mp_num)
        key = self._split_heads(
            key, self.num_attention_heads, self.head_dim, mp_num=mp_num)

        value = self._split_heads(
            value, self.num_attention_heads, self.head_dim, mp_num=mp_num)
        value = ops.permute(value, (0, 2, 1, 3))
        # embed_positions = self.embed_positions
        # if embed_positions.device != position_ids.device:
        #     embed_positions = embed_positions.to(position_ids.device)
        #     self.embed_positions = embed_positions

        embed_positions = self.embed_positions
        sincos = embed_positions[position_ids]
        sin, cos = ops.split(sincos, sincos.shape[-1] // 2, -1)

        if self.rotary_dim is not None:
            k_rot = key[:, :, :, : self.rotary_dim]
            k_pass = key[:, :, :, self.rotary_dim:]

            q_rot = query[:, :, :, : self.rotary_dim]
            q_pass = query[:, :, :, self.rotary_dim:]

            k_rot = apply_rotary_pos_emb(k_rot, sin, cos)
            q_rot = apply_rotary_pos_emb(q_rot, sin, cos)

            key = ops.cat([k_rot, k_pass], axis=-1)
            query = ops.cat([q_rot, q_pass], axis=-1)
        else:
            key = apply_rotary_pos_emb(key, sin, cos)
            query = apply_rotary_pos_emb(query, sin, cos)

        key = ops.permute(key, (0, 2, 1, 3))
        query = ops.permute(query, (0, 2, 1, 3))
        if layer_past is not None:
            past_key = layer_past[0]
            past_value = layer_past[1]
            key = ops.cat((past_key, key), axis=-2)
            value = ops.cat((past_value, value), axis=-2)

        if use_cache is True:
            present = (key, value)
        else:
            present = None

        # compute self-attention: V x Softmax(QK^T)
        attn_output, attn_weights = self._attn(
            query, key, value, attention_mask, head_mask)

        attn_output = self._merge_heads(
            attn_output, self.num_attention_heads, self.head_dim)
        attn_output = self.out_proj(attn_output)
        attn_output = self.resid_dropout(attn_output)

        outputs = (attn_output, present)
        if output_attentions:
            outputs += (attn_weights,)

        return outputs  # a, present, (attentions)


class MossMLP(nn.Cell):
    """
    Copied from transformers.models.gptj.modeling_gptj.GPTJMLP with GPTJ->Moss
    """

    def __init__(self, intermediate_size, config):  # in MLP: intermediate_size= 4 * embed_dim
        super().__init__()
        embed_dim = config.n_embd

        self.fc_in = nn.Dense(embed_dim, intermediate_size)
        self.fc_out = nn.Dense(intermediate_size, embed_dim)

        self.act = ACT2FN[config.activation_function]
        self.dropout = nn.Dropout(p=config.resid_pdrop)

    def construct(self, hidden_states: Optional[Tensor]) -> Tensor:
        hidden_states = self.fc_in(hidden_states)
        hidden_states = self.act(hidden_states)
        hidden_states = self.fc_out(hidden_states)
        hidden_states = self.dropout(hidden_states)
        return hidden_states


class MossBlock(nn.Cell):
    """
    Copied from transformers.models.gptj.modeling_gptj.GPTJBlock with GPTJ->Moss
    """

    def __init__(self, config):
        super().__init__()
        inner_dim = config.n_inner if config.n_inner is not None else 4 * config.n_embd
        self.ln_1 = nn.LayerNorm(
            [config.n_embd], epsilon=config.layer_norm_epsilon)
        self.attn = MossAttention(config)
        self.mlp = MossMLP(inner_dim, config)

    def construct(
            self,
            hidden_states: Optional[Tensor],
            layer_past: Optional[Tuple[Tensor]] = None,
            attention_mask: Optional[Tensor] = None,
            position_ids: Optional[Tensor] = None,
            head_mask: Optional[Tensor] = None,
            use_cache: Optional[bool] = False,
            output_attentions: Optional[bool] = False,
    ) -> Union[Tuple[Tensor], Optional[Tuple[Tensor, Tuple[Tensor, ...]]]]:
        residual = hidden_states
        hidden_states = self.ln_1(hidden_states)
        attn_outputs = self.attn(
            hidden_states=hidden_states,
            layer_past=layer_past,
            attention_mask=attention_mask,
            position_ids=position_ids,
            head_mask=head_mask,
            use_cache=use_cache,
            output_attentions=output_attentions,
        )
        attn_output = attn_outputs[0]  # output_attn: a, present, (attentions)
        outputs = attn_outputs[1:]

        feed_forward_hidden_states = self.mlp(hidden_states)
        hidden_states = attn_output + feed_forward_hidden_states + residual

        if use_cache:
            outputs = (hidden_states,) + outputs
        else:
            outputs = (hidden_states,) + outputs[1:]

        return outputs  # hidden_states, present, (attentions)


class MossPreTrainedModel(PreTrainedModel):
    """
    An abstract class to handle weights initialization and a simple interface for downloading and loading pretrained
    models.
    """

    config_class = MossConfig
    base_model_prefix = "transformer"
    supports_gradient_checkpointing = True
    _no_split_modules = ["MossBlock"]

    # def __init__(self, *inputs, **kwargs):
    #     super().__init__(*inputs, **kwargs)

    def _init_weights(self, cell):
        """Initialize the weight."""
        if isinstance(cell, (nn.Dense,)):
            # Slightly different from Mesh Transformer JAX which uses truncated_normal for initialization
            # cf https://github.com/MindSpore/MindSpore/pull/5617
            # cell.weight.data.normal_(mean=0.0, std=self.config.initializer_range)
            # cell = ops.normal(cell.weight.data.shape,mean=0.0,stddev=self.config.initializer_range)
            cell.weight.set_data(initializer(Normal(self.config.initializer_range),
                                             cell.weight.shape, cell.weight.dtype))
            if cell.bias is not None:
                cell.bias.set_data(initializer(
                    Zero(), cell.bias.shape, cell.bias.dtype))
        elif isinstance(cell, nn.Embedding):

            cell.weight.set_data(initializer(Normal(self.config.initializer_range),
                                                      cell.weight.shape, cell.weight.dtype))
            if cell.padding_idx is not None:
                cell.weight.data[cell.padding_idx].zero_()
        elif isinstance(cell, nn.LayerNorm):

            cell.bias.set_data(initializer(
                Zero(), cell.bias.shape, cell.bias.dtype))
            cell.weight.set_data(initializer(
                One(), cell.weight.shape, cell.weight.dtype))

    def _set_gradient_checkpointing(self, cell, value=False):
        if isinstance(cell, MossModel):
            cell.gradient_checkpointing = value


MOSS_START_DOCSTRING = r"""
    This model is a MindSpore [mindspore.nn.Cell](https://MindSpore.org/docs/stable/nn.html#mindspore.nn.Cell) sub-class. Use
    it as a regular MindSpore Cell and refer to the MindSpore documentation for all matter related to general usage and
    behavior.

    Parameters:
        config ([`MossConfig`]): Model configuration class with all the parameters of the model.
            Initializing with a config file does not load the weights associated with the model, only the
            configuration. Check out the [`~PreTrainedModel.from_pretrained`] method to load the model weights.
"""

MOSS_INPUTS_DOCSTRING = r"""
    Args:
        input_ids (`Tensor(dtype=mindspore.int64)` of shape `({0})`):
            Indices of input sequence tokens in the vocabulary.

            Indices can be obtained using [`AutoProcenizer`]. See [`PreTrainedTokenizer.encode`] and
            [`PreTrainedTokenizer.__call__`] for details.

            [What are input IDs?](../glossary#input-ids)
        attention_mask (`Tensor(dtype=mindspore.float32)` of shape `({0})`, *optional*):
            Mask to avoid performing attention on padding token indices. Mask values selected in `[0, 1]`:

            - 1 for tokens that are **not masked**,
            - 0 for tokens that are **masked**.

            [What are attention masks?](../glossary#attention-mask)
        token_type_ids (`Tensor(dtype=mindspore.int64)` of shape `({0})`, *optional*):
            Segment token indices to indicate first and second portions of the inputs. Indices are selected in `[0,
            1]`:

            - 0 corresponds to a *sentence A* token,
            - 1 corresponds to a *sentence B* token.

            [What are token type IDs?](../glossary#token-type-ids)
        position_ids (`Tensor(dtype=mindspore.int64)` of shape `({0})`, *optional*):
            Indices of positions of each input sequence tokens in the position embeddings. Selected in the range `[0,
            config.n_positions - 1]`.

            [What are position IDs?](../glossary#position-ids)
        head_mask (`Tensor(dtype=mindspore.float32)` of shape `(num_attention_heads,)` or `(n_layer, num_attention_heads)`, *optional*):
            Mask to nullify selected heads of the self-attention modules. Mask values selected in `[0, 1]`:

            - 1 indicates the head is **not masked**,
            - 0 indicates the head is **masked**.

        inputs_embeds (`Tensor(dtype=mindspore.float32)` of shape `({0}, hidden_dim)`, *optional*):
            Optionally, instead of passing `input_ids` you can choose to directly pass an embedded representation. This
            is useful if you want more control over how to convert *input_ids* indices into associated vectors than the
            model's internal embedding lookup matrix.
        output_attentions (`bool`, *optional*):
            Whether or not to return the attentions tensors of all attention layers. See `attentions` under returned
            tensors for more detail.
        output_hidden_states (`bool`, *optional*):
            Whether or not to return the hidden states of all layers. See `hidden_states` under returned tensors for
            more detail.
        return_dict (`bool`, *optional*):
            Whether or not to return a [`~utils.ModelOutput`] instead of a plain tuple.
"""


class MossModel(MossPreTrainedModel):
    """
    Moss model layer
    """

    def __init__(self, config):
        super().__init__(config)

        self.embed_dim = config.n_embd
        self.vocab_size = config.vocab_size
        self.wte = nn.Embedding(config.vocab_size, self.embed_dim)
        self.drop = nn.Dropout(p=config.embd_pdrop)
        self.h = nn.CellList([MossBlock(config) for _ in range(config.n_layer)])
        self.ln_f = nn.LayerNorm([self.embed_dim], epsilon=config.layer_norm_epsilon)
        self.rotary_dim = min(config.rotary_dim, config.n_ctx // config.num_attention_heads)

        self.gradient_checkpointing = False

        # Initialize weights and apply final processing
        self.post_init()

    def get_input_embeddings(self):
        """
        get input embeddings
        """
        return self.wte

    def set_input_embeddings(self, new_embeddings):
        """
        set input embeddings
        """
        self.wte = new_embeddings

    def construct(
            self,
            input_ids: Optional[Tensor] = None,
            past_key_values: Optional[Tuple[Tuple[Tensor]]] = None,
            attention_mask: Optional[Tensor] = None,
            token_type_ids: Optional[Tensor] = None,
            position_ids: Optional[Tensor] = None,
            head_mask: Optional[Tensor] = None,
            inputs_embeds: Optional[Tensor] = None,
            use_cache: Optional[bool] = None,
            output_attentions: Optional[bool] = None,
            output_hidden_states: Optional[bool] = None,
            return_dict: Optional[bool] = None,
    ) -> Union[Tuple, Tuple]:
        """
        Construct moss model
        """
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        use_cache = use_cache if use_cache is not None else self.config.use_cache
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        if input_ids is not None and inputs_embeds is not None:
            raise ValueError(
                "You cannot specify both input_ids and inputs_embeds at the same time")
        if input_ids is not None:
            input_shape = input_ids.shape
            input_ids = input_ids.view(-1, input_shape[-1])
            batch_size = input_ids.shape[0]
        elif inputs_embeds is not None:
            input_shape = inputs_embeds.shape[:-1]
            batch_size = inputs_embeds.shape[0]
        else:
            raise ValueError(
                "You have to specify either input_ids or inputs_embeds")

        # device = input_ids.device if input_ids is not None else inputs_embeds.device

        if token_type_ids is not None:
            token_type_ids = token_type_ids.view(-1, input_shape[-1])

        if position_ids is not None:
            position_ids = position_ids.view(-1, input_shape[-1]).long()

        if past_key_values is None:
            past_length = 0
            past_key_values = tuple([None] * len(self.h))
        else:
            past_length = past_key_values[0][0].size(-2)

        if position_ids is None:
            position_ids = ops.arange(
                past_length, input_shape[-1] + past_length, dtype=mindspore.int64)
            position_ids = Tensor.unsqueeze(
                position_ids, dim=0).view(-1, input_shape[-1])

        # Attention mask.
        if attention_mask is not None:
            if batch_size <= 0:
                raise ValueError("batch_size has to be defined and > 0")
            attention_mask = attention_mask.view(batch_size, -1)
            # We create a 3D attention mask from a 2D tensor mask.
            # Sizes are [batch_size, 1, 1, to_seq_length]
            # So we can broadcast to [batch_size, num_heads, from_seq_length, to_seq_length]
            # this attention mask is more simple than the triangular masking of causal attention
            # used in OpenAI GPT, we just need to prepare the broadcast dimension here.
            attention_mask = attention_mask[:, None, None, :]

            # Since attention_mask is 1.0 for positions we want to attend and 0.0 for
            # masked positions, this operation will create a tensor which is 0.0 for
            # positions we want to attend and the dtype's smallest value for masked positions.
            # Since we are adding it to the raw scores before the softmax, this is
            # effectively the same as removing these entirely.
            attention_mask = attention_mask.to(
                dtype=self.dtype)  # fp16 compatibility
            attention_mask = (1.0 - attention_mask) * \
                             float(np.finfo(mindspore.dtype_to_nptype(self.dtype)).min)

        # Prepare head mask if needed
        # 1.0 in head_mask indicate we keep the head
        # attention_probs has shape bsz x num_attention_heads x N x N
        # head_mask has shape n_layer x batch x num_attention_heads x N x N
        head_mask = self.get_head_mask(head_mask, self.config.n_layer)

        if inputs_embeds is None:
            inputs_embeds = self.wte(input_ids)

        hidden_states = inputs_embeds

        if token_type_ids is not None:
            token_type_embeds = self.wte(token_type_ids)
            hidden_states = hidden_states + token_type_embeds

        hidden_states = self.drop(hidden_states)

        output_shape = input_shape + (hidden_states.shape[-1],)

        # if self.gradient_checkpointing and self.training:
        #     if use_cache:
        #         logger.warning_once(
        #             "`use_cache=True` is incompatible with `config.gradient_checkpointing=True`. Setting "
        #             "`use_cache=False`..."
        #         )
        #         use_cache = False

        presents = () if use_cache else None
        all_self_attentions = () if output_attentions else None
        all_hidden_states = () if output_hidden_states else None
        for i, (block, layer_past) in enumerate(zip(self.h, past_key_values)):
            if output_hidden_states:
                all_hidden_states = all_hidden_states + (hidden_states,)

            # if self.gradient_checkpointing and self.training:

            #     def create_custom_forward(cell):
            #         def custom_forward(*inputs):
            #             # None for past_key_value
            #             return cell(*inputs, use_cache, output_attentions)

            #         return custom_forward
            #     # outputs = torch.utils.checkpoint.checkpoint(
            #     #     create_custom_forward(block),
            #     #     hidden_states,
            #     #     None,
            #     #     attention_mask,
            #     #     position_ids,
            #     #     head_mask[i],
            #     # )
            # else:
            outputs = block(
                hidden_states=hidden_states,
                layer_past=layer_past,
                attention_mask=attention_mask,
                position_ids=position_ids,
                head_mask=head_mask[i],
                use_cache=use_cache,
                output_attentions=output_attentions,
            )

            hidden_states = outputs[0]
            if use_cache:
                presents = presents + (outputs[1],)

            if output_attentions:
                all_self_attentions = all_self_attentions + \
                                      (outputs[2 if use_cache else 1],)

        hidden_states = self.ln_f(hidden_states)

        hidden_states = hidden_states.view(output_shape)
        # Add last hidden state
        if output_hidden_states:
            all_hidden_states = all_hidden_states + (hidden_states,)

        if not return_dict:
            return tuple(v for v in [hidden_states, presents, all_hidden_states, all_self_attentions] if v is not None)

        return BaseModelOutputWithPast(
            last_hidden_state=hidden_states,
            past_key_values=presents,
            hidden_states=all_hidden_states,
            attentions=all_self_attentions,
        )


class MossForCausalLM(MossPreTrainedModel):
    """
    The Moss Model transformer with a language modeling head on top.
    """

    _keys_to_ignore_on_load_missing = [r"h\.\d+\.attn\.causal_mask"]

    def __init__(self, config):
        super().__init__(config)
        if not hasattr(config, 'wbits'):
            config.wbits = 32
            config.groupsize = 128

        # if config.wbits not in [4, 8, 32]:
        #     logger.warning(f'Specify `wbits` with 4, 8 or 32 to load the model. ')
        if config.wbits in [4, 8]:
            def noop():
                pass

            mindspore.common.initializer.HeUniform = noop
            mindspore.ops.uniform = noop
            mindspore.common.initializer.Normal = noop

            # torch.set_default_dtype(mindspore.half)

            self._init_weights = False
            # torch.set_default_dtype(mindspore.half)
        self.transformer = MossModel(config)
        self.lm_head = nn.Dense(config.n_embd, config.vocab_size)
        if config.wbits in [4, 8]:
            # torch.set_default_dtype(mindspore.float32)
            self._init_weights = True
            self.quantize(config.wbits, config.groupsize)
        # Initialize weights and apply final processing
        self.post_init()

    def get_output_embeddings(self):
        """
        get output embeddings
        """
        return self.lm_head

    def set_output_embeddings(self, new_embeddings):
        """
        set output embeddings
        """
        self.lm_head = new_embeddings

    def prepare_inputs_for_generation(self, input_ids, past_key_values=None, **kwargs):
        """
        Prepare inputs for the generation task.
        """
        token_type_ids = kwargs.get("token_type_ids", None)
        # only last token for inputs_ids if past is defined in kwargs
        if past_key_values:
            input_ids = ops.unsqueeze(input_ids[:, -1], dim=-1)
            if token_type_ids is not None:
                ops.unsqueeze(token_type_ids[:, -1], dim=-1)
        attention_mask = kwargs.get("attention_mask", None)
        position_ids = kwargs.get("position_ids", None)

        if attention_mask is not None and position_ids is None:
            # create position_ids on the fly for batch generation
            position_ids = attention_mask.long().cumsum(-1) - 1
            position_ids.masked_fill_(attention_mask == 0, 1)
            if past_key_values:
                position_ids = ops.unsqueeze(position_ids[:, -1], dim=-1)

        return {
            "input_ids": input_ids,
            "past_key_values": past_key_values,
            "use_cache": kwargs.get("use_cache"),
            "position_ids": position_ids,
            "attention_mask": attention_mask,
            "token_type_ids": token_type_ids,
        }

    def construct(
            self,
            input_ids: Optional[Tensor] = None,
            past_key_values: Optional[Tuple[Tuple[Tensor]]] = None,
            attention_mask: Optional[Tensor] = None,
            token_type_ids: Optional[Tensor] = None,
            position_ids: Optional[Tensor] = None,
            head_mask: Optional[Tensor] = None,
            inputs_embeds: Optional[Tensor] = None,
            labels: Optional[Tensor] = None,
            use_cache: Optional[bool] = None,
            output_attentions: Optional[bool] = None,
            output_hidden_states: Optional[bool] = None,
            return_dict: Optional[bool] = None,
    ) -> Union[Tuple, Tuple]:
        r"""
        labels (`Tensor(dtype=mindspore.int64)` of shape `(batch_size, sequence_length)`, *optional*):
            Labels for language modeling. Note that the labels **are shifted** inside the model, i.e. you can set
            `labels = input_ids` Indices are selected in `[-100, 0, ..., config.vocab_size]` All labels set to `-100`
            are ignored (masked), the loss is only computed for labels in `[0, ..., config.vocab_size]`
        """
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        outputs = self.transformer(
            input_ids,
            past_key_values=past_key_values,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )
        hidden_states = outputs[0]

        # make sure sampling in fp16 works correctly and
        # compute loss in fp32 to match with mesh-tf version
        # https://github.com/EleutherAI/gpt-neo/blob/89ce74164da2fb16179106f54e2269b5da8db333/models/gpt2/gpt2.py#L179
        logits = self.lm_head(hidden_states).to(mindspore.float32)

        loss = None
        if labels is not None:
            # Shift so that tokens < n predict n
            shift_logits = logits[..., :-1, :]
            shift_labels = labels[..., 1:]
            # Flatten the tokens
            loss_fct = CrossEntropyLoss()
            loss = loss_fct(
                shift_logits.view(-1, shift_logits.shape[-1]), shift_labels.view(-1))

            loss = loss.to(hidden_states.dtype)

        if not return_dict:
            output = (logits,) + outputs[1:]
            return ((loss,) + output) if loss is not None else output

        return CausalLMOutputWithPast(
            loss=loss,
            logits=logits,
            past_key_values=outputs.past_key_values,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )

    @staticmethod
    def _reorder_cache(
            past_key_values: Tuple[Tuple[Tensor]], beam_idx: Tensor
    ) -> Tuple[Tuple[Tensor]]:
        """
        This function is used to re-order the `past_key_values` cache if [`~PretrainedModel.beam_search`] or
        [`~PretrainedModel.beam_sample`] is called. This is required to match `past_key_values` with the correct
        beam_idx at every generation step.
        """
        return tuple(
            tuple(past_state.index_select(0, beam_idx.to(past_state.device))
                  for past_state in layer_past)
            for layer_past in past_key_values
        )

    def quantize(self, wbits, groupsize):
        """
        Function to quantize a model using GPTQ.
        """
