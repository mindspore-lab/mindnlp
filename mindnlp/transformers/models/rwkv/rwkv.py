# coding=utf-8
# Copyright 2023 Bo Peng and HuggingFace Inc. team.
# Copyright (c) 2018, NVIDIA CORPORATION.  All rights reserved.
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
# ============================================================================

# pylint: disable=C0103
# pylint: disable=C0415
# pylint: disable=W0613
# pylint: disable=W0223

"""MindSpore RWKV model."""

import math
from typing import List, Optional, Tuple
import numpy as np

import mindspore
from mindspore import nn, ops
from mindspore import Tensor, Parameter
from mindspore.nn import CrossEntropyLoss
from mindspore import log as logger

from ...modeling_utils import PreTrainedModel
from .rwkv_config import RwkvConfig

RWKV_PRETRAINED_MODEL_ARCHIVE_LIST = [
    "RWKV/rwkv-4-169m-pile",
    "RWKV/rwkv-4-430m-pile",
    "RWKV/rwkv-4-1b5-pile",
    "RWKV/rwkv-4-3b-pile",
    "RWKV/rwkv-4-7b-pile",
    "RWKV/rwkv-4-14b-pile",
    "RWKV/rwkv-raven-1b5",
    "RWKV/rwkv-raven-3b",
    "RWKV/rwkv-raven-7b",
    "RWKV/rwkv-raven-14b",
    # See all RWKV models at https://huggingface.co/models?filter=rwkv
]

WKV_SHAPE_INFER = {
    'wkv_forward': lambda w, u, k, v: k,
    'wkv_forward_with_state': lambda w, u, k, v, s: k,
    'wkv_backward': lambda w_w, w_u, k, v, gy: ((k[0], k[2]), (k[0], k[2]), k, v),
}

WKV_DTYPE_INFER = {
    'wkv_forward': lambda w, u, k, v: k,
    'wkv_forward_with_state': lambda w, u, k, v, s: k,
    'wkv_backward': lambda w_w, w_u, k, v, gy: (w_w, w_u, k, v),
}

def load_wkv_cuda_kernel(func_name, context_length):
    """load wkv cuda kernel"""
    device_target = mindspore.get_context('device_target')
    if device_target != 'GPU':
        raise RuntimeError('WKV operator only support GPU currently.')

    logger.info(f"Loading CUDA kernel for RWKV at context length of {context_length}.")

    from ...kernel_utils import compile_kernel
    so_path = compile_kernel(Tmax=context_length)
    wkv_op = ops.Custom(
        str(so_path) + ':' + func_name,
        out_shape=WKV_SHAPE_INFER[func_name],
        out_dtype=WKV_DTYPE_INFER[func_name],
        func_type='aot'
    )
    wkv_op.add_prim_attr('primitive_target', device_target)
    return wkv_op


class RwkvLinearAttention(nn.Cell):
    """RWKV linear attention"""
    def __init__(self, config):
        super().__init__()
        self.max_seq_length = config.context_length
        self.return_state = config.use_cache
        if self.return_state:
            self.wkv_forward = load_wkv_cuda_kernel('wkv_forward_with_state', config.context_length)
        else:
            self.wkv_forward = load_wkv_cuda_kernel('wkv_forward', config.context_length)

        self.wkv_backward = load_wkv_cuda_kernel('wkv_backward', config.context_length)

    def construct(self, time_decay, time_first, key, value, state=None):
        batch_size, seq_len, hidden_size = key.shape
        if seq_len > self.max_seq_length:
            raise ValueError(
                f"Cannot process a batch with {seq_len} tokens at the same time, use a maximum of "
                f"{self.max_seq_length} with this model."
            )
        if batch_size * hidden_size % min(hidden_size, 32) != 0:
            raise ValueError(
                f"The product of batch size ({batch_size}) and hidden size ({hidden_size}) needs to be a round "
                f"multiple of {min(hidden_size, 32)}."
            )

        input_dtype = key.dtype

        time_decay = ops.neg(ops.exp(time_decay.astype(mindspore.float32)))
        if key.dtype == mindspore.float16:
            time_first = time_first.astype(mindspore.float32)
            key = key.astype(mindspore.float32)
            value = value.astype(mindspore.float32)
        # The CUDA kernel will fill this tensor.

        if self.return_state:
            if state is None:
                state = ops.zeros((batch_size, hidden_size, 3), dtype=mindspore.float32)
                state[:, :, 2] -= 1e38
            else:
                state = ops.cat([s.expand_dims(2) for s in state], axis=2)
            output = self.wkv_forward(time_decay, time_first, key, value, state)
        else:
            output = self.wkv_forward(time_decay, time_first, key, value)

        if state is not None:
            state = [s.squeeze(2) for s in ops.chunk(state, 3, axis=2)]

        return output.astype(input_dtype), state

    # g stands for grad
    def bprop(self, w, u, k, v, s, y, gy):
        """bporp for wkv"""
        dtype = k.dtype
        k = k.astype(mindspore.float32)
        v = v.astype(mindspore.float32)
        gy = gy.astype(mindspore.float32)
        gw, gu, gk, gv = self.wkv_backward(w, u, k, v, gy)
        gw = ops.sum(gw, 0)
        gu = ops.sum(gu, 0)

        return (gw, gu, gk.astype(dtype), gv.astype(dtype))


class RwkvSelfAttention(nn.Cell):
    """RWKV self attention"""
    def __init__(self, config, layer_id=0):
        super().__init__()
        self.config = config
        self.rwkv_linear_attention = RwkvLinearAttention(config)

        self.layer_id = layer_id
        hidden_size = config.hidden_size
        attention_hidden_size = (
            config.attention_hidden_size if config.attention_hidden_size is not None else hidden_size
        )
        self.attention_hidden_size = attention_hidden_size

        self.time_decay = Parameter(Tensor(np.zeros(attention_hidden_size), mindspore.float32), 'time_decay')
        self.time_first = Parameter(Tensor(np.zeros(attention_hidden_size), mindspore.float32), 'time_decay')

        self.time_mix_key = Parameter(Tensor(np.zeros((1, 1, hidden_size)), mindspore.float32), 'time_mix_key')
        self.time_mix_value = Parameter(Tensor(np.zeros((1, 1, hidden_size)), mindspore.float32), 'time_mix_value')
        self.time_mix_receptance = Parameter(Tensor(np.zeros((1, 1, hidden_size)), mindspore.float32), 'time_mix_receptance')

        self.time_shift = nn.ZeroPad2d((0, 0, 1, -1))
        self.key = nn.Dense(hidden_size, attention_hidden_size, has_bias=False)
        self.value = nn.Dense(hidden_size, attention_hidden_size, has_bias=False)
        self.receptance = nn.Dense(hidden_size, attention_hidden_size, has_bias=False)
        self.output = nn.Dense(attention_hidden_size, hidden_size, has_bias=False)

    def extract_key_value(self, hidden, state=None):
        """extrac key value"""
        # Mix hidden with the previous timestep to produce key, value, receptance
        if hidden.shape[1] == 1 and state is not None:
            shifted = state[1][:, :, self.layer_id]
        else:
            shifted = self.time_shift(hidden)
            if state is not None:
                shifted[:, 0] = state[1][:, :, self.layer_id]
        key = hidden * self.time_mix_key + shifted * (1 - self.time_mix_key)
        value = hidden * self.time_mix_value + shifted * (1 - self.time_mix_value)
        receptance = hidden * self.time_mix_receptance + shifted * (1 - self.time_mix_receptance)

        key = self.key(key)
        value = self.value(value)
        receptance = ops.sigmoid(self.receptance(receptance))
        if state is not None:
            state[1][:, :, self.layer_id] = hidden[:, -1]
        return receptance, key, value, state

    def construct(self, hidden, state=None):
        receptance, key, value, state = self.extract_key_value(hidden, state=state)
        layer_state = tuple(s[:, :, self.layer_id] for s in state[2:]) if state is not None else None
        rwkv, layer_state = self.rwkv_linear_attention(
            self.time_decay,
            self.time_first,
            key,
            value,
            state=layer_state,
        )

        if layer_state is not None:
            print(layer_state)
            state[2][:, :, self.layer_id] = layer_state[0]
            state[3][:, :, self.layer_id] = layer_state[1]
            state[4][:, :, self.layer_id] = layer_state[2]

        return self.output(receptance * rwkv), state


class RwkvFeedForward(nn.Cell):
    """RWKV feed forward"""
    def __init__(self, config, layer_id=0):
        super().__init__()
        self.config = config
        self.layer_id = layer_id
        hidden_size = config.hidden_size
        intermediate_size = (
            config.intermediate_size if config.intermediate_size is not None else 4 * config.hidden_size
        )

        self.time_shift = nn.ZeroPad2d((0, 0, 1, -1))
        self.time_mix_key = Parameter(Tensor(np.zeros((1, 1, hidden_size)), mindspore.float32), 'time_mix_key')
        self.time_mix_receptance =Parameter(Tensor(np.zeros((1, 1, hidden_size)), mindspore.float32), 'time_mix_receptance')

        self.key = nn.Dense(hidden_size, intermediate_size, has_bias=False)
        self.receptance = nn.Dense(hidden_size, hidden_size, has_bias=False)
        self.value = nn.Dense(intermediate_size, hidden_size, has_bias=False)

    def construct(self, hidden, state=None):
        if hidden.shape[1] == 1 and state is not None:
            shifted = state[0][:, :, self.layer_id]
        else:
            shifted = self.time_shift(hidden)
            if state is not None:
                shifted[:, 0] = state[0][:, :, self.layer_id]
        key = hidden * self.time_mix_key + shifted * (1 - self.time_mix_key)
        receptance = hidden * self.time_mix_receptance + shifted * (1 - self.time_mix_receptance)

        key = ops.square(ops.relu(self.key(key)))
        value = self.value(key)
        receptance = ops.sigmoid(self.receptance(receptance))

        if state is not None:
            state[0][:, :, self.layer_id] = hidden[:, -1]

        return receptance * value, state


class RwkvBlock(nn.Cell):
    """RWKV block"""
    def __init__(self, config, layer_id):
        super().__init__()
        self.config = config
        self.layer_id = layer_id

        if layer_id == 0:
            self.pre_ln = nn.LayerNorm([config.hidden_size], epsilon=config.layer_norm_epsilon)

        self.ln1 = nn.LayerNorm([config.hidden_size], epsilon=config.layer_norm_epsilon)
        self.ln2 = nn.LayerNorm([config.hidden_size], epsilon=config.layer_norm_epsilon)

        self.attention = RwkvSelfAttention(config, layer_id)
        self.feed_forward = RwkvFeedForward(config, layer_id)
        self.output_attentions = self.config.output_attentions

    def construct(self, hidden, state=None):
        if self.layer_id == 0:
            hidden = self.pre_ln(hidden)

        attention, state = self.attention(self.ln1(hidden), state=state)
        hidden = hidden + attention

        feed_forward, state = self.feed_forward(self.ln2(hidden), state=state)
        hidden = hidden + feed_forward

        outputs = (hidden, state)
        if self.output_attentions:
            outputs += (attention,)
        else:
            outputs += (None,)

        return outputs


class RwkvPreTrainedModel(PreTrainedModel):
    """
    An abstract class to handle weights initialization and a simple interface for downloading and loading pretrained
    models.
    """

    config_class = RwkvConfig
    base_model_prefix = "rwkv"
    _no_split_modules = ["RwkvBlock"]
    _keep_in_fp32_modules = ["time_decay", "time_first"]

    def _init_weights(self, cell):
        """Initialize the weights."""
        if isinstance(cell, RwkvSelfAttention):
            layer_id = cell.layer_id
            num_hidden_layers = cell.config.num_hidden_layers
            hidden_size = cell.config.hidden_size
            attention_hidden_size = cell.attention_hidden_size

            ratio_0_to_1 = layer_id / (num_hidden_layers - 1)  # 0 to 1
            ratio_1_to_almost0 = 1.0 - (layer_id / num_hidden_layers)  # 1 to ~0

            time_weight = Tensor(
                [i / hidden_size for i in range(hidden_size)],
                dtype=cell.time_mix_key.dtype,
            )
            time_weight = time_weight[None, None, :]

            decay_speed = [
                -5 + 8 * (h / (attention_hidden_size - 1)) ** (0.7 + 1.3 * ratio_0_to_1)
                for h in range(attention_hidden_size)
            ]
            decay_speed = Tensor(decay_speed, dtype=cell.time_decay.dtype)
            zigzag = (
                Tensor(
                    [(i + 1) % 3 - 1 for i in range(attention_hidden_size)],
                    dtype=cell.time_first.dtype
                )
                * 0.5
            )

            cell.time_decay.set_data(decay_speed)
            cell.time_first.set_data(ops.ones_like(cell.time_first * math.log(0.3) + zigzag))

            cell.time_mix_key.set_data(ops.pow(time_weight, ratio_1_to_almost0))
            cell.time_mix_value.set_data(ops.pow(time_weight, ratio_1_to_almost0) + 0.3 * ratio_0_to_1)
            cell.time_mix_receptance.set_data(ops.pow(time_weight, 0.5 * ratio_1_to_almost0))

        elif isinstance(cell, RwkvFeedForward):
            layer_id = cell.layer_id
            num_hidden_layers = cell.config.num_hidden_layers
            hidden_size = cell.config.hidden_size

            ratio_1_to_almost0 = 1.0 - (layer_id / num_hidden_layers)  # 1 to ~0

            time_weight = Tensor(
                [i / hidden_size for i in range(hidden_size)],
                dtype=cell.time_mix_key.dtype
            )
            time_weight = time_weight[None, None, :]

            cell.time_mix_key.set_data(ops.pow(time_weight, ratio_1_to_almost0))
            cell.time_mix_receptance.set_data(ops.pow(time_weight, ratio_1_to_almost0))


class RwkvModel(RwkvPreTrainedModel):
    """RWKV Model"""
    def __init__(self, config):
        super().__init__(config)

        self.embeddings = nn.Embedding(config.vocab_size, config.hidden_size)
        self.blocks = nn.CellList([RwkvBlock(config, layer_id=idx) for idx in range(config.num_hidden_layers)])
        self.ln_out = nn.LayerNorm([config.hidden_size])

        self.layers_are_rescaled = False
        self.output_attentions = self.config.output_attentions
        self.output_hidden_states = self.config.output_hidden_states
        self.use_cache = self.config.use_cache
        # Initialize weights and apply final processing
        self.post_init()

    def get_input_embeddings(self):
        return self.embeddings

    def set_input_embeddings(self, new_embeddings):
        self.embeddings = new_embeddings

    def __call__(self, *args, **kwargs):
        if self.training == self.layers_are_rescaled:
            self._rescale_layers()
        return super().__call__(*args, **kwargs)

    def construct(
        self,
        input_ids: Optional[Tensor] = None,
        attention_mask: Optional[Tensor] = None,  # noqa
        inputs_embeds: Optional[Tensor] = None,
        state: Optional[List[Tensor]] = None,
        use_cache: Optional[bool] = None,
    ) -> Tuple:
        if input_ids is not None and inputs_embeds is not None:
            raise ValueError("You cannot specify both input_ids and inputs_embeds at the same time")
        if input_ids is None and inputs_embeds is None:
            raise ValueError("You have to specify either input_ids or inputs_embeds")

        if inputs_embeds is None:
            inputs_embeds = self.embeddings(input_ids)

        if use_cache and state is None:
            shape = (inputs_embeds.shape[0], self.config.hidden_size, self.config.num_hidden_layers)
            state = [
                ops.zeros(shape, dtype=inputs_embeds.dtype if i <= 1 else mindspore.float32)
                for i in range(5)
            ]
            state[4] -= 1e30

        hidden_states = inputs_embeds

        all_self_attentions = ()
        all_hidden_states = ()
        for idx, block in enumerate(self.blocks):
            hidden_states, state, attentions = block(hidden_states, state=state)
            if (
                self.layers_are_rescaled
                and self.config.rescale_every > 0
                and (idx + 1) % self.config.rescale_every == 0
            ):
                hidden_states = hidden_states / 2

            if self.output_hidden_states:
                all_hidden_states = all_hidden_states + (hidden_states,)

            if self.output_attentions:
                all_self_attentions = all_self_attentions + (attentions,)

        hidden_states = self.ln_out(hidden_states)

        if self.output_hidden_states:
            all_hidden_states = all_hidden_states + (hidden_states,)

        return tuple(x for x in [hidden_states, state, all_hidden_states, all_self_attentions] if x is not None)

    def _rescale_layers(self):
        # Layers should be rescaled for inference only.
        if self.layers_are_rescaled == (not self.training):
            return
        if self.config.rescale_every > 0:
            for block_id, block in enumerate(self.blocks):
                if self.training:
                    block.attention.output.weight.set_data(block.attention.output.weight * \
                                                           (2 ** int(block_id // self.config.rescale_every)))
                    block.feed_forward.value.weight.set_data(block.feed_forward.value.weight * \
                                                             (2 ** int(block_id // self.config.rescale_every)))
                else:
                    # Deal with quantization statistics
                    if hasattr(block.attention.output.weight, "SCB"):
                        block.attention.output.weight.SCB.set_data(block.attention.output.weight.SCB / \
                                                                   (2 ** int(block_id // self.config.rescale_every)))
                        block.feed_forward.value.weight.SCB.set_data(block.feed_forward.value.weight.SCB / \
                                                                     (2 ** int(block_id // self.config.rescale_every)))
                    else:
                        block.attention.output.weight.set_data(block.attention.output.weight / \
                                                            (2 ** int(block_id // self.config.rescale_every)))
                        block.feed_forward.value.weight.set_data(block.feed_forward.value.weight / \
                                                                (2 ** int(block_id // self.config.rescale_every)))
        self.layers_are_rescaled = not self.training


class RwkvForCausalLM(RwkvPreTrainedModel):
    """RWKV for causal LM"""
    def __init__(self, config):
        super().__init__(config)
        self.rwkv = RwkvModel(config)
        self.head = nn.Dense(config.hidden_size, config.vocab_size, has_bias=False)

        # Initialize weights and apply final processing
        self.post_init()

    def get_output_embeddings(self):
        """get output embeddings"""
        return self.head

    def set_output_embeddings(self, new_embeddings):
        """set output embeddings"""
        self.head = new_embeddings

    def prepare_inputs_for_generation(self, input_ids, state=None, inputs_embeds=None, **kwargs):
        """prepare inputs"""
        # only last token for inputs_ids if the state is passed along.
        if state is not None:
            input_ids = input_ids[:, -1].unsqueeze(-1)

        # if `inputs_embeds` are passed, we only want to use them in the 1st generation step
        if inputs_embeds is not None and state is None:
            model_inputs = {"inputs_embeds": inputs_embeds}
        else:
            model_inputs = {"input_ids": input_ids}

        model_inputs["state"] = state
        return model_inputs

    def construct(
        self,
        input_ids: Optional[Tensor] = None,
        attention_mask: Optional[Tensor] = None,  # noqa
        inputs_embeds: Optional[Tensor] = None,
        state: Optional[List[Tensor]] = None,
        labels: Optional[Tensor] = None
    ) -> Tuple:
        r"""
        labels (`Tensor` of shape `(batch_size, sequence_length)`, *optional*):
            Labels for language modeling. Note that the labels **are shifted** inside the model, i.e. you can set
            `labels = input_ids` Indices are selected in `[-100, 0, ..., config.vocab_size]` All labels set to `-100`
            are ignored (masked), the loss is only computed for labels in `[0, ..., config.vocab_size]`
        """
        rwkv_outputs = self.rwkv(
            input_ids,
            inputs_embeds=inputs_embeds,
            state=state,
        )
        hidden_states = rwkv_outputs[0]

        logits = self.head(hidden_states)

        loss = None
        if labels is not None:
            # move labels to correct device to enable model parallelism
            # Shift so that tokens < n predict n
            shift_logits = logits[..., :-1, :]
            shift_labels = labels[..., 1:]
            # Flatten the tokens
            loss_fct = CrossEntropyLoss()
            loss = loss_fct(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))

        output = (logits,) + rwkv_outputs[1:]
        return ((loss,) + output) if loss is not None else output

__all__ = ['RwkvModel', 'RwkvForCausalLM']
