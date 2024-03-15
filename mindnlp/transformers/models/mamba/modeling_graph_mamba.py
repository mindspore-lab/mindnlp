# coding=utf-8
# Copyright 2024 state-spaces/mamba org and HuggingFace Inc. team.
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
# pylint: disable=invalid-name
# pylint: disable=invalid-unary-operand-type
# pylint: disable=missing-class-docstring
# pylint: disable=unused-argument
"""MindSpore MAMBA model."""

import math
from typing import Any, Dict, List, Optional, Tuple, Union
from easydict import EasyDict

import mindspore
from mindspore import nn, ops, Parameter
from mindspore.common.initializer import initializer, Normal, Uniform, HeUniform

from mindnlp.utils import (
    ModelOutput,
    logging,
)
from ...activations import ACT2FN
from ...modeling_utils import PreTrainedModel
from .configuration_mamba import MambaConfig


logger = logging.get_logger(__name__)


_CHECKPOINT_FOR_DOC = "state-spaces/mamba-130m-hf"
_CONFIG_FOR_DOC = "MambaConfig"

MAMBA_PRETRAINED_MODEL_ARCHIVE_LIST = []  # See all MSMamba models at https://huggingface.co/models?filter=mamba

class MambaDense(nn.Dense):
    def construct(self, x):
        x_shape = x.shape
        if len(x_shape) != 2:
            x = x.reshape(-1, x.shape[-1])
        x = ops.matmul(x, self.weight.T)
        if self.has_bias:
            x = ops.add(x, self.bias)
        if len(x_shape) != 2:
            out_shape = x_shape[:-1] + (x.shape[-1],)
            x = x.reshape(out_shape)
        return x

class MSMambaMixer(nn.Cell):
    """
    Compute ∆, A, B, C, and D the state space parameters and compute the `contextualized_states`.
    A, D are input independent (see MSMamba paper [1] Section 3.5.2 "Interpretation of A" for why A isn't selective)
    ∆, B, C are input-dependent (this is a key difference between MSMamba and the linear time invariant S4,
    and is why MSMamba is called **selective** state spaces)
    """

    def __init__(self, config, layer_idx):
        super().__init__()
        self.hidden_size = config.hidden_size
        self.ssm_state_size = config.state_size
        self.conv_kernel_size = config.conv_kernel
        self.intermediate_size = config.intermediate_size
        self.time_step_rank = config.time_step_rank
        self.layer_idx = layer_idx
        self.use_conv_bias = config.use_conv_bias
        self.conv1d = nn.Conv1d(
            in_channels=self.intermediate_size,
            out_channels=self.intermediate_size,
            has_bias=config.use_conv_bias,
            kernel_size=config.conv_kernel,
            group=self.intermediate_size,
            padding=config.conv_kernel - 1,
            pad_mode='pad'
        )

        self.activation = config.hidden_act
        self.act = ACT2FN[config.hidden_act]

        # projection of the input hidden states
        self.in_proj = MambaDense(self.hidden_size, self.intermediate_size * 2, has_bias=config.use_bias)
        # selective projection used to make dt, B and C input dependant
        self.x_proj = MambaDense(self.intermediate_size, self.time_step_rank + self.ssm_state_size * 2, has_bias=False)
        # time step projection (discretization)
        self.dt_proj = MambaDense(self.time_step_rank, self.intermediate_size, has_bias=True)

        # S4D real initialization. These are not discretized!
        # The core is to load them, compute the discrete states, then write the updated state. Keeps the memory bounded
        A = ops.arange(1, self.ssm_state_size + 1, dtype=mindspore.float32)[None, :]
        A = A.expand(self.intermediate_size, -1)

        self.A_log = Parameter(ops.log(A))
        self.D = Parameter(ops.ones(self.intermediate_size))
        self.out_proj = MambaDense(self.intermediate_size, self.hidden_size, has_bias=config.use_bias)
        self.use_bias = config.use_bias

    # fmt: off
    def construct(self, input_states, cache_params=None):
        batch_size, seq_len, _ = input_states.shape
        dtype = input_states.dtype
        # 1. Gated MLP's linear projection
        projected_states = self.in_proj(input_states).swapaxes(1, 2)                   # [batch, 2 * intermediate_size, seq_len]
        hidden_states, gate = projected_states.chunk(2, axis=1)

        # 2. Convolution sequence transformation
        if cache_params is not None:
            ssm_state = cache_params.ssm_states[self.layer_idx]
            if cache_params.seqlen_offset > 0:
                conv_state = cache_params.conv_states[self.layer_idx]                   # [batch, intermediate_size, conv_kernel_size]
                conv_state = mindspore.numpy.roll(conv_state, shift=-1, axis=-1)
                conv_state[:, :, -1] = hidden_states[:, :, 0]
                cache_params.conv_states[self.layer_idx] = conv_state
                hidden_states = ops.sum(conv_state * self.conv1d.weight[:, 0, :], dim=-1)
                if self.use_conv_bias:
                    hidden_states += self.conv1d.bias
                hidden_states = self.act(hidden_states).to(dtype).unsqueeze(-1)         # [batch, intermediate_size, 1] : decoding
            else:
                conv_state = ops.pad(
                    hidden_states,
                    (self.conv_kernel_size - hidden_states.shape[-1], 0)
                )
                cache_params.conv_states[self.layer_idx] = conv_state
                hidden_states = self.act(self.conv1d(hidden_states)[..., :seq_len])     # [batch, intermediate_size, seq_len]
        else:
            ssm_state = ops.zeros(
                (batch_size, self.intermediate_size, self.ssm_state_size), dtype=dtype
            )
            hidden_states = self.act(self.conv1d(hidden_states)[..., :seq_len])         # [batch, intermediate_size, seq_len]

        # 3. State Space Model sequence transformation
        # 3.a. Selection:  [batch, seq_len, self.time_step_rank + self.ssm_state_size * 2]
        ssm_parameters = self.x_proj(hidden_states.swapaxes(1, 2))
        time_step, B, C = ops.split(
            ssm_parameters, [self.time_step_rank, self.ssm_state_size, self.ssm_state_size], axis=-1
        )
        discrete_time_step = self.dt_proj(time_step)                                    # [batch, seq_len, intermediate_size]
        discrete_time_step = ops.softplus(discrete_time_step).swapaxes(1, 2) # [batch, intermediate_size, seq_len]

        # 3.b. Discretization: B and C to [batch, seq_len, intermediate_size, ssm_state_size] (SRAM)
        A = -ops.exp(self.A_log.float())                                              # [intermediate_size, ssm_state_size]
        discrete_A = ops.exp(A[None, :, None, :] * discrete_time_step[:, :, :, None]) # [batch, intermediate_size, seq_len, ssm_state_size]
        discrete_B = discrete_time_step[:, :, :, None] * B[:, None, :, :].float()       # [batch, intermediade_size, seq_len, ssm_state_size]
        deltaB_u = discrete_B * hidden_states[:, :, :, None].float()

        # 3.c perform the recurrence y ← SSM(A, B, C)(x)
        scan_outputs = []
        for i in range(seq_len):
            ssm_state = discrete_A[:, :, i, :] * ssm_state + deltaB_u[:, :, i, :]      # [batch, intermediade_size, ssm_state]
            scan_output = ops.matmul(ssm_state.to(dtype), C[:, i, :].unsqueeze(-1))  # [batch, intermediade_size, 1]
            scan_outputs.append(scan_output[:, :, 0])
        scan_output = ops.stack(scan_outputs, axis=-1)                                # [batch, seq_len, intermediade_size]
        scan_output = scan_output + (hidden_states * self.D[None, :, None])
        scan_output = scan_output * self.act(gate)

        if cache_params is not None:
            cache_params.ssm_states[self.layer_idx] = ssm_state

        # 4. Final linear projection
        contextualized_states = self.out_proj(scan_output.swapaxes(1, 2))             # [batch, seq_len, hidden_size]
        return contextualized_states

@mindspore.jit_class
class MSMambaCache:
    def __init__(self, config, batch_size, dtype=mindspore.float16):
        self.seqlen_offset = Parameter(0)
        self.dtype = dtype
        intermediate_size = config.intermediate_size
        ssm_state_size = config.state_size
        conv_kernel_size = config.conv_kernel

        self.conv_states = Parameter(ops.zeros(config.num_hidden_layers, batch_size, intermediate_size, conv_kernel_size, dtype=dtype), name='conv_states')
        # {
        #     i: ops.zeros(batch_size, intermediate_size, conv_kernel_size, dtype=dtype)
        #     for i in range(config.num_hidden_layers)
        # }
        self.ssm_states = Parameter(ops.zeros(config.num_hidden_layers, batch_size, intermediate_size, ssm_state_size, dtype=dtype), name='ssm_states')
        # {
        #     i: ops.zeros(batch_size, intermediate_size, ssm_state_size, dtype=dtype)
        #     for i in range(config.num_hidden_layers)
        # }


class MSMambaRMSNorm(nn.Cell):
    def __init__(self, hidden_size, eps=1e-6):
        """
        LlamaRMSNorm is equivalent to T5LayerNorm
        """
        super().__init__()
        self.weight = Parameter(ops.ones(hidden_size))
        self.variance_epsilon = eps

    def construct(self, hidden_states):
        input_dtype = hidden_states.dtype
        hidden_states = hidden_states.to(mindspore.float32)
        variance = hidden_states.pow(2).mean(-1, keep_dims=True)
        hidden_states = hidden_states * ops.rsqrt(variance + self.variance_epsilon)
        return self.weight * hidden_states.to(input_dtype)


class MSMambaBlock(nn.Cell):
    def __init__(self, config, layer_idx):
        super().__init__()
        self.config = config
        self.layer_idx = layer_idx
        self.residual_in_fp32 = config.residual_in_fp32
        self.norm = MSMambaRMSNorm(config.hidden_size, eps=config.layer_norm_epsilon)
        self.mixer = MSMambaMixer(config, layer_idx=layer_idx)

    def construct(self, hidden_states, cache_params=None):
        residual = hidden_states
        hidden_states = self.norm(hidden_states.to(dtype=self.norm.weight.dtype))
        if self.residual_in_fp32:
            residual = residual.to(mindspore.float32)

        hidden_states = self.mixer(hidden_states, cache_params=cache_params)
        hidden_states = residual + hidden_states
        return hidden_states


class MSMambaPreTrainedModel(PreTrainedModel):
    """
    An abstract class to handle weights initialization and a simple interface for downloading and loading pretrained
    models.
    """

    config_class = MambaConfig
    base_model_prefix = "backbone"
    _no_split_modules = ["MSMambaBlock"]
    supports_gradient_checkpointing = True

    def _init_weights(self, cell):
        """Initialize the weights."""
        if isinstance(cell, MSMambaMixer):
            cell.A_log._no_weight_decay = True
            cell.D._no_weight_decay = True

            dt_init_std = self.config.time_step_rank**-0.5 * self.config.time_step_scale
            if self.config.time_step_init_scheme == "constant":
                cell.dt_proj.weight[:] = dt_init_std
            elif self.config.time_step_init_scheme == "random":
                cell.dt_proj.weight.set_data(initializer(Uniform(dt_init_std), cell.dt_proj.weight.shape, cell.dt_proj.weight.dtype))

            dt = ops.exp(
                ops.rand(self.config.intermediate_size)
                * (math.log(self.config.time_step_max) - math.log(self.config.time_step_min))
                + math.log(self.config.time_step_min)
            ).clamp(min=self.config.time_step_floor)
            # # Inverse of softplus: https://github.com/pytorch/pytorch/issues/72759
            inv_dt = dt + ops.log(-ops.expm1(-dt))
            cell.dt_proj.bias[:] = inv_dt
            cell.dt_proj.bias._no_reinit = True

        if isinstance(cell, MambaDense):
            if cell.bias is not None:
                if not getattr(cell.bias, "_no_reinit", False):
                    cell.bias[:] = 0
        elif isinstance(cell, nn.Embedding):
            cell.weight.set_data(initializer(Normal(self.config.initializer_range), cell.weight.shape, cell.weight.dtype))

        if self.config.rescale_prenorm_residual:
            # Reinitialize selected weights subject to the OpenAI GPT-2 Paper Scheme:
            #   > A modified initialization which accounts for the accumulation on the residual path with model depth. Scale
            #   > the weights of residual layers at initialization by a factor of 1/√N where N is the # of residual layers.
            #   >   -- GPT-2 :: https://openai.com/blog/better-language-models/
            #
            # Reference (Megatron-LM): https://github.com/NVIDIA/Megatron-LM/blob/main/megatron/model/gpt_model.py
            for name, p in cell.parameters_and_names():
                if name in ["out_proj.weight"]:
                    # Special Scaled Initialization --> There are 2 Layer Norms per Transformer Block
                    # Following Pytorch init, except scale by 1/sqrt(2 * n_layer)
                    # We need to reinit p since this code could be called multiple times
                    # Having just p *= scale would repeatedly scale it down
                    p.set_data(initializer(HeUniform(math.sqrt(5)), p.shape, p.dtype) / math.sqrt(self.config.num_layers))

    def __call__(self, *args, **kwargs):
        outputs = super().__call__(*args, **kwargs)
        if isinstance(outputs, dict):
            return EasyDict(outputs)
        return outputs

class MSMambaModel(MSMambaPreTrainedModel):
    def __init__(self, config):
        super().__init__(config)

        self.embeddings = nn.Embedding(config.vocab_size, config.hidden_size)
        self.layers = nn.CellList([MSMambaBlock(config, layer_idx=idx) for idx in range(config.num_hidden_layers)])

        self.gradient_checkpointing = False
        self.norm_f = MSMambaRMSNorm(config.hidden_size, eps=config.layer_norm_epsilon)
        # Initialize weights and apply final processing
        self.post_init()

    def get_input_embeddings(self):
        return self.embeddings

    def set_input_embeddings(self, new_embeddings):
        self.embeddings = new_embeddings

    def construct(
        self,
        input_ids: Optional[mindspore.Tensor] = None,
        inputs_embeds: Optional[mindspore.Tensor] = None,
        cache_params: Optional[List[mindspore.Tensor]] = None,
        use_cache: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        **kwargs,  # `attention_mask` is passed by the tokenizer and we don't want it
    ) -> Union[Tuple, Dict]:
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        use_cache = use_cache if use_cache is not None else (self.config.use_cache if not self.training else False)
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        # if (input_ids is None) ^ (inputs_embeds is not None):  # ^ is python for xor
        #     raise ValueError(
        #         "You cannot specify both input_ids and inputs_embeds at the same time, and must specify either one"
        #     )

        if inputs_embeds is None:
            inputs_embeds = self.embeddings(input_ids)

        if self.gradient_checkpointing and self.training and use_cache:
            use_cache = False

        if cache_params is None and use_cache:
            cache_params = MSMambaCache(
                self.config, inputs_embeds.shape[0], dtype=inputs_embeds.dtype
            )

        hidden_states = inputs_embeds
        all_hidden_states = () if output_hidden_states else None
        for mixer_block in self.layers:
            hidden_states = mixer_block(hidden_states, cache_params=cache_params)

            if output_hidden_states:
                all_hidden_states = all_hidden_states + (hidden_states,)

        if use_cache:
            ops.assign(cache_params.seqlen_offset, cache_params.seqlen_offset + inputs_embeds.shape[1])

        hidden_states = self.norm_f(hidden_states)

        if output_hidden_states:
            all_hidden_states = all_hidden_states + (hidden_states,)

        if not return_dict:
            return tuple(v for v in [hidden_states, cache_params, all_hidden_states] if v is not None)

        return {
            'last_hidden_state': hidden_states,
            'cache_params': cache_params if use_cache else None,
            'hidden_states': all_hidden_states,
        }


class MSMambaForCausalLM(MSMambaPreTrainedModel):
    _tied_weights_keys = ["lm_head.weight"]

    def __init__(self, config):
        super().__init__(config)
        self.backbone = MSMambaModel(config)
        self.lm_head = MambaDense(config.hidden_size, config.vocab_size, has_bias=False)
        # Initialize weights and apply final processing
        self.post_init()

    def get_output_embeddings(self):
        return self.lm_head

    def set_output_embeddings(self, new_embeddings):
        self.lm_head = new_embeddings

    def get_input_embeddings(self):
        return self.backbone.get_input_embeddings()

    def set_input_embeddings(self, new_embeddings):
        return self.backbone.set_input_embeddings(new_embeddings)

    def _update_model_kwargs_for_generation(
        self, outputs: ModelOutput, model_kwargs: Dict[str, Any], **kwargs
    ) -> Dict[str, Any]:
        model_kwargs["cache_params"] = outputs.get("cache_params", None)
        return model_kwargs

    def prepare_inputs_for_generation(
        self, input_ids, cache_params=None, inputs_embeds=None, **kwargs
    ):
        # only last token for inputs_ids if the state is passed along.
        if cache_params is not None:
            input_ids = input_ids[:, -1].unsqueeze(-1)

        if inputs_embeds is not None and cache_params is None:
            model_inputs = {"inputs_embeds": inputs_embeds}
        else:
            model_inputs = {"input_ids": input_ids}

        model_inputs["cache_params"] = cache_params
        return model_inputs

    @mindspore.jit
    def construct(
        self,
        input_ids: Optional[mindspore.Tensor] = None,
        inputs_embeds: Optional[mindspore.Tensor] = None,
        cache_params: Optional[mindspore.Tensor] = None,
        labels: Optional[mindspore.Tensor] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        **kwargs,  # for now we need this for generation
    ) -> Union[Tuple, Dict]:
        r"""
        labels (`mindspore.Tensor` of shape `(batch_size, sequence_length)`, *optional*):
            Labels for language modeling. Note that the labels **are shifted** inside the model, i.e. you can set
            `labels = input_ids` Indices are selected in `[-100, 0, ..., config.vocab_size]` All labels set to `-100`
            are ignored (masked), the loss is only computed for labels in `[0, ..., config.vocab_size]`
        """
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        mamba_outputs = self.backbone(
            input_ids,
            cache_params=cache_params,
            inputs_embeds=inputs_embeds,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )
        if return_dict:
            hidden_states = mamba_outputs['last_hidden_state']
        else:
            hidden_states = mamba_outputs[0]

        logits = self.lm_head(hidden_states.to(self.lm_head.weight.dtype)).float()

        loss = None
        if labels is not None:
            # move labels to correct device to enable model parallelism
            # Shift so that tokens < n predict n
            shift_logits = logits[..., :-1, :]
            shift_labels = labels[..., 1:]
            # Flatten the tokens
            loss = ops.cross_entropy(shift_logits.view(-1, shift_logits.shape[-1]), shift_labels.view(-1))

        if not return_dict:
            output = (logits,) + mamba_outputs[1:]
            return ((loss,) + output) if loss is not None else output

        return {
            'loss': loss,
            'logits': logits,
            'cache_params': mamba_outputs['cache_params'],
            'hidden_states': mamba_outputs['hidden_states'],
        }

__all__ = [
    "MAMBA_PRETRAINED_MODEL_ARCHIVE_LIST",
    "MSMambaForCausalLM",
    "MSMambaModel",
    "MSMambaPreTrainedModel",
]
