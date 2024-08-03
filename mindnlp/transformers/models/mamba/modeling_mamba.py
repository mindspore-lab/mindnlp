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
"""MindSpore MAMBA model."""

import math
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple, Union

import mindspore
from mindspore import Parameter
from mindspore.common.initializer import initializer, Normal, Uniform, HeUniform

from mindnlp.core import nn, ops
from mindnlp.core.nn import functional as F
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

MAMBA_PRETRAINED_MODEL_ARCHIVE_LIST = []  # See all Mamba models at https://hf-mirror.com/models?filter=mamba


class MambaMixer(nn.Module):
    """
    Compute ∆, A, B, C, and D the state space parameters and compute the `contextualized_states`.
    A, D are input independent (see Mamba paper [1] Section 3.5.2 "Interpretation of A" for why A isn't selective)
    ∆, B, C are input-dependent (this is a key difference between Mamba and the linear time invariant S4,
    and is why Mamba is called **selective** state spaces)
    """
    def __init__(self, config, layer_idx):
        """
        Initializes a MambaMixer instance.
        
        Args:
            self (MambaMixer): The MambaMixer instance.
            config:
                A configuration object containing the following attributes:

                - hidden_size (int): The size of the hidden layer.
                - state_size (int): The size of the state.
                - conv_kernel (int): The size of the convolutional kernel.
                - intermediate_size (int): The size of the intermediate layer.
                - time_step_rank (int): The rank of the time step.
                - use_conv_bias (bool): Specifies whether to use bias in convolution.
                - hidden_act (str): The activation function for the hidden layer.
            layer_idx (int): The index of the layer.

        Returns:
            None.

        Raises:
            ValueError: If any of the configuration attributes are invalid or missing.
            TypeError: If the input types are incorrect.
        """
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
            bias=config.use_conv_bias,
            kernel_size=config.conv_kernel,
            groups=self.intermediate_size,
            padding=config.conv_kernel - 1,
        )

        self.activation = config.hidden_act
        self.act = ACT2FN[config.hidden_act]

        # projection of the input hidden states-
        self.in_proj = nn.Linear(self.hidden_size, self.intermediate_size * 2, bias=config.use_bias)
        # selective projection used to make dt, B and C input dependant
        self.x_proj = nn.Linear(self.intermediate_size, self.time_step_rank + self.ssm_state_size * 2, bias=False)
        # time step projection (discretization)
        self.dt_proj = nn.Linear(self.time_step_rank, self.intermediate_size, bias=True)

        # S4D real initialization. These are not discretized!
        # The core is to load them, compute the discrete states, then write the updated state. Keeps the memory bounded
        A = ops.arange(1, self.ssm_state_size + 1, dtype=mindspore.float32)[None, :]
        A = A.broadcast_to((self.intermediate_size, -1))

        self.A_log = Parameter(ops.log(A))
        self.D = Parameter(ops.ones(self.intermediate_size))
        self.out_proj = nn.Linear(self.intermediate_size, self.hidden_size, bias=config.use_bias)
        self.use_bias = config.use_bias

    # fmt: off
    def forward(self, input_states, cache_params=None):

        """
        Constructs contextualized states based on input states and cache parameters.

        Args:
            self (MambaMixer): The instance of the MambaMixer class.
            input_states (Tensor): The input states with shape (batch_size, seq_len, _),
                where batch_size is the number of sequences, seq_len is the maximum sequence length, and _ is
                the dimension of the input feature.
            cache_params (CacheParams, optional): The cache parameters containing states for caching computations,
                defaults to None.

        Returns:
            Tensor: The contextualized states with shape (batch_size, seq_len, output_size),
                where output_size is the size of the output.

        Raises:
            ValueError: If the input_states shape is invalid or if the cache_params is not None and
                does not contain the required states.
            TypeError: If the input_states or cache_params are not of the expected types.
        """
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
                conv_state = ops.roll(conv_state, shifts=-1, dims=-1)
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
            ssm_parameters, [self.time_step_rank, self.ssm_state_size, self.ssm_state_size], dim=-1
        )
        discrete_time_step = self.dt_proj(time_step)                                    # [batch, seq_len, intermediate_size]
        discrete_time_step = F.softplus(discrete_time_step).swapaxes(1, 2) # [batch, intermediate_size, seq_len]

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
        scan_output = ops.stack(scan_outputs, dim=-1)                                # [batch, seq_len, intermediade_size]
        scan_output = scan_output + (hidden_states * self.D[None, :, None])
        scan_output = scan_output * self.act(gate)

        if cache_params is not None:
            cache_params.ssm_states[self.layer_idx] = ssm_state

        # 4. Final linear projection
        contextualized_states = self.out_proj(scan_output.swapaxes(1, 2))             # [batch, seq_len, hidden_size]
        return contextualized_states

class MambaCache:

    """
    MambaCache class represents a cache for storing intermediate states during Mamba model training.

    This class inherits from [parent class].

    Attributes:
        seqlen_offset (int): The offset for sequence length.
        dtype (mindspore.dtype): The data type used for the cache.
        conv_states (dict): A dictionary storing convolutional states for each hidden layer.
        ssm_states (dict): A dictionary storing SSM (Spatio-spectral modulation) states for each hidden layer.

    Args:
        config: The configuration for the Mamba model.
        batch_size (int): The size of the input batch.
        dtype: The data type to be used, default is mindspore.float16.
    """
    def __init__(self, config, batch_size, dtype=mindspore.float16):

        """
        Initialize the MambaCache class.

        Args:
            self: The instance of the class.
            config (Config): Configuration object containing model parameters.
            batch_size (int): The size of the input batch.
            dtype (mindspore.dtype, optional): Data type for the tensors (default: mindspore.float16).

        Returns:
            None.

        Raises:
            None.
        """
        self.seqlen_offset = 0
        self.dtype = dtype
        intermediate_size = config.intermediate_size
        ssm_state_size = config.state_size
        conv_kernel_size = config.conv_kernel

        self.conv_states = {
            i: ops.zeros(batch_size, intermediate_size, conv_kernel_size, dtype=dtype)
            for i in range(config.num_hidden_layers)
        }
        self.ssm_states = {
            i: ops.zeros(batch_size, intermediate_size, ssm_state_size, dtype=dtype)
            for i in range(config.num_hidden_layers)
        }


class MambaRMSNorm(nn.Module):

    """
    MambaRMSNorm is a neural network cell that represents a modified version of the RMS normalization layer.
    It inherits from the nn.Module class and provides functionality for normalizing hidden states in a neural network.

    This class initializes the MambaRMSNorm layer with the specified hidden size and epsilon value for variance.
    The hidden_size parameter determines the size of the input hidden states, while the eps parameter sets the variance
    epsilon value for numerical stability.

    The forward method of MambaRMSNorm takes hidden_states as input and performs RMS normalization on the input
    hidden states. It first converts the input hidden states to float32 data type, calculates the variance of the
    hidden states, and then applies the RMS normalization using the variance and epsilon values.
    The normalized hidden states are then multiplied by the weight parameter and converted back to the original input
    data type before being returned.

    Note:
        The implementation details and usage of this class should be referenced from the source code and any
        related documentation.
    """
    def __init__(self, hidden_size, eps=1e-6):
        """
        LlamaRMSNorm is equivalent to T5LayerNorm
        """
        super().__init__()
        self.weight = Parameter(ops.ones(hidden_size))
        self.variance_epsilon = eps

    def forward(self, hidden_states):

        """
        This method forwards the MambaRMSNorm operation by normalizing the hidden states.

        Args:
            self (MambaRMSNorm): The instance of the MambaRMSNorm class.
            hidden_states (Tensor): The input hidden states to be normalized.
                It should be a tensor containing the hidden states data.

        Returns:
            None.

        Raises:
            ValueError: If the input hidden_states tensor is not valid.
            RuntimeError: If there is an issue during the normalization process.
        """
        input_dtype = hidden_states.dtype
        hidden_states = hidden_states.to(mindspore.float32)
        variance = hidden_states.pow(2).mean(-1, keep_dims=True)
        hidden_states = hidden_states * ops.rsqrt(variance + self.variance_epsilon)
        return self.weight * hidden_states.to(input_dtype)


class MambaBlock(nn.Module):

    """
    The MambaBlock class represents a block used in the Mamba neural network model for processing hidden states.
    This class inherits from nn.Module and contains methods for initializing the block and forwarding the
    block's operations.

    Attributes:
        config: A dictionary containing configuration parameters for the block.
        layer_idx: An integer representing the index of the layer within the neural network.
        residual_in_fp32: A boolean indicating whether the residual input is in float32 format.
        norm: An instance of MambaRMSNorm for performing layer normalization on hidden states.
        mixer: An instance of MambaMixer for mixing the normalized hidden states.

    Methods:
        __init__: Initializes the MambaBlock instance with the provided configuration and layer index.
        forward: Constructs the block by processing hidden states through normalization, mixing,
            and addition of residuals.

    Example:
        ```python
        >>> # Example of initializing and using the MambaBlock class
        >>> config = {'hidden_size': 512, 'layer_norm_epsilon': 1e-5, 'residual_in_fp32': True}
        >>> block = MambaBlock(config, layer_idx=1)
        >>> output = block.forward(hidden_states)
        ```
    """
    def __init__(self, config, layer_idx):

        """
        Initializes a MambaBlock instance.

        Args:
            self (MambaBlock): The MambaBlock instance itself.
            config (Config): An object containing configuration settings for the block.
            layer_idx (int): Index of the layer within the block.

        Returns:
            None.

        Raises:
            None.
        """
        super().__init__()
        self.config = config
        self.layer_idx = layer_idx
        self.residual_in_fp32 = config.residual_in_fp32
        self.norm = MambaRMSNorm(config.hidden_size, eps=config.layer_norm_epsilon)
        self.mixer = MambaMixer(config, layer_idx=layer_idx)

    def forward(self, hidden_states, cache_params=None):

        """
        This method forwards a MambaBlock by performing a series of operations on the input hidden_states.

        Args:
            self: The instance of the MambaBlock class.
            hidden_states: A tensor representing the input hidden states. It is the main input to the forward method.
            cache_params: Optional parameter. A dictionary containing cache parameters. Default is None.

        Returns:
            None: This method does not return any value directly, but it updates the hidden_states tensor in place.

        Raises:
            TypeError: If the hidden_states parameter is not a valid tensor.
            ValueError: If the cache_params parameter is provided but is not a valid dictionary.
            RuntimeError: If there is a runtime error during the execution of the method.
        """
        residual = hidden_states
        hidden_states = self.norm(hidden_states.to(dtype=self.norm.weight.dtype))
        if self.residual_in_fp32:
            residual = residual.to(mindspore.float32)

        hidden_states = self.mixer(hidden_states, cache_params=cache_params)
        hidden_states = residual + hidden_states
        return hidden_states


class MambaPreTrainedModel(PreTrainedModel):
    """
    An abstract class to handle weights initialization and a simple interface for downloading and loading pretrained
    models.
    """
    config_class = MambaConfig
    base_model_prefix = "backbone"
    _no_split_modules = ["MambaBlock"]
    supports_gradient_checkpointing = True

    def _init_weights(self, cell):
        """Initialize the weights."""
        if isinstance(cell, MambaMixer):
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

        if isinstance(cell, nn.Linear):
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
                    # Following MindSpore init, except scale by 1/sqrt(2 * n_layer)
                    # We need to reinit p since this code could be called multiple times
                    # Having just p *= scale would repeatedly scale it down
                    p.set_data(initializer(HeUniform(math.sqrt(5)), p.shape, p.dtype) / math.sqrt(self.config.num_layers))


@dataclass
class MambaOutput(ModelOutput):
    """
    Class for the MAMBA model outputs.

    Args:
        last_hidden_state (`mindspore.Tensor` of shape `(batch_size, sequence_length, hidden_size)`):
            Sequence of hidden-states at the output of the last layer of the model.
        cache_params (list of five `mindspore.Tensor` of shape `(batch_size, hidden_size, num_hidden_layers)`):
            The state of the model at the last time step. Can be used in a forward method with the next `input_ids` to
            avoid providing the old `input_ids`.

            Includes both the State space model states weights after the selective scan, and the Convolutional states
        hidden_states (`tuple(mindspore.Tensor)`, *optional*, returned when `output_hidden_states=True`
            is passed or when `config.output_hidden_states=True`):
            Tuple of `mindspore.Tensor` (one for the output of the embeddings, if the model has an embedding layer, +
            one for the output of each layer) of shape `(batch_size, sequence_length, hidden_size)`.

            Hidden-states of the model at the output of each layer plus the optional initial embedding outputs.
    """
    last_hidden_state: mindspore.Tensor = None
    cache_params: Optional[List[mindspore.Tensor]] = None
    hidden_states: Optional[Tuple[mindspore.Tensor]] = None


@dataclass
class MambaCausalLMOutput(ModelOutput):
    """
    Base class for causal language model (or autoregressive) outputs.

    Args:
        loss (`mindspore.Tensor` of shape `(1,)`, *optional*, returned when `labels` is provided):
            Language modeling loss (for next-token prediction).
        logits (`mindspore.Tensor` of shape `(batch_size, sequence_length, config.vocab_size)`):
            Prediction scores of the language modeling head (scores for each vocabulary token before SoftMax).
        cache_params (list of five `mindspore.Tensor` of shape `(batch_size, hidden_size, num_hidden_layers)`):
            The state of the model at the last time step. Can be used in a forward method with the next `input_ids` to
            avoid providing the old `input_ids`.
        hidden_states (`tuple(mindspore.Tensor)`, *optional*, returned when `output_hidden_states=True` is passed or
            when `config.output_hidden_states=True`):
            Tuple of `mindspore.Tensor` (one for the output of the embeddings, if the model has an embedding layer, +
            one for the output of each layer) of shape `(batch_size, sequence_length, hidden_size)`.

            Hidden-states of the model at the output of each layer plus the optional initial embedding outputs.
    """
    loss: Optional[mindspore.Tensor] = None
    logits: mindspore.Tensor = None
    cache_params: Optional[List[mindspore.Tensor]] = None
    hidden_states: Optional[Tuple[mindspore.Tensor]] = None


class MambaModel(MambaPreTrainedModel):

    """
    A class representing the MambaModel.

    This class is a Python implementation of the MambaModel, which is a deep learning model used for various
    natural language processing tasks. The MambaModel inherits from the MambaPreTrainedModel class.

    Attributes:
        embeddings (nn.Embedding): An instance of the nn.Embedding class representing the input embeddings.
        layers (nn.ModuleList): A list of MambaBlock instances representing the layers of the model.
        gradient_checkpointing (bool): A flag indicating whether gradient checkpointing is used during training.
        norm_f (MambaRMSNorm): An instance of the MambaRMSNorm class representing the normalization function.

    Methods:
        __init__: Initializes the MambaModel instance.
        get_input_embeddings: Returns the input embeddings.
        set_input_embeddings: Sets the input embeddings to the specified value.
        forward: Constructs the MambaModel.
    """
    def __init__(self, config):

        """
        Initializes an instance of the MambaModel class.

        Args:
            self: The instance of the class.
            config: An object that holds the configuration parameters for the model.
                It should have the following attributes:

                - vocab_size (int): The size of the vocabulary.
                - hidden_size (int): The size of the hidden state.
                - num_hidden_layers (int): The number of hidden layers in the model.
                - layer_norm_epsilon (float): The epsilon value used in layer normalization.

        Returns:
            None.

        Raises:
            None.
        """
        super().__init__(config)

        self.embeddings = nn.Embedding(config.vocab_size, config.hidden_size)
        self.layers = nn.ModuleList([MambaBlock(config, layer_idx=idx) for idx in range(config.num_hidden_layers)])

        self.gradient_checkpointing = False
        self.norm_f = MambaRMSNorm(config.hidden_size, eps=config.layer_norm_epsilon)
        # Initialize weights and apply final processing
        self.post_init()

    def get_input_embeddings(self):

        """
        Method to retrieve the input embeddings from the MambaModel instance.

        Args:
            self: The MambaModel instance itself.

        Returns:
            embeddings:
                The input embeddings associated with the MambaModel instance.

        Raises:
            None.
        """
        return self.embeddings

    def set_input_embeddings(self, new_embeddings):

        """
        Sets the input embeddings for the MambaModel.

        Args:
            self (MambaModel): The instance of the MambaModel class.
            new_embeddings (Any): The new input embeddings to be set. This parameter can be of any type.

        Returns:
            None.

        Raises:
            None.
        """
        self.embeddings = new_embeddings

    def forward(
        self,
        input_ids: Optional[mindspore.Tensor] = None,
        inputs_embeds: Optional[mindspore.Tensor] = None,
        cache_params: Optional[List[mindspore.Tensor]] = None,
        use_cache: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        **kwargs,  # `attention_mask` is passed by the tokenizer and we don't want it
    ) -> Union[Tuple, MambaOutput]:

        """
        This method forwards the MambaModel by processing input data through multiple mixer blocks.

        Args:
            self: The instance of the MambaModel class.
            input_ids (Optional[mindspore.Tensor]): The input tensor containing token indices. Default is None.
            inputs_embeds (Optional[mindspore.Tensor]): The input tensor containing pre-computed embeddings.
                Default is None.
            cache_params (Optional[List[mindspore.Tensor]]): List of tensors used for caching intermediate states.
                Default is None.
            use_cache (Optional[bool]): Flag indicating whether to use caching. Default is None.
            output_hidden_states (Optional[bool]): Flag indicating whether to output hidden states. Default is None.
            return_dict (Optional[bool]): Flag indicating whether to return the output as a dictionary. Default is None.

        Returns:
            Union[Tuple, MambaOutput]: The return value can be either a tuple containing hidden states, cache parameters,
                and all hidden states (if not None), or a MambaOutput object containing the last hidden state,
                cache parameters (if caching is enabled), and all hidden states.

        Raises:
            ValueError: Raised if both input_ids and inputs_embeds are specified simultaneously.
        """
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        use_cache = use_cache if use_cache is not None else (self.config.use_cache if not self.training else False)
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        if (input_ids is None) ^ (inputs_embeds is not None):  # ^ is python for xor
            raise ValueError(
                "You cannot specify both input_ids and inputs_embeds at the same time, and must specify either one"
            )

        if inputs_embeds is None:
            inputs_embeds = self.embeddings(input_ids)

        if self.gradient_checkpointing and self.training and use_cache:
            use_cache = False

        if cache_params is None and use_cache:
            cache_params = MambaCache(
                self.config, inputs_embeds.shape[0], dtype=inputs_embeds.dtype
            )

        hidden_states = inputs_embeds
        all_hidden_states = () if output_hidden_states else None
        for mixer_block in self.layers:
            hidden_states = mixer_block(hidden_states, cache_params=cache_params)

            if output_hidden_states:
                all_hidden_states = all_hidden_states + (hidden_states,)

        if use_cache:
            cache_params.seqlen_offset += inputs_embeds.shape[1]

        hidden_states = self.norm_f(hidden_states)

        if output_hidden_states:
            all_hidden_states = all_hidden_states + (hidden_states,)

        if not return_dict:
            return tuple(v for v in [hidden_states, cache_params, all_hidden_states] if v is not None)

        return MambaOutput(
            last_hidden_state=hidden_states,
            cache_params=cache_params if use_cache else None,
            hidden_states=all_hidden_states,
        )


class MambaForCausalLM(MambaPreTrainedModel):

    """
    This class represents a Mamba model for Causal Language Modeling (LM), which is a subclass of MambaPreTrainedModel.

    The class includes methods for initializing the model, getting and setting the output embeddings, getting and
    setting the input embeddings, updating model keyword arguments for generation, preparing inputs for generation,
    and forwarding the model for LM tasks.

    The 'forward' method takes input_ids, inputs_embeds, cache_params, labels, output_hidden_states, and return_dict
    as input parameters, and returns the model output for Causal LM tasks. It calculates the loss if labels are provided
    and returns the loss along with the logits and other relevant outputs.

    The class also provides functionality to handle cache_params, hidden states, and embedding tensors during the
    model's execution for LM tasks.
    """
    _tied_weights_keys = ["lm_head.weight"]

    def __init__(self, config):

        """
        Initializes the MambaForCausalLM class.

        Args:
            self: The instance of the MambaForCausalLM class.
            config: An object containing the configuration settings for the model.

        Returns:
            None.

        Raises:
            None.
        """
        super().__init__(config)
        self.backbone = MambaModel(config)
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
        # Initialize weights and apply final processing
        self.post_init()

    def get_output_embeddings(self):

        """
        Returns the output embeddings for the MambaForCausalLM model.

        Args:
            self (MambaForCausalLM): The instance of the MambaForCausalLM class.

        Returns:
            lm_head: This method returns the 'lm_head' attribute of the MambaForCausalLM instance,
                which represents the output embeddings.

        Raises:
            None.
        """
        return self.lm_head

    def set_output_embeddings(self, new_embeddings):

        """
        Sets the output embeddings for the MambaForCausalLM model.

        Args:
            self (MambaForCausalLM): An instance of the MambaForCausalLM class.
            new_embeddings (torch.Tensor): A tensor containing the new output embeddings to be set.

        Returns:
            None: This method updates the output embeddings of the MambaForCausalLM model in-place.

        Raises:
            None.
        """
        self.lm_head = new_embeddings

    def get_input_embeddings(self):

        """
        This method retrieves the input embeddings from the MambaForCausalLM model's backbone.

        Args:
            self (MambaForCausalLM): The MambaForCausalLM instance itself.

        Returns:
            None.

        Raises:
            None.
        """
        return self.backbone.get_input_embeddings()

    def set_input_embeddings(self, new_embeddings):

        """
        Sets the input embeddings for the MambaForCausalLM model.

        Args:
            self (MambaForCausalLM): The instance of the MambaForCausalLM class.
            new_embeddings (object): The new input embeddings to be set for the model.
                It can be of any valid type that is compatible with the model's input requirements.

        Returns:
            None.

        Raises:
            TypeError: If the new_embeddings parameter is of an incompatible type.
            ValueError: If the new_embeddings parameter does not meet the required criteria for input embeddings.
            RuntimeError: If an unexpected error occurs while setting the input embeddings.
        """
        return self.backbone.set_input_embeddings(new_embeddings)

    def _update_model_kwargs_for_generation(
        self, outputs: ModelOutput, model_kwargs: Dict[str, Any], **kwargs
    ) -> Dict[str, Any]:

        """
        Updates the model keyword arguments for generation based on the provided outputs.

        Args:
            self (MambaForCausalLM): The instance of the MambaForCausalLM class.
            outputs (ModelOutput): The output object containing the generated model outputs.
            model_kwargs (Dict[str, Any]): The dictionary of keyword arguments for the model.
            **kwargs: Additional keyword arguments.

        Returns:
            Dict[str, Any]: The updated model keyword arguments.

        Raises:
            None.

        This method updates the model_kwargs dictionary with the 'cache_params' key-value pair from the outputs object,
        if present. If 'cache_params' is not present in the outputs object, it sets the corresponding value in
        model_kwargs to None. The method then returns the updated model_kwargs dictionary.
        """
        model_kwargs["cache_params"] = outputs.get("cache_params", None)
        return model_kwargs

    def prepare_inputs_for_generation(
        self, input_ids, cache_params=None, inputs_embeds=None, **kwargs
    ):

        """
        This method prepares inputs for text generation in the MambaForCausalLM class.

        Args:
            self (MambaForCausalLM): The instance of the MambaForCausalLM class.
            input_ids (Tensor): The input tensor containing token indices of the input sequence.
            cache_params (Optional[Dict]): A dictionary containing parameters for caching computations.
            inputs_embeds (Optional[Tensor]): The embeddings of the input tokens, if provided.

        Returns:
            dict: A dictionary containing either 'input_ids' or 'inputs_embeds' based on the conditions
                specified in the method. Additionally, 'cache_params' is included in the dictionary.

        Raises:
            None
        """
        # only last token for inputs_ids if the state is passed along.
        if cache_params is not None:
            input_ids = input_ids[:, -1].unsqueeze(-1)

        if inputs_embeds is not None and cache_params is None:
            model_inputs = {"inputs_embeds": inputs_embeds}
        else:
            model_inputs = {"input_ids": input_ids}

        model_inputs["cache_params"] = cache_params
        return model_inputs

    def forward(
        self,
        input_ids: Optional[mindspore.Tensor] = None,
        inputs_embeds: Optional[mindspore.Tensor] = None,
        cache_params: Optional[mindspore.Tensor] = None,
        labels: Optional[mindspore.Tensor] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        **kwargs,  # for now we need this for generation
    ) -> Union[Tuple, MambaCausalLMOutput]:
        r"""
        Args:
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
        hidden_states = mamba_outputs[0]

        logits = self.lm_head(hidden_states.to(self.lm_head.weight.dtype)).float()

        loss = None
        if labels is not None:
            # move labels to correct device to enable model parallelism
            # Shift so that tokens < n predict n
            shift_logits = logits[..., :-1, :]
            shift_labels = labels[..., 1:]
            # Flatten the tokens
            loss = F.cross_entropy(shift_logits.view(-1, shift_logits.shape[-1]), shift_labels.view(-1))

        if not return_dict:
            output = (logits,) + mamba_outputs[1:]
            return ((loss,) + output) if loss is not None else output

        return MambaCausalLMOutput(
            loss=loss,
            logits=logits,
            cache_params=mamba_outputs.cache_params,
            hidden_states=mamba_outputs.hidden_states,
        )

__all__ = [
    "MAMBA_PRETRAINED_MODEL_ARCHIVE_LIST",
    "MambaForCausalLM",
    "MambaModel",
    "MambaPreTrainedModel",
]
