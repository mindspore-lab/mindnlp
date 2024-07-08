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
"""
MindNLP Graphormer model
"""

import math
import logging
from typing import Iterable, Iterator, List, Optional, Tuple, Union

import numpy as np
import mindspore as ms

from mindspore import nn, ops
from mindspore.nn import BCEWithLogitsLoss, CrossEntropyLoss, MSELoss
from mindspore import Parameter, Tensor
from mindspore.common.initializer import Uniform

from ...activations import ACT2FN
from ...modeling_outputs import (
    BaseModelOutputWithNoAttention,
    SequenceClassifierOutput,
)
from ...modeling_utils import PreTrainedModel
from .configuration_graphormer import GraphormerConfig
from .utils import init_zero, init_normal, init_xavier_uniform


logger = logging.getLogger(__name__)

_CHECKPOINT_FOR_DOC = "graphormer-base-pcqm4mv1"
_CONFIG_FOR_DOC = "GraphormerConfig"


GRAPHORMER_PRETRAINED_MODEL_ARCHIVE_LIST = [
    "clefourrier/graphormer-base-pcqm4mv1",
    "clefourrier/graphormer-base-pcqm4mv2",
    # See all Graphormer models at https://hf-mirror.com/models?filter=graphormer
]


def quant_noise(module: nn.Cell, q_noise: float, block_size: int):
    """
    From:
    https://github.com/facebookresearch/fairseq/blob/dd0079bde7f678b0cd0715cbd0ae68d661b7226d/fairseq/modules/quant_noise.py

    Wraps modules and applies quantization noise to the weights for subsequent quantization with Iterative Product
    Quantization as described in "Training with Quantization Noise for Extreme Model Compression"

    Args:
        module: nn.Cell
        q_noise: amount of Quantization Noise
        block_size: size of the blocks for subsequent quantization with iPQ

    Notes:
        - Module weights must have the right sizes wrt the block size
        - Only Linear, Embedding and Conv2d modules are supported for the moment
        - For more detail on how to quantize by blocks with convolutional weights, see "And the Bit Goes Down:
        Revisiting the Quantization of Neural Networks"
        - We implement the simplest form of noise here as stated in the paper which consists in randomly dropping
        blocks
    """
    # if no quantization noise, don't register hook
    if q_noise <= 0:
        return module

    # supported modules
    if not isinstance(module, (nn.Dense, nn.Embedding, nn.Conv2d)):
        raise NotImplementedError("Module unsupported for quant_noise.")

    # test whether module.weight has the right sizes wrt block_size
    is_conv = module.weight.ndim == 4

    # 2D matrix
    if not is_conv:
        if module.weight.shape[1] % block_size != 0:
            raise AssertionError("Input features must be a multiple of block sizes")

    # 4D matrix
    else:
        # 1x1 convolutions
        if module.kernel_size == (1, 1):
            if module.in_channels % block_size != 0:
                raise AssertionError("Input channels must be a multiple of block sizes")
        # regular convolutions
        else:
            k = module.kernel_size[0] * module.kernel_size[1]
            if k % block_size != 0:
                raise AssertionError("Kernel size must be a multiple of block size")

    def _forward_pre_hook(mod, input_var):
                # no noise for evaluation
        if mod.training:
            if not is_conv:
                # gather weight and sizes
                weight = mod.weight
                in_features = weight.shape[1]
                out_features = weight.shape[0]

                # split weight matrix into blocks and randomly drop selected blocks
                mask = ops.zeros(in_features // block_size * out_features)
                mask.bernoulli_(q_noise)
                mask = mask.repeat_interleave(block_size, -1).view(-1, in_features)

            else:
                # gather weight and sizes
                weight = mod.weight
                in_channels = mod.in_channels
                out_channels = mod.out_channels

                # split weight matrix into blocks and randomly drop selected blocks
                if mod.kernel_size == (1, 1):
                    mask = ops.zeros(
                        int(in_channels // block_size * out_channels))
                    mask.bernoulli_(q_noise)
                    mask = mask.repeat_interleave(block_size, -1).view(-1, in_channels)
                else:
                    mask = ops.zeros(weight.shape[0], weight.shape[1])
                    mask.bernoulli_(q_noise)
                    mask = mask.unsqueeze(2).unsqueeze(3).repeat(1, 1, mod.kernel_size[0], mod.kernel_size[1])

            # scale weights and apply mask
            mask = mask.bool()
            mod.weight.data = 1 / (1 - q_noise) * weight.masked_fill(mask, 0)

    module.register_forward_pre_hook(_forward_pre_hook)
    return module


class LayerDropModuleList(nn.CellList):
    """
    From:
    https://github.com/facebookresearch/fairseq/blob/dd0079bde7f678b0cd0715cbd0ae68d661b7226d/fairseq/modules/layer_drop.py
    A LayerDrop implementation based on [`mindspore.nn.CellList`]. LayerDrop as described in
    https://arxiv.org/abs/1909.11556.

    We refresh the choice of which layers to drop every time we iterate over the LayerDropModuleList instance.
    During evaluation we always iterate over all layers.

    Example:
        ```python
        >>> layers = LayerDropList(p_drop=0.5, modules=[layer1, layer2, layer3])
        >>> for layer in layers:  # this might iterate over layers 1 and 3
        >>>     x = layer(x
        >>> for layer in layers:  # this might iterate over all layers
        >>>     x = layer(x)
        >>> for layer in layers:  # this might not iterate over any layers
        >>>     x = layer(x)
        ```

    Args:
        p_drop (float): probability of dropping out each layer
        modules (iterable, optional): an iterable of modules to add
    """
    def __init__(self, p_drop: float, modules: Optional[Iterable[nn.Cell]] = None):
        """
        Initialize a LayerDropModuleList object with the provided parameters.

        Args:
            self (object): The current instance of the LayerDropModuleList class.
            p_drop (float): The probability of dropping a module during training. Must be a float value.
            modules (Optional[Iterable[nn.Cell]]): An optional iterable of neural network modules to be included
                in the list. Defaults to None if not provided. Each module should be an instance of nn.Cell.

        Returns:
            None.

        Raises:
            None.
        """
        super().__init__(modules)
        self.p_drop = p_drop

    def __iter__(self) -> Iterator[nn.Cell]:
        """
        This method '__iter__' in the class 'LayerDropModuleList' serves as an iterator to iterate through the cells
        within the module list.

        Args:
            self: The instance of the 'LayerDropModuleList' class.
                It is used to access the attributes and methods of the class.

        Returns:
            An iterator of type 'Iterator[nn.Cell]' that yields the cells within the module list.

        Raises:
            None.
        """
        dropout_probs = Tensor(shape=(len(self)), dtype=ms.float32, init=Uniform())
        for i, cell in enumerate(super().__iter__()):
            if not self.training or (dropout_probs[i] > self.p_drop):
                yield cell


class GraphormerGraphNodeFeature(nn.Cell):
    """
    Compute node features for each node in the graph.
    """
    def __init__(self, config: GraphormerConfig):
        """
        Initialize the GraphormerGraphNodeFeature class.

        Args:
            self (GraphormerGraphNodeFeature): The instance of the GraphormerGraphNodeFeature class.
            config (GraphormerConfig):
                An instance of GraphormerConfig containing the configuration parameters.

                - num_attention_heads (int): Number of attention heads.
                - num_atoms (int): Number of atoms.
                - num_in_degree (int): Number of in-degrees.
                - num_out_degree (int): Number of out-degrees.
                - hidden_size (int): Size of the hidden layers.
                - pad_token_id (int): Token ID for padding.

        Returns:
            None.

        Raises:
            None.
        """
        super().__init__()
        self.num_heads = config.num_attention_heads
        self.num_atoms = config.num_atoms

        self.atom_encoder = nn.Embedding(config.num_atoms + 1, config.hidden_size, padding_idx=config.pad_token_id)
        self.in_degree_encoder = nn.Embedding(
            config.num_in_degree, config.hidden_size, padding_idx=config.pad_token_id
        )
        self.out_degree_encoder = nn.Embedding(
            config.num_out_degree, config.hidden_size, padding_idx=config.pad_token_id
        )

        self.graph_token = nn.Embedding(1, config.hidden_size)

    def construct(
        self,
        input_nodes: Tensor,
        in_degree: Tensor,
        out_degree: Tensor,
    ) -> Tensor:
        """
        Constructs graph node features based on input nodes, in-degree, and out-degree information.

        Args:
            self: Instance of the GraphormerGraphNodeFeature class.
            input_nodes (Tensor): Tensor containing input node features for each graph.
                Shape: (n_graph, n_nodes, n_features).
            in_degree (Tensor): Tensor representing the in-degree of each node in the graph.
                Shape: (n_graph, n_nodes).
            out_degree (Tensor): Tensor representing the out-degree of each node in the graph.
                Shape: (n_graph, n_nodes).

        Returns:
            Tensor: A tensor representing the graph node features after encoding and aggregation.
                Shape: (n_graph, n_nodes, feature_dim).

        Raises:
            None
        """
        n_graph, _ = input_nodes.shape[:2]

        node_feature = (  # node feature + graph token
            self.atom_encoder(input_nodes).sum(axis=-2)  # [n_graph, n_node, n_hidden]
            + self.in_degree_encoder(in_degree)
            + self.out_degree_encoder(out_degree)
        )

        graph_token_feature = self.graph_token.weight.unsqueeze(0).tile((n_graph, 1, 1))

        graph_node_feature = ops.cat([graph_token_feature, node_feature], axis=1)

        return graph_node_feature


class GraphormerGraphAttnBias(nn.Cell):
    """
    Compute attention bias for each head.
    """
    def __init__(self, config: GraphormerConfig):
        """
        This method initializes the GraphormerGraphAttnBias class with the provided configuration.

        Args:
            self: The instance of the GraphormerGraphAttnBias class.
            config (GraphormerConfig): An instance of GraphormerConfig class containing the configuration parameters
                for the Graphormer model.

        Returns:
            None.

        Raises:
            None.
        """
        super().__init__()
        self.num_heads = config.num_attention_heads
        self.multi_hop_max_dist = config.multi_hop_max_dist

        # We do not change edge feature embedding learning, as edge embeddings are represented as a combination of the original features
        # + shortest path
        self.edge_encoder = nn.Embedding(config.num_edges + 1, config.num_attention_heads, padding_idx=0)

        self.edge_type = config.edge_type
        if self.edge_type == "multi_hop":
            self.edge_dis_encoder = nn.Embedding(
                config.num_edge_dis * config.num_attention_heads * config.num_attention_heads,
                1,
            )

        self.spatial_pos_encoder = nn.Embedding(config.num_spatial, config.num_attention_heads, padding_idx=0)

        self.graph_token_virtual_distance = nn.Embedding(1, config.num_attention_heads)

    def construct(
        self,
        input_nodes: Tensor,
        attn_bias: Tensor,
        spatial_pos: Tensor,
        input_edges: Tensor,
        attn_edge_type: Tensor,
    ) -> Tensor:
        """
        This method constructs the graph attention bias tensor for the Graphormer model.

        Args:
            self: The instance of the GraphormerGraphAttnBias class.
            input_nodes (Tensor): The input nodes tensor representing the nodes in the graph.
                Shape should be (n_graph, n_node, ...).
            attn_bias (Tensor): The attention bias tensor. Should have the same shape as input_nodes.
            spatial_pos (Tensor): The spatial positional encoding tensor. Should have the same shape as input_nodes.
            input_edges (Tensor): The input edges tensor representing the edges in the graph.
                Shape should be (n_graph, n_node, n_node, ...).
            attn_edge_type (Tensor): The attention edge type tensor. Should have the same shape as input_edges.

        Returns:
            Tensor: The constructed graph attention bias tensor after performing the specified operations.

        Raises:
            ValueError: If the input shapes of input_nodes, attn_bias, spatial_pos, input_edges, or attn_edge_type are
                incompatible for the operations within the method.
            RuntimeError: If any runtime error occurs during the execution of the method.
        """
        n_graph, n_node = input_nodes.shape[:2]
        graph_attn_bias = attn_bias.copy()
        graph_attn_bias = graph_attn_bias.unsqueeze(1).tile(
            (1, self.num_heads, 1, 1)
        )  # [n_graph, n_head, n_node+1, n_node+1]

        # spatial pos
        # [n_graph, n_node, n_node, n_head] -> [n_graph, n_head, n_node, n_node]
        spatial_pos_bias = self.spatial_pos_encoder(spatial_pos).permute(0, 3, 1, 2)
        graph_attn_bias[:, :, 1:, 1:] = graph_attn_bias[:, :, 1:, 1:] + spatial_pos_bias

        # reset spatial pos here
        tvd = self.graph_token_virtual_distance.weight.view(1, self.num_heads, 1)
        graph_attn_bias[:, :, 1:, 0] = graph_attn_bias[:, :, 1:, 0] + tvd
        graph_attn_bias[:, :, 0, :] = graph_attn_bias[:, :, 0, :] + tvd

        # edge feature
        if self.edge_type == "multi_hop":
            spatial_pos_ = spatial_pos.copy()

            spatial_pos_[spatial_pos_ == 0] = 1  # set pad to 1
            # set 1 to 1, input_nodes > 1 to input_nodes - 1
            spatial_pos_ = ops.where(spatial_pos_ > 1, spatial_pos_ - 1, spatial_pos_)
            if self.multi_hop_max_dist > 0:
                spatial_pos_ = spatial_pos_.clamp(0, self.multi_hop_max_dist)
                input_edges = input_edges[:, :, :, : self.multi_hop_max_dist, :]
            # [n_graph, n_node, n_node, max_dist, n_head]

            input_edges = self.edge_encoder(input_edges).mean(-2)
            max_dist = input_edges.shape[-2]
            edge_input_flat = input_edges.permute(3, 0, 1, 2, 4).reshape(max_dist, -1, self.num_heads)
            edge_input_flat = ops.bmm(
                edge_input_flat,
                self.edge_dis_encoder.weight.reshape(-1, self.num_heads, self.num_heads)[:max_dist, :, :],
            )
            input_edges = edge_input_flat.reshape(max_dist, n_graph, n_node, n_node, self.num_heads).permute(
                1, 2, 3, 0, 4
            )
            input_edges = (input_edges.sum(-2) / (spatial_pos_.float().unsqueeze(-1))).permute(0, 3, 1, 2)
        else:
            # [n_graph, n_node, n_node, n_head] -> [n_graph, n_head, n_node, n_node]
            input_edges = self.edge_encoder(attn_edge_type).mean(-2).permute(0, 3, 1, 2)

        graph_attn_bias[:, :, 1:, 1:] = graph_attn_bias[:, :, 1:, 1:] + input_edges
        graph_attn_bias = graph_attn_bias + attn_bias.unsqueeze(1)  # reset

        return graph_attn_bias


class GraphormerMultiheadAttention(nn.Cell):
    """Multi-headed attention.

    See "Attention Is All You Need" for more details.
    """
    def __init__(self, config: GraphormerConfig):
        """
        Initializes an instance of the GraphormerMultiheadAttention class.

        Args:
            self: The instance of the class.
            config (GraphormerConfig):
                An object containing the configuration parameters for the GraphormerMultiheadAttention.

                - embedding_dim (int): The dimension of the input embeddings.
                - kdim (int, optional): The dimension of the key embeddings.
                If not provided, it defaults to embedding_dim.
                - vdim (int, optional): The dimension of the value embeddings.
                If not provided, it defaults to embedding_dim.
                - num_attention_heads (int): The number of attention heads.
                - attention_dropout (float): The dropout rate for attention weights.
                - bias (bool): Whether to include bias in the linear transformations.
                - q_noise (float): The standard deviation of the noise added to the query, key, and value projections.
                - qn_block_size (int): The block size for quantization noise.

        Returns:
            None.

        Raises:
            AssertionError: If embedding_dim is not divisible by num_attention_heads.
            NotImplementedError: If self-attention is disabled.
            AssertionError: If self-attention is enabled but query, key, and value dimensions are not the same.
        """
        super().__init__()
        self.embedding_dim = config.embedding_dim
        self.kdim = config.kdim if config.kdim is not None else config.embedding_dim
        self.vdim = config.vdim if config.vdim is not None else config.embedding_dim
        self.qkv_same_dim = self.kdim == config.embedding_dim and self.vdim == config.embedding_dim

        self.num_heads = config.num_attention_heads
        self.attention_dropout_module = nn.Dropout(p=config.attention_dropout)

        self.head_dim = config.embedding_dim // config.num_attention_heads
        if not self.head_dim * config.num_attention_heads == self.embedding_dim:
            raise AssertionError("The embedding_dim must be divisible by num_heads.")
        self.scaling = self.head_dim**-0.5

        self.self_attention = True  # config.self_attention
        if not self.self_attention:
            raise NotImplementedError("The Graphormer model only supports self attention for now.")
        if self.self_attention and not self.qkv_same_dim:
            raise AssertionError("Self-attention requires query, key and value to be of the same size.")

        self.k_proj = quant_noise(
            nn.Dense(self.kdim, config.embedding_dim, has_bias=config.bias),
            config.q_noise,
            config.qn_block_size,
        )
        self.v_proj = quant_noise(
            nn.Dense(self.vdim, config.embedding_dim, has_bias=config.bias),
            config.q_noise,
            config.qn_block_size,
        )
        self.q_proj = quant_noise(
            nn.Dense(config.embedding_dim, config.embedding_dim, has_bias=config.bias),
            config.q_noise,
            config.qn_block_size,
        )

        self.out_proj = quant_noise(
            nn.Dense(config.embedding_dim, config.embedding_dim, has_bias=config.bias),
            config.q_noise,
            config.qn_block_size,
        )

        self.onnx_trace = False

    def reset_parameters(self):
        """
        Reset parameters
        """
        if self.qkv_same_dim:
            # Empirically observed the convergence to be much better with
            # the scaled initialization
            gain = 1 / math.sqrt(2)
        else:
            gain = 1

        self.k_proj.weight.set_data(init_xavier_uniform(self.k_proj.weight, gain))
        self.v_proj.weight.set_data(init_xavier_uniform(self.v_proj.weight, gain))
        self.q_proj.weight.set_data(init_xavier_uniform(self.q_proj.weight, gain))

        self.out_proj.weight.set_data(init_xavier_uniform(self.out_proj.weight, gain))
        if self.out_proj.bias is not None:
            self.out_proj.bias.set_data(init_zero(self.out_proj.bias))

    def construct(
        self,
        query: Tensor,
        key: Optional[Tensor],
        value: Optional[Tensor],
        attn_bias: Optional[Tensor],
        key_padding_mask: Optional[Tensor] = None,
        need_weights: bool = True,
        attn_mask: Optional[Tensor] = None,
        before_softmax: bool = False,
        need_head_weights: bool = False,
    ) -> Tuple[Tensor, Optional[Tensor]]:
        """
        Args:
            key_padding_mask (Bytetorch.Tensor, optional): mask to exclude
                keys that are pads, of shape `(batch, src_len)`, where padding elements are indicated by 1s.
            need_weights (bool, optional): return the attention weights,
                averaged over heads (default: False).
            attn_mask (Bytetorch.Tensor, optional): typically used to
                implement causal attention, where the mask prevents the attention from looking forward in time
                (default: None).
            before_softmax (bool, optional): return the raw attention
                weights and values before the attention softmax.
            need_head_weights (bool, optional): return the attention
                weights for each head. Implies *need_weights*. Default: return the average attention weights over all
                heads.
        """
        if need_head_weights:
            need_weights = True

        tgt_len, bsz, embedding_dim = query.shape
        src_len = tgt_len
        if not embedding_dim == self.embedding_dim:
            raise AssertionError(
                f"The query embedding dimension {embedding_dim} is not equal to the expected embedding_dim"
                f" {self.embedding_dim}."
            )
        if list(query.shape) != [tgt_len, bsz, embedding_dim]:
            raise AssertionError("Query size incorrect in Graphormer, compared to model dimensions.")

        if key is not None:
            src_len, key_bsz, _ = key.shape
            # Only do this assertion outside jit
            if (key_bsz != bsz) or (value is None) or not (src_len, bsz == value.shape[:2]):
                raise AssertionError(
                    "The batch shape does not match the key or value shapes provided to the attention."
                )

        qproj = self.q_proj(query)
        kproj = self.k_proj(query)
        vproj = self.v_proj(query)

        qproj *= self.scaling

        qproj = qproj.view(tgt_len, bsz * self.num_heads, self.head_dim).swapaxes(0, 1)
        if kproj is not None:
            kproj = kproj.view(-1, bsz * self.num_heads, self.head_dim).swapaxes(0, 1)
        if vproj is not None:
            vproj = vproj.view(-1, bsz * self.num_heads, self.head_dim).swapaxes(0, 1)

        if (kproj is None) or not kproj.shape[1] == src_len:
            raise AssertionError("The shape of the key generated in the attention is incorrect")

        # This is part of a workaround to get around fork/join parallelism
        # not supporting Optional types.
        if key_padding_mask is not None and key_padding_mask.dim() == 0:
            key_padding_mask = None

        if key_padding_mask is not None:
            if key_padding_mask.shape[0] != bsz or key_padding_mask.shape[1] != src_len:
                raise AssertionError(
                    "The shape of the generated padding mask for the key does not match expected dimensions."
                )
        attn_weights = ops.bmm(qproj, kproj.swapaxes(1, 2))
        attn_weights = self.apply_sparse_mask(attn_weights)

        if list(attn_weights.shape) != [bsz * self.num_heads, tgt_len, src_len]:
            raise AssertionError("The attention weights generated do not match the expected dimensions.")

        if attn_bias is not None:
            attn_weights += attn_bias.view(bsz * self.num_heads, tgt_len, src_len)

        if attn_mask is not None:
            attn_mask = attn_mask.unsqueeze(0)
            attn_weights += attn_mask

        if key_padding_mask is not None:
            # don't attend to padding symbols
            attn_weights = attn_weights.view(bsz, self.num_heads, tgt_len, src_len)
            attn_weights = attn_weights.masked_fill(
                key_padding_mask.unsqueeze(1).unsqueeze(2).bool(), float("-inf")
            )
            attn_weights = attn_weights.view(bsz * self.num_heads, tgt_len, src_len)

        if before_softmax:
            return attn_weights, vproj

        attn_weights_float = ops.softmax(attn_weights, axis=-1)
        attn_weights = attn_weights_float.type_as(attn_weights)
        attn_probs = self.attention_dropout_module(attn_weights)

        if vproj is None:
            raise AssertionError("No value generated")
        attn = ops.bmm(attn_probs, vproj)
        if list(attn.shape) != [bsz * self.num_heads, tgt_len, self.head_dim]:
            raise AssertionError("The attention generated do not match the expected dimensions.")

        attn = attn.swapaxes(0, 1).view(tgt_len, bsz, embedding_dim)
        attn: Tensor = self.out_proj(attn)

        attn_weights = None
        if need_weights:
            attn_weights = attn_weights_float.view(bsz, self.num_heads, tgt_len, src_len).swapaxes(1, 0)
            if not need_head_weights:
                # average attention weights over heads
                attn_weights = attn_weights.mean(axis=0)

        return attn, attn_weights

    def apply_sparse_mask(self, attn_weights: Tensor) -> Tensor:
        """
        Apply sparse mask (seems did not do anythin)
        """
        return attn_weights


class GraphormerGraphEncoderLayer(nn.Cell):
    """
    Graphormer Graph Encoder Layer
    """
    def __init__(self, config: GraphormerConfig) -> None:
        """
        Initializes a GraphormerGraphEncoderLayer object with the provided configuration.

        Args:
            self (GraphormerGraphEncoderLayer): The instance of the GraphormerGraphEncoderLayer class.
            config (GraphormerConfig):
                An instance of GraphormerConfig containing the configuration parameters for the encoder layer.

                - embedding_dim (int): The dimension of the input embeddings.
                - num_attention_heads (int): The number of attention heads in the multi-head attention mechanism.
                - q_noise (bool): A flag indicating whether to use quantization noise.
                - qn_block_size (int): The block size for quantization noise.
                - pre_layernorm (bool): A flag indicating whether to apply LayerNorm before the self-attention module.
                - dropout (float): The dropout probability for the encoder layer.
                - activation_dropout (float): The dropout probability for the activation function.
                - activation_fn (str): The activation function to be used.
                - ffn_embedding_dim (int): The dimension of the feed-forward neural network layer.

        Returns:
            None.

        Raises:
            None.
        """
        super().__init__()

        # Initialize parameters
        self.embedding_dim = config.embedding_dim
        self.num_attention_heads = config.num_attention_heads
        self.q_noise = config.q_noise
        self.qn_block_size = config.qn_block_size
        self.pre_layernorm = config.pre_layernorm

        self.dropout_module = nn.Dropout(p=config.dropout)

        self.activation_dropout_module = nn.Dropout(p=config.activation_dropout)

        # Initialize blocks
        self.activation_fn = ACT2FN[config.activation_fn]
        self.self_attn = GraphormerMultiheadAttention(config)

        # layer norm associated with the self attention layer
        self.self_attn_layer_norm = nn.LayerNorm([self.embedding_dim])

        self.fc1 = self.build_fc(
            self.embedding_dim,
            config.ffn_embedding_dim,
            q_noise=config.q_noise,
            qn_block_size=config.qn_block_size,
        )
        self.fc2 = self.build_fc(
            config.ffn_embedding_dim,
            self.embedding_dim,
            q_noise=config.q_noise,
            qn_block_size=config.qn_block_size,
        )

        # layer norm associated with the position wise feed-forward NN
        self.final_layer_norm = nn.LayerNorm([self.embedding_dim])

    def build_fc(
        self, input_dim: int, output_dim: int, q_noise: float, qn_block_size: int
    ) -> Union[nn.Cell, nn.Dense, nn.Embedding, nn.Conv2d]:
        """
        Build function
        """
        return quant_noise(nn.Dense(input_dim, output_dim), q_noise, qn_block_size)

    def construct(
        self,
        input_nodes: Tensor,
        self_attn_bias: Optional[Tensor] = None,
        self_attn_mask: Optional[Tensor] = None,
        self_attn_padding_mask: Optional[Tensor] = None,
    ) -> Tuple[Tensor, Optional[Tensor]]:
        """
        nn.LayerNorm is applied either before or after the self-attention/ffn modules similar to the original
        Transformer implementation.
        """
        residual = input_nodes
        if self.pre_layernorm:
            input_nodes = self.self_attn_layer_norm(input_nodes)

        input_nodes, attn = self.self_attn(
            query=input_nodes,
            key=input_nodes,
            value=input_nodes,
            attn_bias=self_attn_bias,
            key_padding_mask=self_attn_padding_mask,
            need_weights=False,
            attn_mask=self_attn_mask,
        )
        input_nodes = self.dropout_module(input_nodes)
        input_nodes = residual + input_nodes
        if not self.pre_layernorm:
            input_nodes = self.self_attn_layer_norm(input_nodes)

        residual = input_nodes
        if self.pre_layernorm:
            input_nodes = self.final_layer_norm(input_nodes)
        input_nodes = self.activation_fn(self.fc1(input_nodes))
        input_nodes = self.activation_dropout_module(input_nodes)
        input_nodes = self.fc2(input_nodes)
        input_nodes = self.dropout_module(input_nodes)
        input_nodes = residual + input_nodes
        if not self.pre_layernorm:
            input_nodes = self.final_layer_norm(input_nodes)

        return input_nodes, attn


class GraphormerGraphEncoder(nn.Cell):
    """
    Graphormer Graph Encoder
    """
    def __init__(self, config: GraphormerConfig):
        """
        Initializes the GraphormerGraphEncoder class.

        Args:
            self: The object itself.
            config (GraphormerConfig):
                An instance of the GraphormerConfig class containing the configuration parameters for the graph encoder.

                - dropout (float): The dropout probability.
                - layerdrop (float): The layer drop probability.
                - embedding_dim (int): The dimension of the input embeddings.
                - apply_graphormer_init (bool): Indicates whether to apply Graphormer initialization.
                - traceable (bool): Indicates whether the model should be traceable.
                - graph_node_feature (GraphormerGraphNodeFeature): An instance of the GraphormerGraphNodeFeature class.
                - graph_attn_bias (GraphormerGraphAttnBias): An instance of the GraphormerGraphAttnBias class.
                - embed_scale (float): The scale factor for the input embeddings.
                - q_noise (float): The quantization noise.
                - qn_block_size (int): The block size for quantization noise.
                - encoder_normalize_before (bool): Indicates whether to normalize before the encoder layers.
                - pre_layernorm (bool): Indicates whether to apply layer normalization before the encoder layers.
                - num_hidden_layers (int): The number of hidden layers in the encoder.
                - freeze_embeddings (bool): Indicates whether to freeze the embeddings.
                - num_trans_layers_to_freeze (int): The number of transformer layers to freeze.

        Returns:
            None.

        Raises:
            NotImplementedError: If the configuration specifies freezing embeddings, as this feature is not yet implemented.
            TypeError: If the parameters provided to the method are not of the expected types.
        """
        super().__init__()

        self.dropout_module = nn.Dropout(p=config.dropout)
        self.layerdrop = config.layerdrop
        self.embedding_dim = config.embedding_dim
        self.apply_graphormer_init = config.apply_graphormer_init
        self.traceable = config.traceable

        self.graph_node_feature = GraphormerGraphNodeFeature(config)
        self.graph_attn_bias = GraphormerGraphAttnBias(config)

        self.embed_scale = config.embed_scale

        if config.q_noise > 0:
            self.quant_noise = quant_noise(
                nn.Dense(self.embedding_dim, self.embedding_dim, has_bias=False),
                config.q_noise,
                config.qn_block_size,
            )
        else:
            self.quant_noise = None

        if config.encoder_normalize_before:
            self.emb_layer_norm = nn.LayerNorm([self.embedding_dim])
        else:
            self.emb_layer_norm = None

        if config.pre_layernorm:
            self.final_layer_norm = nn.LayerNorm([self.embedding_dim])

        if self.layerdrop > 0.0:
            self.layers = LayerDropModuleList(p_drop=self.layerdrop)
        else:
            self.layers = nn.CellList([])
        self.layers.extend([GraphormerGraphEncoderLayer(config) for _ in range(config.num_hidden_layers)])

        # Apply initialization of model params after building the model
        if config.freeze_embeddings:
            raise NotImplementedError("Freezing embeddings is not implemented yet.")

        for layer in range(config.num_trans_layers_to_freeze):
            mod = self.layers[layer]
            if mod is not None:
                for par in mod.parameters():
                    par.requires_grad = False

    def construct(
        self,
        input_nodes: Tensor,
        input_edges: Tensor,
        attn_bias: Tensor,
        in_degree: Tensor,
        out_degree: Tensor,
        spatial_pos: Tensor,
        attn_edge_type: Tensor,
        perturb=None,
        last_state_only: bool = False,
        token_embeddings: Optional[Tensor] = None,
        attn_mask: Optional[Tensor] = None,
    ) -> Tuple[Union[Tensor, List[Tensor]], Tensor]:
        '''
        Constructs the graph encoder for Graphormer model.

        Args:
            self: The instance of the GraphormerGraphEncoder class.
            input_nodes (Tensor): The input nodes of the graph. Shape (batch_size, num_nodes, input_dim).
            input_edges (Tensor): The input edges of the graph. Shape (batch_size, num_edges, input_dim).
            attn_bias (Tensor): The attention bias tensor. Shape (batch_size, num_heads, num_nodes, num_nodes).
            in_degree (Tensor): The in-degree tensor of the graph nodes. Shape (batch_size, num_nodes).
            out_degree (Tensor): The out-degree tensor of the graph nodes. Shape (batch_size, num_nodes).
            spatial_pos (Tensor): The spatial position tensor of the graph nodes.
                Shape (batch_size, num_nodes, spatial_dim).
            attn_edge_type (Tensor): The attention edge type tensor. Shape (batch_size, num_edges).
            perturb (Optional[Tensor]): The optional perturbation tensor.
                If provided, shape (batch_size, num_nodes, input_dim).
            last_state_only (bool): A flag indicating whether to return only the last state. Default is False.
            token_embeddings (Optional[Tensor]): Optional token embeddings tensor.
                If provided, shape (batch_size, num_nodes, input_dim).
            attn_mask (Optional[Tensor]): Optional attention mask tensor.
                If provided, shape (batch_size, num_heads, num_nodes, num_nodes).

        Returns:
            Tuple[Union[Tensor, List[Tensor]], Tensor]:
                A tuple containing the inner states as a list of tensors and the graph representation tensor.

        Raises:
            None.
        '''
        # compute padding mask. This is needed for multi-head attention
        data_x = input_nodes
        n_graph, _ = data_x.shape[:2]
        padding_mask = (data_x[:, :, 0]).eq(0)
        padding_mask_cls = ops.zeros((n_graph, 1), dtype=padding_mask.dtype)
        padding_mask = ops.cat((padding_mask_cls, padding_mask), axis=1)

        attn_bias = self.graph_attn_bias(input_nodes, attn_bias, spatial_pos, input_edges, attn_edge_type)

        if token_embeddings is not None:
            input_nodes = token_embeddings
        else:
            input_nodes = self.graph_node_feature(input_nodes, in_degree, out_degree)

        if perturb is not None:
            input_nodes[:, 1:, :] += perturb

        if self.embed_scale is not None:
            input_nodes = input_nodes * self.embed_scale

        if self.quant_noise is not None:
            input_nodes = self.quant_noise(input_nodes)

        if self.emb_layer_norm is not None:
            input_nodes = self.emb_layer_norm(input_nodes)

        input_nodes = self.dropout_module(input_nodes)

        input_nodes = input_nodes.swapaxes(0, 1)

        inner_states = []
        if not last_state_only:
            inner_states.append(input_nodes)
        for layer in self.layers:
            input_nodes, _ = layer(
                input_nodes,
                self_attn_padding_mask=padding_mask,
                self_attn_mask=attn_mask,
                self_attn_bias=attn_bias,
            )
            if not last_state_only:
                inner_states.append(input_nodes)

        graph_rep = input_nodes[0, :, :]

        if last_state_only:
            inner_states = [input_nodes]

        if self.traceable:
            return ops.stack(inner_states), graph_rep
        return inner_states, graph_rep


class GraphormerDecoderHead(nn.Cell):
    """
    Graphormer Decoder Head
    """
    def __init__(self, embedding_dim: int, num_classes: int):
        """
        Initializes the GraphormerDecoderHead class.

        Args:
            self: The instance of the GraphormerDecoderHead class.
            embedding_dim (int): The dimension of the embedding space.
            num_classes (int): The number of classes in the classification task.

        Returns:
            None.

        Raises:
            None.
        """
        super().__init__()
        # num_classes should be 1 for regression, or the number of classes for classification
        self.lm_output_learned_bias = Parameter(ops.zeros(1))
        self.classifier = nn.Dense(embedding_dim, num_classes, has_bias=False)
        self.num_classes = num_classes

    def construct(self, input_nodes: Tensor, **kwargs) -> Tensor:
        """
        Construct the GraphormerDecoderHead.

        Args:
            self: The instance of the GraphormerDecoderHead class.
            input_nodes (Tensor): The input nodes to be processed by the method.

        Returns:
            Tensor: The processed output nodes.

        Raises:
            None.
        """
        input_nodes = self.classifier(input_nodes)
        input_nodes = input_nodes + self.lm_output_learned_bias
        return input_nodes


class GraphormerPreTrainedModel(PreTrainedModel):
    """
    An abstract class to handle weights initialization and a simple interface for downloading and loading pretrained
    models.
    """
    config_class = GraphormerConfig
    base_model_prefix = "graphormer"
    supports_gradient_checkpointing = True
    main_input_name_nodes = "input_nodes"
    main_input_name_edges = "input_edges"

    def init_graphormer_params(self, module: Union[nn.Dense, nn.Embedding, GraphormerMultiheadAttention]):
        """
        Initialize the weights specific to the Graphormer Model.
        """
        if isinstance(module, nn.Dense):
            module.weight.set_data(init_normal(module.weight, sigma=0.02, mean=0.0))
            if module.has_bias:
                module.bias.set_data(init_zero(module.bias))
        if isinstance(module, nn.Embedding):
            weight = np.random.normal(loc=0.0, scale=0.02, size=module.weight.shape)
            if module.padding_idx:
                weight[module.padding_idx] = 0

            module.weight.set_data(Tensor(weight, module.weight.dtype))
        if isinstance(module, GraphormerMultiheadAttention):
            module.q_proj.weight.set_data(init_normal(module.q_proj.weight,
                                                      sigma=0.02, mean=0.0))
            module.k_proj.weight.set_data(init_normal(module.k_proj.weight,
                                                      sigma=0.02, mean=0.0))
            module.v_proj.weight.set_data(init_normal(module.v_proj.weight,
                                                      sigma=0.02, mean=0.0))

    def _init_weights(
        self,
        cell
    ):
        """
        Initialize the weights
        """
        if isinstance(cell, (nn.Dense, nn.Conv2d)):
            # We might be missing part of the Linear init, dependant on the layer num
            cell.weight.set_data(init_normal(cell.weight, sigma=0.02, mean=0.0))
            if cell.has_bias:
                cell.bias.set_data(init_zero(cell.bias))
        elif isinstance(cell, nn.Embedding):
            weight = np.random.normal(loc=0.0, scale=0.02, size=cell.weight.shape)
            if cell.padding_idx:
                weight[cell.padding_idx] = 0

            cell.weight.set_data(Tensor(weight, cell.weight.dtype))
        elif isinstance(cell, GraphormerMultiheadAttention):
            cell.q_proj.weight.set_data(init_normal(cell.q_proj.weight,
                                                      sigma=0.02, mean=0.0))
            cell.k_proj.weight.set_data(init_normal(cell.k_proj.weight,
                                                      sigma=0.02, mean=0.0))
            cell.v_proj.weight.set_data(init_normal(cell.v_proj.weight,
                                                      sigma=0.02, mean=0.0))
        elif isinstance(cell, GraphormerGraphEncoder):
            if cell.apply_graphormer_init:
                cell.apply(self.init_graphormer_params)

    def _set_gradient_checkpointing(self, module, value=False):
        """
        Set the gradient checkpointing option for a given module in a GraphormerPreTrainedModel.

        Args:
            self (GraphormerPreTrainedModel): The instance of the GraphormerPreTrainedModel class.
            module (GraphormerModel): The module for which the gradient checkpointing option is being set.
            value (bool): The value indicating whether gradient checkpointing is enabled or disabled.

        Returns:
            None.

        Raises:
            TypeError: If the provided module is not an instance of GraphormerModel.
        """
        if isinstance(module, GraphormerModel):
            module.gradient_checkpointing = value


class GraphormerModel(GraphormerPreTrainedModel):
    """
    The Graphormer model is a graph-encoder model.

    It goes from a graph to its representation. If you want to use the model for a downstream classification task, use
    GraphormerForGraphClassification instead. For any other downstream task, feel free to add a new class, or combine
    this model with a downstream model of your choice, following the example in GraphormerForGraphClassification.
    """
    def __init__(self, config: GraphormerConfig):
        """
        Initializes a new instance of the GraphormerModel class.

        Args:
            self: The instance of the GraphormerModel class.
            config (GraphormerConfig):
                An object of type GraphormerConfig containing the configuration settings for the model.
                The config parameter is used to set various attributes of the GraphormerModel instance,
                such as max_nodes, graph_encoder, share_input_output_embed, lm_output_learned_bias, load_softmax,
                lm_head_transform_weight, activation_fn, and layer_norm.

        Returns:
            None.

        Raises:
            None.
        """
        super().__init__(config)
        self.max_nodes = config.max_nodes

        self.graph_encoder = GraphormerGraphEncoder(config)

        self.share_input_output_embed = config.share_input_output_embed
        self.lm_output_learned_bias = None

        # Remove head is set to true during fine-tuning
        self.load_softmax = not getattr(config, "remove_head", False)

        self.lm_head_transform_weight = nn.Dense(config.embedding_dim, config.embedding_dim)
        self.activation_fn = ACT2FN[config.activation_fn]
        self.layer_norm = nn.LayerNorm([config.embedding_dim])

        self.post_init()

    def reset_output_layer_parameters(self):
        """
        Reset output layer parameters
        """
        self.lm_output_learned_bias = Parameter(ops.zeros(1))

    def construct(
        self,
        input_nodes: Tensor,
        input_edges: Tensor,
        attn_bias: Tensor,
        in_degree: Tensor,
        out_degree: Tensor,
        spatial_pos: Tensor,
        attn_edge_type: Tensor,
        perturb: Optional[Tensor] = None,
        masked_tokens: None = None,
        return_dict: Optional[bool] = None,
        **kwargs,
    ) -> Union[Tuple[Tensor], BaseModelOutputWithNoAttention]:
        """
        Construct method in the GraphormerModel class.

        Args:
            self: The instance of the class.
            input_nodes (Tensor): The input nodes tensor for the graph.
            input_edges (Tensor): The input edges tensor for the graph.
            attn_bias (Tensor): The attention bias tensor.
            in_degree (Tensor): The in-degree tensor for nodes in the graph.
            out_degree (Tensor): The out-degree tensor for nodes in the graph.
            spatial_pos (Tensor): The spatial position tensor for nodes in the graph.
            attn_edge_type (Tensor): The attention edge type tensor.
            perturb (Optional[Tensor], default=None): A tensor for perturbation.
            masked_tokens (None): Not implemented; should be None.
            return_dict (Optional[bool], default=None): If True, returns a BaseModelOutputWithNoAttention object.

        Returns:
            Union[Tuple[Tensor], BaseModelOutputWithNoAttention]:
                Depending on the value of return_dict, either a tuple containing input_nodes and inner_states
                or a BaseModelOutputWithNoAttention object.

        Raises:
            NotImplementedError: If masked_tokens is not None, indicating that the functionality is not implemented.
        """
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        inner_states, _ = self.graph_encoder(
            input_nodes, input_edges, attn_bias, in_degree, out_degree, spatial_pos, attn_edge_type, perturb=perturb
        )

        # last inner state, then revert Batch and Graph len
        input_nodes = inner_states[-1].swapaxes(0, 1)

        # project masked tokens only
        if masked_tokens is not None:
            raise NotImplementedError

        input_nodes = self.layer_norm(self.activation_fn(self.lm_head_transform_weight(input_nodes)))

        # project back to size of vocabulary
        if self.share_input_output_embed and hasattr(self.graph_encoder.embed_tokens, "weight"):
            input_nodes = ops.dense(input_nodes, self.graph_encoder.embed_tokens.weight)

        if not return_dict:
            return tuple(x for x in [input_nodes, inner_states] if x is not None)
        return BaseModelOutputWithNoAttention(last_hidden_state=input_nodes, hidden_states=inner_states)


class GraphormerForGraphClassification(GraphormerPreTrainedModel):
    """
    This model can be used for graph-level classification or regression tasks.

    It can be trained on

    - regression (by setting config.num_classes to 1); there should be one float-type label per graph
    - one task classification (by setting config.num_classes to the number of classes); there should be one integer
    label per graph
    - binary multi-task classification (by setting config.num_classes to the number of labels); there should be a list
    of integer labels for each graph.
    """
    def __init__(self, config: GraphormerConfig):
        """
        Initializes a new instance of GraphormerForGraphClassification.

        Args:
            self: The instance of the class.
            config (GraphormerConfig):
                An instance of GraphormerConfig containing the configuration settings for the Graphormer model.

        Returns:
            None.

        Raises:
            None.
        """
        super().__init__(config)
        self.encoder = GraphormerModel(config)
        self.embedding_dim = config.embedding_dim
        self.num_classes = config.num_classes
        self.classifier = GraphormerDecoderHead(self.embedding_dim, self.num_classes)
        self.is_encoder_decoder = True

        # Initialize weights and apply final processing
        self.post_init()

    def construct(
        self,
        input_nodes: Tensor,
        input_edges: Tensor,
        attn_bias: Tensor,
        in_degree: Tensor,
        out_degree: Tensor,
        spatial_pos: Tensor,
        attn_edge_type: Tensor,
        labels: Optional[Tensor] = None,
        return_dict: Optional[bool] = None,
        **kwargs,
    ) -> Union[Tuple[Tensor], SequenceClassifierOutput]:
        """Constructs a Graphormer for graph classification.

        This method takes the following parameters:

        - self: The object instance.
        - input_nodes: A Tensor representing the input nodes.
        - input_edges: A Tensor representing the input edges.
        - attn_bias: A Tensor representing the attention bias.
        - in_degree: A Tensor representing the in-degree of the nodes.
        - out_degree: A Tensor representing the out-degree of the nodes.
        - spatial_pos: A Tensor representing the spatial positions of the nodes.
        - attn_edge_type: A Tensor representing the attention edge types.
        - labels: An optional Tensor representing the labels for classification. Defaults to None.
        - return_dict: An optional boolean indicating whether to return a dictionary.
        If not provided, it uses the value from the configuration. Defaults to None.
        - **kwargs: Additional keyword arguments.

        The method returns a value of type Union[Tuple[Tensor], SequenceClassifierOutput].

        Args:
            self: The object instance.
            input_nodes: A Tensor representing the input nodes. Shape: [batch_size, sequence_length, hidden_size].
            input_edges: A Tensor representing the input edges.
                Shape: [batch_size, sequence_length, sequence_length, hidden_size].
            attn_bias: A Tensor representing the attention bias. Shape: [batch_size, sequence_length, sequence_length].
            in_degree: A Tensor representing the in-degree of the nodes. Shape: [batch_size, sequence_length].
            out_degree: A Tensor representing the out-degree of the nodes. Shape: [batch_size, sequence_length].
            spatial_pos: A Tensor representing the spatial positions of the nodes.
                Shape: [batch_size, sequence_length, hidden_size].
            attn_edge_type: A Tensor representing the attention edge types.
                Shape: [batch_size, sequence_length, sequence_length].
            labels: An optional Tensor representing the labels for classification. Shape: [batch_size, num_classes].
                Defaults to None.
            return_dict: An optional boolean indicating whether to return a dictionary.
                If not provided, it uses the value from the configuration. Defaults to None.
            **kwargs: Additional keyword arguments.

        Returns:
            Conditional Return:

                - If 'return_dict' is False, the method returns a tuple containing the following elements (if not None):

                    - loss: A Tensor representing the calculated loss. Shape: [batch_size].
                    - logits: A Tensor representing the output logits. Shape: [batch_size, num_classes].
                    - hidden_states: A list of Tensors representing the hidden states. Each Tensor has shape
                    [batch_size, sequence_length, hidden_size].

                - If 'return_dict' is True, the method returns a SequenceClassifierOutput object with the following
                attributes (if not None):

                    - loss: A Tensor representing the calculated loss. Shape: [batch_size].
                    - logits: A Tensor representing the output logits. Shape: [batch_size, num_classes].
                    - hidden_states: A list of Tensors representing the hidden states. Each Tensor has shape
                    [batch_size, sequence_length, hidden_size].
                    - attentions: None.

        Raises:
            MSELossError: If 'labels' is not None and 'num_classes' is 1, but the shape of 'labels' is not compatible
                with logits.
            CrossEntropyLossError: If 'labels' is not None and 'num_classes' is greater than 1, but the shape of
                'labels' is not compatible with logits.
            BCEWithLogitsLossError: If 'labels' is not None and 'num_classes' is greater than 1, but the shape of
                'labels' is not compatible with logits.
        """
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        encoder_outputs = self.encoder(
            input_nodes,
            input_edges,
            attn_bias,
            in_degree,
            out_degree,
            spatial_pos,
            attn_edge_type,
            return_dict=True,
        )

        outputs, hidden_states = encoder_outputs["last_hidden_state"], encoder_outputs["hidden_states"]

        head_outputs = self.classifier(outputs)
        logits = head_outputs[:, 0, :]

        loss = None
        if labels is not None:
            mask = 1 - ops.isnan(labels) # invert True and False

            if self.num_classes == 1:  # regression
                loss_fct = MSELoss()
                loss = loss_fct(logits[mask].squeeze(), labels[mask].squeeze().float())
            elif self.num_classes > 1 and len(labels.shape) == 1:  # One task classification
                loss_fct = CrossEntropyLoss()
                loss = loss_fct(logits[mask].view(-1, self.num_classes), labels[mask].view(-1))
            else:  # Binary multi-task classification
                loss_fct = BCEWithLogitsLoss(reduction="sum")
                loss = loss_fct(logits[mask], labels[mask])

        if not return_dict:
            return tuple(x for x in [loss, logits, hidden_states] if x is not None)
        return SequenceClassifierOutput(loss=loss, logits=logits, hidden_states=hidden_states, attentions=None)
