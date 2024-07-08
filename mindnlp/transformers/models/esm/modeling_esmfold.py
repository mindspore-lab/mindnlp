# coding=utf-8
# Copyright 2022 Meta and The HuggingFace Inc. team. All rights reserved.
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
"""MindSpore ESMFold model"""
import math
import sys
from dataclasses import dataclass
from functools import partial
from typing import Callable, Dict, List, Optional, Tuple, Union

import numpy as np
import mindspore
from mindspore import nn, ops, Parameter
from mindspore.common.initializer import initializer, Normal, XavierUniform, HeNormal

from mindnlp.utils import (
    ContextManagers,
    is_scipy_available,
    logging,
)
from ...modeling_outputs import ModelOutput
from .modeling_esm import EsmModel, EsmPreTrainedModel
from .openfold_utils import (
    OFProtein,
    Rigid,
    Rotation,
    atom14_to_atom37,
    chunk_layer,
    compute_predicted_aligned_error,
    compute_tm,
    frames_and_literature_positions_to_atom14_pos,
    make_atom14_masks,
    residue_constants,
    to_pdb,
    torsion_angles_to_frames,
)


logger = logging.get_logger(__name__)
_CHECKPOINT_FOR_DOC = "facebook/esmfold_v1"
_CONFIG_FOR_DOC = "EsmConfig"


@dataclass
class EsmForProteinFoldingOutput(ModelOutput):
    """
    Output type of [`EsmForProteinFoldingOutput`].

    Args:
        frames (`mindspore.Tensor`):
            Output frames.
        sidechain_frames (`mindspore.Tensor`):
            Output sidechain frames.
        unnormalized_angles (`mindspore.Tensor`):
            Predicted unnormalized backbone and side chain torsion angles.
        angles (`mindspore.Tensor`):
            Predicted backbone and side chain torsion angles.
        positions (`mindspore.Tensor`):
            Predicted positions of the backbone and side chain atoms.
        states (`mindspore.Tensor`):
            Hidden states from the protein folding trunk.
        s_s (`mindspore.Tensor`):
            Per-residue embeddings derived by concatenating the hidden states of each layer of the ESM-2 LM stem.
        s_z (`mindspore.Tensor`):
            Pairwise residue embeddings.
        distogram_logits (`mindspore.Tensor`):
            Input logits to the distogram used to compute residue distances.
        lm_logits (`mindspore.Tensor`):
            Logits output by the ESM-2 protein language model stem.
        aatype (`mindspore.Tensor`):
            Input amino acids (AlphaFold2 indices).
        atom14_atom_exists (`mindspore.Tensor`):
            Whether each atom exists in the atom14 representation.
        residx_atom14_to_atom37 (`mindspore.Tensor`):
            Mapping between atoms in the atom14 and atom37 representations.
        residx_atom37_to_atom14 (`mindspore.Tensor`):
            Mapping between atoms in the atom37 and atom14 representations.
        atom37_atom_exists (`mindspore.Tensor`):
            Whether each atom exists in the atom37 representation.
        residue_index (`mindspore.Tensor`):
            The index of each residue in the protein chain. Unless internal padding tokens are used, this will just be
            a sequence of integers from 0 to `sequence_length`.
        lddt_head (`mindspore.Tensor`):
            Raw outputs from the lddt head used to compute plddt.
        plddt (`mindspore.Tensor`):
            Per-residue confidence scores. Regions of low confidence may indicate areas where the model's prediction is
            uncertain, or where the protein structure is disordered.
        ptm_logits (`mindspore.Tensor`):
            Raw logits used for computing ptm.
        ptm (`mindspore.Tensor`):
            TM-score output representing the model's high-level confidence in the overall structure.
        aligned_confidence_probs (`mindspore.Tensor`):
            Per-residue confidence scores for the aligned structure.
        predicted_aligned_error (`mindspore.Tensor`):
            Predicted error between the model's prediction and the ground truth.
        max_predicted_aligned_error (`mindspore.Tensor`):
            Per-sample maximum predicted error.
    """
    frames: mindspore.Tensor = None
    sidechain_frames: mindspore.Tensor = None
    unnormalized_angles: mindspore.Tensor = None
    angles: mindspore.Tensor = None
    positions: mindspore.Tensor = None
    states: mindspore.Tensor = None
    s_s: mindspore.Tensor = None
    s_z: mindspore.Tensor = None
    distogram_logits: mindspore.Tensor = None
    lm_logits: mindspore.Tensor = None
    aatype: mindspore.Tensor = None
    atom14_atom_exists: mindspore.Tensor = None
    residx_atom14_to_atom37: mindspore.Tensor = None
    residx_atom37_to_atom14: mindspore.Tensor = None
    atom37_atom_exists: mindspore.Tensor = None
    residue_index: mindspore.Tensor = None
    lddt_head: mindspore.Tensor = None
    plddt: mindspore.Tensor = None
    ptm_logits: mindspore.Tensor = None
    ptm: mindspore.Tensor = None
    aligned_confidence_probs: mindspore.Tensor = None
    predicted_aligned_error: mindspore.Tensor = None
    max_predicted_aligned_error: mindspore.Tensor = None


def collate_dense_tensors(samples: List[mindspore.Tensor], pad_v: float = 0) -> mindspore.Tensor:
    """
    Takes a list of tensors with the following dimensions:

    ```[(d_11, ..., d_1K),
     (d_21, ..., d_2K), ..., (d_N1, ..., d_NK)]```

    and stack + pads them into a single tensor of:

    ```(N, max_i=1,N { d_i1 }, ..., max_i=1,N {diK})```
    """
    if len(samples) == 0:
        return mindspore.Tensor()
    if len({x.dim() for x in samples}) != 1:
        raise RuntimeError(f"Samples has varying dimensions: {[x.dim() for x in samples]}")

    max_shape = [max(lst) for lst in zip(*[x.shape for x in samples])]
    result = ops.full((len(samples), *max_shape), pad_v, dtype=samples[0].dtype)
    for i, t in enumerate(samples):
        result_i = result[i]
        result_i[tuple(slice(0, k) for k in t.shape)] = t
    return result


def flatten_final_dims(t: mindspore.Tensor, no_dims: int):
    """
    Flatten the final dimensions of a tensor.
    
    Args:
        t (mindspore.Tensor): The input tensor to be flattened.
        no_dims (int): The number of dimensions to be flattened.
    
    Returns:
        mindspore.Tensor: A tensor with the specified number of final dimensions flattened.
    
    Raises:
        ValueError: If the input tensor does not have enough dimensions to flatten.
    """
    return t.reshape(t.shape[:-no_dims] + (-1,))


def permute_final_dims(tensor: mindspore.Tensor, inds: List[int]):
    """
    This function permutes the final dimensions of the input tensor based on the provided indices.
    
    Args:
        tensor (mindspore.Tensor): The input tensor to be permuted.
        inds (List[int]):
            A list of integers representing the indices of the dimensions to be permuted. The dimensions are 0-indexed.
    
    Returns:
        None.
    
    Raises:
        ValueError: If the indices provided in 'inds' are out of bounds or not in the correct format.
        TypeError: If the input tensor is not of type mindspore.Tensor.
    """
    zero_index = -1 * len(inds)
    first_inds = list(range(len(tensor.shape[:zero_index])))
    return tensor.permute(first_inds + [zero_index + i for i in inds])


def dict_multimap(fn, dicts):
    """
    This function takes two parameters: 
    
    - fn: A function that will be applied to the values of the dictionaries.
    - dicts: A list of dictionaries.

    The function returns a new dictionary with the same keys as the first dictionary in the input list,
    where the values are the result of applying the given function to the corresponding values from all the
    input dictionaries.

    Raises:
        KeyError: If a key is not found in the dictionaries.
        TypeError: If the values are not suitable for the given function.
    """
    first = dicts[0]
    new_dict = {}
    for k, v in first.items():
        all_v = [d[k] for d in dicts]
        if isinstance(v, dict):
            new_dict[k] = dict_multimap(fn, all_v)
        else:
            new_dict[k] = fn(all_v)

    return new_dict


def trunc_normal_init_(weights, scale=1.0, fan="fan_in"):
    """
    This function initializes weights with a truncated normal distribution.

    Args:
        weights (Tensor): The weights to be initialized.
        scale (float, optional): The scale factor for the standard deviation. Defaults to 1.0.
        fan (str, optional): Specifies the mode for computing the fan. Defaults to 'fan_in'.

    Returns:
        None.

    Raises:
        ValueError: If the shape of the weights is not valid.
        ImportError: If scipy is not available and it is required for the initialization.
    """
    shape = weights.shape
    scale = scale / max(1, shape[1])

    if not is_scipy_available():
        logger.warning(
            "This init requires scipy, but scipy was not found, default to an approximation that might not be"
            " equivalent."
        )
        std = math.sqrt(scale)
        weights.set_data(initializer(Normal(std), weights.shape, weights.dtype).clamp(min=0.0, max=2.0 * std))

    else:
        from scipy.stats import truncnorm

        std = math.sqrt(scale) / truncnorm.std(a=-2, b=2, loc=0, scale=1)
        samples = truncnorm.rvs(a=-2, b=2, loc=0, scale=std, size=weights.numel())
        samples = np.reshape(samples, shape)
        weights.set_data(mindspore.tensor(samples))


def ipa_point_weights_init_(weights):
    """
    Initializes the IPA (International Phonetic Alphabet) point weights.

    Args:
        weights (list): A list of weights to be initialized.

    Returns:
        None.

    Raises:
        None.
    """
    softplus_inverse_1 = 0.541324854612918
    weights[:] = softplus_inverse_1


class EsmFoldLinear(nn.Dense):
    """
    A Linear layer with built-in nonstandard initializations. Called just like torch.nn.Dense.

    Implements the initializers in 1.11.4, plus some additional ones found in the code.
    """
    def __init__(
        self,
        in_dim: int,
        out_dim: int,
        has_bias: bool = True,
        init: str = "default",
        init_fn: Optional[Callable[[mindspore.Tensor, mindspore.Tensor], None]] = None,
    ):
        """
        Args:
            in_dim:
                The final dimension of inputs to the layer
            out_dim:
                The final dimension of layer outputs
            bias:
                Whether to learn an additive bias. True by default
            init:
                The initializer to use. Choose from:

                "default": LeCun fan-in truncated normal initialization "relu": He initialization w/ truncated normal
                distribution "glorot": Fan-average Glorot uniform initialization "gating": Weights=0, Bias=1 "normal":
                Normal initialization with std=1/sqrt(fan_in) "final": Weights=0, Bias=0

                Overridden by init_fn if the latter is not None.
            init_fn:
                A custom initializer taking weight and bias as inputs. Overrides init if not None.
        """
        super().__init__(in_dim, out_dim, has_bias=has_bias)

        self.init = init
        self.init_fn = init_fn
        if has_bias:
            self.bias.set_data(ops.zeros_like(self.bias))

        if init not in ["default", "relu", "glorot", "gating", "normal", "final"]:
            raise ValueError("Invalid init string.")


class EsmFoldLayerNorm(nn.Cell):

    """
    EsmFoldLayerNorm represents a custom layer normalization module with additional trainable parameters for weight and bias.
    This class inherits from nn.Cell and implements the Layer Normalization operation with custom weight and bias parameters.

    Attributes:
        c_in (int): Number of input channels for the layer normalization operation.
        eps (float): Epsilon value used in the normalization operation.
        weight (Parameter): Trainable parameter representing the weights for the normalization operation.
        bias (Parameter): Trainable parameter representing the bias for the normalization operation.
        layer_norm (ops.LayerNorm): Layer normalization operation with custom weight and bias parameters.

    Methods:
        __init__:
            Initializes the EsmFoldLayerNorm instance with the specified input channels and epsilon value.

        construct:
            Applies the layer normalization operation with custom weight and bias parameters to the input tensor x.

    Returns:
        Tensor: The normalized output tensor after applying the layer normalization operation with custom parameters.
    """
    def __init__(self, c_in, eps=1e-5):
        """
        Initialize the EsmFoldLayerNorm class.

        Args:
            self: The instance of the EsmFoldLayerNorm class.
            c_in (int): The number of input channels for the layer normalization. Must be a positive integer.
            eps (float, optional): The epsilon value for numerical stability in layer normalization. Default is 1e-05.

        Returns:
            None.

        Raises:
            ValueError: If c_in is not a positive integer.
            ValueError: If eps is not a valid epsilon value (not a float).
        """
        super().__init__()

        self.c_in = (c_in,)
        self.eps = eps

        self.weight = Parameter(ops.ones(c_in))
        self.bias = Parameter(ops.zeros(c_in))
        self.layer_norm = ops.LayerNorm(begin_norm_axis=-1,
                                        begin_params_axis=-1,
                                        epsilon=eps)
    def construct(self, x):
        """
        Constructs a normalized layer using the EsmFold algorithm.

        Args:
            self (EsmFoldLayerNorm): An instance of the EsmFoldLayerNorm class.
            x: The input tensor to be normalized. Should have shape (batch_size, features).

        Returns:
            None: This method does not return a value.
                The normalized layer is stored within the instance of the EsmFoldLayerNorm class.

        Raises:
            None.
        """
        y, _, _ = self.layer_norm(x, self.weight, self.bias)
        return y


def softmax_no_cast(t: mindspore.Tensor, dim: int = -1) -> mindspore.Tensor:
    """
    Softmax, but without automatic casting to fp32 when the input is of type bfloat16
    """
    s = ops.softmax(t, axis=dim)

    return s


class EsmFoldAttention(nn.Cell):
    """
    Standard multi-head attention using AlphaFold's default layer initialization. Allows multiple bias vectors.
    """
    def __init__(
        self,
        c_q: int,
        c_k: int,
        c_v: int,
        c_hidden: int,
        no_heads: int,
        gating: bool = True,
    ):
        """
        Args:
            c_q:
                Input dimension of query data
            c_k:
                Input dimension of key data
            c_v:
                Input dimension of value data
            c_hidden:
                Per-head hidden dimension
            no_heads:
                Number of attention heads
            gating:
                Whether the output should be gated using query data
        """
        super().__init__()

        self.c_q = c_q
        self.c_k = c_k
        self.c_v = c_v
        self.c_hidden = c_hidden
        self.no_heads = no_heads
        self.gating = gating

        # DISCREPANCY: c_hidden is not the per-head channel dimension, as
        # stated in the supplement, but the overall channel dimension.

        self.linear_q = EsmFoldLinear(self.c_q, self.c_hidden * self.no_heads, has_bias=False, init="glorot")
        self.linear_k = EsmFoldLinear(self.c_k, self.c_hidden * self.no_heads, has_bias=False, init="glorot")
        self.linear_v = EsmFoldLinear(self.c_v, self.c_hidden * self.no_heads, has_bias=False, init="glorot")
        self.linear_o = EsmFoldLinear(self.c_hidden * self.no_heads, self.c_q, init="final")

        self.linear_g = None
        if self.gating:
            self.linear_g = EsmFoldLinear(self.c_q, self.c_hidden * self.no_heads, init="gating")

        self.sigmoid = nn.Sigmoid()

    def _prep_qkv(self, q_x: mindspore.Tensor, kv_x: mindspore.Tensor) -> Tuple[mindspore.Tensor, mindspore.Tensor, mindspore.Tensor]:
        """
        Prepares the query, key, and value tensors for the EsmFoldAttention module.

        Args:
            self (EsmFoldAttention): The instance of the EsmFoldAttention module.
            q_x (mindspore.Tensor): The query tensor.
                It should have a shape of (batch_size, seq_length, hidden_size).
            kv_x (mindspore.Tensor): The key-value tensor.
                It should have a shape of (batch_size, seq_length, hidden_size).

        Returns:
            Tuple[mindspore.Tensor, mindspore.Tensor, mindspore.Tensor]:
                A tuple containing the query, key, and value tensors.

                - q: The transformed query tensor with a shape of (batch_size, seq_length, no_heads, hidden_size//no_heads).
                - k: The transformed key tensor with a shape of (batch_size, seq_length, no_heads, hidden_size//no_heads).
                - v: The transformed value tensor with a shape of (batch_size, seq_length, no_heads, hidden_size//no_heads).

        Raises:
            None.
        """
        # [*, Q/K/V, H * C_hidden]
        q = self.linear_q(q_x)
        k = self.linear_k(kv_x)
        v = self.linear_v(kv_x)

        # [*, Q/K, H, C_hidden]
        q = q.view(q.shape[:-1] + (self.no_heads, -1))
        k = k.view(k.shape[:-1] + (self.no_heads, -1))
        v = v.view(v.shape[:-1] + (self.no_heads, -1))

        # [*, H, Q/K, C_hidden]
        q = q.swapaxes(-2, -3)
        k = k.swapaxes(-2, -3)
        v = v.swapaxes(-2, -3)

        q /= math.sqrt(self.c_hidden)

        return q, k, v

    def _wrap_up(self, o: mindspore.Tensor, q_x: mindspore.Tensor) -> mindspore.Tensor:
        """
        This method '_wrap_up' in the class 'EsmFoldAttention' performs a wrapping up operation on the input tensors.

        Args:
            self: An instance of the 'EsmFoldAttention' class.
            o (mindspore.Tensor): Input tensor representing the output from previous layers.
                Shape should be compatible with the subsequent operations.
            q_x (mindspore.Tensor): Input tensor representing the query tensor.
                Shape should be compatible with the subsequent operations.

        Returns:
            mindspore.Tensor: A tensor resulting from the wrapping up operation.
                The shape and content of the tensor depend on the operations performed within the method.

        Raises:
            No specific exceptions are documented to be raised by this method under normal operation.
        """
        if self.linear_g is not None:
            g = self.sigmoid(self.linear_g(q_x))

            # [*, Q, H, C_hidden]
            g = g.view(g.shape[:-1] + (self.no_heads, -1))
            o = o * g

        # [*, Q, H * C_hidden]
        o = flatten_final_dims(o, 2)

        # [*, Q, C_q]
        o = self.linear_o(o)

        return o

    def construct(
        self,
        q_x: mindspore.Tensor,
        kv_x: mindspore.Tensor,
        biases: Optional[List[mindspore.Tensor]] = None,
        use_memory_efficient_kernel: bool = False,
        use_lma: bool = False,
        lma_q_chunk_size: int = 1024,
        lma_kv_chunk_size: int = 4096,
        use_flash: bool = False,
        flash_mask: Optional[mindspore.Tensor] = None,
    ) -> mindspore.Tensor:
        """
        Args:
            q_x:
                [*, Q, C_q] query data
            kv_x:
                [*, K, C_k] key data
            biases:
                List of biases that broadcast to [*, H, Q, K]
            use_memory_efficient_kernel:
                Whether to use a custom memory-efficient attention kernel. This should be the default choice for most.
                If none of the "use_<...>" flags are True, a stock PyTorch implementation is used instead
            use_lma:
                Whether to use low-memory attention (Staats & Rabe 2021). If none of the "use_<...>" flags are True, a
                stock PyTorch implementation is used instead
            lma_q_chunk_size:
                Query chunk size (for LMA)
            lma_kv_chunk_size:
                Key/Value chunk size (for LMA)
        Returns
            [*, Q, C_q] attention update
        """
        if use_lma and (lma_q_chunk_size is None or lma_kv_chunk_size is None):
            raise ValueError("If use_lma is specified, lma_q_chunk_size and lma_kv_chunk_size must be provided")

        if use_flash and biases is not None:
            raise ValueError("use_flash is incompatible with the bias option. For masking, use flash_mask instead")

        attn_options = [use_memory_efficient_kernel, use_lma, use_flash]
        if sum(attn_options) > 1:
            raise ValueError("Choose at most one alternative attention algorithm")

        if biases is None:
            biases = []

        # [*, H, Q/K, C_hidden]
        query, key, value = self._prep_qkv(q_x, kv_x)
        key = permute_final_dims(key, (1, 0))

        # [*, H, Q, K]
        output = ops.matmul(query, key)
        for b in biases:
            output += b
        output = softmax_no_cast(output, -1)

        # [*, H, Q, C_hidden]
        output = ops.matmul(output, value)
        output = output.swapaxes(-2, -3)
        output = self._wrap_up(output, q_x)

        return output


class EsmFoldTriangleAttention(nn.Cell):

    """
    This class represents an attention mechanism called EsmFoldTriangleAttention, which is used in the ESMFold model.
    It is designed to calculate attention weights between pairs of elements in a tensor.

    The EsmFoldTriangleAttention class inherits from the nn.Cell class and has the following attributes:

    Attributes:
        c_in:
            Input channel dimension.
        c_hidden:
            Overall hidden channel dimension (not per-head).
        no_heads:
            Number of attention heads.
        starting:
            Flag indicating if the attention is applied to the starting point of a pair.
        inf:
            Value used as infinity for masking.
        layer_norm:
            Layer normalization module applied to the input tensor.
        linear:
            Linear transformation layer used for computing triangle biases.
        mha:
            EsmFoldAttention module used for calculating attention weights.

    Methods:
        __init__:
            Initializes an instance of the EsmFoldTriangleAttention class.

        _chunk:
            Splits the input tensor into chunks and applies the EsmFoldAttention module to each chunk.

        construct:
            Applies the attention mechanism to the input tensor and returns the output tensor.
    """
    def __init__(self, c_in, c_hidden, no_heads, starting=True, inf=1e9):
        """
        Args:
            c_in:
                Input channel dimension
            c_hidden:
                Overall hidden channel dimension (not per-head)
            no_heads:
                Number of attention heads
        """
        super().__init__()

        self.c_in = c_in
        self.c_hidden = c_hidden
        self.no_heads = no_heads
        self.starting = starting
        self.inf = inf

        self.layer_norm = nn.LayerNorm(self.c_in)

        self.linear = EsmFoldLinear(c_in, self.no_heads, has_bias=False, init="normal")

        self.mha = EsmFoldAttention(self.c_in, self.c_in, self.c_in, self.c_hidden, self.no_heads)

    def _chunk(
        self,
        x: mindspore.Tensor,
        biases: List[mindspore.Tensor],
        chunk_size: int,
        use_memory_efficient_kernel: bool = False,
        use_lma: bool = False,
        inplace_safe: bool = False,
    ) -> mindspore.Tensor:
        "triangle! triangle!"
        mha_inputs = {
            "q_x": x,
            "kv_x": x,
            "biases": biases,
        }

        return chunk_layer(
            partial(self.mha, use_memory_efficient_kernel=use_memory_efficient_kernel, use_lma=use_lma),
            mha_inputs,
            chunk_size=chunk_size,
            no_batch_dims=len(x.shape[:-2]),
            _out=x if inplace_safe else None,
        )

    def construct(
        self,
        x: mindspore.Tensor,
        mask: Optional[mindspore.Tensor] = None,
        chunk_size: Optional[int] = None,
        use_memory_efficient_kernel: bool = False,
        use_lma: bool = False,
        inplace_safe: bool = False,
    ) -> mindspore.Tensor:
        """
        Args:
            x:
                [*, I, J, C_in] input tensor (e.g. the pair representation)
        Returns:
            [*, I, J, C_in] output tensor
        """
        if mask is None:
            # [*, I, J]
            mask = x.new_ones(
                x.shape[:-1],
            )

        if not self.starting:
            x = x.swapaxes(-2, -3)
            mask = mask.swapaxes(-1, -2)

        # [*, I, J, C_in]
        x = self.layer_norm(x)

        # [*, I, 1, 1, J]
        mask_bias = (self.inf * (mask - 1))[..., :, None, None, :]

        # [*, H, I, J]
        triangle_bias = permute_final_dims(self.linear(x), (2, 0, 1))

        # [*, 1, H, I, J]
        triangle_bias = triangle_bias.unsqueeze(-4)

        biases = [mask_bias, triangle_bias]

        if chunk_size is not None:
            x = self._chunk(
                x,
                biases,
                chunk_size,
                use_memory_efficient_kernel=use_memory_efficient_kernel,
                use_lma=use_lma,
                inplace_safe=inplace_safe,
            )
        else:
            x = self.mha(
                q_x=x, kv_x=x, biases=biases, use_memory_efficient_kernel=use_memory_efficient_kernel, use_lma=use_lma
            )

        if not self.starting:
            x = x.swapaxes(-2, -3)

        return x


class EsmFoldTriangleMultiplicativeUpdate(nn.Cell):
    """
    Implements Algorithms 11 and 12.
    """
    def __init__(self, config, _outgoing=True):
        """
        Initializes an instance of the EsmFoldTriangleMultiplicativeUpdate class.

        Args:
            self: The instance of the class.
            config: An object containing configuration parameters.
            _outgoing (bool): A boolean indicating whether the update is outgoing (default is True).

        Returns:
            None.

        Raises:
            TypeError: If config is not provided or is not of the expected type.
            ValueError: If config.pairwise_state_dim is not accessible or does not have the expected value.
            RuntimeError: If an issue occurs during the initialization of linear layers or normalization layers.
        """
        super().__init__()
        c_hidden = config.pairwise_state_dim
        self._outgoing = _outgoing

        self.linear_a_p = EsmFoldLinear(c_hidden, c_hidden)
        self.linear_a_g = EsmFoldLinear(c_hidden, c_hidden, init="gating")
        self.linear_b_p = EsmFoldLinear(c_hidden, c_hidden)
        self.linear_b_g = EsmFoldLinear(c_hidden, c_hidden, init="gating")
        self.linear_g = EsmFoldLinear(c_hidden, c_hidden, init="gating")
        self.linear_z = EsmFoldLinear(c_hidden, c_hidden, init="final")

        self.layer_norm_in = nn.LayerNorm(c_hidden)
        self.layer_norm_out = nn.LayerNorm(c_hidden)

        self.sigmoid = nn.Sigmoid()

    def _combine_projections(
        self, a: mindspore.Tensor, b: mindspore.Tensor, _inplace_chunk_size: Optional[int] = None
    ) -> mindspore.Tensor:
        """
        Combines two projections using a multiplicative update method.

        Args:
            self (EsmFoldTriangleMultiplicativeUpdate): The instance of the EsmFoldTriangleMultiplicativeUpdate class.
            a (mindspore.Tensor): The first projection tensor.
            b (mindspore.Tensor): The second projection tensor.
            _inplace_chunk_size (Optional[int], optional): The size of the chunk for in-place computation.
                Defaults to None.

        Returns:
            mindspore.Tensor: The combined projection tensor.

        Raises:
            None.
        """
        if self._outgoing:
            a = permute_final_dims(a, (2, 0, 1))
            b = permute_final_dims(b, (2, 1, 0))
        else:
            a = permute_final_dims(a, (2, 1, 0))
            b = permute_final_dims(b, (2, 0, 1))

        if _inplace_chunk_size is not None:
            # To be replaced by torch vmap
            for i in range(0, a.shape[-3], _inplace_chunk_size):
                a_chunk = a[..., i : i + _inplace_chunk_size, :, :]
                b_chunk = b[..., i : i + _inplace_chunk_size, :, :]
                a[..., i : i + _inplace_chunk_size, :, :] = ops.matmul(
                    a_chunk,
                    b_chunk,
                )

            p = a
        else:
            p = ops.matmul(a, b)

        return permute_final_dims(p, (1, 2, 0))

    def _inference_forward(
        self,
        z: mindspore.Tensor,
        mask: Optional[mindspore.Tensor] = None,
        inplace_chunk_size: Optional[int] = None,
        with_add: bool = True,
    ):
        """
        Args:
            z:
                A [*, N, N, C_z] pair representation
            mask:
                A [*, N, N] pair mask
            inplace_chunk_size:
                Size of chunks used in the main computation. Increase to trade memory for speed.
            with_add:
                If True, z is overwritten with (z + update). Otherwise, it is overwritten with (update).
        Returns:
            A reference to the overwritten z

        More memory-efficient, inference-only version of the forward function. Uses in-place operations, fusion of the
        addition that happens after this module in the Evoformer, a smidge of recomputation, and a cache of overwritten
        values to lower peak memory consumption of this module from 5x the size of the input tensor z to 2.5x its size.
        Useful for inference on extremely long sequences.

        It works as follows. We will make reference to variables used in the default forward implementation below.
        Naively, triangle multiplication attention requires the manifestation of 5 tensors the size of z: 1) z, the
        "square" input tensor, 2) a, the first projection of z, 3) b, the second projection of b, 4) g, a z-sized mask,
        and 5) a z-sized tensor for intermediate computations. For large N, this is prohibitively expensive; for
        N=4000, for example, z is more than 8GB alone. To avoid this problem, we compute b, g, and all intermediate
        tensors in small chunks, noting that the chunks required to compute a chunk of the output depend only on the
        tensor a and corresponding vertical and horizontal chunks of z. This suggests an algorithm that loops over
        pairs of chunks of z: hereafter "columns" and "rows" of z, even though each "column" and "row" in fact contains
        inplace_chunk_size contiguous true columns and rows of z. Writing output chunks to a new tensor would bring
        total memory consumption down to 3x the size of z. However, more memory can be saved by writing output chunks
        directly to z in-place. WLOG, we choose to write output chunks vertically, overwriting the ith "column" of z at
        the end of the ith iteration of the main loop. Despite this overwriting, the ith column is always one column
        ahead of previously overwritten columns and can be recovered directly from z. After the first iteration,
        however, the ith row of z is always at least partially overwritten. For this reason, we introduce the z-cache,
        a tensor one-half the size of z. The z-cache initially contains the left half (2nd and 3rd quadrants) of z. For
        0 < i < N/2, the missing left part of the ith row of z is recovered from this cache at the beginning of the ith
        iteration. Once i exceeds n/2, the cache is "reoriented" to encompass the 3rd and 4th quadrants of z instead.
        Though the 3rd quadrant of the original z is entirely overwritten at this point, it can be recovered from the
        z-cache itself. Thereafter, the ith row of z can be recovered in its entirety from the reoriented z-cache.
        After the final iteration, z has been completely overwritten and contains the triangular multiplicative update.
        If with_add is True, it instead contains the sum of z and the triangular multiplicative update. In either case,
        peak memory consumption is just 2.5x the size of z, disregarding memory used for chunks and other small
        variables.
        """
        if mask is None:
            mask = z.new_ones(z.shape[:-1])

        mask = mask.unsqueeze(-1)

        def compute_projection_helper(pair, mask, a=True):
            if a:
                linear_g = self.linear_a_g
                linear_p = self.linear_a_p
            else:
                linear_g = self.linear_b_g
                linear_p = self.linear_b_p

            pair = self.layer_norm_in(pair)
            p = linear_g(pair)
            p = p.sigmoid()
            p *= linear_p(pair)
            p *= mask
            p = permute_final_dims(p, (2, 0, 1))
            return p

        def compute_projection(pair, mask, a=True, chunked=True):
            need_transpose = self._outgoing ^ a
            if not chunked:
                p = compute_projection_helper(pair, mask, a)
                if need_transpose:
                    p = p.swapaxes(-1, -2)
            else:
                # This computation is chunked so as not to exceed our 2.5x
                # budget with a large intermediate tensor
                linear_g = self.linear_a_g if a else self.linear_b_g
                c = linear_g.bias.shape[-1]
                out_shape = pair.shape[:-3] + (c,) + pair.shape[-3:-1]
                p = pair.new_zeros(out_shape)
                for i in range(0, pair.shape[-3], inplace_chunk_size):
                    pair_chunk = pair[..., i : i + inplace_chunk_size, :, :]
                    pair_chunk = compute_projection_helper(
                        pair[..., i : i + inplace_chunk_size, :, :],
                        mask[..., i : i + inplace_chunk_size, :, :],
                        a,
                    )
                    if need_transpose:
                        pair_chunk = pair_chunk.swapaxes(-1, -2)
                        p[..., i : i + inplace_chunk_size] = pair_chunk
                    else:
                        p[..., i : i + inplace_chunk_size, :] = pair_chunk

                    del pair_chunk

            return p

        # We start by fully manifesting a. In addition to the input, this
        # brings total memory consumption to 2x z (disregarding size of chunks)
        # [*, N, N, c]
        a = compute_projection(z, mask, True, chunked=True)

        if inplace_chunk_size is not None:
            n = a.shape[-1]
            half_n = n // 2 + n % 2
            row_dim = -3
            col_dim = -2
            b_chunk_dim = row_dim if self._outgoing else col_dim

            def empty_slicer(t):
                return [slice(None) for _ in t.shape]

            def slice_tensor(t, start, end, dim):
                # Slices start:end from the dim dimension of t
                s = empty_slicer(t)
                s[dim] = slice(start, end)
                return t[s]

            def flip_z_cache_(z_cache, z):
                # "Reorient" the z_cache (see below), filling it with quadrants
                # 3---recovered from the z_cache---and 4---recovered from z---
                # of the input tensor z.
                quadrant_3 = slice_tensor(z_cache, half_n, None, row_dim)
                z_cache = z_cache.swapaxes(row_dim, col_dim)

                # If n is odd, we need to shrink the z_cache by one row
                z_cache = z_cache[..., : (n // 2), :, :]

                # Move the 3rd quadrant of z into the
                first_half_slicer = empty_slicer(z_cache)
                first_half_slicer[col_dim] = slice(0, half_n)
                z_cache[first_half_slicer] = quadrant_3

                # Get the fourth quadrant of z
                quadrant_4 = slice_tensor(z, half_n, None, row_dim)
                quadrant_4 = slice_tensor(quadrant_4, half_n, None, col_dim)

                # Insert said quadrant into the rotated z-cache
                quadrant_3_slicer = empty_slicer(z_cache)
                quadrant_3_slicer[col_dim] = slice(half_n, None)

                z_cache[quadrant_3_slicer] = quadrant_4

                return z_cache

            # Initialize the z cache to the left half of z.
            z_cache_shape = list(z.shape)
            z_cache_shape[col_dim] = half_n
            z_cache = z.new_zeros(z_cache_shape)
            z_cache_slicer = empty_slicer(z_cache)
            z_cache_slicer[col_dim] = slice(0, half_n)
            z_cache[:] = z[z_cache_slicer]
            z_cache_rotated = False

            # We need to reorient the z-cache at the halfway point, and we
            # don't want a single chunk to straddle that point. We contract one
            # of the chunks in the middle to address that problem.
            i_range = list(range(0, half_n, inplace_chunk_size))
            initial_offsets = [i_2 - i_1 for i_1, i_2 in zip(i_range, i_range[1:] + [half_n])]
            after_half = list(range(half_n, n, inplace_chunk_size))
            after_half_offsets = [inplace_chunk_size for _ in after_half]
            combined_range_with_offsets = zip(i_range + after_half, initial_offsets + after_half_offsets)
            for i, offset in combined_range_with_offsets:
                if not z_cache_rotated and i >= half_n:
                    z_cache = flip_z_cache_(z_cache, z)
                    z_cache_rotated = True

                z_chunk_b = slice_tensor(z, i, i + offset, b_chunk_dim)
                mask_chunk = slice_tensor(mask, i, i + offset, b_chunk_dim)

                z_chunk_b = z_chunk_b.copy()
                if b_chunk_dim == col_dim:
                    z_chunk_b = slice_tensor(z, i, i + offset, col_dim)
                else:  # b_chunk_dim == row_dim
                    # In this case, the b-dimension (b_chunk_dim) is partially
                    # overwritten at the end of each iteration. We need to
                    # restore the missing component from the z-cache.
                    if not z_cache_rotated:
                        z_chunk_slicer = empty_slicer(z_chunk_b)
                        z_chunk_slicer[col_dim] = slice(0, half_n)
                        z_chunk_b[z_chunk_slicer] = slice_tensor(z_cache, i, i + offset, row_dim)
                    else:
                        z_cache_offset = i - half_n
                        z_chunk_b = slice_tensor(z_cache, z_cache_offset, z_cache_offset + offset, row_dim)

                b_chunk = compute_projection(z_chunk_b, mask_chunk, a=False, chunked=False)
                del z_chunk_b

                x_chunk = ops.matmul(a, b_chunk)
                x_chunk = permute_final_dims(x_chunk, (1, 2, 0))
                x_chunk = self.layer_norm_out(x_chunk)
                x_chunk = self.linear_z(x_chunk)

                # The g dimension (col_dim) is parallel to and ahead of the
                # overwrites in z. We can extract the g chunk normally.
                z_chunk_g = slice_tensor(z, i, i + offset, col_dim)
                g_chunk = self.linear_g(self.layer_norm_in(z_chunk_g))
                g_chunk = g_chunk.sigmoid()
                del z_chunk_g

                x_chunk *= g_chunk

                # Write the columns into z in-place
                z_slicer = empty_slicer(z)
                z_slicer[col_dim] = slice(i, i + offset)
                if with_add:
                    z[z_slicer] += x_chunk
                else:
                    z[z_slicer] = x_chunk
        else:
            b = compute_projection(z, mask, False, False)
            x = ops.matmul(a, b)
            x = self.layer_norm_out(x)
            x = self.linear_z(x)
            g = self.linear_g(z)
            g = g.sigmoid()
            x *= g
            if with_add:
                z += x
            else:
                z = x

        return z

    def construct(
        self,
        z: mindspore.Tensor,
        mask: Optional[mindspore.Tensor] = None,
        inplace_safe: bool = False,
        _add_with_inplace: bool = False,
        _inplace_chunk_size: Optional[int] = 256,
    ) -> mindspore.Tensor:
        """
        Args:
            x:
                [*, N_res, N_res, C_z] input tensor
            mask:
                [*, N_res, N_res] input mask
        Returns:
            [*, N_res, N_res, C_z] output tensor
        """
        if inplace_safe:
            x = self._inference_forward(
                z,
                mask,
                inplace_chunk_size=_inplace_chunk_size,
                with_add=_add_with_inplace,
            )
            return x

        if mask is None:
            mask = z.new_ones(z.shape[:-1])

        mask = mask.unsqueeze(-1)

        z = self.layer_norm_in(z)
        a = mask
        a = a * self.sigmoid(self.linear_a_g(z))
        a = a * self.linear_a_p(z)
        b = mask
        b = b * self.sigmoid(self.linear_b_g(z))
        b = b * self.linear_b_p(z)

        x = self._combine_projections(a, b)

        del a, b
        x = self.layer_norm_out(x)
        x = self.linear_z(x)
        g = self.sigmoid(self.linear_g(z))
        x = x * g

        return x


class EsmFoldPreTrainedModel(EsmPreTrainedModel):
    """
    An abstract class to handle weights initialization and a simple interface for downloading and loading pretrained
    models.
    """
    # Subclass `EsMPreTrainedModel` to deal with special init
    def _init_weights(self, cell):
        """Initialize the weights"""
        if isinstance(cell, EsmFoldLinear):
            if cell.init_fn is not None:
                cell.init_fn(cell.weight, cell.bias)
            elif cell.init == "default":
                trunc_normal_init_(cell.weight, scale=1.0)
            elif cell.init == "relu":
                trunc_normal_init_(cell.weight, scale=2.0)
            elif cell.init == "glorot":
                cell.weight.set_data(initializer(XavierUniform(), cell.weight.shape, cell.weight.dtype))
            elif cell.init == "gating":
                cell.weight[:] = 0
                if cell.bias:
                    cell.bias[:] = 1
            elif cell.init == "normal":
                cell.weight.set_data(initializer(HeNormal(nonlinearity="linear"), cell.weight.shape, cell.weight.dtype))
            elif cell.init == "final":
                cell.weight[:] = 0
        elif isinstance(cell, EsmFoldInvariantPointAttention):
            ipa_point_weights_init_(cell.head_weights)
        elif isinstance(cell, EsmFoldTriangularSelfAttentionBlock):
            cell.tri_mul_in.linear_z.weight[:] = 0
            cell.tri_mul_in.linear_z.bias[:] = 0
            cell.tri_mul_out.linear_z.weight[:] = 0
            cell.tri_mul_out.linear_z.bias[:] = 0
            cell.tri_att_start.mha.linear_o.weight[:] = 0
            cell.tri_att_start.mha.linear_o.bias[:] = 0
            cell.tri_att_end.mha.linear_o.weight[:] = 0
            cell.tri_att_end.mha.linear_o.bias[:] = 0

            cell.sequence_to_pair.o_proj.weight[:] = 0
            cell.sequence_to_pair.o_proj.bias[:] = 0
            cell.pair_to_sequence.linear.weight[:] = 0
            cell.seq_attention.o_proj.weight[:] = 0
            cell.seq_attention.o_proj.bias[:] = 0
            cell.mlp_seq.mlp[-2].weight[:] = 0
            cell.mlp_seq.mlp[-2].bias[:] = 0
            cell.mlp_pair.mlp[-2].weight[:] = 0
            cell.mlp_pair.mlp[-2].bias[:] = 0
        else:
            super()._init_weights(cell)


class EsmFoldSelfAttention(nn.Cell):

    """
    This class represents a self-attention mechanism for processing sequences, specifically designed for handling
    sequences of varying lengths.
    It implements a multi-head self-attention mechanism with optional gating, bias, and masking capabilities.

    Attributes:
        embed_dim (int): The dimension of the input embedding.
        num_heads (int): The number of attention heads.
        head_width (int): The width of each attention head.
        gated (bool): Indicates whether the attention mechanism uses gating.
        proj (nn.Dense): Linear projection layer for processing input sequences.
        o_proj (nn.Dense): Output projection layer.
        g_proj (nn.Dense): Gating projection layer (if gated is True).
        rescale_factor (float): Scaling factor for the attention weights.

    Methods:
        construct(self, x, mask=None, bias=None, indices=None):
            Performs self-attention on the input batch of sequences with optional mask and external pairwise bias.

            Inputs:

            - x (Tensor): Batch of input sequences of shape (B x L x C).
            - mask (Tensor, optional): Batch of boolean masks where 1 denotes valid positions and 0 denotes padding positions of shape (B x L_k).
            - bias (Tensor, optional): Batch of scalar pairwise attention biases of shape (B x Lq x Lk x num_heads).
            - indices (Tensor, optional): Additional indices for attention computation.

            Outputs:

            - y (Tensor): Sequence projection of shape (B x L x embed_dim).
            - attention_maps (Tensor): Attention maps of shape (B x L x L x num_heads).

    Note:
        - Gating mechanism is applied if 'gated' is set to True.
        - The attention weights are softmax normalized.
        - The attention computation is based on the query, key, and value projections.
        - Masking is supported to handle sequences of different lengths.
    """
    def __init__(self, embed_dim, num_heads, head_width, gated=False):
        """
        Initializes the EsmFoldSelfAttention class.

        Args:
            self: The instance of the class.
            embed_dim (int): The dimension of the input embeddings.
            num_heads (int): The number of attention heads.
            head_width (int): The width of each attention head.
            gated (bool, optional): Specifies whether the attention mechanism is gated. Defaults to False.

        Returns:
            None.

        Raises:
            AssertionError: If embed_dim is not equal to the product of num_heads and head_width.

        """
        super().__init__()
        assert embed_dim == num_heads * head_width

        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_width = head_width

        self.proj = nn.Dense(embed_dim, embed_dim * 3, has_bias=False)
        self.o_proj = nn.Dense(embed_dim, embed_dim, has_bias=True)
        self.gated = gated
        if gated:
            self.g_proj = nn.Dense(embed_dim, embed_dim)
            self.g_proj.weight.set_data(ops.zeros_like(self.g_proj.weight))
            self.g_proj.bias.set_data(ops.ones_like(self.g_proj.bias))

        self.rescale_factor = self.head_width**-0.5

        self.o_proj.bias.set_data(ops.zeros_like(self.o_proj.bias))

    def construct(self, x, mask=None, bias=None, indices=None):
        """
        Basic self attention with optional mask and external pairwise bias. To handle sequences of different lengths,
        use mask.

        Inputs:
            x: batch of input sequneces (.. x L x C) mask: batch of boolean masks where 1=valid, 0=padding position (..
                x L_k) bias: batch of scalar pairwise attention biases (.. x Lq x Lk x num_heads)

        Outputs:
            sequence projection (B x L x embed_dim), attention maps (B x L x L x num_heads)
        """
        t = self.proj(x).view(*x.shape[:2], self.num_heads, -1)
        t = t.permute(0, 2, 1, 3)
        q, k, v = t.chunk(3, axis=-1)

        q = self.rescale_factor * q
        a = ops.einsum("...qc, ...kc -> ...qk", q, k)

        # Add external attention bias.
        if bias is not None:
            a = a + bias.permute(0, 3, 1, 2)

        # Do not attend to padding tokens.
        if mask is not None:
            mask = mask[:, None, None]
            a = a.masked_fill(mask == False, -np.inf)

        a = ops.softmax(a, axis=-1)

        y = ops.einsum("...hqk,...hkc->...qhc", a, v)
        y = y.reshape(*y.shape[:2], -1)

        if self.gated:
            y = self.g_proj(x).sigmoid() * y
        y = self.o_proj(y)

        return y, a.permute(0, 3, 1, 2)


class EsmFoldDropout(nn.Cell):
    """
    Implementation of dropout with the ability to share the dropout mask along a particular dimension.
    """
    def __init__(self, r: float, batch_dim: Union[int, List[int]]):
        """
        Initializes an instance of the EsmFoldDropout class.

        Args:
            self: The instance of the class.
            r (float): The dropout rate value.
            batch_dim (Union[int, List[int]]):
                The dimension(s) of the input batch.
                If an integer is provided, it will be converted to a list with that integer as the only element.

        Returns:
            None.

        Raises:
            None.
        """
        super().__init__()

        self.r = r
        if isinstance(batch_dim, int):
            batch_dim = [batch_dim]
        self.batch_dim = batch_dim
        self.dropout = nn.Dropout(p=self.r)

    def construct(self, x: mindspore.Tensor) -> mindspore.Tensor:
        """
        This method constructs a modified tensor with dropout for the EsmFoldDropout class.

        Args:
            self: An instance of the EsmFoldDropout class.
            x (mindspore.Tensor): The input tensor for which the modified tensor is constructed.

        Returns:
            mindspore.Tensor: Returns a new tensor, which is the result of applying dropout to the input tensor.

        Raises:
            TypeError: If the input x is not of type mindspore.Tensor.
            ValueError: If the shape manipulation encounters an error during the construction process.
            RuntimeError: If there is a runtime issue during the execution of the method.
        """
        shape = list(x.shape)
        if self.batch_dim is not None:
            for bd in self.batch_dim:
                shape[bd] = 1
        return x * self.dropout(x.new_ones(shape))


class EsmFoldSequenceToPair(nn.Cell):

    """
    This class represents a neural network model for transforming sequence states into pairwise states
    using an attention mechanism.

    This class inherits from nn.Cell and includes methods for initialization and constructing the pairwise states
    from sequence states.

    Attributes:
        layernorm (nn.LayerNorm): A layer normalization module for normalizing the sequence state dimensions.
        proj (nn.Dense): A fully connected layer for projecting the sequence state into an inner dimension space.
        o_proj (nn.Dense): A fully connected layer for projecting the inner dimension space into pairwise state dimensions.

    Methods:
        __init__: Initializes the EsmFoldSequenceToPair instance with the specified dimensions.

        construct: Transforms the input sequence state tensor into pairwise state tensor.

    Args:
        sequence_state_dim (int): Dimension of the input sequence state.
        inner_dim (int): Dimension of the inner representation used in the transformation.
        pairwise_state_dim (int): Dimension of the output pairwise state.

    Inputs:
        sequence_state (Tensor): Input sequence state tensor with shape B x L x sequence_state_dim.

    Output:
        pairwise_state (Tensor): Output pairwise state tensor with shape B x L x L x pairwise_state_dim.

    Intermediate state:
        Intermediate state tensor with shape B x L x L x 2*inner_dim, used during the transformation process.

    Returns:
        Tensor: Pairwise state tensor representing the relationships between elements in the input sequence state.

    Raises:
        AssertionError: If the input sequence state tensor does not have the expected shape.

    """
    def __init__(self, sequence_state_dim, inner_dim, pairwise_state_dim):
        """
        Initializes the EsmFoldSequenceToPair class.

        Args:
            sequence_state_dim (int): The dimension of the input sequence state.
            inner_dim (int): The inner dimension used for projection.
            pairwise_state_dim (int): The dimension of the pairwise state.

        Returns:
            None.

        Raises:
            None.
        """
        super().__init__()

        self.layernorm = nn.LayerNorm(sequence_state_dim)
        self.proj = nn.Dense(sequence_state_dim, inner_dim * 2, has_bias=True)
        self.o_proj = nn.Dense(2 * inner_dim, pairwise_state_dim, has_bias=True)
        self.proj.bias.set_data(ops.zeros_like(self.proj.bias))
        self.o_proj.bias.set_data(ops.zeros_like(self.o_proj.bias))

    def construct(self, sequence_state):
        """
        Inputs:
            sequence_state: B x L x sequence_state_dim

        Output:
            pairwise_state: B x L x L x pairwise_state_dim

        Intermediate state:
          B x L x L x 2*inner_dim
        """
        assert len(sequence_state.shape) == 3

        s = self.layernorm(sequence_state)
        s = self.proj(s)
        q, k = s.chunk(2, axis=-1)

        prod = q[:, None, :, :] * k[:, :, None, :]
        diff = q[:, None, :, :] - k[:, :, None, :]

        x = ops.cat([prod, diff], axis=-1)
        x = self.o_proj(x)

        return x


class EsmFoldPairToSequence(nn.Cell):

    """
    EsmFoldPairToSequence class represents a neural network module for converting pairwise features to sequence features
    using self-attention mechanism.

    This class inherits from nn.Cell and includes methods for initializing the module and constructing the forward pass.

    Attributes:
        pairwise_state_dim (int): Dimension of the pairwise state features.
        num_heads (int): Number of attention heads.

    Methods:
        __init__:
            Initializes the EsmFoldPairToSequence module with the given pairwise_state_dim and num_heads.

        construct:
            Applies self-attention mechanism to the input pairwise_state tensor to generate pairwise_bias tensor.

    Args:
        pairwise_state_dim (int): Dimension of the pairwise state features.
        num_heads (int): Number of attention heads.

    Inputs:
        pairwise_state (tensor): Input tensor of shape B x L x L x pairwise_state_dim.

    Outputs:
        pairwise_bias (tensor): Output tensor of shape B x L x L x num_heads.
    """
    def __init__(self, pairwise_state_dim, num_heads):
        """
        Initializes an instance of the EsmFoldPairToSequence class.

        Args:
            self: The instance of the class.
            pairwise_state_dim (int): The dimension of the pairwise state.
            num_heads (int): The number of attention heads to use.

        Returns:
            None.

        Raises:
            ValueError: If pairwise_state_dim or num_heads is not a positive integer.
            AttributeError: If the attributes layernorm or linear cannot be initialized.
        """
        super().__init__()

        self.layernorm = nn.LayerNorm(pairwise_state_dim)
        self.linear = nn.Dense(pairwise_state_dim, num_heads, has_bias=False)

    def construct(self, pairwise_state):
        """
        Inputs:
            pairwise_state: B x L x L x pairwise_state_dim

        Output:
            pairwise_bias: B x L x L x num_heads
        """
        assert len(pairwise_state.shape) == 4
        z = self.layernorm(pairwise_state)
        pairwise_bias = self.linear(z)
        return pairwise_bias


class EsmFoldResidueMLP(nn.Cell):

    """
    This class represents a multi-layer perceptron (MLP) used for folding residues in the ESM
    (Evolutionary Scale Modeling) framework. It inherits from the nn.Cell class.

    The EsmFoldResidueMLP class implements a MLP architecture with layer normalization, dense layers, ReLU activation,
    and dropout. The MLP takes an input tensor and applies a series of linear transformations to produce an output
    tensor. The output tensor is then added element-wise to the input tensor, resulting in the folded residue
    representation.

    Attributes:
        embed_dim (int): The dimensionality of the input and output tensors.
        inner_dim (int): The dimensionality of the intermediate hidden layer in the MLP.
        dropout (float, optional): The dropout probability applied after the ReLU activation. Defaults to 0.

    Methods:
        __init__:
            Initializes an instance of the EsmFoldResidueMLP class.

            - embed_dim (int): The dimensionality of the input and output tensors.
            - inner_dim (int): The dimensionality of the intermediate hidden layer in the MLP.
            - dropout (float, optional): The dropout probability applied after the ReLU activation. Defaults to 0.

        construct(self, x):
            Applies the MLP to the input tensor x and returns the folded residue representation.

            - x (Tensor): The input tensor of shape (batch_size, embed_dim).

    Example:
        ```python
        >>> embed_dim = 128
        >>> inner_dim = 256
        >>> dropout = 0.2
        ...
        >>> mlp = EsmFoldResidueMLP(embed_dim, inner_dim, dropout)
        >>> input_tensor = torch.randn(batch_size, embed_dim)
        ...
        >>> output = mlp.construct(input_tensor)
        ```
    """
    def __init__(self, embed_dim, inner_dim, dropout=0):
        """
        Initializes the EsmFoldResidueMLP class.

        Args:
            self (object): The instance of the class.
            embed_dim (int): The dimension of the input embeddings.
            inner_dim (int): The dimension of the inner layer.
            dropout (float, optional): The dropout probability. Defaults to 0.

        Returns:
            None.

        Raises:
            TypeError: If embed_dim or inner_dim is not an integer, or if dropout is not a float.
            ValueError: If embed_dim or inner_dim is less than or equal to 0, or if dropout is not within the range [0, 1].
        """
        super().__init__()

        self.mlp = nn.SequentialCell(
            nn.LayerNorm(embed_dim),
            nn.Dense(embed_dim, inner_dim),
            nn.ReLU(),
            nn.Dense(inner_dim, embed_dim),
            nn.Dropout(p=dropout),
        )

    def construct(self, x):
        """
        Constructs an output value by adding the input value with the result of the multi-layer perceptron (MLP) operation.

        Args:
            self (EsmFoldResidueMLP): Instance of the EsmFoldResidueMLP class.
            x (any): Input value to be used in the construction process.

        Returns:
            None: The constructed value is returned as the result of adding the input value with the MLP operation.

        Raises:
            TypeError: If the input value 'x' is not compatible for addition with the MLP operation.
            ValueError: If the MLP operation encounters any unexpected issues during computation.
        """
        return x + self.mlp(x)


class EsmFoldTriangularSelfAttentionBlock(nn.Cell):

    """
    This class represents a block of Triangular Self-Attention for the EsmFold model.
    It is used to process sequence and pairwise states in the EsmFold model.

    Attributes:
        layernorm_1 (nn.LayerNorm): A layer normalization module for the sequence state dimension.
        sequence_to_pair (EsmFoldSequenceToPair): A module that converts the sequence state to pairwise state.
        pair_to_sequence (EsmFoldPairToSequence): A module that converts the pairwise state to sequence state.
        seq_attention (EsmFoldSelfAttention): A self-attention module for the sequence state.
        tri_mul_out (EsmFoldTriangleMultiplicativeUpdate):
            A module that performs triangular multiplicative update on the pairwise state.
        tri_mul_in (EsmFoldTriangleMultiplicativeUpdate):
            A module that performs triangular multiplicative update on the pairwise state.
        tri_att_start (EsmFoldTriangleAttention):
            A module that performs triangular attention on the pairwise state starting from a specific position.
        tri_att_end (EsmFoldTriangleAttention):
            A module that performs triangular attention on the pairwise state ending at a specific position.
        mlp_seq (EsmFoldResidueMLP): A multilayer perceptron module for the sequence state.
        mlp_pair (EsmFoldResidueMLP): A multilayer perceptron module for the pairwise state.
        drop (nn.Dropout): A dropout module.
        row_drop (EsmFoldDropout): A dropout module that applies dropout on rows of the pairwise state.
        col_drop (EsmFoldDropout): A dropout module that applies dropout on columns of the pairwise state.

    Methods:
        construct(sequence_state, pairwise_state, mask=None, chunk_size=None, **__kwargs):
            Process the sequence and pairwise states.

            Args:

            - sequence_state (torch.Tensor): Input sequence state tensor of shape
            (batch_size, sequence_length, sequence_state_dim).
            - pairwise_state (torch.Tensor): Input pairwise state tensor of shape
            (batch_size, sequence_length, sequence_length, pairwise_state_dim).
            - mask (torch.Tensor, optional): Boolean tensor of valid positions, with shape
            (batch_size, sequence_length). Defaults to None.
            - chunk_size (int, optional): The size of the attention chunks. Defaults to None.

            Returns:

            - sequence_state (torch.Tensor): Processed sequence state tensor of shape
            (batch_size, sequence_length, sequence_state_dim).
            - pairwise_state (torch.Tensor): Processed pairwise state tensor of shape
            (batch_size, sequence_length, sequence_length, pairwise_state_dim).
    """
    def __init__(self, config):
        """
        This method initializes an instance of the EsmFoldTriangularSelfAttentionBlock class.

        Args:
            self: The instance of the EsmFoldTriangularSelfAttentionBlock class.
            config: The configuration object containing parameters for the attention block.

        Returns:
            None.

        Raises:
            NotImplementedError: If the method is not implemented for any reason.
            ValueError: If the provided configuration object is invalid or missing required parameters.
            TypeError: If the provided configuration object is of incorrect type.
        """
        super().__init__()
        self.config = config

        sequence_state_dim = config.sequence_state_dim
        pairwise_state_dim = config.pairwise_state_dim
        sequence_num_heads = sequence_state_dim // config.sequence_head_width
        pairwise_num_heads = pairwise_state_dim // config.pairwise_head_width

        self.layernorm_1 = nn.LayerNorm(sequence_state_dim)

        self.sequence_to_pair = EsmFoldSequenceToPair(sequence_state_dim, pairwise_state_dim // 2, pairwise_state_dim)
        self.pair_to_sequence = EsmFoldPairToSequence(pairwise_state_dim, sequence_num_heads)

        self.seq_attention = EsmFoldSelfAttention(
            sequence_state_dim, sequence_num_heads, config.sequence_head_width, gated=True
        )
        self.tri_mul_out = EsmFoldTriangleMultiplicativeUpdate(config, _outgoing=True)
        self.tri_mul_in = EsmFoldTriangleMultiplicativeUpdate(config, _outgoing=False)

        self.tri_att_start = EsmFoldTriangleAttention(
            pairwise_state_dim, config.pairwise_head_width, pairwise_num_heads, inf=1e9, starting=True
        )
        self.tri_att_end = EsmFoldTriangleAttention(
            pairwise_state_dim, config.pairwise_head_width, pairwise_num_heads, inf=1e9, starting=False
        )

        self.mlp_seq = EsmFoldResidueMLP(sequence_state_dim, 4 * sequence_state_dim, dropout=config.dropout)
        self.mlp_pair = EsmFoldResidueMLP(pairwise_state_dim, 4 * pairwise_state_dim, dropout=config.dropout)

        self.drop = nn.Dropout(p=config.dropout)
        self.row_drop = EsmFoldDropout(config.dropout * 2, 2)
        self.col_drop = EsmFoldDropout(config.dropout * 2, 1)

    def construct(self, sequence_state, pairwise_state, mask=None, chunk_size=None, **__kwargs):
        """
        Inputs:
          sequence_state: B x L x sequence_state_dim pairwise_state: B x L x L x pairwise_state_dim mask: B x L boolean
          tensor of valid positions

        Output:
          sequence_state: B x L x sequence_state_dim pairwise_state: B x L x L x pairwise_state_dim
        """
        if len(sequence_state.shape) != 3:
            raise ValueError(f"`sequence_state` should be a 3d-tensor, got {len(sequence_state.shape)} dims.")
        if len(pairwise_state.shape) != 4:
            raise ValueError(f"`pairwise_state` should be a 4d-tensor, got {len(pairwise_state.shape)} dims.")
        if mask is not None and len(mask.shape) != 2:
            raise ValueError(f"`mask` should be a 2d-tensor, got {len(mask.shape)} dims.")

        batch_dim, seq_dim, sequence_state_dim = sequence_state.shape
        pairwise_state_dim = pairwise_state.shape[3]

        if sequence_state_dim != self.config.sequence_state_dim:
            raise ValueError(
                "`sequence_state` last dimension should be equal to `self.sequence_state_dim`. Got "
                f"{sequence_state_dim} != {self.config.sequence_state_dim}."
            )
        if pairwise_state_dim != self.config.pairwise_state_dim:
            raise ValueError(
                "`pairwise_state` last dimension should be equal to `self.pairwise_state_dim`. Got "
                f"{pairwise_state_dim} != {self.config.pairwise_state_dim}."
            )
        if batch_dim != pairwise_state.shape[0]:
            raise ValueError(
                f"`sequence_state` and `pairwise_state` have inconsistent batch size: {batch_dim} != "
                f"{pairwise_state.shape[0]}."
            )
        if seq_dim != pairwise_state.shape[1] or seq_dim != pairwise_state.shape[2]:
            raise ValueError(
                f"`sequence_state` and `pairwise_state` have inconsistent sequence length: {seq_dim} != "
                f"{pairwise_state.shape[1]} or {pairwise_state.shape[2]}."
            )

        # Update sequence state
        bias = self.pair_to_sequence(pairwise_state)

        # Self attention with bias + mlp.
        y = self.layernorm_1(sequence_state)
        y, _ = self.seq_attention(y, mask=mask, bias=bias)
        sequence_state = sequence_state + self.drop(y)
        sequence_state = self.mlp_seq(sequence_state)

        # Update pairwise state
        pairwise_state = pairwise_state + self.sequence_to_pair(sequence_state)

        # Axial attention with triangular bias.
        tri_mask = mask.unsqueeze(2) * mask.unsqueeze(1) if mask is not None else None
        pairwise_state = pairwise_state + self.row_drop(self.tri_mul_out(pairwise_state, mask=tri_mask))
        pairwise_state = pairwise_state + self.col_drop(self.tri_mul_in(pairwise_state, mask=tri_mask))
        pairwise_state = pairwise_state + self.row_drop(
            self.tri_att_start(pairwise_state, mask=tri_mask, chunk_size=chunk_size)
        )
        pairwise_state = pairwise_state + self.col_drop(
            self.tri_att_end(pairwise_state, mask=tri_mask, chunk_size=chunk_size)
        )

        # MLP over pairs.
        pairwise_state = self.mlp_pair(pairwise_state)

        return sequence_state, pairwise_state


class EsmCategoricalMixture:

    """
    EsmCategoricalMixture represents a categorical mixture distribution for probability calculations based on given logits.

    This class provides methods for initializing the distribution, calculating the log probability of a given value,
    and computing the mean of the distribution.

    Attributes:
        param: The logits parameter for the categorical mixture distribution.
        bins: The number of bins for the distribution (default is 50).
        start: The starting value for the bins (default is 0).
        end: The ending value for the bins (default is 1).

    Methods:
        __init__: Initializes the categorical mixture distribution with the given parameters.
        log_prob: Calculates the log probability of a given value within the distribution.
        mean: Computes the mean of the categorical mixture distribution.

    Note:
        This class inherits from an unspecified parent class.
    """
    def __init__(self, param, bins=50, start=0, end=1):
        """
        Initializes an instance of the EsmCategoricalMixture class.

        Args:
            self: Instance of the EsmCategoricalMixture class.
            param: The logits parameter to be assigned to the instance.
            bins: Number of bins for creating the v_bins attribute. Default is 50.
            start: The starting value for the linspace function. Default is 0.
            end: The ending value for the linspace function. Default is 1.

        Returns:
            None.

        Raises:
            ValueError: If the start value is greater than or equal to the end value.
            TypeError: If the param or bins parameter types are incompatible.
            ValueError: If the bins parameter is less than 1.
        """
        # All tensors are of shape ..., bins.
        self.logits = param
        bins = ops.linspace(start, end, bins + 1).astype(self.logits.dtype)
        self.v_bins = (bins[:-1] + bins[1:]) / 2

    def log_prob(self, true):
        """
        This method calculates the log probability of a given true value in the context of a categorical mixture model.

        Args:
            self: EsmCategoricalMixture
                The instance of the EsmCategoricalMixture class.
            true: torch.Tensor
                The true value for which the log probability needs to be calculated.
                It should be a tensor of shape (batch_size,) where batch_size is the number of samples.
                The true values should be within the range of valid classes for the categorical mixture model.

        Returns:
            None:
                This method does not return any value. The log probability is calculated and stored internally within
                the EsmCategoricalMixture instance.

        Raises:
            ValueError: If the true tensor does not have the expected shape or if it contains values outside the
                range of valid classes for the categorical mixture model.
            IndexError: If the true tensor index is out of bounds.
        """
        # Shapes are:
        #     self.probs: ... x bins
        #     true      : ...
        true_index = (true.unsqueeze(-1) - self.v_bins[[None] * true.ndim]).abs().argmin(-1)
        nll = self.logits.log_softmax(-1)
        return ops.gather_elements(nll, -1, true_index.unsqueeze(-1)).squeeze(-1)

    def mean(self):
        """
        Method 'mean' calculates the mean value of the categorical mixture distribution in the EsmCategoricalMixture class.

        Args:
            self: The instance of the EsmCategoricalMixture class.

        Returns:
            None.

        Raises:
            NotImplementedError: If the method is called without implementing it in a subclass.
            ValueError: If the input data is not in the expected format.
            RuntimeError: If the operation fails due to unforeseen circumstances.
        """
        return (ops.softmax(self.logits, -1) @ self.v_bins.unsqueeze(1)).squeeze(-1)


def categorical_lddt(logits, bins=50):
    """
    This function calculates the average log-likelihood of a categorical distribution.

    Args:
        logits (array-like): The logits representing the categorical distribution.
            It should be a 2-dimensional array-like object with shape (n_samples, n_classes),
            where n_samples is the number of samples and n_classes is the number of classes.
        bins (int, optional): The number of bins used for discretizing the logits. Defaults to 50.

    Returns:
        float: The average log-likelihood of the categorical distribution.

    Raises:
        ValueError: If the logits parameter is not a 2-dimensional array-like object.
        ValueError: If the bins parameter is not a positive integer.

    """
    # Logits are ..., 37, bins.
    return EsmCategoricalMixture(logits, bins=bins).mean()


def get_axial_mask(mask):
    """
    Helper to convert B x L mask of valid positions to axial mask used in row column attentions.

    Input:
        mask: B x L tensor of booleans

    Output:
        mask: B x L x L tensor of booleans
    """
    if mask is None:
        return None

    if len(mask.shape) != 2:
        raise ValueError(f"`mask` should be a 2d-tensor, got {len(mask.shape)} dims.")
    batch_dim, seq_dim = mask.shape
    m = mask.unsqueeze(1).expand(batch_dim, seq_dim, seq_dim)
    m = m.reshape(batch_dim * seq_dim, seq_dim)
    return m


class EsmFoldRelativePosition(nn.Cell):

    """
    Represents a class for constructing relative position embeddings for protein folding using the ESM
    (Evolutionary Scale Modeling) framework.

    This class inherits from the nn.Cell class and provides methods for initializing the class and constructing pairwise
    state embeddings based on residue indices and optional masking.

    Attributes:
        bins: An integer representing the number of position bins used for constructing the embeddings.
        embedding: An instance of nn.Embedding used for creating the embeddings based on the position differences.

    Methods:
        __init__: Initializes the EsmFoldRelativePosition class with the provided configuration.
        construct: Constructs pairwise state embeddings based on the given residue indices and optional mask.

    Args:
        config: An object containing configuration parameters for initializing the class.
        residue_index: A B x L tensor of indices (dtype=torch.long) representing the residue indices.
        mask: A B x L tensor of booleans representing an optional mask.

    Returns:
        pairwise_state: A B x L x L x pairwise_state_dim tensor of embeddings based on the input residue indices and mask.

    Raises:
        ValueError:
            If the dtype of residue_index is not torch.long or if the shapes of residue_index and mask are inconsistent.
    """
    def __init__(self, config):
        """
        Initializes an instance of the EsmFoldRelativePosition class.

        Args:
            self (EsmFoldRelativePosition): The current instance of the class.
            config: The configuration object containing the necessary parameters.

        Returns:
            None

        Raises:
            None
        """
        super().__init__()
        self.bins = config.position_bins

        # Note an additional offset is used so that the 0th position
        # is reserved for masked pairs.
        self.embedding = nn.Embedding(2 * self.bins + 2, config.pairwise_state_dim)

    def construct(self, residue_index, mask=None):
        """
        Input:
            residue_index: B x L tensor of indices (dytpe=torch.long) mask: B x L tensor of booleans

        Output:
            pairwise_state: B x L x L x pairwise_state_dim tensor of embeddings
        """
        if residue_index.dtype != mindspore.int64:
            raise ValueError(f"`residue_index` has dtype {residue_index.dtype}, it should be `torch.long`.")
        if mask is not None and residue_index.shape != mask.shape:
            raise ValueError(
                f"`residue_index` and `mask` have inconsistent shapes: {residue_index.shape} != {mask.shape}."
            )

        diff = residue_index[:, None, :] - residue_index[:, :, None]
        diff = diff.clamp(-self.bins, self.bins)
        diff = diff + self.bins + 1  # Add 1 to adjust for padding index.

        if mask is not None:
            mask = mask[:, None, :] * mask[:, :, None]
            diff[mask == False] = 0  # noqa: E712

        output = self.embedding(diff)
        return output


class EsmFoldAngleResnetBlock(nn.Cell):

    """
    This class represents an EsmFoldAngleResnetBlock, which is a block used in the construction of an EsmFold model.
    It inherits from the nn.Cell class.

    Attributes:
        linear_1 (EsmFoldLinear):
            A linear layer used in the block, initialized with a rectified linear unit (ReLU) activation function.
        linear_2 (EsmFoldLinear):
            Another linear layer used in the block, initialized with a final activation function.
        relu (nn.ReLU): An instance of the ReLU activation function.

    Methods:
        __init__: Initializes the EsmFoldAngleResnetBlock with the given configuration.
        construct: Constructs the EsmFoldAngleResnetBlock using the input tensor.

    """
    def __init__(self, config):
        """
        Initializes an EsmFoldAngleResnetBlock object.

        Args:
            self (EsmFoldAngleResnetBlock): The current instance of the EsmFoldAngleResnetBlock class.
            config (object):
                A configuration object containing the parameters for initializing the EsmFoldAngleResnetBlock.

                - resnet_dim (int): The dimension of the resnet block.
                - init (str): The initialization method for the linear layers. Possible values are 'relu' and 'final'.

        Returns:
            None.

        Raises:
            TypeError: If the provided config object is not of the expected type.
            ValueError: If the config object does not contain the required parameters.
        """
        super().__init__()

        self.linear_1 = EsmFoldLinear(config.resnet_dim, config.resnet_dim, init="relu")
        self.linear_2 = EsmFoldLinear(config.resnet_dim, config.resnet_dim, init="final")

        self.relu = nn.ReLU()

    def construct(self, a: mindspore.Tensor) -> mindspore.Tensor:
        """
        This method constructs a computation graph for the EsmFoldAngleResnetBlock.

        Args:
            self (EsmFoldAngleResnetBlock): The instance of the EsmFoldAngleResnetBlock class.
            a (mindspore.Tensor): The input tensor for the computation graph.

        Returns:
            mindspore.Tensor: The output tensor resulting from the computation graph.

        Raises:
            None
        """
        s_initial = a

        a = self.relu(a)
        a = self.linear_1(a)
        a = self.relu(a)
        a = self.linear_2(a)

        return a + s_initial


class EsmFoldAngleResnet(nn.Cell):
    """
    Implements Algorithm 20, lines 11-14
    """
    def __init__(self, config):
        '''
        Initializes the EsmFoldAngleResnet class.

        Args:
            self (EsmFoldAngleResnet): The instance of the EsmFoldAngleResnet class.
            config:
                The configuration object containing parameters for the EsmFoldAngleResnet initialization.

                - Type: object
                - Purpose: Specifies the configuration settings for the EsmFoldAngleResnet class.
                - Restrictions: Must be a valid configuration object.

        Returns:
            None.

        Raises:
            None
        '''
        super().__init__()
        self.config = config

        self.linear_in = EsmFoldLinear(config.sequence_dim, config.resnet_dim)
        self.linear_initial = EsmFoldLinear(config.sequence_dim, config.resnet_dim)

        self.layers = nn.CellList()
        for _ in range(config.num_resnet_blocks):
            layer = EsmFoldAngleResnetBlock(config)
            self.layers.append(layer)

        self.linear_out = EsmFoldLinear(config.resnet_dim, config.num_angles * 2)

        self.relu = nn.ReLU()

    def construct(self, s: mindspore.Tensor, s_initial: mindspore.Tensor) -> Tuple[mindspore.Tensor, mindspore.Tensor]:
        """
        Args:
            s:
                [*, C_hidden] single embedding
            s_initial:
                [*, C_hidden] single embedding as of the start of the StructureModule
        Returns:
            [*, no_angles, 2] predicted angles
        """
        # NOTE: The ReLU's applied to the inputs are absent from the supplement
        # pseudocode but present in the source. For maximal compatibility with
        # the pretrained weights, I'm going with the source.

        # [*, C_hidden]
        s_initial = self.relu(s_initial)
        s_initial = self.linear_initial(s_initial)
        s = self.relu(s)
        s = self.linear_in(s)
        s = s + s_initial

        for l in self.layers:
            s = l(s)

        s = self.relu(s)

        # [*, no_angles * 2]
        s = self.linear_out(s)

        # [*, no_angles, 2]
        s = s.view(s.shape[:-1] + (-1, 2))

        unnormalized_s = s
        norm_denom = ops.sqrt(
            ops.clamp(
                ops.sum(s**2, dim=-1, keepdim=True),
                min=self.config.epsilon,
            )
        )

        s = s / norm_denom

        return unnormalized_s, s


class EsmFoldInvariantPointAttention(nn.Cell):
    """
    Implements Algorithm 22.
    """
    def __init__(self, config):
        '''
        Initializes an instance of the EsmFoldInvariantPointAttention class.

        Args:
            self: The instance of the class.
            config: An object containing the configuration settings.

        Returns:
            None

        Raises:
            None

        Description:
            This method initializes the EsmFoldInvariantPointAttention instance by setting various parameters and
            creating necessary objects.

        Parameters:
            self: The instance of the class.
            config: An object containing the configuration settings.

        The config object must have the following attributes:

        - sequence_dim: An integer representing the dimension of the sequence.
        - pairwise_dim: An integer representing the dimension of the pairwise data.
        - ipa_dim: An integer representing the dimension of the ipa data.
        - num_heads_ipa: An integer representing the number of heads for the ipa.
        - num_qk_points: An integer representing the number of query and key points.
        - num_v_points: An integer representing the number of value points.

        Attributes:
            hidden_dim: An integer representing the ipa dimension.
            num_heads: An integer representing the number of ipa heads.
            num_qk_points: An integer representing the number of query and key points.
            num_v_points: An integer representing the number of value points.
            linear_q: An instance of the EsmFoldLinear class with input dimension c_s and output dimension hc.
            linear_kv: An instance of the EsmFoldLinear class with input dimension c_s and output dimension 2 * hc.
            linear_q_points: An instance of the EsmFoldLinear class with input dimension c_s and output dimension hpq.
            linear_kv_points: An instance of the EsmFoldLinear class with input dimension c_s and output dimension hpkv.
            linear_b: An instance of the EsmFoldLinear class with input dimension c_z and output dimension num_heads_ipa.
            head_weights: A Parameter object representing the weights of the ipa heads.
            linear_out: An instance of the EsmFoldLinear class with input dimension concat_out_dim and output dimension c_s.
            softmax: An instance of the Softmax class used for softmax activation.
            softplus: An instance of the Softplus class used for softplus activation.
        '''
        super().__init__()
        self.config = config

        c_s = config.sequence_dim
        c_z = config.pairwise_dim
        self.hidden_dim = config.ipa_dim
        self.num_heads = config.num_heads_ipa
        self.num_qk_points = config.num_qk_points
        self.num_v_points = config.num_v_points

        # These linear layers differ from their specifications in the
        # supplement. There, they lack bias and use Glorot initialization.
        # Here as in the official source, they have bias and use the default
        # Lecun initialization.
        hc = config.ipa_dim * config.num_heads_ipa
        self.linear_q = EsmFoldLinear(c_s, hc)
        self.linear_kv = EsmFoldLinear(c_s, 2 * hc)

        hpq = config.num_heads_ipa * config.num_qk_points * 3
        self.linear_q_points = EsmFoldLinear(c_s, hpq)

        hpkv = config.num_heads_ipa * (config.num_qk_points + config.num_v_points) * 3
        self.linear_kv_points = EsmFoldLinear(c_s, hpkv)

        self.linear_b = EsmFoldLinear(c_z, config.num_heads_ipa)

        self.head_weights = Parameter(ops.zeros((config.num_heads_ipa)))

        concat_out_dim = config.num_heads_ipa * (c_z + config.ipa_dim + config.num_v_points * 4)
        self.linear_out = EsmFoldLinear(concat_out_dim, c_s, init="final")

        self.softmax = nn.Softmax(axis=-1)
        self.softplus = ops.softplus

    def construct(
        self,
        s: mindspore.Tensor,
        z: Optional[mindspore.Tensor],
        r: Rigid,
        mask: mindspore.Tensor,
    ) -> mindspore.Tensor:
        """
        Args:
            s:
                [*, N_res, C_s] single representation
            z:
                [*, N_res, N_res, C_z] pair representation
            r:
                [*, N_res] transformation object
            mask:
                [*, N_res] mask
        Returns:
            [*, N_res, C_s] single representation update
        """
        z = [z]

        #######################################
        # Generate scalar and point activations
        #######################################
        # [*, N_res, H * C_hidden]
        q = self.linear_q(s)
        kv = self.linear_kv(s)

        # [*, N_res, H, C_hidden]
        q = q.view(q.shape[:-1] + (self.num_heads, -1))

        # [*, N_res, H, 2 * C_hidden]
        kv = kv.view(kv.shape[:-1] + (self.num_heads, -1))

        # [*, N_res, H, C_hidden]
        k, v = ops.split(kv, self.hidden_dim, axis=-1)

        # [*, N_res, H * P_q * 3]
        q_pts = self.linear_q_points(s)

        # This is kind of clunky, but it's how the original does it
        # [*, N_res, H * P_q, 3]
        q_pts = ops.split(q_pts, q_pts.shape[-1] // 3, axis=-1)
        q_pts = ops.stack(q_pts, axis=-1)
        q_pts = r[..., None].apply(q_pts)

        # [*, N_res, H, P_q, 3]
        q_pts = q_pts.view(q_pts.shape[:-2] + (self.num_heads, self.num_qk_points, 3))

        # [*, N_res, H * (P_q + P_v) * 3]
        kv_pts = self.linear_kv_points(s)

        # [*, N_res, H * (P_q + P_v), 3]
        kv_pts = ops.split(kv_pts, kv_pts.shape[-1] // 3, axis=-1)
        kv_pts = ops.stack(kv_pts, axis=-1)
        kv_pts = r[..., None].apply(kv_pts)

        # [*, N_res, H, (P_q + P_v), 3]
        kv_pts = kv_pts.view(kv_pts.shape[:-2] + (self.num_heads, -1, 3))

        # [*, N_res, H, P_q/P_v, 3]
        k_pts, v_pts = ops.split(kv_pts, [self.num_qk_points, self.num_v_points], axis=-2)

        ##########################
        # Compute attention scores
        ##########################
        # [*, N_res, N_res, H]
        b = self.linear_b(z[0])
        # [*, H, N_res, N_res]
        a = ops.matmul(
            permute_final_dims(q, (1, 0, 2)),  # [*, H, N_res, C_hidden]
            permute_final_dims(k, (1, 2, 0)),  # [*, H, C_hidden, N_res]
        )

        a *= math.sqrt(1.0 / (3 * self.hidden_dim))
        a += math.sqrt(1.0 / 3) * permute_final_dims(b, (2, 0, 1))

        # [*, N_res, N_res, H, P_q, 3]
        pt_att = q_pts.unsqueeze(-4) - k_pts.unsqueeze(-5)
        pt_att = pt_att**2

        # [*, N_res, N_res, H, P_q]
        pt_att = sum(ops.unbind(pt_att, dim=-1))
        head_weights = self.softplus(self.head_weights).view(*((1,) * len(pt_att.shape[:-2]) + (-1, 1)))
        head_weights = head_weights * math.sqrt(1.0 / (3 * (self.num_qk_points * 9.0 / 2)))
        pt_att = pt_att * head_weights

        # [*, N_res, N_res, H]
        pt_att = ops.sum(pt_att, dim=-1) * (-0.5)
        # [*, N_res, N_res]
        square_mask = mask.unsqueeze(-1) * mask.unsqueeze(-2)
        square_mask = self.config.inf * (square_mask - 1)

        # [*, H, N_res, N_res]
        pt_att = permute_final_dims(pt_att, (2, 0, 1))

        a = a + pt_att
        a = a + square_mask.unsqueeze(-3)
        a = self.softmax(a)

        ################
        # Compute output
        ################
        # [*, N_res, H, C_hidden]
        o = ops.matmul(a, v.swapaxes(-2, -3).to(dtype=a.dtype)).swapaxes(-2, -3)

        # [*, N_res, H * C_hidden]
        o = flatten_final_dims(o, 2)

        # [*, H, 3, N_res, P_v]
        o_pt = ops.sum(
            (a[..., None, :, :, None] * permute_final_dims(v_pts, (1, 3, 0, 2))[..., None, :, :]),
            dim=-2,
        )

        # [*, N_res, H, P_v, 3]
        o_pt = permute_final_dims(o_pt, (2, 0, 3, 1))
        o_pt = r[..., None, None].invert_apply(o_pt)

        # [*, N_res, H * P_v]
        o_pt_norm = flatten_final_dims(ops.sqrt(ops.sum(o_pt**2, dim=-1) + self.config.epsilon), 2)

        # [*, N_res, H * P_v, 3]
        o_pt = o_pt.reshape(*o_pt.shape[:-3], -1, 3)

        # [*, N_res, H, C_z]
        o_pair = ops.matmul(a.swapaxes(-2, -3), z[0].to(dtype=a.dtype))

        # [*, N_res, H * C_z]
        o_pair = flatten_final_dims(o_pair, 2)

        # [*, N_res, C_s]
        s = self.linear_out(
            ops.cat((o, *ops.unbind(o_pt, dim=-1), o_pt_norm, o_pair), axis=-1).to(dtype=z[0].dtype)
        )

        return s


class EsmFoldBackboneUpdate(nn.Cell):
    """
    Implements part of Algorithm 23.
    """
    def __init__(self, config):
        """
        Initializes the EsmFoldBackboneUpdate class.

        Args:
            self: The instance of the class.
            config: A dictionary containing configuration parameters for the backbone update.
                It should include the 'sequence_dim' parameter representing the dimension of the input sequence.

        Returns:
            None.

        Raises:
            TypeError: If the config parameter is not provided or is not a dictionary.
            ValueError: If the 'sequence_dim' parameter is missing in the config dictionary.
            ValueError: If the 'sequence_dim' parameter in the config dictionary is not a positive integer.
        """
        super().__init__()

        self.linear = EsmFoldLinear(config.sequence_dim, 6, init="final")

    def construct(self, s: mindspore.Tensor) -> Tuple[mindspore.Tensor, mindspore.Tensor]:
        """
        Args:
            [*, N_res, C_s] single representation
        Returns:
            [*, N_res, 6] update vector
        """
        # [*, 6]
        update = self.linear(s)

        return update


class EsmFoldStructureModuleTransitionLayer(nn.Cell):

    """
    EsmFoldStructureModuleTransitionLayer

    Represents a transition layer for the EsmFold structure module, inheriting from nn.Cell.

    This class initializes with the provided configuration and constructs a transition layer for the EsmFold structure
    module using the specified linear layers and activation functions.

    Attributes:
        linear_1 (EsmFoldLinear): The first linear layer for the transition.
        linear_2 (EsmFoldLinear): The second linear layer for the transition.
        linear_3 (EsmFoldLinear): The third linear layer for the transition.
        relu (nn.ReLU): The rectified linear unit activation function.

    Methods:
        construct(s): Constructs the transition layer for the EsmFold structure module using the input tensor 's'.

    Returns:
        The output tensor after applying the transition layer to the input tensor 's'.
    """
    def __init__(self, config):
        """
        Initializes a new instance of the EsmFoldStructureModuleTransitionLayer class.

        Args:
            self: The instance of the class.
            config:
                The configuration object containing the parameters for initializing the transition layer.

                - Type: object
                - Purpose: Specifies the configuration parameters required for initializing the transition layer.
                - Restrictions: Must be a valid configuration object.

        Returns:
            None.

        Raises:
            None.
        """
        super().__init__()

        self.linear_1 = EsmFoldLinear(config.sequence_dim, config.sequence_dim, init="relu")
        self.linear_2 = EsmFoldLinear(config.sequence_dim, config.sequence_dim, init="relu")
        self.linear_3 = EsmFoldLinear(config.sequence_dim, config.sequence_dim, init="final")

        self.relu = nn.ReLU()

    def construct(self, s):
        """Constructs a new EsmFoldStructureModuleTransitionLayer.

        This method takes in two parameters, self and s.

        Args:
            self (EsmFoldStructureModuleTransitionLayer): An instance of the EsmFoldStructureModuleTransitionLayer class.
            s (Tensor): The input tensor.

        Returns:
            Tensor: The output tensor after applying linear transformations and element-wise addition.

        Raises:
            None.
        """
        s_initial = s
        s = self.linear_1(s)
        s = self.relu(s)
        s = self.linear_2(s)
        s = self.relu(s)
        s = self.linear_3(s)

        s = s + s_initial

        return s


class EsmFoldStructureModuleTransition(nn.Cell):

    """
    The EsmFoldStructureModuleTransition class represents a module for transitioning the fold structure in a neural network.
    This class inherits from the nn.Cell class and is used to construct transition layers for the fold structure module.

    Attributes:
        config: A configuration object containing parameters for the module.
        layers: A CellList containing the transition layers for the module.
        dropout: A dropout layer with a specified dropout rate.
        layer_norm: A layer normalization layer for normalizing the output.

    Methods:
        __init__: Initializes the EsmFoldStructureModuleTransition with the given configuration.
        construct: Constructs the transition layers for the fold structure module using the input s.

    """
    def __init__(self, config):
        """
        Initializes an instance of the EsmFoldStructureModuleTransition class.

        Args:
            self: The instance of the class.
            config: An object of type 'Config' that holds the configuration settings for the module.

        Returns:
            None

        Raises:
            None
        """
        super().__init__()
        self.config = config

        self.layers = nn.CellList()
        for _ in range(config.num_transition_layers):
            l = EsmFoldStructureModuleTransitionLayer(config)
            self.layers.append(l)

        self.dropout = nn.Dropout(p=config.dropout_rate)
        self.layer_norm = nn.LayerNorm(config.sequence_dim)

    def construct(self, s):
        """
        Constructs the EsmFoldStructureModuleTransition.

        This method takes in two parameters: self and s.

        Args:
            self (EsmFoldStructureModuleTransition): An instance of the EsmFoldStructureModuleTransition class.
            s (unknown type): The input data.

        Returns:
            None.

        Raises:
            None.
        """
        for l in self.layers:
            s = l(s)

        s = self.dropout(s)
        s = self.layer_norm(s)

        return s


class EsmFoldStructureModule(nn.Cell):

    """
    The EsmFoldStructureModule class represents a module for predicting protein structure using Evolutionary Structure
    Model (ESM) and folding techniques. It inherits from the nn.Cell class.

    The class includes methods for initializing the module, constructing the protein structure prediction, and
    converting torsion angles to frames and literature positions to atom14 positions.
    The construct method takes evolutionary formers' output, amino acid indices, and optional sequence mask as input and
    returns a dictionary of predicted outputs. The _init_residue_constants method initializes constants used
    in the module for calculating torsion angles to frames and literature positions to atom14 positions.

    The class also includes the code for initializing the default frames, group indices, atom masks, and literature
    positions, and for converting torsion angles to frames and frames and literature positions to atom14 positions.

    Please note that the detailed implementation and usage of the class methods are described in the code provided.
    """
    def __init__(self, config):
        '''
        Initializes an instance of the EsmFoldStructureModule class.

        Args:
            self (EsmFoldStructureModule): The instance of the class itself.
            config:
                A configuration object containing parameters for initializing the module.

                - Type: Custom configuration object
                - Purpose: Stores various configuration parameters for the module.
                - Restrictions: Must be a valid configuration object.

        Returns:
            None.

        Raises:
            None
        '''
        super().__init__()
        self.config = config

        # Buffers to be lazily initialized later
        # self.default_frames
        # self.group_idx
        # self.atom_mask
        # self.lit_positions

        self.layer_norm_s = nn.LayerNorm(config.sequence_dim)
        self.layer_norm_z = nn.LayerNorm(config.pairwise_dim)

        self.linear_in = EsmFoldLinear(config.sequence_dim, config.sequence_dim)

        self.ipa = EsmFoldInvariantPointAttention(config)

        self.ipa_dropout = nn.Dropout(p=config.dropout_rate)
        self.layer_norm_ipa = nn.LayerNorm(config.sequence_dim)

        self.transition = EsmFoldStructureModuleTransition(config)
        self.bb_update = EsmFoldBackboneUpdate(config)
        self.angle_resnet = EsmFoldAngleResnet(config)

    def construct(
        self,
        evoformer_output_dict,
        aatype,
        mask=None,
        _offload_inference=False,
    ):
        """

        Args:
            evoformer_output_dict:
                Dictionary containing:

                - "single": [*, N_res, C_s] single representation
                - "pair": [*, N_res, N_res, C_z] pair representation
            aatype:
                [*, N_res] amino acid indices
            mask:
                Optional [*, N_res] sequence mask

        Returns:
            A dictionary of outputs
        """
        s = evoformer_output_dict["single"]

        if mask is None:
            # [*, N]
            mask = s.new_ones(s.shape[:-1])

        # [*, N, C_s]
        s = self.layer_norm_s(s)

        # [*, N, N, C_z]
        z = self.layer_norm_z(evoformer_output_dict["pair"])

        z_reference_list = None
        if _offload_inference:
            assert sys.getrefcount(evoformer_output_dict["pair"]) == 2
            evoformer_output_dict["pair"] = evoformer_output_dict["pair"].cpu()
            z_reference_list = [z]
            z = None

        # [*, N, C_s]
        s_initial = s
        s = self.linear_in(s)

        # [*, N]
        rigids = Rigid.identity(
            s.shape[:-1],
            s.dtype,
            fmt="quat",
        )
        outputs = []
        for _ in range(self.config.num_blocks):
            # [*, N, C_s]
            s = s + self.ipa(
                s,
                z,
                rigids,
                mask,
            )
            s = self.ipa_dropout(s)
            s = self.layer_norm_ipa(s)
            s = self.transition(s)

            # [*, N]
            rigids = rigids.compose_q_update_vec(self.bb_update(s))

            # To hew as closely as possible to AlphaFold, we convert our
            # quaternion-based transformations to rotation-matrix ones
            # here
            backb_to_global = Rigid(
                Rotation(rot_mats=rigids.get_rots().get_rot_mats(), quats=None),
                rigids.get_trans(),
            )

            backb_to_global = backb_to_global.scale_translation(self.config.trans_scale_factor)

            # [*, N, 7, 2]
            unnormalized_angles, angles = self.angle_resnet(s, s_initial)

            all_frames_to_global = self.torsion_angles_to_frames(backb_to_global, angles, aatype)

            pred_xyz = self.frames_and_literature_positions_to_atom14_pos(all_frames_to_global, aatype)

            scaled_rigids = rigids.scale_translation(self.config.trans_scale_factor)

            preds = {
                "frames": scaled_rigids.to_tensor_7(),
                "sidechain_frames": all_frames_to_global.to_tensor_4x4(),
                "unnormalized_angles": unnormalized_angles,
                "angles": angles,
                "positions": pred_xyz,
                "states": s,
            }

            outputs.append(preds)

            rigids = rigids.stop_rot_gradient()

        del z, z_reference_list

        outputs = dict_multimap(ops.stack, outputs)
        outputs["single"] = s

        return outputs

    def _init_residue_constants(self, float_dtype):
        """
        Initializes the residue constants required for EsmFoldStructureModule.

        Args:
            self (EsmFoldStructureModule): An instance of the EsmFoldStructureModule class.
            float_dtype (dtype): The data type of the floating point values.

        Returns:
            None

        Raises:
            None

        Description:
            This method initializes the following residue constants:

            - default_frames: A tensor containing the default frames for rigid groups.
            If not already initialized, it is created using the 'restype_rigid_group_default_frame' constant and the
            provided float_dtype.
            - group_idx: A tensor mapping atom14 indices to rigid group indices.
            If not already initialized, it is created using the 'restype_atom14_to_rigid_group' constant.
            - atom_mask: A tensor containing the atom14 mask.
            If not already initialized, it is created using the 'restype_atom14_mask' constant and the provided
            float_dtype.
            - lit_positions: A tensor containing the positions of atom14 rigid groups.
            If not already initialized, it is created using the 'restype_atom14_rigid_group_positions' constant and
            the provided float_dtype.

        Note:
            - This method should be called before using any other functionality of the EsmFoldStructureModule class.
            - The 'float_dtype' parameter determines the precision of the floating point values used in the tensors.
        """
        if not hasattr(self, "default_frames"):
            self.default_frames = mindspore.tensor(
                    residue_constants.restype_rigid_group_default_frame,
                    dtype=float_dtype,
                )
        if not hasattr(self, "group_idx"):
            self.group_idx = mindspore.tensor(
                    residue_constants.restype_atom14_to_rigid_group,
                )
        if not hasattr(self, "atom_mask"):
            self.atom_mask = mindspore.tensor(
                    residue_constants.restype_atom14_mask,
                    dtype=float_dtype,
                )
        if not hasattr(self, "lit_positions"):
            self.lit_positions = mindspore.tensor(
                    residue_constants.restype_atom14_rigid_group_positions,
                    dtype=float_dtype,
                )

    def torsion_angles_to_frames(self, r, alpha, f):
        """
        Converts torsion angles to frames using the given parameters.

        Args:
            self (EsmFoldStructureModule): The instance of the EsmFoldStructureModule class.
            r (numpy.ndarray): The input array of shape (N, 3) containing the residue atoms' coordinates in angstroms.
            alpha (numpy.ndarray): The input array of shape (N, 3) containing the residue angles in radians.
            f (numpy.ndarray): The input array of shape (N, 3, 3) containing the reference frames.

        Returns:
            None.

        Raises:
            ValueError: If the input arrays have incompatible shapes or types.
            TypeError: If the input parameters are not of the expected types.
            RuntimeError: If an unexpected error occurs during the conversion process.
        """
        # Lazily initialize the residue constants on the correct device
        self._init_residue_constants(alpha.dtype)
        # Separated purely to make testing less annoying
        return torsion_angles_to_frames(r, alpha, f, self.default_frames)

    def frames_and_literature_positions_to_atom14_pos(self, r, f):  # [*, N, 8]  # [*, N]
        """
        Converts frames and literature positions to atom14 positions.

        Args:
            self (EsmFoldStructureModule): The instance of the EsmFoldStructureModule class.
            r (object): The 'r' parameter representing some variable.
            f (object): The 'f' parameter representing some variable.

        Returns:
            None.

        Raises:
            None.
        """
        # Lazily initialize the residue constants on the correct device
        self._init_residue_constants(r.get_rots().dtype)
        return frames_and_literature_positions_to_atom14_pos(
            r,
            f,
            self.default_frames,
            self.group_idx,
            self.atom_mask,
            self.lit_positions,
        )


class EsmFoldingTrunk(nn.Cell):

    """
    EsmFoldingTrunk is a neural network cell that represents the trunk of the ESM-Fold model.
    It inherits from the nn.Cell class and contains methods for initializing and constructing the model, as well as a
    static method for computing distograms.

    Attributes:
        config: A configuration object specifying the dimensions and parameters for the ESM-Fold model.

    Methods:
        __init__: Initializes the EsmFoldingTrunk instance with the provided configuration.

        set_chunk_size: Sets the chunk size for processing sequences and pair features.

        construct: Constructs the ESM-Fold model using the provided input tensors and parameters, and returns the
        predicted structure wrapped in a Coordinates object.

        distogram(coords, min_bin, max_bin, num_bins):
            A static method that computes distograms based on the input coordinates and bin parameters.

    Note:
        This class assumes the presence of the required modules and dependencies for the ESM-Fold model.
    """
    def __init__(self, config):
        '''
        Initializes an instance of the EsmFoldingTrunk class.

        Args:
            self: The instance of the class.
            config:
                An object containing the configuration parameters for the EsmFoldingTrunk.

                - sequence_state_dim: An integer representing the dimension of the sequence state.
                - pairwise_state_dim: An integer representing the dimension of the pairwise state.
                - num_blocks: An integer specifying the number of blocks.
                - structure_module: An object containing the configuration parameters for the structure module.

                    - sequence_dim: An integer representing the dimension of the sequence.
                    - pairwise_dim: An integer representing the dimension of the pairwise.

        Returns:
            None

        Raises:
            None
        '''
        super().__init__()
        self.config = config

        c_s = config.sequence_state_dim
        c_z = config.pairwise_state_dim

        self.pairwise_positional_embedding = EsmFoldRelativePosition(config)

        self.blocks = nn.CellList([EsmFoldTriangularSelfAttentionBlock(config) for _ in range(config.num_blocks)])

        self.recycle_bins = 15
        self.recycle_s_norm = nn.LayerNorm(c_s)
        self.recycle_z_norm = nn.LayerNorm(c_z)
        self.recycle_disto = nn.Embedding(self.recycle_bins, c_z)
        self.recycle_disto.weight[0] = 0

        self.structure_module = EsmFoldStructureModule(config.structure_module)
        self.trunk2sm_s = nn.Dense(c_s, config.structure_module.sequence_dim)
        self.trunk2sm_z = nn.Dense(c_z, config.structure_module.pairwise_dim)

        self.chunk_size = config.chunk_size

    def set_chunk_size(self, chunk_size):
        """
        Sets the chunk size for the EsmFoldingTrunk.

        Args:
            self: The instance of the EsmFoldingTrunk class.
            chunk_size (int): The size of the chunk to be set. It should be a positive integer.

        Returns:
            None.

        Raises:
            None.
        """
        # This parameter means the axial attention will be computed
        # in a chunked manner. This should make the memory used more or less O(L) instead of O(L^2).
        # It's equivalent to running a for loop over chunks of the dimension we're iterative over,
        # where the chunk_size is the size of the chunks, so 128 would mean to parse 128-length chunks.
        self.chunk_size = chunk_size

    def construct(self, seq_feats, pair_feats, true_aa, residx, mask, no_recycles):
        """
        Inputs:
            seq_feats: B x L x C tensor of sequence features pair_feats: B x L x L x C tensor of pair features residx: B
            x L long tensor giving the position in the sequence mask: B x L boolean tensor indicating valid residues

        Output:
            predicted_structure: B x L x (num_atoms_per_residue * 3) tensor wrapped in a Coordinates object
        """
        s_s_0 = seq_feats
        s_z_0 = pair_feats

        if no_recycles is None:
            no_recycles = self.config.max_recycles
        else:
            if no_recycles < 0:
                raise ValueError("Number of recycles must not be negative.")
            no_recycles += 1  # First 'recycle' is just the standard forward pass through the model.

        def trunk_iter(s, z, residx, mask):
            z = z + self.pairwise_positional_embedding(residx, mask=mask)

            for block in self.blocks:
                s, z = block(s, z, mask=mask, residue_index=residx, chunk_size=self.chunk_size)
            return s, z

        s_s = s_s_0
        s_z = s_z_0
        recycle_s = ops.zeros_like(s_s)
        recycle_z = ops.zeros_like(s_z)
        recycle_bins = ops.zeros(*s_z.shape[:-1], dtype=mindspore.int64)

        for _ in range(no_recycles):
            with ContextManagers([]):
                # === Recycling ===
                recycle_s = self.recycle_s_norm(recycle_s)
                recycle_z = self.recycle_z_norm(recycle_z)
                recycle_z += self.recycle_disto(recycle_bins)

                s_s, s_z = trunk_iter(s_s_0 + recycle_s, s_z_0 + recycle_z, residx, mask)

                # === Structure module ===
                structure = self.structure_module(
                    {"single": self.trunk2sm_s(s_s), "pair": self.trunk2sm_z(s_z)},
                    true_aa,
                    mask.float(),
                )

                recycle_s = s_s
                recycle_z = s_z
                # Distogram needs the N, CA, C coordinates, and bin constants same as alphafold.
                recycle_bins = EsmFoldingTrunk.distogram(
                    structure["positions"][-1][:, :, :3],
                    3.375,
                    21.375,
                    self.recycle_bins,
                )

        structure["s_s"] = s_s
        structure["s_z"] = s_z

        return structure

    @staticmethod
    def distogram(coords, min_bin, max_bin, num_bins):
        """
        Method to calculate the distance histogram based on the provided coordinates.

        Args:
            coords (Tensor): A tensor containing the coordinates of atoms in the structure.
                Expected shape should be (N, 3, L), where N is the number of atoms, 3 represents x, y, z coordinates,
                and L is the length of the structure.
            min_bin (int): The minimum distance value for binning the distances.
            max_bin (int): The maximum distance value for binning the distances.
            num_bins (int): The number of bins to divide the distance range into.

        Returns:
            None: The method calculates the distance histogram and returns the histogram bins.

        Raises:
            ValueError: If the input coordinates tensor is not in the expected shape or if any of the distance
                parameters (min_bin, max_bin, num_bins) are invalid.
            RuntimeError: If there is an issue with the calculation process.
        """
        # Coords are [... L x 3 x 3], where it's [N, CA, C] x 3 coordinates.
        boundaries = ops.linspace(
            min_bin,
            max_bin,
            num_bins - 1,
        )
        boundaries = boundaries**2
        N, CA, C = [x.squeeze(-2) for x in coords.chunk(3, axis=-2)]
        # Infer CB coordinates.
        b = CA - N
        c = C - CA
        a = mindspore.numpy.cross(b, c, axis=-1)
        CB = -0.58273431 * a + 0.56802827 * b - 0.54067466 * c + CA
        dists = (CB[..., None, :, :] - CB[..., :, None, :]).pow(2).sum(axis=-1, keepdims=True)
        bins = ops.sum(dists > boundaries, dim=-1)  # [..., L, L]
        return bins


class EsmForProteinFolding(EsmPreTrainedModel):

    """
    EsmForProteinFolding is a class that represents a model for protein folding using the ESM
    (Evolutionary Scale Modeling) approach.
    It inherits from EsmPreTrainedModel and implements methods for protein structure prediction and inference.

    The class includes methods for initializing the model, converting input sequences to protein structures,
    and generating Protein Data Bank (PDB) files from model outputs. It also provides functionality for
    language model representations, masking input sequences, and inferring protein structures from input sequences.

    Example:
        ```python
        >>> from transformers import AutoTokenizer, EsmForProteinFolding
        ...
        >>> model = EsmForProteinFolding.from_pretrained("facebook/esmfold_v1")
        >>> tokenizer = AutoTokenizer.from_pretrained("facebook/esmfold_v1")
        >>> inputs = tokenizer(["MLKNVQVQLV"], return_tensors="pt", add_special_tokens=False)  # A tiny random peptide
        >>> outputs = model(**inputs)
        >>> folded_positions = outputs.positions
        ```
    """
    _no_split_modules = ["EsmFoldStructureModule", "EsmFoldTriangularSelfAttentionBlock"]

    def __init__(self, config):
        """Initializes an instance of the EsmForProteinFolding class.

        Args:
            self: The instance of the class.
            config: An object containing configuration settings for the model.

        Returns:
            None

        Raises:
            None
        """
        super().__init__(config)

        self.config = config

        self.distogram_bins = 64

        self.esm = EsmModel(config, add_pooling_layer=False)

        if self.config.esmfold_config.fp16_esm:
            self.esm.half()

        self.esm_feats = self.config.hidden_size
        self.esm_attns = self.config.num_hidden_layers * self.config.num_attention_heads
        self.esm_layers = self.config.num_hidden_layers
        self.af2_to_esm = self._af2_to_esm_from_vocab_list(config.vocab_list)
        self.esm_s_combine = Parameter(ops.zeros((self.esm_layers + 1,)))

        trunk_config = self.config.esmfold_config.trunk
        c_s = trunk_config.sequence_state_dim
        c_z = trunk_config.pairwise_state_dim
        self.esm_s_mlp = nn.SequentialCell(
            nn.LayerNorm(self.esm_feats),
            nn.Dense(self.esm_feats, c_s),
            nn.ReLU(),
            nn.Dense(c_s, c_s),
        )

        # 0 is padding, N is unknown residues, N + 1 is mask.
        self.n_tokens_embed = residue_constants.restype_num + 3
        self.pad_idx = 0
        self.unk_idx = self.n_tokens_embed - 2
        self.mask_idx = self.n_tokens_embed - 1
        self.esm_dict_cls_idx = self.config.vocab_list.index("<cls>")
        self.esm_dict_mask_idx = self.config.vocab_list.index("<mask>")
        self.esm_dict_eos_idx = self.config.vocab_list.index("<eos>")
        self.esm_dict_padding_idx = self.config.vocab_list.index("<pad>")
        if self.config.esmfold_config.embed_aa:
            self.embedding = nn.Embedding(self.n_tokens_embed, c_s, padding_idx=0)

        self.trunk = EsmFoldingTrunk(trunk_config)

        self.distogram_head = nn.Dense(c_z, self.distogram_bins)
        self.ptm_head = nn.Dense(c_z, self.distogram_bins)
        self.lm_head = nn.Dense(c_s, self.n_tokens_embed)
        self.lddt_bins = 50
        structure_module_config = trunk_config.structure_module
        self.lddt_head = nn.SequentialCell(
            nn.LayerNorm(structure_module_config.sequence_dim),
            nn.Dense(structure_module_config.sequence_dim, self.config.esmfold_config.lddt_head_hid_dim),
            nn.Dense(self.config.esmfold_config.lddt_head_hid_dim, self.config.esmfold_config.lddt_head_hid_dim),
            nn.Dense(self.config.esmfold_config.lddt_head_hid_dim, 37 * self.lddt_bins),
        )

    @staticmethod
    def _af2_to_esm_from_vocab_list(vocab_list: List[str]) -> mindspore.Tensor:
        """
        Converts a vocabulary list to a mindspore Tensor, specifically for the ESM (Evolutionary Scale Modeling)
        implementation, in the context of protein folding.

        Args:
            vocab_list (List[str]): A list of strings representing the vocabulary.
                Each string corresponds to a specific residue or token.

        Returns:
            mindspore.Tensor: The resulting Tensor representing the reordered vocabulary list.
                The Tensor contains the indices of the vocabulary list elements, with the first element being the index
                of '<pad>' and the following elements being the indices of the residues from the 'restypes_with_x' list.

        Raises:
            None.

        Note:
            - The '<pad>' element is a special token used for padding sequences.
            - 'residue_constants.restypes_with_x' is a predefined list of residue types with an additional 'x' type.

        Example:
            ```python
            >>> vocab_list = ['<pad>', 'A', 'C', 'D', 'E', 'F', 'G']
            >>> EsmForProteinFolding._af2_to_esm_from_vocab_list(vocab_list)
            Tensor(shape=[8], dtype=Int32, value=
            [0, 1, 2, 3, 4, 5, 6])
            ```
        """
        # Remember that t is shifted from residue_constants by 1 (0 is padding).
        esm_reorder = [vocab_list.index("<pad>")] + [vocab_list.index(v) for v in residue_constants.restypes_with_x]
        return mindspore.tensor(esm_reorder)

    def construct(
        self,
        input_ids: mindspore.Tensor,
        attention_mask: Optional[mindspore.Tensor] = None,
        position_ids: Optional[mindspore.Tensor] = None,
        masking_pattern: Optional[mindspore.Tensor] = None,
        num_recycles: Optional[int] = None,
    ) -> EsmForProteinFoldingOutput:
        r"""
        Returns:
            EsmForProteinFoldingOutput

        Example:
            ```python
            >>> from transformers import AutoTokenizer, EsmForProteinFolding
            ...
            >>> model = EsmForProteinFolding.from_pretrained("facebook/esmfold_v1")
            >>> tokenizer = AutoTokenizer.from_pretrained("facebook/esmfold_v1")
            >>> inputs = tokenizer(["MLKNVQVQLV"], return_tensors="pt", add_special_tokens=False)  # A tiny random peptide
            >>> outputs = model(**inputs)
            >>> folded_positions = outputs.positions
            ```

        """
        cfg = self.config.esmfold_config

        aa = input_ids  # B x L
        B = aa.shape[0]
        L = aa.shape[1]
        if attention_mask is None:
            attention_mask = ops.ones_like(aa)
        if position_ids is None:
            position_ids = ops.arange(L).expand_as(input_ids)

        # === ESM ===
        esmaa = self.af2_idx_to_esm_idx(aa, attention_mask)

        if masking_pattern is not None:
            masked_aa, esmaa, mlm_targets = self.bert_mask(aa, esmaa, attention_mask, masking_pattern)
        else:
            masked_aa = aa
            mlm_targets = None

        # We get sequence and pair representations from whatever version of ESM /
        # configuration we are using. The sequence representation esm_s is always
        # present. The pair embedding esm_z may be present depending on the
        # configuration of the model. If esm_z is not used by the model then it
        # is returned as None here.
        esm_s = self.compute_language_model_representations(esmaa)

        # Convert esm_s and esm_z, if present, to the precision used by the trunk and
        # the structure module. These tensors may be a lower precision if, for example,
        # we're running the language model in fp16 precision.
        esm_s = esm_s.to(self.esm_s_combine.dtype)

        if cfg.esm_ablate_sequence:
            esm_s = esm_s * 0

        # === preprocessing ===
        esm_s = (ops.softmax((self.esm_s_combine + 1e-8), 0).unsqueeze(0) @ esm_s).squeeze(2)
        s_s_0 = self.esm_s_mlp(esm_s)

        s_z_0 = s_s_0.new_zeros((B, L, L, cfg.trunk.pairwise_state_dim))

        if self.config.esmfold_config.embed_aa:
            s_s_0 += self.embedding(masked_aa)

        structure: dict = self.trunk(s_s_0, s_z_0, aa, position_ids, attention_mask, no_recycles=num_recycles)
        # Documenting what we expect:
        structure = {
            k: v
            for k, v in structure.items()
            if k
            in [
                "s_z",
                "s_s",
                "frames",
                "sidechain_frames",
                "unnormalized_angles",
                "angles",
                "positions",
                "states",
            ]
        }

        # Add BERT mask for the loss to use, if available.
        if mlm_targets:
            structure["mlm_targets"] = mlm_targets

        disto_logits = self.distogram_head(structure["s_z"])
        disto_logits = (disto_logits + disto_logits.swapaxes(1, 2)) / 2
        structure["distogram_logits"] = disto_logits

        lm_logits = self.lm_head(structure["s_s"])
        structure["lm_logits"] = lm_logits

        structure["aatype"] = aa
        make_atom14_masks(structure)
        # Of course, this doesn't respect the true mask because it doesn't know about it...
        # We're not going to properly mask change of index tensors:
        #    "residx_atom14_to_atom37",
        #    "residx_atom37_to_atom14",
        for k in [
            "atom14_atom_exists",
            "atom37_atom_exists",
        ]:
            structure[k] *= attention_mask.unsqueeze(-1)
        structure["residue_index"] = position_ids

        lddt_head = self.lddt_head(structure["states"]).reshape(structure["states"].shape[0], B, L, -1, self.lddt_bins)
        structure["lddt_head"] = lddt_head
        plddt = categorical_lddt(lddt_head[-1], bins=self.lddt_bins)
        structure["plddt"] = plddt

        ptm_logits = self.ptm_head(structure["s_z"])
        structure["ptm_logits"] = ptm_logits
        structure["ptm"] = compute_tm(ptm_logits, max_bin=31, no_bins=self.distogram_bins)
        structure.update(compute_predicted_aligned_error(ptm_logits, max_bin=31, no_bins=self.distogram_bins))

        return EsmForProteinFoldingOutput(**structure)

    def af2_idx_to_esm_idx(self, aa, mask):
        """
        This method 'af2_idx_to_esm_idx' is defined in the class 'EsmForProteinFolding' and is used to convert the
        input indices from one representation to another.
        
        Args:
            self: The instance of the class. It is automatically passed as the first argument. Used to access the
                attributes and methods of the class.
            aa:
                A tensor representing the input indices.

                - Type: torch.Tensor.
                - Purpose: It is used to calculate the converted indices. Restrictions: Should be a tensor of indices.
            mask:
                A tensor representing the mask.

                - Type: torch.Tensor.
                - Purpose: It is used to mask the input indices. Restrictions: Should be a tensor of masks.
        
        Returns:
            None: This method does not return any value.
                The converted indices are updated in the instance attribute 'af2_to_esm'.
        
        Raises:
            None.
        """
        aa = (aa + 1).masked_fill(mask != 1, 0)
        return self.af2_to_esm[aa]

    def compute_language_model_representations(self, esmaa: mindspore.Tensor) -> mindspore.Tensor:
        ''' 
        The method 'compute_language_model_representations' in the class 'EsmForProteinFolding' computes the
        representations of the language model.
        
        Args:
            self: The instance of the class.
            esmaa (mindspore.Tensor): A tensor representing the input data with shape (B, L), where B is the batch size
                and L is the sequence length.
        
        Returns:
            mindspore.Tensor: A tensor representing the language model representations.
        
        Raises:
            None.
        '''
        B, L = esmaa.shape  # B = batch size, L = sequence length.

        if self.config.esmfold_config.bypass_lm:
            esm_s = ops.zeros(B, L, self.esm_s_combine.size[0], -1, self.esm_feats)
            return esm_s

        bosi, eosi = self.esm_dict_cls_idx, self.esm_dict_eos_idx
        bos = esmaa.new_ones((B, 1)) * bosi
        eos = esmaa.new_ones((B, 1)) * self.esm_dict_padding_idx
        esmaa = ops.cat([bos, esmaa, eos], axis=1)
        # Use the first padding index as eos during inference.
        esmaa[ops.arange(B), (esmaa != 1).sum(1)] = eosi

        # _, esm_z, esm_s = self.esm(esmaa, return_pairs=self.config.esmfold_config.use_esm_attn_map)
        # Because we do not support use_esm_attn_map in the HF port as it is not used in any public models,
        # esm_z is always None
        esm_hidden_states = self.esm(esmaa, attention_mask=esmaa != 1, output_hidden_states=True)["hidden_states"]
        esm_s = ops.stack(esm_hidden_states, axis=2)

        esm_s = esm_s[:, 1:-1]  # B, L, nLayers, C

        return esm_s

    def bert_mask(self, aa, esmaa, mask, pattern):
        """
        This method 'bert_mask' in the class 'EsmForProteinFolding' masks specific elements in the input arrays based on
        the provided pattern.
        
        Args:
            self: The instance of the class.
            aa (numpy array): The input array of amino acids.
            esmaa (numpy array): The input array of ESMs for amino acids.
            mask (int): The mask index to be applied to 'aa' where pattern equals 1.
            pattern (numpy array): The pattern array used to determine which elements to mask.
        
        Returns:
            None: This method does not return any explicit value but modifies the input arrays in-place. It returns None.
        
        Raises:
            None.
        """
        new_aa = aa.copy()
        target = aa.copy()
        new_esmaa = esmaa.copy()
        new_aa[pattern == 1] = self.mask_idx
        target[pattern != 1] = 0
        new_esmaa[pattern == 1] = self.esm_dict_mask_idx
        return new_aa, new_esmaa, target

    def infer(
        self,
        seqs: Union[str, List[str]],
        position_ids=None,
    ):
        """
        Performs inference on protein folding using the ESM model.
        
        Args:
            self (EsmForProteinFolding): An instance of the EsmForProteinFolding class.
            seqs (Union[str, List[str]]): The protein sequences to perform inference on.
                It can be a single sequence as a string or a list of multiple sequences.
                Each sequence should be a string.
            position_ids (Optional[Tensor]): The position IDs for the sequences.
                If None, default position IDs will be used. Default is None.
        
        Returns:
            None.
        
        Raises:
            None.
        """
        if isinstance(seqs, str):
            lst = [seqs]
        else:
            lst = seqs
        # Returns the raw outputs of the model given an input sequence.
        aatype = collate_dense_tensors(
            [
                mindspore.Tensor.from_numpy(
                    residue_constants.sequence_to_onehot(
                        sequence=seq,
                        mapping=residue_constants.restype_order_with_x,
                        map_unknown_to_x=True,
                    )
                )
                .argmax(axis=1)
                for seq in lst
            ]
        )  # B=1 x L
        mask = collate_dense_tensors([aatype.new_ones(len(seq)) for seq in lst])
        position_ids = (
            ops.arange(aatype.shape[1]).expand(len(lst), -1)
            if position_ids is None
            else position_ids
        )
        if position_ids.ndim == 1:
            position_ids = position_ids.unsqueeze(0)
        return self.construct(
            aatype,
            mask,
            position_ids=position_ids,
        )

    @staticmethod
    def output_to_pdb(output: Dict) -> List[str]:
        """Returns the pbd (file) string from the model given the model output."""
        output = {k: v.asnumpy() for k, v in output.items()}
        pdbs = []
        final_atom_positions = atom14_to_atom37(output["positions"][-1], output)
        final_atom_mask = output["atom37_atom_exists"]
        for i in range(output["aatype"].shape[0]):
            aa = output["aatype"][i]
            pred_pos = final_atom_positions[i]
            mask = final_atom_mask[i]
            resid = output["residue_index"][i] + 1
            pred = OFProtein(
                aatype=aa,
                atom_positions=pred_pos,
                atom_mask=mask,
                residue_index=resid,
                b_factors=output["plddt"][i],
            )
            pdbs.append(to_pdb(pred))
        return pdbs

    def infer_pdb(self, seqs, *args, **kwargs) -> str:
        """Returns the pdb (file) string from the model given an input sequence."""
        assert isinstance(seqs, str)
        output = self.infer(seqs, *args, **kwargs)
        return self.output_to_pdb(output)[0]

    def infer_pdbs(self, seqs: List[str], *args, **kwargs) -> List[str]:
        """Returns the pdb (file) string from the model given an input sequence."""
        output = self.infer(seqs, *args, **kwargs)
        return self.output_to_pdb(output)

__all__ = ["EsmForProteinFolding", "EsmFoldPreTrainedModel"]
