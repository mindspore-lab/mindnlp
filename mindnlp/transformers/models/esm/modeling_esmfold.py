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
# pylint: disable=missing-class-docstring
# pylint: disable=missing-function-docstring
# pylint: disable=invalid-name
# pylint: disable=import-outside-toplevel
# pylint: disable=unused-argument
# pylint: disable=singleton-comparison
# pylint: disable=attribute-defined-outside-init
# pylint: disable=unexpected-keyword-arg
# pylint: disable=arguments-renamed
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
        [(d_11, ..., d_1K),
         (d_21, ..., d_2K), ..., (d_N1, ..., d_NK)]
    and stack + pads them into a single tensor of:
    (N, max_i=1,N { d_i1 }, ..., max_i=1,N {diK})
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
    return t.reshape(t.shape[:-no_dims] + (-1,))


def permute_final_dims(tensor: mindspore.Tensor, inds: List[int]):
    zero_index = -1 * len(inds)
    first_inds = list(range(len(tensor.shape[:zero_index])))
    return tensor.permute(first_inds + [zero_index + i for i in inds])


def dict_multimap(fn, dicts):
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
    def __init__(self, c_in, eps=1e-5):
        super().__init__()

        self.c_in = (c_in,)
        self.eps = eps

        self.weight = Parameter(ops.ones(c_in))
        self.bias = Parameter(ops.zeros(c_in))
        self.layer_norm = ops.LayerNorm(begin_norm_axis=-1,
                                        begin_params_axis=-1,
                                        epsilon=eps)
    def construct(self, x):
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
    def __init__(self, embed_dim, num_heads, head_width, gated=False):
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
        super().__init__()

        self.r = r
        if isinstance(batch_dim, int):
            batch_dim = [batch_dim]
        self.batch_dim = batch_dim
        self.dropout = nn.Dropout(p=self.r)

    def construct(self, x: mindspore.Tensor) -> mindspore.Tensor:
        shape = list(x.shape)
        if self.batch_dim is not None:
            for bd in self.batch_dim:
                shape[bd] = 1
        return x * self.dropout(x.new_ones(shape))


class EsmFoldSequenceToPair(nn.Cell):
    def __init__(self, sequence_state_dim, inner_dim, pairwise_state_dim):
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
    def __init__(self, pairwise_state_dim, num_heads):
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
    def __init__(self, embed_dim, inner_dim, dropout=0):
        super().__init__()

        self.mlp = nn.SequentialCell(
            nn.LayerNorm(embed_dim),
            nn.Dense(embed_dim, inner_dim),
            nn.ReLU(),
            nn.Dense(inner_dim, embed_dim),
            nn.Dropout(p=dropout),
        )

    def construct(self, x):
        return x + self.mlp(x)


class EsmFoldTriangularSelfAttentionBlock(nn.Cell):
    def __init__(self, config):
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
    def __init__(self, param, bins=50, start=0, end=1):
        # All tensors are of shape ..., bins.
        self.logits = param
        bins = ops.linspace(start, end, bins + 1).astype(self.logits.dtype)
        self.v_bins = (bins[:-1] + bins[1:]) / 2

    def log_prob(self, true):
        # Shapes are:
        #     self.probs: ... x bins
        #     true      : ...
        true_index = (true.unsqueeze(-1) - self.v_bins[[None] * true.ndim]).abs().argmin(-1)
        nll = self.logits.log_softmax(-1)
        return ops.gather_elements(nll, -1, true_index.unsqueeze(-1)).squeeze(-1)

    def mean(self):
        return (ops.softmax(self.logits, -1) @ self.v_bins.unsqueeze(1)).squeeze(-1)


def categorical_lddt(logits, bins=50):
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
    def __init__(self, config):
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
    def __init__(self, config):
        super().__init__()

        self.linear_1 = EsmFoldLinear(config.resnet_dim, config.resnet_dim, init="relu")
        self.linear_2 = EsmFoldLinear(config.resnet_dim, config.resnet_dim, init="final")

        self.relu = nn.ReLU()

    def construct(self, a: mindspore.Tensor) -> mindspore.Tensor:
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
    def __init__(self, config):
        super().__init__()

        self.linear_1 = EsmFoldLinear(config.sequence_dim, config.sequence_dim, init="relu")
        self.linear_2 = EsmFoldLinear(config.sequence_dim, config.sequence_dim, init="relu")
        self.linear_3 = EsmFoldLinear(config.sequence_dim, config.sequence_dim, init="final")

        self.relu = nn.ReLU()

    def construct(self, s):
        s_initial = s
        s = self.linear_1(s)
        s = self.relu(s)
        s = self.linear_2(s)
        s = self.relu(s)
        s = self.linear_3(s)

        s = s + s_initial

        return s


class EsmFoldStructureModuleTransition(nn.Cell):
    def __init__(self, config):
        super().__init__()
        self.config = config

        self.layers = nn.CellList()
        for _ in range(config.num_transition_layers):
            l = EsmFoldStructureModuleTransitionLayer(config)
            self.layers.append(l)

        self.dropout = nn.Dropout(p=config.dropout_rate)
        self.layer_norm = nn.LayerNorm(config.sequence_dim)

    def construct(self, s):
        for l in self.layers:
            s = l(s)

        s = self.dropout(s)
        s = self.layer_norm(s)

        return s


class EsmFoldStructureModule(nn.Cell):
    def __init__(self, config):
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
                    "single":
                        [*, N_res, C_s] single representation
                    "pair":
                        [*, N_res, N_res, C_z] pair representation
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
        # Lazily initialize the residue constants on the correct device
        self._init_residue_constants(alpha.dtype)
        # Separated purely to make testing less annoying
        return torsion_angles_to_frames(r, alpha, f, self.default_frames)

    def frames_and_literature_positions_to_atom14_pos(self, r, f):  # [*, N, 8]  # [*, N]
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
    def __init__(self, config):
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
    _no_split_modules = ["EsmFoldStructureModule", "EsmFoldTriangularSelfAttentionBlock"]

    def __init__(self, config):
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

        Example:

        ```python
        >>> from transformers import AutoTokenizer, EsmForProteinFolding

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
        aa = (aa + 1).masked_fill(mask != 1, 0)
        return self.af2_to_esm[aa]

    def compute_language_model_representations(self, esmaa: mindspore.Tensor) -> mindspore.Tensor:
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
