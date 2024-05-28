# Copyright 2021 AlQuraishi Laboratory
# Copyright 2021 DeepMind Technologies Limited
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ============================================================================
"""loss"""
from typing import Dict, Optional, Tuple

import mindspore
from mindspore import ops


def _calculate_bin_centers(boundaries: mindspore.Tensor) -> mindspore.Tensor:
    """
    Args:
        boundaries (mindspore.Tensor): A 1D tensor representing the boundaries of the bins. It defines the range for binning data. It should be a sorted tensor.
    
    Returns:
        mindspore.Tensor: A 1D tensor containing the calculated bin centers based on the input boundaries. The bin centers are used for binning data.
    
    Raises:
        None
    """
    step = boundaries[1] - boundaries[0]
    bin_centers = boundaries + step / 2
    bin_centers = ops.cat([bin_centers, (bin_centers[-1] + step).unsqueeze(-1)], axis=0)
    return bin_centers


def _calculate_expected_aligned_error(
    alignment_confidence_breaks: mindspore.Tensor,
    aligned_distance_error_probs: mindspore.Tensor,
) -> Tuple[mindspore.Tensor, mindspore.Tensor]:
    """
    Args:
        alignment_confidence_breaks (mindspore.Tensor): A tensor containing the confidence breaks for alignment.
        aligned_distance_error_probs (mindspore.Tensor): A tensor containing the probabilities of aligned distance errors.
    
    Returns:
        Tuple[mindspore.Tensor, mindspore.Tensor]: A tuple containing the sum of aligned distance error probabilities multiplied by bin centers and the last element of the bin centers.
    
    Raises:
        None
    """
    bin_centers = _calculate_bin_centers(alignment_confidence_breaks)
    return (
        ops.sum(aligned_distance_error_probs * bin_centers, dim=-1),
        bin_centers[-1],
    )


def compute_predicted_aligned_error(
    logits: mindspore.Tensor,
    max_bin: int = 31,
    no_bins: int = 64,
    **kwargs,
) -> Dict[str, mindspore.Tensor]:
    """Computes aligned confidence metrics from logits.

    Args:
      logits: [*, num_res, num_res, num_bins] the logits output from
        PredictedAlignedErrorHead.
      max_bin: Maximum bin value
      no_bins: Number of bins
    Returns:
      aligned_confidence_probs: [*, num_res, num_res, num_bins] the predicted
        aligned error probabilities over bins for each residue pair.
      predicted_aligned_error: [*, num_res, num_res] the expected aligned distance
        error for each pair of residues.
      max_predicted_aligned_error: [*] the maximum predicted error possible.
    """
    boundaries = ops.linspace(0, max_bin, steps=(no_bins - 1))

    aligned_confidence_probs = ops.softmax(logits, axis=-1)
    predicted_aligned_error, max_predicted_aligned_error = _calculate_expected_aligned_error(
        alignment_confidence_breaks=boundaries,
        aligned_distance_error_probs=aligned_confidence_probs,
    )

    return {
        "aligned_confidence_probs": aligned_confidence_probs,
        "predicted_aligned_error": predicted_aligned_error,
        "max_predicted_aligned_error": max_predicted_aligned_error,
    }


def compute_tm(
    logits: mindspore.Tensor,
    residue_weights: Optional[mindspore.Tensor] = None,
    max_bin: int = 31,
    no_bins: int = 64,
    eps: float = 1e-8,
    **kwargs,
) -> mindspore.Tensor:
    """
    Compute the predicted transmembrane (TM) term for a given set of logits.
    
    Args:
        logits (mindspore.Tensor): The input logits tensor of shape (..., N), where N is the number of classes.
        residue_weights (Optional[mindspore.Tensor], optional): The weights assigned to each residue. If not provided, default weights of ones will be used. Defaults to None.
        max_bin (int, optional): The maximum bin value. Defaults to 31.
        no_bins (int, optional): The number of bins. Defaults to 64.
        eps (float, optional): A small value added to the denominator to avoid division by zero. Defaults to 1e-08.
    
    Returns:
        mindspore.Tensor: The predicted TM term of shape (...,), representing the TM term for the most probable alignment.
    
    Raises:
        None
    
    Note:
        - The residue_weights argument is used to assign weights to each residue in the computation of the TM term.
        - The boundaries and bin_centers are calculated based on the max_bin and no_bins parameters.
        - The computation of the predicted TM term involves applying a softmax function to the logits, followed by calculating the TM term for each bin and summing them up.
        - The normed_residue_mask is calculated by normalizing the residue_weights by their sum.
        - The weighted array is obtained by multiplying the per_alignment values with the corresponding residue_weights.
        - The argmax represents the index of the maximum value in the weighted array.
        - The return value is the per_alignment value corresponding to the argmax index.
    """
    if residue_weights is None:
        residue_weights = logits.new_ones(logits.shape[-2])

    boundaries = ops.linspace(0, max_bin, steps=(no_bins - 1))

    bin_centers = _calculate_bin_centers(boundaries)
    ops.sum(residue_weights)
    n = logits.shape[-2]
    clipped_n = max(n, 19)

    d0 = 1.24 * (clipped_n - 15) ** (1.0 / 3) - 1.8

    probs = ops.softmax(logits, axis=-1)

    tm_per_bin = 1.0 / (1 + (bin_centers**2) / (d0**2))
    predicted_tm_term = ops.sum(probs * tm_per_bin, dim=-1)

    normed_residue_mask = residue_weights / (eps + residue_weights.sum())
    per_alignment = ops.sum(predicted_tm_term * normed_residue_mask, dim=-1)

    weighted = per_alignment * residue_weights

    argmax = (weighted == ops.max(weighted)[0]).nonzero()[0]
    return per_alignment[tuple(argmax)]
