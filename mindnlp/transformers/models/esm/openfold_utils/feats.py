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

"""feats"""
from typing import Dict, Tuple, overload

import mindspore
from mindspore import ops

from . import residue_constants as rc
from .rigid_utils import Rigid, Rotation
from .tensor_utils import batched_gather


@overload
def pseudo_beta_fn(aatype: mindspore.Tensor, all_atom_positions: mindspore.Tensor, all_atom_masks: None) -> mindspore.Tensor:
    """
    This function calculates the pseudo beta value using the given input parameters.
    
    Args:
        aatype (mindspore.Tensor): A tensor containing the amino acid type information.
        all_atom_positions (mindspore.Tensor): A tensor containing the positions of all atoms.
        all_atom_masks (None): This parameter is not used in the function and is optional.
    
    Returns:
        mindspore.Tensor: A tensor containing the calculated pseudo beta value.
    
    Raises:
        None
    """


@overload
def pseudo_beta_fn(
    aatype: mindspore.Tensor, all_atom_positions: mindspore.Tensor, all_atom_masks: mindspore.Tensor
) -> Tuple[mindspore.Tensor, mindspore.Tensor]:
    '''
    This function calculates pseudo beta values based on the provided amino acid type, atom positions, and atom masks.
    
    Args:
        aatype (mindspore.Tensor): A tensor containing the amino acid type information.
        all_atom_positions (mindspore.Tensor): A tensor containing positions of all atoms.
        all_atom_masks (mindspore.Tensor): A tensor containing masks for all atoms.
    
    Returns:
        Tuple[mindspore.Tensor, mindspore.Tensor]: A tuple containing the calculated pseudo beta values.
    
    Raises:
        None
    '''


def pseudo_beta_fn(aatype, all_atom_positions, all_atom_masks):
    """
    Args:
        aatype (ndarray): An array representing the amino acid type.
        all_atom_positions (ndarray): An array containing the positions of all atoms.
        all_atom_masks (ndarray): An array containing masks for all atoms.
    
    Returns:
        If all_atom_masks is not None:
            tuple: A tuple containing pseudo_beta and pseudo_beta_mask represented as ndarrays.
        Otherwise:
            ndarray: An array representing the pseudo beta value.
    
    Raises:
        None
    """
    is_gly = aatype == rc.restype_order["G"]
    ca_idx = rc.atom_order["CA"]
    cb_idx = rc.atom_order["CB"]
    pseudo_beta = ops.where(
        is_gly[..., None].expand(*((-1,) * len(is_gly.shape)), 3),
        all_atom_positions[..., ca_idx, :],
        all_atom_positions[..., cb_idx, :],
    )

    if all_atom_masks is not None:
        pseudo_beta_mask = ops.where(
            is_gly,
            all_atom_masks[..., ca_idx],
            all_atom_masks[..., cb_idx],
        )
        return pseudo_beta, pseudo_beta_mask
    return pseudo_beta


def atom14_to_atom37(atom14: mindspore.Tensor, batch: Dict[str, mindspore.Tensor]) -> mindspore.Tensor:
    """
    Args:
        atom14 (mindspore.Tensor): The input tensor representing the atom data in a 14-dimensional space.
        batch (Dict[str, mindspore.Tensor]): A dictionary containing batched data, including 'residx_atom37_to_atom14' and 'atom37_atom_exists' for gathering atom37 data.
    
    Returns:
        mindspore.Tensor: A tensor representing the atom data in a 37-dimensional space, based on the input atom14 and batch data.
    
    Raises:
        None
    """
    atom37_data = batched_gather(
        atom14,
        batch["residx_atom37_to_atom14"],
        dim=-2,
        no_batch_dims=len(atom14.shape[:-2]),
    )

    atom37_data = atom37_data * batch["atom37_atom_exists"][..., None]

    return atom37_data


def build_template_angle_feat(template_feats: Dict[str, mindspore.Tensor]) -> mindspore.Tensor:
    '''
    This function builds a template angle feature tensor based on the input template features.
    
    Args:
        template_feats (Dict[str, mindspore.Tensor]): A dictionary containing the following keys:
            - 'template_aatype': A tensor representing the template amino acid types.
            - 'template_torsion_angles_sin_cos': A tensor representing the sin and cosine of the template torsion angles.
            - 'template_alt_torsion_angles_sin_cos': A tensor representing the sin and cosine of the alternative template torsion angles.
            - 'template_torsion_angles_mask': A tensor representing the mask for template torsion angles.
    
    Returns:
        mindspore.Tensor: The template angle feature tensor constructed by concatenating one-hot encoded template amino acid types, template torsion angles sin and cosine, alternative template torsion angles
sin and cosine, and template torsion angles mask.
    
    Raises:
        None
    '''
    template_aatype = template_feats["template_aatype"]
    torsion_angles_sin_cos = template_feats["template_torsion_angles_sin_cos"]
    alt_torsion_angles_sin_cos = template_feats["template_alt_torsion_angles_sin_cos"]
    torsion_angles_mask = template_feats["template_torsion_angles_mask"]
    template_angle_feat = ops.cat(
        [
            ops.one_hot(template_aatype, 22),
            torsion_angles_sin_cos.reshape(*torsion_angles_sin_cos.shape[:-2], 14),
            alt_torsion_angles_sin_cos.reshape(*alt_torsion_angles_sin_cos.shape[:-2], 14),
            torsion_angles_mask,
        ],
        axis=-1,
    )

    return template_angle_feat


def build_template_pair_feat(
    batch: Dict[str, mindspore.Tensor],
    min_bin,
    max_bin,
    no_bins: int,
    use_unit_vector: bool = False,
    eps: float = 1e-20,
    inf: float = 1e8,
) -> mindspore.Tensor:
    """
    Builds a template pair feature tensor based on the input batch data.
    
    Args:
        batch (Dict[str, mindspore.Tensor]): A dictionary containing the input data tensors.
        min_bin (int): The minimum bin value for the histogram calculation.
        max_bin (int): The maximum bin value for the histogram calculation.
        no_bins (int): The number of bins for the histogram calculation.
        use_unit_vector (bool, optional): A flag indicating whether to use unit vectors. Defaults to False.
        eps (float, optional): A small value to prevent division by zero. Defaults to 1e-20.
        inf (float, optional): A large value representing infinity. Defaults to 100000000.0.
    
    Returns:
        mindspore.Tensor: A tensor representing the calculated template pair feature.
    
    Raises:
        None
    """
    template_mask = batch["template_pseudo_beta_mask"]
    template_mask_2d = template_mask[..., None] * template_mask[..., None, :]

    # Compute distogram (this seems to differ slightly from Alg. 5)
    tpb = batch["template_pseudo_beta"]
    dgram = ops.sum((tpb[..., None, :] - tpb[..., None, :, :]) ** 2, dim=-1, keepdim=True)
    lower = ops.linspace(min_bin, max_bin, no_bins) ** 2
    upper = ops.cat([lower[1:], lower.new_tensor([inf])], axis=-1)
    dgram = ((dgram > lower) * (dgram < upper)).astype(dgram.dtype)

    to_concat = [dgram, template_mask_2d[..., None]]

    aatype_one_hot: mindspore.Tensor = ops.one_hot(
        batch["template_aatype"],
        rc.restype_num + 2,
    )

    n_res = batch["template_aatype"].shape[-1]
    to_concat.append(aatype_one_hot[..., None, :, :].expand(*aatype_one_hot.shape[:-2], n_res, -1, -1))
    to_concat.append(aatype_one_hot[..., None, :].expand(*aatype_one_hot.shape[:-2], -1, n_res, -1))

    n, ca, c = [rc.atom_order[a] for a in ["N", "CA", "C"]]
    rigids = Rigid.make_transform_from_reference(
        n_xyz=batch["template_all_atom_positions"][..., n, :],
        ca_xyz=batch["template_all_atom_positions"][..., ca, :],
        c_xyz=batch["template_all_atom_positions"][..., c, :],
        eps=eps,
    )
    points = rigids.get_trans()[..., None, :, :]
    rigid_vec = rigids[..., None].invert_apply(points)

    inv_distance_scalar = ops.rsqrt(eps + ops.sum(rigid_vec**2, dim=-1))

    t_aa_masks = batch["template_all_atom_mask"]
    template_mask = t_aa_masks[..., n] * t_aa_masks[..., ca] * t_aa_masks[..., c]
    template_mask_2d = template_mask[..., None] * template_mask[..., None, :]

    inv_distance_scalar = inv_distance_scalar * template_mask_2d
    unit_vector = rigid_vec * inv_distance_scalar[..., None]

    if not use_unit_vector:
        unit_vector = unit_vector * 0.0

    to_concat.extend(ops.unbind(unit_vector[..., None, :], dim=-1))
    to_concat.append(template_mask_2d[..., None])

    act = ops.cat(to_concat, axis=-1)
    act = act * template_mask_2d[..., None]

    return act


def build_extra_msa_feat(batch: Dict[str, mindspore.Tensor]) -> mindspore.Tensor:
    """
    This function builds additional features using the input batch data for multiple sequence alignment (MSA).
    
    Args:
        batch (Dict[str, mindspore.Tensor]): A dictionary containing input tensors for MSA processing.
            - 'extra_msa': Tensor representing the MSA data encoded as integers.
            - 'extra_has_deletion': Tensor indicating the presence of deletions in MSA sequences.
            - 'extra_deletion_value': Tensor containing values of deletions in MSA sequences.
    
    Returns:
        mindspore.Tensor: A concatenated tensor containing additional MSA features constructed from the input batch data.
    
    Raises:
        None
    """
    msa_1hot: mindspore.Tensor = ops.one_hot(batch["extra_msa"], 23)
    msa_feat = [
        msa_1hot,
        batch["extra_has_deletion"].unsqueeze(-1),
        batch["extra_deletion_value"].unsqueeze(-1),
    ]
    return ops.cat(msa_feat, axis=-1)


def torsion_angles_to_frames(
    r: Rigid,
    alpha: mindspore.Tensor,
    aatype: mindspore.Tensor,
    rrgdf: mindspore.Tensor,
) -> Rigid:
    """
    Converts torsion angles to frames in a Rigid object.
    
    Args:
        r (Rigid): The original Rigid object to base the transformation on.
        alpha (mindspore.Tensor): Tensor representing the torsion angles.
        aatype (mindspore.Tensor): Tensor representing the amino acid type.
        rrgdf (mindspore.Tensor): Tensor containing the default 4x4 transformation matrix.
    
    Returns:
        Rigid: A new Rigid object representing the frames converted from the torsion angles.
    
    Raises:
        None
    """
    # [*, N, 8, 4, 4]
    default_4x4 = rrgdf[aatype, ...]

    # [*, N, 8] transformations, i.e.
    #   One [*, N, 8, 3, 3] rotation matrix and
    #   One [*, N, 8, 3]    translation matrix
    default_r = r.from_tensor_4x4(default_4x4)

    bb_rot = alpha.new_zeros((*((1,) * len(alpha.shape[:-1])), 2))
    bb_rot[..., 1] = 1

    # [*, N, 8, 2]
    alpha = ops.cat([bb_rot.expand(*alpha.shape[:-2], -1, -1), alpha], axis=-2)

    # [*, N, 8, 3, 3]
    # Produces rotation matrices of the form:
    # [
    #   [1, 0  , 0  ],
    #   [0, a_2,-a_1],
    #   [0, a_1, a_2]
    # ]
    # This follows the original code rather than the supplement, which uses
    # different indices.

    all_rots = alpha.new_zeros(default_r.get_rots().get_rot_mats().shape)
    all_rots[..., 0, 0] = 1
    all_rots[..., 1, 1] = alpha[..., 1]
    all_rots[..., 1, 2] = -alpha[..., 0]
    all_rots[..., 2, 1:] = alpha

    all_frames = default_r.compose(Rigid(Rotation(rot_mats=all_rots), None))

    chi2_frame_to_frame = all_frames[..., 5]
    chi3_frame_to_frame = all_frames[..., 6]
    chi4_frame_to_frame = all_frames[..., 7]

    chi1_frame_to_bb = all_frames[..., 4]
    chi2_frame_to_bb = chi1_frame_to_bb.compose(chi2_frame_to_frame)
    chi3_frame_to_bb = chi2_frame_to_bb.compose(chi3_frame_to_frame)
    chi4_frame_to_bb = chi3_frame_to_bb.compose(chi4_frame_to_frame)

    all_frames_to_bb = Rigid.cat(
        [
            all_frames[..., :5],
            chi2_frame_to_bb.unsqueeze(-1),
            chi3_frame_to_bb.unsqueeze(-1),
            chi4_frame_to_bb.unsqueeze(-1),
        ],
        dim=-1,
    )

    all_frames_to_global = r[..., None].compose(all_frames_to_bb)

    return all_frames_to_global


def frames_and_literature_positions_to_atom14_pos(
    r: Rigid,
    aatype: mindspore.Tensor,
    default_frames: mindspore.Tensor,
    group_idx: mindspore.Tensor,
    atom_mask: mindspore.Tensor,
    lit_positions: mindspore.Tensor,
) -> mindspore.Tensor:
    """
    Converts frames and literature positions to atom14 positions.
    
    Args:
        r (Rigid): The rigid transformation matrix.
        aatype (mindspore.Tensor): The tensor representing the amino acid type.
        default_frames (mindspore.Tensor): The tensor representing the default frames.
        group_idx (mindspore.Tensor): The tensor representing the group index.
        atom_mask (mindspore.Tensor): The tensor representing the atom mask.
        lit_positions (mindspore.Tensor): The tensor representing the literature positions.
    
    Returns:
        mindspore.Tensor: The tensor representing the predicted atom14 positions.
    
    Raises:
        None.
    """
    # [*, N, 14]
    group_mask = group_idx[aatype, ...]

    # [*, N, 14, 8]
    group_mask_one_hot: mindspore.Tensor = ops.one_hot(
        group_mask,
        default_frames.shape[-3],
    )

    # [*, N, 14, 8]
    t_atoms_to_global = r[..., None, :] * group_mask_one_hot

    # [*, N, 14]
    t_atoms_to_global = t_atoms_to_global.map_tensor_fn(lambda x: ops.sum(x, dim=-1))

    # [*, N, 14, 1]
    atom_mask = atom_mask[aatype, ...].unsqueeze(-1)

    # [*, N, 14, 3]
    lit_positions = lit_positions[aatype, ...]
    pred_positions = t_atoms_to_global.apply(lit_positions)
    pred_positions = pred_positions * atom_mask

    return pred_positions
