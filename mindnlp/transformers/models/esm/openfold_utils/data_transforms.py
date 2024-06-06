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
"""data transforms"""
from typing import Dict

import numpy as np
import mindspore
from mindspore import ops

from . import residue_constants as rc
from .tensor_utils import tensor_tree_map, tree_map


def make_atom14_masks(protein: Dict[str, mindspore.Tensor]) -> Dict[str, mindspore.Tensor]:
    """Construct denser atom positions (14 dimensions instead of 37)."""
    restype_atom14_to_atom37_list = []
    restype_atom37_to_atom14_list = []
    restype_atom14_mask_list = []

    for rt in rc.restypes:
        atom_names = rc.restype_name_to_atom14_names[rc.restype_1to3[rt]]
        restype_atom14_to_atom37_list.append([(rc.atom_order[name] if name else 0) for name in atom_names])
        atom_name_to_idx14 = {name: i for i, name in enumerate(atom_names)}
        restype_atom37_to_atom14_list.append(
            [(atom_name_to_idx14[name] if name in atom_name_to_idx14 else 0) for name in rc.atom_types]
        )

        restype_atom14_mask_list.append([(1.0 if name else 0.0) for name in atom_names])

    # Add dummy mapping for restype 'UNK'
    restype_atom14_to_atom37_list.append([0] * 14)
    restype_atom37_to_atom14_list.append([0] * 37)
    restype_atom14_mask_list.append([0.0] * 14)

    restype_atom14_to_atom37 = mindspore.tensor(
        restype_atom14_to_atom37_list,
        dtype=mindspore.int32,
    )
    restype_atom37_to_atom14 = mindspore.tensor(
        restype_atom37_to_atom14_list,
        dtype=mindspore.int32,
    )
    restype_atom14_mask = mindspore.tensor(
        restype_atom14_mask_list,
        dtype=mindspore.float32,
    )
    protein_aatype = protein["aatype"].to(mindspore.int64)

    # create the mapping for (residx, atom14) --> atom37, i.e. an array
    # with shape (num_res, 14) containing the atom37 indices for this protein
    residx_atom14_to_atom37 = restype_atom14_to_atom37[protein_aatype]
    residx_atom14_mask = restype_atom14_mask[protein_aatype]

    protein["atom14_atom_exists"] = residx_atom14_mask
    protein["residx_atom14_to_atom37"] = residx_atom14_to_atom37.long()

    # create the gather indices for mapping back
    residx_atom37_to_atom14 = restype_atom37_to_atom14[protein_aatype]
    protein["residx_atom37_to_atom14"] = residx_atom37_to_atom14.long()

    # create the corresponding mask
    restype_atom37_mask = ops.zeros((21, 37), dtype=mindspore.float32)
    for restype, restype_letter in enumerate(rc.restypes):
        restype_name = rc.restype_1to3[restype_letter]
        atom_names = rc.residue_atoms[restype_name]
        for atom_name in atom_names:
            atom_type = rc.atom_order[atom_name]
            restype_atom37_mask[restype, atom_type] = 1

    residx_atom37_mask = restype_atom37_mask[protein_aatype]
    protein["atom37_atom_exists"] = residx_atom37_mask

    return protein


def make_atom14_masks_np(batch: Dict[str, mindspore.Tensor]) -> Dict[str, np.ndarray]:
    """
    Converts a batch of MindSpore tensors to NumPy arrays and applies a function to create atom14 masks.
    
    Args:
        batch (Dict[str, mindspore.Tensor]): A dictionary containing MindSpore tensors representing the batch data.
    
    Returns:
        Dict[str, np.ndarray]: A dictionary where the keys are strings and the values are NumPy arrays representing the atom14 masks.
    
    Raises:
        None
    """
    batch = tree_map(lambda n: mindspore.tensor(n), batch, np.ndarray)
    out = tensor_tree_map(lambda t: np.array(t), make_atom14_masks(batch))
    return out
