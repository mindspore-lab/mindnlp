# Copyright 2021 AlQuraishi Laboratory
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
"""openfold utils"""
from .chunk_utils import chunk_layer
from .data_transforms import make_atom14_masks
from .feats import atom14_to_atom37, frames_and_literature_positions_to_atom14_pos, torsion_angles_to_frames
from .loss import compute_predicted_aligned_error, compute_tm
from .protein import Protein as OFProtein
from .protein import to_pdb
from .rigid_utils import Rigid, Rotation
from .tensor_utils import dict_multimap, flatten_final_dims, permute_final_dims
