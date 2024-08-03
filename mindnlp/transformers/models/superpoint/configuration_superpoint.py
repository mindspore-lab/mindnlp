# Copyright 2024 Huawei Technologies Co., Ltd
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

"""SuperPoint model configuration"""
from typing import List
from mindnlp.utils import logging
from ...configuration_utils import PretrainedConfig


logger = logging.get_logger(__name__)


class SuperPointConfig(PretrainedConfig):

    model_type = "superpoint"

    def __init__(
        self,
        encoder_hidden_sizes: List[int] = [64, 64, 128, 128],
        decoder_hidden_size: int = 256,
        keypoint_decoder_dim: int = 65,
        descriptor_decoder_dim: int = 256,
        keypoint_threshold: float = 0.005,
        max_keypoints: int = -1,
        nms_radius: int = 4,
        border_removal_distance: int = 4,
        initializer_range=0.02,
        **kwargs,
    ):
        self.encoder_hidden_sizes = encoder_hidden_sizes
        self.decoder_hidden_size = decoder_hidden_size
        self.keypoint_decoder_dim = keypoint_decoder_dim
        self.descriptor_decoder_dim = descriptor_decoder_dim
        self.keypoint_threshold = keypoint_threshold
        self.max_keypoints = max_keypoints
        self.nms_radius = nms_radius
        self.border_removal_distance = border_removal_distance
        self.initializer_range = initializer_range

        super().__init__(**kwargs)


__all__ = ["SuperPointConfig"]
