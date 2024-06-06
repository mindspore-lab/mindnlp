'''
Author: skyous 1019364238@qq.com
Date: 2024-06-05 19:01:20
LastEditors: skyous 1019364238@qq.com
LastEditTime: 2024-06-05 19:05:41
FilePath: /huawei2024/mindnlp/mindnlp/transformers/models/videomae/feature_extraction_videomae.py
Description: 

Copyright (c) 2024 by 1019364238@qq.com, All Rights Reserved. 
'''
# coding=utf-8
# Copyright 2022 The HuggingFace Inc. team. All rights reserved.
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
"""Feature extractor class for VideoMAE."""

import warnings

from mindnlp.utils import logging
from .image_processing_videomae import VideoMAEImageProcessor


logger = logging.get_logger(__name__)


class VideoMAEFeatureExtractor(VideoMAEImageProcessor):
    def __init__(self, *args, **kwargs) -> None:
        warnings.warn(
            "The class VideoMAEFeatureExtractor is deprecated and will be removed in version 5 of Transformers."
            " Please use VideoMAEImageProcessor instead.",
            FutureWarning,
        )
        super().__init__(*args, **kwargs)
        
__all__ = ["VideoMAEFeatureExtractor"]