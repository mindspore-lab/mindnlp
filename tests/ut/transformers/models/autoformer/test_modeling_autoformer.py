# coding=utf-8
# Copyright 2023 The HuggingFace Inc. team. All rights reserved.
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
""" Testing suite for the Mindspore Autoformer model. """

import inspect
import tempfile
import unittest

from huggingface_hub import hf_hub_download

from mindnlp.utils.testing_utils import is_flaky, is_mindspore_available

from ...test_configuration_common import ConfigTester
from ...test_modeling_common import ModelTesterMixin, floats_tensor, ids_tensor

TOLERANCE = 1e-4

if is_mindspore_available():
    import mindspore
    from mindspore import ops
    from mindnlp.transformers.models.autoformer.modeling_autoformer import AutoformerConfig#, AutoformerForPrediction, AutoformerModel

    #from mindnlp.transformers.models.autoformer.modeling_autoformer import AutoformerDecoder, AutoformerEncoder
