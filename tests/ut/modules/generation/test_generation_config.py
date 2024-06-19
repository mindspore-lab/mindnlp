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
Test GenerationConfig
"""

import unittest
from mindnlp.transformers.generation import GenerationConfig
from mindnlp.transformers import T5Config

class TestGenerationConfig(unittest.TestCase):
    r"""
    Test module.generation GenerationConfig
    """
    def test_generation_config_from_model_config(self):
        """test GenerationConfig.from_model_config()"""
        config = T5Config()
        generation_config = GenerationConfig.from_model_config(config)
        assert config.eos_token_id == generation_config.eos_token_id

    def test_generation_config_from_dict(self):
        """test GenerationConfig.from_dict()"""
        config_dict = T5Config().__dict__
        generation_config = GenerationConfig.from_dict(config_dict)
        assert config_dict['eos_token_id'] == generation_config.eos_token_id

    def test_generation_config_update(self):
        """test GenerationConfig.update()"""
        config = T5Config()
        generation_config = GenerationConfig.from_model_config(config)
        generation_config.update(eos_token_id=666)
        assert generation_config.eos_token_id == 666
