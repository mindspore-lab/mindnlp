# Copyright 2022 Huawei Technologies Co., Ltd
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
Test HF_GLUE
"""

import os
import unittest
import shutil
import pytest
from mindnlp.dataset import HF_CMMLU, CMMLU

class TestHFCMMLU(unittest.TestCase):
    r"""
    Test HF_CMMLU
    """
    all_valid_tasks = ['agronomy', 'anatomy', 'ancient_chinese', 'arts', 'astronomy', 'business_ethics', 'chinese_civil_service_exam', 'chinese_driving_rule', 'chinese_food_culture', 'chinese_foreign_policy', 'chinese_history', 'chinese_literature', 
'chinese_teacher_qualification', 'clinical_knowledge', 'college_actuarial_science', 'college_education', 'college_engineering_hydrology', 'college_law', 'college_mathematics', 'college_medical_statistics', 'college_medicine', 'computer_science',
'computer_security', 'conceptual_physics', 'construction_project_management', 'economics', 'education', 'electrical_engineering', 'elementary_chinese', 'elementary_commonsense', 'elementary_information_and_technology', 'elementary_mathematics', 
'ethnology', 'food_science', 'genetics', 'global_facts', 'high_school_biology', 'high_school_chemistry', 'high_school_geography', 'high_school_mathematics', 'high_school_physics', 'high_school_politics', 'human_sexuality',
'international_law', 'journalism', 'jurisprudence', 'legal_and_moral_basis', 'logical', 'machine_learning', 'management', 'marketing', 'marxist_theory', 'modern_chinese', 'nutrition', 'philosophy', 'professional_accounting', 'professional_law', 
'professional_medicine', 'professional_psychology', 'public_relations', 'security_study', 'sociology', 'sports_science', 'traditional_chinese_medicine', 'virology', 'world_history', 'world_religions']

    @classmethod
    def setUpClass(cls):
        cls.root = os.path.join(os.path.expanduser("~"), ".mindnlp", "datasets", "hf_datasets", "CMMLU")
    
    @classmethod
    def tearDownClass(cls):
        if os.path.exists(cls.root):
            shutil.rmtree(cls.root)

    @pytest.mark.download
    def test_hf_cmmlu_ancient_chinese(self):
        """Test load cmmlu:ancient_chinese"""
        test, dev = HF_CMMLU(
            name = "ancient_chinese",
            root = self.root,
            split = ("test", "dev"),
        )

        num_lines = {
            "test": 164,
            "dev": 5,
        }
        assert test.get_dataset_size() == num_lines["test"]
        assert dev.get_dataset_size() == num_lines["dev"]

    @pytest.mark.download
    def test_hf_cmmlu_ancient_chinese_dev(self):
        """Test load cmmlu:ancient_chinese"""
        dev = HF_CMMLU(
            name = "ancient_chinese",
            root = self.root,
            split = "dev",
        )

        assert dev.get_dataset_size() == 5  

    @pytest.mark.download
    def test_hf_cmmlu_ancient_agronomy(self):
        """Test load cmmlu:agronomy"""
        test, dev = HF_CMMLU(
            name = "agronomy",
            root = self.root,
            split = ("test", "dev"),
        )

        num_lines = {
            "test": 169,
            "dev": 5,
        }

        assert test.get_dataset_size() == num_lines['test']
        assert dev.get_dataset_size() == num_lines['dev']
    
    @pytest.mark.local
    def test_cmmlu_agronomy(self):
        """Test load cmmlu:agronomy"""
        test, dev = CMMLU(
            name = "agronomy",
            root = "/home/cjl/code/chat/collections/datasets/CMMLU/data",
            split = ("test", "dev"),
        )

        num_lines = {
            "test": 169,
            "dev": 5,
        }

        assert test.get_dataset_size() == num_lines['test']
        assert dev.get_dataset_size() == num_lines['dev']
