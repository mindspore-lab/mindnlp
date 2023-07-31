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
from mindnlp.dataset import HF_CMMLU

['agronomy', 'anatomy', 'ancient_chinese', 'arts', 'astronomy', 'business_ethics', 'chinese_civil_service_exam', 'chinese_driving_rule', 'chinese_food_culture', 'chinese_foreign_policy', 'chinese_history', 'chinese_literature', 
'chinese_teacher_qualification', 'clinical_knowledge', 'college_actuarial_science', 'college_education', 'college_engineering_hydrology', 'college_law', 'college_mathematics', 'college_medical_statistics', 'college_medicine', 'computer_science',
'computer_security', 'conceptual_physics', 'construction_project_management', 'economics', 'education', 'electrical_engineering', 'elementary_chinese', 'elementary_commonsense', 'elementary_information_and_technology', 'elementary_mathematics', 
'ethnology', 'food_science', 'genetics', 'global_facts', 'high_school_biology', 'high_school_chemistry', 'high_school_geography', 'high_school_mathematics', 'high_school_physics', 'high_school_politics', 'human_sexuality',
'international_law', 'journalism', 'jurisprudence', 'legal_and_moral_basis', 'logical', 'machine_learning', 'management', 'marketing', 'marxist_theory', 'modern_chinese', 'nutrition', 'philosophy', 'professional_accounting', 'professional_law', 
'professional_medicine', 'professional_psychology', 'public_relations', 'security_study', 'sociology', 'sports_science', 'traditional_chinese_medicine', 'virology', 'world_history', 'world_religions']
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
        shutil.rmtree(cls.root)
        pass

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
    
#     @pytest.mark.download
#     def test_hf_cmmlu_download_all_tasks(self):
#         """Test download cmmlu: all subtask, time-costly -- uncomment to test"""
#         all_valid_tasks = ['agronomy', 'anatomy', 'ancient_chinese', 'arts', 'astronomy', 'business_ethics', 'chinese_civil_service_exam', 'chinese_driving_rule', 'chinese_food_culture', 'chinese_foreign_policy', 'chinese_history', 'chinese_literature', 
# 'chinese_teacher_qualification', 'clinical_knowledge', 'college_actuarial_science', 'college_education', 'college_engineering_hydrology', 'college_law', 'college_mathematics', 'college_medical_statistics', 'college_medicine', 'computer_science',
# 'computer_security', 'conceptual_physics', 'construction_project_management', 'economics', 'education', 'electrical_engineering', 'elementary_chinese', 'elementary_commonsense', 'elementary_information_and_technology', 'elementary_mathematics', 
# 'ethnology', 'food_science', 'genetics', 'global_facts', 'high_school_biology', 'high_school_chemistry', 'high_school_geography', 'high_school_mathematics', 'high_school_physics', 'high_school_politics', 'human_sexuality',
# 'international_law', 'journalism', 'jurisprudence', 'legal_and_moral_basis', 'logical', 'machine_learning', 'management', 'marketing', 'marxist_theory', 'modern_chinese', 'nutrition', 'philosophy', 'professional_accounting', 'professional_law', 
# 'professional_medicine', 'professional_psychology', 'public_relations', 'security_study', 'sociology', 'sports_science', 'traditional_chinese_medicine', 'virology', 'world_history', 'world_religions']
#         for task in all_valid_tasks:
#             test, dev = HF_CMMLU(
#             name = task,
#             root = self.root,
#             split = ("test", "dev"),
#         )

    # @pytest.mark.download
    # def test_hf_glue_sst2(self):
    #     """Test glue_sst2"""
    #     num_lines = {
    #         "train": 67349,
    #         "test": 1821,
    #         "validation": 872,
    #     }
    #     dataset_train, dataset_validation, dataset_test = HF_GLUE(
    #         name="sst2", root=self.root, split=("train", "validation", "test")
    #     )
    #     assert dataset_train.get_dataset_size() == num_lines["train"]
    #     assert dataset_test.get_dataset_size() == num_lines["test"]
    #     assert dataset_validation.get_dataset_size() == num_lines["validation"]

    # @pytest.mark.download
    # def test_hf_glue_mrpc(self):
    #     """Test glue_mrpc"""
    #     num_lines = {
    #         "train": 3668,
    #         "test": 1725,
    #         "validation": 408,
    #     }
    #     dataset_train, dataset_validation, dataset_test = HF_GLUE(
    #         name="mrpc", root=self.root, split=("train", "validation", "test")
    #     )
    #     assert dataset_train.get_dataset_size() == num_lines["train"]
    #     assert dataset_test.get_dataset_size() == num_lines["test"]
    #     assert dataset_validation.get_dataset_size() == num_lines["validation"]

    # @pytest.mark.download
    # def test_hf_glue_qqp(self):
    #     """Test glue_qqp"""
    #     num_lines = {
    #         "train": 363846,
    #         "test": 390965,
    #         "validation": 40430,
    #     }
    #     dataset_train, dataset_validation, dataset_test = HF_GLUE(
    #         name="qqp", root=self.root, split=("train", "validation", "test")
    #     )
    #     assert dataset_train.get_dataset_size() == num_lines["train"]
    #     assert dataset_test.get_dataset_size() == num_lines["test"]
    #     assert dataset_validation.get_dataset_size() == num_lines["validation"]

    # @pytest.mark.download
    # def test_hf_glue_stsb(self):
    #     """Test glue_stsb"""
    #     num_lines = {
    #         "train": 5749,
    #         "test": 1379,
    #         "validation": 1500,
    #     }
    #     dataset_train, dataset_validation, dataset_test = HF_GLUE(
    #         name="stsb", root=self.root, split=("train", "validation", "test")
    #     )
    #     assert dataset_train.get_dataset_size() == num_lines["train"]
    #     assert dataset_test.get_dataset_size() == num_lines["test"]
    #     assert dataset_validation.get_dataset_size() == num_lines["validation"]

    # @pytest.mark.download
    # def test_hf_glue_mnli(self):
    #     """Test glue_mnli"""
    #     num_lines = {
    #         "test_matched": 9796,
    #         "test_mismatched": 9847,
    #         "train": 392702,
    #         "validation_matched": 9815,
    #         "validation_mismatched": 9832,
    #     }
    #     dataset_train, dataset_validation_matched, dataset_validation_mismatched, dataset_test_matched, dataset_test_mismatched = HF_GLUE(
    #         name="mnli", root=self.root, split=("train",  "validation_matched", "validation_mismatched","test_matched", "test_mismatched",)
    #     )
    #     assert dataset_train.get_dataset_size() == num_lines["train"]
    #     assert dataset_test_matched.get_dataset_size() == num_lines["test_matched"]
    #     assert dataset_test_mismatched.get_dataset_size() == num_lines["test_mismatched"]
    #     assert dataset_validation_matched.get_dataset_size() == num_lines["validation_matched"]
    #     assert dataset_validation_mismatched.get_dataset_size() == num_lines["validation_mismatched"]

    # @pytest.mark.download
    # def test_hf_glue_mnli_mismatched(self):
    #     """Test glue_mnli_mismatched"""
    #     num_lines = {
    #         "test": 9847,
    #         "validation": 9832,
    #     }
    #     dataset_validation, dataset_test = HF_GLUE(
    #         name="mnli_mismatched", root=self.root,
    #         split=("validation", "test")
    #     )
    #     assert dataset_test.get_dataset_size() == num_lines["test"]
    #     assert dataset_validation.get_dataset_size() == num_lines["validation"]

    # @pytest.mark.download
    # def test_hf_glue_mnli_matched(self):
    #     """Test glue_mnli_matched"""
    #     num_lines = {
    #         "test": 9796,
    #         "validation": 9815,
    #     }
    #     dataset_validation, dataset_test = HF_GLUE(
    #         name="mnli_matched", root=self.root,
    #         split=("validation", "test")
    #     )
    #     assert dataset_test.get_dataset_size() == num_lines["test"]
    #     assert dataset_validation.get_dataset_size() == num_lines["validation"]

    # @pytest.mark.download
    # def test_hf_glue_qnli(self):
    #     """Test glue_qnli"""
    #     num_lines = {
    #         "train": 104743,
    #         "test": 5463,
    #         "validation": 5463,
    #     }
    #     dataset_train, dataset_validation, dataset_test = HF_GLUE(
    #         name="qnli", root=self.root, split=("train", "validation", "test")
    #     )
    #     assert dataset_train.get_dataset_size() == num_lines["train"]
    #     assert dataset_test.get_dataset_size() == num_lines["test"]
    #     assert dataset_validation.get_dataset_size() == num_lines["validation"]

    # @pytest.mark.download
    # def test_hf_glue_rte(self):
    #     """Test glue_rte"""
    #     num_lines = {
    #         "train": 2490,
    #         "test": 3000,
    #         "validation": 277,
    #     }
    #     dataset_train, dataset_validation, dataset_test = HF_GLUE(
    #         name="rte", root=self.root, split=("train", "validation", "test")
    #     )
    #     assert dataset_train.get_dataset_size() == num_lines["train"]
    #     assert dataset_test.get_dataset_size() == num_lines["test"]
    #     assert dataset_validation.get_dataset_size() == num_lines["validation"]

    # @pytest.mark.download
    # def test_hf_glue_wnli(self):
    #     """Test glue_wnli"""
    #     num_lines = {
    #         "train": 635,
    #         "test": 146,
    #         "validation": 71,
    #     }
    #     dataset_train, dataset_validation, dataset_test = HF_GLUE(
    #         name="wnli", root=self.root, split=("train", "validation", "test")
    #     )
    #     assert dataset_train.get_dataset_size() == num_lines["train"]
    #     assert dataset_test.get_dataset_size() == num_lines["test"]
    #     assert dataset_validation.get_dataset_size() == num_lines["validation"]

    # @pytest.mark.download
    # def test_hf_glue_ax(self):
    #     """Test glue_ax"""
    #     num_lines = {
    #         "test": 1104,
    #     }
    #     dataset_test = HF_GLUE(
    #         name="ax", root=self.root, split="test"
    #     )
    #     assert dataset_test.get_dataset_size() == num_lines["test"]
