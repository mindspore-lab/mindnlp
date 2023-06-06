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
Test Xnli
"""

import os
import unittest
import shutil
import pytest
from mindnlp.dataset import HF_Xnli


class TestHFXnli(unittest.TestCase):
    r"""
    Test Xnli
    """

    @classmethod
    def setUpClass(cls):
        cls.root = os.path.join(os.path.expanduser("~"), ".mindnlp")

    @classmethod
    def tearDownClass(cls):
        shutil.rmtree(cls.root)

    @pytest.mark.download
    def test_hf_xnli_ar(self):
        """Test xnli_ar"""
        num_lines = {
            "train": 392702,
            "test": 5010,
            "validation": 2490,
        }
        dataset_train, dataset_validation, dataset_test = HF_Xnli(
            name="ar", root=self.root, split=("train", "validation", "test")
        )
        assert dataset_train.get_dataset_size() == num_lines["train"]
        assert dataset_test.get_dataset_size() == num_lines["test"]
        assert dataset_validation.get_dataset_size() == num_lines["validation"]

    @pytest.mark.download
    def test_hf_xnli_bg(self):
        """Test xnli_bg"""
        num_lines = {
            "train": 392702,
            "test": 5010,
            "validation": 2490,
        }
        dataset_train, dataset_validation, dataset_test = HF_Xnli(
            name="bg", root=self.root, split=("train", "validation", "test"))
        assert dataset_train.get_dataset_size() == num_lines["train"]
        assert dataset_test.get_dataset_size() == num_lines["test"]
        assert dataset_validation.get_dataset_size() == num_lines["validation"]

    @pytest.mark.download
    def test_hf_xnli_de(self):
        """Test xnli_de"""
        num_lines = {
            "train": 392702,
            "test": 5010,
            "validation": 2490,
        }
        dataset_train, dataset_validation, dataset_test = HF_Xnli(
            name="de", root=self.root, split=("train", "validation", "test")
        )
        assert dataset_train.get_dataset_size() == num_lines["train"]
        assert dataset_test.get_dataset_size() == num_lines["test"]
        assert dataset_validation.get_dataset_size() == num_lines["validation"]

    @pytest.mark.download
    def test_hf_xnli_el(self):
        """Test xnli_el"""
        num_lines = {
            "train": 392702,
            "test": 5010,
            "validation": 2490,
        }
        dataset_train, dataset_validation, dataset_test = HF_Xnli(
            name="el", root=self.root, split=("train", "validation", "test")
        )
        assert dataset_train.get_dataset_size() == num_lines["train"]
        assert dataset_test.get_dataset_size() == num_lines["test"]
        assert dataset_validation.get_dataset_size() == num_lines["validation"]

    @pytest.mark.download
    def test_hf_xnli_en(self):
        """Test xnli_en"""
        num_lines = {
            "train": 392702,
            "test": 5010,
            "validation": 2490,
        }
        dataset_train, dataset_validation, dataset_test = HF_Xnli(
            name="en", root=self.root, split=("train", "validation", "test")
        )
        assert dataset_train.get_dataset_size() == num_lines["train"]
        assert dataset_test.get_dataset_size() == num_lines["test"]
        assert dataset_validation.get_dataset_size() == num_lines["validation"]

    @pytest.mark.download
    def test_hf_xnli_es(self):
        """Test xnli_es"""
        num_lines = {
            "train": 392702,
            "test": 5010,
            "validation": 2490,
        }
        dataset_train, dataset_validation, dataset_test = HF_Xnli(
            name="es", root=self.root, split=("train", "validation", "test")
        )
        assert dataset_train.get_dataset_size() == num_lines["train"]
        assert dataset_test.get_dataset_size() == num_lines["test"]
        assert dataset_validation.get_dataset_size() == num_lines["validation"]

    @pytest.mark.download
    def test_hf_xnli_fr(self):
        """Test xnli_fr"""
        num_lines = {
            "train": 392702,
            "test": 5010,
            "validation": 2490,
        }
        dataset_train, dataset_validation, dataset_test = HF_Xnli(
            name="fr", root=self.root, split=("train", "validation", "test")
        )
        assert dataset_train.get_dataset_size() == num_lines["train"]
        assert dataset_test.get_dataset_size() == num_lines["test"]
        assert dataset_validation.get_dataset_size() == num_lines["validation"]

    @pytest.mark.download
    def test_hf_xnli_hi(self):
        """Test xnli_hi"""
        num_lines = {
            "train": 392702,
            "test": 5010,
            "validation": 2490,
        }
        dataset_train, dataset_validation, dataset_test = HF_Xnli(
            name="hi", root=self.root, split=("train", "validation", "test")
        )
        assert dataset_train.get_dataset_size() == num_lines["train"]
        assert dataset_test.get_dataset_size() == num_lines["test"]
        assert dataset_validation.get_dataset_size() == num_lines["validation"]

    @pytest.mark.download
    def test_hf_xnli_ru(self):
        """Test xnli_ru"""
        num_lines = {
            "train": 392702,
            "test": 5010,
            "validation": 2490,
        }
        dataset_train, dataset_validation, dataset_test = HF_Xnli(
            name="ru", root=self.root, split=("train", "validation", "test")
        )
        assert dataset_train.get_dataset_size() == num_lines["train"]
        assert dataset_test.get_dataset_size() == num_lines["test"]
        assert dataset_validation.get_dataset_size() == num_lines["validation"]

    @pytest.mark.download
    def test_hf_xnli_sw(self):
        """Test xnli_sw"""
        num_lines = {
            "train": 392702,
            "test": 5010,
            "validation": 2490,
        }
        dataset_train, dataset_validation, dataset_test = HF_Xnli(
            name="sw", root=self.root, split=("train", "validation", "test")
        )
        assert dataset_train.get_dataset_size() == num_lines["train"]
        assert dataset_test.get_dataset_size() == num_lines["test"]
        assert dataset_validation.get_dataset_size() == num_lines["validation"]

    @pytest.mark.download
    def test_hf_xnli_th(self):
        """Test xnli_th"""
        num_lines = {
            "train": 392702,
            "test": 5010,
            "validation": 2490,
        }
        dataset_train, dataset_validation, dataset_test = HF_Xnli(
            name="th", root=self.root, split=("train", "validation", "test")
        )
        assert dataset_train.get_dataset_size() == num_lines["train"]
        assert dataset_test.get_dataset_size() == num_lines["test"]
        assert dataset_validation.get_dataset_size() == num_lines["validation"]

    @pytest.mark.download
    def test_hf_xnli_tr(self):
        """Test xnli_tr"""
        num_lines = {
            "train": 392702,
            "test": 5010,
            "validation": 2490,
        }
        dataset_train, dataset_validation, dataset_test = HF_Xnli(
            name="tr", root=self.root, split=("train", "validation", "test")
        )
        assert dataset_train.get_dataset_size() == num_lines["train"]
        assert dataset_test.get_dataset_size() == num_lines["test"]
        assert dataset_validation.get_dataset_size() == num_lines["validation"]

    @pytest.mark.download
    def test_hf_xnli_ur(self):
        """Test xnli_ur"""
        num_lines = {
            "train": 392702,
            "test": 5010,
            "validation": 2490,
        }
        dataset_train, dataset_validation, dataset_test = HF_Xnli(
            name="ur", root=self.root, split=("train", "validation", "test")
        )
        assert dataset_train.get_dataset_size() == num_lines["train"]
        assert dataset_test.get_dataset_size() == num_lines["test"]
        assert dataset_validation.get_dataset_size() == num_lines["validation"]

    @pytest.mark.download
    def test_hf_xnli_vi(self):
        """Test xnli_vi"""
        num_lines = {
            "train": 392702,
            "test": 5010,
            "validation": 2490,
        }
        dataset_train, dataset_validation, dataset_test = HF_Xnli(
            name="vi", root=self.root, split=("train", "validation", "test")
        )
        assert dataset_train.get_dataset_size() == num_lines["train"]
        assert dataset_test.get_dataset_size() == num_lines["test"]
        assert dataset_validation.get_dataset_size() == num_lines["validation"]

    @pytest.mark.download
    def test_hf_xnli_zh(self):
        """Test xnli_zh"""
        num_lines = {
            "train": 392702,
            "test": 5010,
            "validation": 2490,
        }
        dataset_train, dataset_validation, dataset_test = HF_Xnli(
            name="zh", root=self.root, split=("train", "validation", "test")
        )
        assert dataset_train.get_dataset_size() == num_lines["train"]
        assert dataset_test.get_dataset_size() == num_lines["test"]
        assert dataset_validation.get_dataset_size() == num_lines["validation"]

    @pytest.mark.download
    def test_hf_xnli_all_languages(self):
        """Test xnli_all_languages"""
        num_lines = {
            "train": 392702*15,
            "test": 5010*15,
            "validation": 2490*15,
        }
        dataset_train, dataset_validation, dataset_test = HF_Xnli(
            name="all_languages", root=self.root, split=("train", "validation", "test")
        )
        assert dataset_train.get_dataset_size() == num_lines["train"]
        assert dataset_test.get_dataset_size() == num_lines["test"]
        assert dataset_validation.get_dataset_size() == num_lines["validation"]
        