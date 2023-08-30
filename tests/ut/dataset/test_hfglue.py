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
import mindspore
from mindnlp.dataset import HF_GLUE, HF_GLUE_Process

class TestHFGLUE(unittest.TestCase):
    r"""
    Test HF_GLUE
    """

    @classmethod
    def setUpClass(cls):
        cls.root = os.path.join(os.path.expanduser("~"), ".mindnlp")

    @classmethod
    def tearDownClass(cls):
        shutil.rmtree(cls.root)

    @pytest.mark.download
    def test_hf_glue_cola(self):
        """Test glue_cola"""
        num_lines = {
            "train": 8551,
            "test": 1063,
            "validation": 1043,
        }
        dataset_train, dataset_validation, dataset_test = HF_GLUE(
            name="cola", root=self.root, split=("train", "validation", "test")
        )
        assert dataset_train.get_dataset_size() == num_lines["train"]
        assert dataset_test.get_dataset_size() == num_lines["test"]
        assert dataset_validation.get_dataset_size() == num_lines["validation"]

    @pytest.mark.download
    def test_hf_glue_sst2(self):
        """Test glue_sst2"""
        num_lines = {
            "train": 67349,
            "test": 1821,
            "validation": 872,
        }
        dataset_train, dataset_validation, dataset_test = HF_GLUE(
            name="sst2", root=self.root, split=("train", "validation", "test")
        )
        assert dataset_train.get_dataset_size() == num_lines["train"]
        assert dataset_test.get_dataset_size() == num_lines["test"]
        assert dataset_validation.get_dataset_size() == num_lines["validation"]

    @pytest.mark.download
    def test_hf_glue_mrpc(self):
        """Test glue_mrpc"""
        num_lines = {
            "train": 3668,
            "test": 1725,
            "validation": 408,
        }
        dataset_train, dataset_validation, dataset_test = HF_GLUE(
            name="mrpc", root=self.root, split=("train", "dev", "test")
        )
        assert dataset_train.get_dataset_size() == num_lines["train"]
        assert dataset_test.get_dataset_size() == num_lines["test"]
        assert dataset_validation.get_dataset_size() == num_lines["validation"]

    @pytest.mark.download
    def test_hf_glue_qqp(self):
        """Test glue_qqp"""
        num_lines = {
            "train": 363846,
            "test": 390965,
            "validation": 40430,
        }
        dataset_train, dataset_validation, dataset_test = HF_GLUE(
            name="qqp", root=self.root, split=("train", "validation", "test")
        )
        assert dataset_train.get_dataset_size() == num_lines["train"]
        assert dataset_test.get_dataset_size() == num_lines["test"]
        assert dataset_validation.get_dataset_size() == num_lines["validation"]

    @pytest.mark.download
    def test_hf_glue_stsb(self):
        """Test glue_stsb"""
        num_lines = {
            "train": 5749,
            "test": 1379,
            "validation": 1500,
        }
        dataset_train, dataset_validation, dataset_test = HF_GLUE(
            name="stsb", root=self.root, split=("train", "dev", "test")
        )
        assert dataset_train.get_dataset_size() == num_lines["train"]
        assert dataset_test.get_dataset_size() == num_lines["test"]
        assert dataset_validation.get_dataset_size() == num_lines["validation"]

    @pytest.mark.download
    def test_hf_glue_mnli(self):
        """Test glue_mnli"""
        num_lines = {
            "test_matched": 9796,
            "test_mismatched": 9847,
            "train": 392702,
            "validation_matched": 9815,
            "validation_mismatched": 9832,
        }
        dataset_train, dataset_validation_matched, dataset_validation_mismatched, dataset_test_matched, dataset_test_mismatched = HF_GLUE(
            name="mnli", root=self.root, split=("train",  "validation_matched", "validation_mismatched","test_matched", "test_mismatched",)
        )
        assert dataset_train.get_dataset_size() == num_lines["train"]
        assert dataset_test_matched.get_dataset_size() == num_lines["test_matched"]
        assert dataset_test_mismatched.get_dataset_size() == num_lines["test_mismatched"]
        assert dataset_validation_matched.get_dataset_size() == num_lines["validation_matched"]
        assert dataset_validation_mismatched.get_dataset_size() == num_lines["validation_mismatched"]

    @pytest.mark.download
    def test_hf_glue_mnli_mismatched(self):
        """Test glue_mnli_mismatched"""
        num_lines = {
            "test": 9847,
            "validation": 9832,
        }
        dataset_validation, dataset_test = HF_GLUE(
            name="mnli_mismatched", root=self.root,
            split=("validation", "test")
        )
        assert dataset_test.get_dataset_size() == num_lines["test"]
        assert dataset_validation.get_dataset_size() == num_lines["validation"]

    @pytest.mark.download
    def test_hf_glue_mnli_matched(self):
        """Test glue_mnli_matched"""
        num_lines = {
            "test": 9796,
            "validation": 9815,
        }
        dataset_validation, dataset_test = HF_GLUE(
            name="mnli_matched", root=self.root,
            split=("validation", "test")
        )
        assert dataset_test.get_dataset_size() == num_lines["test"]
        assert dataset_validation.get_dataset_size() == num_lines["validation"]

    @pytest.mark.download
    def test_hf_glue_qnli(self):
        """Test glue_qnli"""
        num_lines = {
            "train": 104743,
            "test": 5463,
            "validation": 5463,
        }
        dataset_train, dataset_validation, dataset_test = HF_GLUE(
            name="qnli", root=self.root, split=("train", "dev", "test")
        )
        assert dataset_train.get_dataset_size() == num_lines["train"]
        assert dataset_test.get_dataset_size() == num_lines["test"]
        assert dataset_validation.get_dataset_size() == num_lines["validation"]

    @pytest.mark.download
    def test_hf_glue_rte(self):
        """Test glue_rte"""
        num_lines = {
            "train": 2490,
            "test": 3000,
            "validation": 277,
        }
        dataset_train, dataset_validation, dataset_test = HF_GLUE(
            name="rte", root=self.root, split=("train", "dev", "test")
        )
        assert dataset_train.get_dataset_size() == num_lines["train"]
        assert dataset_test.get_dataset_size() == num_lines["test"]
        assert dataset_validation.get_dataset_size() == num_lines["validation"]

    @pytest.mark.download
    def test_hf_glue_wnli(self):
        """Test glue_wnli"""
        num_lines = {
            "train": 635,
            "test": 146,
            "validation": 71,
        }
        dataset_train, dataset_validation, dataset_test = HF_GLUE(
            name="wnli", root=self.root, split=("train", "dev", "test")
        )
        assert dataset_train.get_dataset_size() == num_lines["train"]
        assert dataset_test.get_dataset_size() == num_lines["test"]
        assert dataset_validation.get_dataset_size() == num_lines["validation"]

    @pytest.mark.download
    def test_hf_glue_ax(self):
        """Test glue_ax"""
        num_lines = {
            "test": 1104,
        }
        dataset_test = HF_GLUE(
            name="ax", root=self.root, split="test"
        )
        assert dataset_test.get_dataset_size() == num_lines["test"]

    @pytest.mark.download
    def test_hf_glue_process(self):
        """
        Test hf_glue process
        """

        train_dataset = HF_GLUE(name="sst2", root=self.root, split="train")
        train_dataset, vocab = HF_GLUE_Process("sst2", train_dataset)

        train_dataset = train_dataset.create_tuple_iterator()
        assert (next(train_dataset)[1]).dtype == mindspore.int32

        for _, value in vocab.vocab().items():
            assert isinstance(value, int)
            break
        dataset = HF_GLUE(name="qnli", root=self.root, split=["test"])
        dataset, vocab = HF_GLUE_Process("qnli", dataset)

        dataset = dataset.create_tuple_iterator()
        assert (next(dataset)[0]).dtype == mindspore.int32
        assert (next(dataset)[1]).dtype == mindspore.int32

        for _, value in vocab.vocab().items():
            assert isinstance(value, int)
            break
