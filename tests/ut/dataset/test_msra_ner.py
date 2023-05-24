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
Test Msra_ner
"""
import os
import shutil
import unittest
import pytest
import mindspore as ms
from mindnlp.transforms import BertTokenizer
from mindnlp.dataset import HF_Msra_ner, HF_Msra_ner_Process
from mindnlp import load_dataset, process


class TestMsraNer(unittest.TestCase):
    r"""
    Test Msra_ner
    """

    @classmethod
    def setUpClass(cls):
        cls.root = os.path.join(os.path.expanduser("~"), ".mindnlp")

    @classmethod
    def tearDownClass(cls):
        shutil.rmtree(cls.root)

    @pytest.mark.download
    def test_hf_msra_ner(self):
        """Test HF_Msra_ner"""
        num_lines = {
            "train": 45001,
            "test": 3443,
        }
        dataset_train, dataset_test = HF_Msra_ner(
            root=self.root, split=("train", "test")
        )
        assert dataset_train.get_dataset_size() == num_lines["train"]
        assert dataset_test.get_dataset_size() == num_lines["test"]

        dataset_train = HF_Msra_ner(root=self.root, split="train")
        dataset_test = HF_Msra_ner(root=self.root, split="test")
        assert dataset_train.get_dataset_size() == num_lines["train"]
        assert dataset_test.get_dataset_size() == num_lines["test"]

    @pytest.mark.download
    def test_hf_msra_ner_by_register(self):
        """test HF_Msra_ner by register"""
        _ = load_dataset(
            "HF_Msra_ner",
            root=self.root,
            split=("train", "test"),
        )

    @pytest.mark.download
    def test_hf_msra_ner_process(self):
        r"""
        Test HF_Msra_ner_Process
        """

        test_dataset = HF_Msra_ner(split='test')
        tokenizer = BertTokenizer.from_pretrained('bert-base-chinese')
        agnews_dataset = HF_Msra_ner_Process(
            test_dataset, tokenizer=tokenizer)

        agnews_dataset = agnews_dataset.create_tuple_iterator()
        assert (next(agnews_dataset)[1]).dtype == ms.int64

    @pytest.mark.download
    def test_hf_msra_ner_process_by_register(self):
        """test HF_Msra_ner_Process process by register"""
        test_dataset = HF_Msra_ner(split='test')
        tokenizer = BertTokenizer.from_pretrained('bert-base-chinese')
        test_dataset = process('HF_Msra_ner',
                               test_dataset, tokenizer=tokenizer)

        test_dataset = test_dataset.create_tuple_iterator()
        assert (next(test_dataset)[1]).dtype == ms.int64
