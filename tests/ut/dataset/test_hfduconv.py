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
Test DuConv
"""

import os
import shutil
import unittest
import pytest
import mindspore as ms
from mindnlp import load_dataset, process, Vocab
from mindnlp.dataset import hf_duconv, hfduconv_process

char_dic = {"<unk>": 0, "<pad>": 1, "e": 2, "t": 3, "a": 4, "i": 5, "n": 6,\
                    "o": 7, "s": 8, "r": 9, "h": 10, "l": 11, "d": 12, "c": 13, "u": 14,\
                    "m": 15, "f": 16, "p": 17, "g": 18, "w": 19, "y": 20, "b": 21, ",": 22,\
                    "v": 23, ".": 24, "k": 25, "1": 26, "0": 27, "x": 28, "2": 29, "\"": 30, \
                    "-": 31, "j": 32, "9": 33, "'": 34, ")": 35, "(": 36, "?": 37, "z": 38,\
                    "5": 39, "8": 40, "q": 41, "3": 42, "4": 43, "7": 44, "6": 45, ";": 46,\
                    ":": 47, "\u2013": 48, "%": 49, "/": 50, "]": 51, "[": 52}
char_vocab = Vocab(char_dic)


class TestDuConv(unittest.TestCase):
    r"""
    Test DuConv 
    """

    @classmethod
    def setUpClass(cls):
        cls.root = os.path.join(os.path.expanduser("~"), ".mindnlp")

    @classmethod
    def tearDownClass(cls):
        shutil.rmtree(cls.root)

    @pytest.mark.download
    @pytest.mark.local
    def test_duconv(self):
        """Test DuConv"""
        num_lines = {
            "train": 19900,
            "dev": 2000,
            "test1":5000,
            "test2":10100,
        }
        dataset_train, dataset_dev, dataset_test1, dataset_test2 = hf_duconv(root=self.root, split=('train', 'dev','test1','test2'))
        assert dataset_train.get_dataset_size() == num_lines["train"]
        assert dataset_dev.get_dataset_size() == num_lines["dev"]
        assert dataset_test1.get_dataset_size() == num_lines["test1"]
        assert dataset_test2.get_dataset_size() == num_lines["test2"]

        dataset_train = hf_duconv(root=self.root, split='train')
        dataset_dev = hf_duconv(root=self.root, split='dev')
        dataset_test1 = hf_duconv(root=self.root, split='test1')
        dataset_test2 = hf_duconv(root=self.root, split='test2')

        assert dataset_train.get_dataset_size() == num_lines["train"]
        assert dataset_dev.get_dataset_size() == num_lines["dev"]
        assert dataset_test1.get_dataset_size() == num_lines["test1"]
        assert dataset_test2.get_dataset_size() == num_lines["test2"]

    @pytest.mark.download
    def test_duconv_by_register(self):
        """test Duconv by register"""
        _ = load_dataset('hf_duconv', root=self.root, split=('train', 'dev', 'test1', 'test2'))
    @pytest.mark.download
    def test_duconv_process(self):
        """
        Test Duconv_Process
        """
        dev_dataset = hf_duconv(split='dev')
        duconv_dev = hfduconv_process(dataset=dev_dataset, char_vocab=char_vocab)
        duconv_dev = duconv_dev.create_tuple_iterator()
        assert (next(duconv_dev)[1]).dtype == ms.int32
        assert (next(duconv_dev)[1]).shape == (64, 768)
        assert (next(duconv_dev)[2]).dtype == ms.int32
        assert (next(duconv_dev)[2]).shape == (64, 64)
        assert (next(duconv_dev)[3]).dtype == ms.int32
        assert (next(duconv_dev)[3]).shape == (64, 768, 48)
        assert (next(duconv_dev)[4]).dtype == ms.int32
        assert (next(duconv_dev)[4]).shape == (64, 64, 48)
        assert (next(duconv_dev)[5]).dtype == ms.int32

    @pytest.mark.download
    def test_squad1_process_by_register(self):
        """
        Test Duconv_Process by register
        """
        dev_dataset = hf_duconv(split='dev')
        duconv_dev = process('hf_duconv', dataset=dev_dataset, char_vocab=char_vocab)
        duconv_dev = duconv_dev.create_tuple_iterator()
        assert (next(duconv_dev)[1]).dtype == ms.int32
        assert (next(duconv_dev)[1]).shape == (64, 768)
        assert (next(duconv_dev)[2]).dtype == ms.int32
        assert (next(duconv_dev)[2]).shape == (64, 64)
        assert (next(duconv_dev)[3]).dtype == ms.int32
        assert (next(duconv_dev)[3]).shape == (64, 768, 48)
        assert (next(duconv_dev)[4]).dtype == ms.int32
        assert (next(duconv_dev)[4]).shape == (64, 64, 48)
        assert (next(duconv_dev)[5]).dtype == ms.int32
        