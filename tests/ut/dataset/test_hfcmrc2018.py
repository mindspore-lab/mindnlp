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
Test HF_CMRC2018
"""

import os
import shutil
import unittest

import mindspore as ms
import pytest

from mindnlp import load_dataset, process, Vocab
from mindnlp.dataset import HF_CMRC2018, HF_CMRC2018_Process

char_dic = {"<unk>": 0, "<pad>": 1, "e": 2, "t": 3, "a": 4, "i": 5, "n": 6, \
            "o": 7, "s": 8, "r": 9, "h": 10, "l": 11, "d": 12, "c": 13, "u": 14, \
            "m": 15, "f": 16, "p": 17, "g": 18, "w": 19, "y": 20, "b": 21, ",": 22, \
            "v": 23, ".": 24, "k": 25, "1": 26, "0": 27, "x": 28, "2": 29, "\"": 30, \
            "-": 31, "j": 32, "9": 33, "'": 34, ")": 35, "(": 36, "?": 37, "z": 38, \
            "5": 39, "8": 40, "q": 41, "3": 42, "4": 43, "7": 44, "6": 45, ";": 46, \
            ":": 47, "\u2013": 48, "%": 49, "/": 50, "]": 51, "[": 52}
char_vocab = Vocab(char_dic)


class TestHFCMRC2018(unittest.TestCase):
    r"""
    Test HF_CMRC2018
    """

    @classmethod
    def setUpClass(cls):
        cls.root = os.path.join(os.path.expanduser("~"), ".mindnlp")

    @classmethod
    def tearDownClass(cls):
        shutil.rmtree(cls.root)

    @pytest.mark.download
    @pytest.mark.local
    def test_hf_cmrc2018(self):
        """Test cmrc2018"""
        num_lines = {
            "train": 10142,
            "validation": 3219,
            "test": 1002,
        }
        dataset_train, dataset_validation, dataset_test = HF_CMRC2018(root=self.root,
                                                                   split=('train', 'validation', 'test'))
        assert dataset_train.get_dataset_size() == num_lines["train"]
        assert dataset_validation.get_dataset_size() == num_lines["validation"]
        assert dataset_test.get_dataset_size() == num_lines["test"]
        dataset_train = HF_CMRC2018(root=self.root, split='train')
        dataset_validation = HF_CMRC2018(root=self.root, split='validation')
        dataset_test = HF_CMRC2018(root=self.root, split='test')

        assert dataset_train.get_dataset_size() == num_lines["train"]
        assert dataset_validation.get_dataset_size() == num_lines["validation"]
        assert dataset_test.get_dataset_size() == num_lines["test"]

    @pytest.mark.download
    def test_hf_cmrc2018_by_register(self):
        """test cmrc2018 by register"""
        _ = load_dataset('hf_cmrc2018',
                         root=self.root,
                         split=('train', 'validation', 'test')
                         )

    @pytest.mark.download
    def test_hf_cmrc2018_process(self):
        """
        Test CMRC2018_Process
        """
        dataset_validation = HF_CMRC2018(split='validation')
        dataset_validation = HF_CMRC2018_Process(dataset=dataset_validation, char_vocab=char_vocab)
        dataset_validation = dataset_validation.create_tuple_iterator()
        assert (next(dataset_validation)[1]).dtype == ms.int32
        assert (next(dataset_validation)[1]).shape == (64, 768)
        assert (next(dataset_validation)[2]).dtype == ms.int32
        assert (next(dataset_validation)[2]).shape == (64, 64)
        assert (next(dataset_validation)[3]).dtype == ms.int32
        assert (next(dataset_validation)[3]).shape == (64, 768, 48)
        assert (next(dataset_validation)[4]).dtype == ms.int32
        assert (next(dataset_validation)[4]).shape == (64, 64, 48)
        assert (next(dataset_validation)[5]).dtype == ms.int32
        assert (next(dataset_validation)[6]).dtype == ms.int32
        assert (next(dataset_validation)[7]).dtype == ms.int32
        assert (next(dataset_validation)[8]).dtype == ms.int32

    @pytest.mark.download
    def test_hf_cmrc2018_process_by_register(self):
        """
        Test CMRC2018_Process by register
        """
        dataset_validation = HF_CMRC2018(split='validation')
        dataset_validation = process('HF_CMRC2018', dataset=dataset_validation, char_vocab=char_vocab)
        dataset_validation = dataset_validation.create_tuple_iterator()
        assert (next(dataset_validation)[1]).dtype == ms.int32
        assert (next(dataset_validation)[1]).shape == (64, 768)
        assert (next(dataset_validation)[2]).dtype == ms.int32
        assert (next(dataset_validation)[2]).shape == (64, 64)
        assert (next(dataset_validation)[3]).dtype == ms.int32
        assert (next(dataset_validation)[3]).shape == (64, 768, 48)
        assert (next(dataset_validation)[4]).dtype == ms.int32
        assert (next(dataset_validation)[4]).shape == (64, 64, 48)
        assert (next(dataset_validation)[5]).dtype == ms.int32
        assert (next(dataset_validation)[6]).dtype == ms.int32
        assert (next(dataset_validation)[7]).dtype == ms.int32
        assert (next(dataset_validation)[8]).dtype == ms.int32
