# Copyright 2024 Huawei Technologies Co., Ltd
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
JiebaTokenizer Python version
"""
import logging
import os

import jieba

jieba.setLogLevel(log_level="ERROR")

class JiebaTokenizer:

    r""" 
        A class for tokenizing Chinese text using the Jieba library.
    
        This class provides methods for initializing the tokenizer with custom word dictionaries and settings, as well as for tokenizing Chinese text with options for different cutting modes and Named Entity
Recognition (NER) recognition.
    
        Args:
            dict_path (str): Path to a custom dictionary file. Default is an empty string.
            custom_word_freq_dict (dict): A dictionary mapping custom words to their frequencies.
    
        Attributes:
            model (object): The Jieba model for tokenization.
    
        Methods:
            __init__(dict_path='', custom_word_freq_dict=None):
                Initializes the JiebaTokenizer instance with a custom dictionary and word frequencies.
    
            tokenize(sentence, cut_all=False, HMM=True):
                Tokenizes the input sentence and returns a list of strings representing the segmented words.
    
        Example Usage:
            tokenizer = JiebaTokenizer()
            tokens = tokenizer.tokenize('我爱自然语言处理', cut_all=True)
            print(tokens)  # Output: ['我', '爱', '自然', '自然语言', '言', '处理']
    """
    def __init__(self, dict_path='', custom_word_freq_dict=None):
        r"""
        Initializes a new instance of the JiebaTokenizer class.
        
        Args:
            self: The current instance of the JiebaTokenizer class.
            dict_path (str): The path to the custom dictionary file. Defaults to an empty string.
            custom_word_freq_dict (dict): A dictionary containing custom word frequencies. Defaults to None.
        
        Returns:
            None. This method does not return any value.
        
        Raises:
            None. This method does not raise any exceptions.
        """
        self.model = jieba
        self.model.default_logger.setLevel(logging.ERROR)
        # 初始化大词典
        if os.path.exists(dict_path):
            self.model.set_dictionary(dict_path)
        # 加载用户自定义词典
        if custom_word_freq_dict:
            for w, f in custom_word_freq_dict.items():
                self.model.add_word(w, freq=f)

    def tokenize(self, sentence, cut_all=False, HMM=True):
        """
        切词并返回切词位置
        :param sentence: 句子
        :param cut_all: 全模式，默认关闭
        :param HMM: 是否打开NER识别，默认打开
        :return:  A list of strings.
        """
        return self.model.lcut(sentence, cut_all=cut_all, HMM=HMM)
