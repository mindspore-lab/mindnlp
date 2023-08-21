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
# pylint: disable=C0301
# pylint: disable=W4902
"""Test ChatGLM"""
import random
import unittest
import pytest
import numpy as np

import mindspore
from mindspore import Tensor

from mindnlp.models.glm.chatglm import ChatGLMForConditionalGeneration
from mindnlp.transforms.tokenizers import ChatGLMTokenizer

def set_random_seed(seed):
    """set random seed"""
    random.seed(seed)

    # mindspore RNGs
    mindspore.set_seed(seed)

    # numpy RNG
    np.random.seed(seed)


def ids_tensor(shape, vocab_size):
    """Creates a random int32 tensor of the shape within the vocab size"""
    total_dims = 1
    for dim in shape:
        total_dims *= dim

    values = []
    for _ in range(total_dims):
        values.append(random.randint(0, vocab_size - 1))

    return mindspore.Tensor(values, dtype=mindspore.int64).view(shape)


def get_model_and_tokenizer():
    """get model and tokenizer"""
    model = ChatGLMForConditionalGeneration.from_pretrained("chatglm-6b")

    tokenizer = ChatGLMTokenizer.from_pretrained("chatglm-6b")
    return model, tokenizer


class ChatGLMGenerationTest(unittest.TestCase):
    """ChatGLM generation test."""
    @pytest.mark.skipif(True, reason="not ready")
    def test_chat(self):
        """test chat"""
        model, tokenizer = get_model_and_tokenizer()
        prompts = ["你好", "介绍一下清华大学", "它创建于哪一年"]
        history = []
        set_random_seed(42)
        expected_responses = [
            '你好👋！我是人工智能助手 ChatGLM-6B，很高兴见到你，欢迎问我任何问题。',
            '清华大学是中国著名的综合性研究型大学，位于中国北京市海淀区，创建于 1911 年，前身是清华学堂。作为我国顶尖高等教育机构之一，清华大学在科学研究、工程技术、信息技术、经济管理等领域处于领先地位，也是世界上最著名的工程学府之一。\n\n清华大学拥有世界一流的教学设施和科学研究平台，设有多个学院和研究中心，包括工程学院、自然科学学院、社会科学学院、人文学院、法学院、经济管理学院等。学校拥有众多知名教授和研究团队，其中包括多位院士、国家杰出青年科学基金获得者、长江学者等。\n\n清华大学的本科生招生范围为全国中学毕业生，本科生入学要求严格，考试成绩优秀。同时，清华大学也提供研究生和博士生招生，包括硕士研究生和博士研究生。',
            '清华大学创建于 1911 年。'
        ]
        for (prompt, expected_response) in zip(prompts, expected_responses):
            response, history = model.chat(tokenizer, prompt, history=history)
            print(repr(response))
            self.assertEquals(expected_response, response)

    @pytest.mark.skipif(True, reason="not ready")
    def test_stream_chat(self):
        """test steam chat"""
        model, tokenizer = get_model_and_tokenizer()
        prompts = ["你好", "介绍一下清华大学", "它创建于哪一年"]
        history = []
        expected_responses = [
            '你好👋！我是人工智能助手 ChatGLM-6B，很高兴见到你，欢迎问我任何问题。',
            '清华大学是中国著名的综合性研究型大学，位于中国北京市海淀区，创建于 1911 年，前身是清华学堂。作为我国顶尖高等教育机构之一，清华大学在科学研究、工程技术、信息技术、经济管理等领域处于领先地位，也是世界上最著名的工程学府之一。\n\n清华大学拥有世界一流的教学设施和科学研究平台，设有多个学院和研究中心，包括工程学院、自然科学学院、社会科学学院、人文学院、法学院、经济管理学院等。学校拥有众多知名教授和研究团队，其中包括多位院士、国家杰出青年科学基金获得者、长江学者等。\n\n清华大学的本科生招生范围为全国中学毕业生，本科生入学要求严格，考试成绩优秀。同时，清华大学也提供研究生和博士生招生，包括硕士研究生和博士研究生。',
            '清华大学创建于 1911 年。'
        ]
        set_random_seed(42)
        for prompt, expected_response in zip(prompts, expected_responses):
            response = ""
            for _, (response, history) in enumerate(model.stream_chat(tokenizer, prompt, history=history)):
                pass
            print(repr(response))
            self.assertEquals(expected_response, response)

    @pytest.mark.download
    def test_generation(self):
        """test_generation"""
        model, tokenizer = get_model_and_tokenizer()
        parameters = [
                    ("晚上睡不着怎么办", False, 2048, 1, True, 4),
                    ("介绍一下清华大学", False, 64, 1, False, 1),
                    ("推荐几个电影", False, 2048, 1, True, 4),
                    ("怎么用Pytorch写一个模型？", False, 2048, 1, True, 4),
                    #   (True, 2048, 1),
                    #   (True, 64, 1),
                    #   (True, 2048, 4)
                      ]
        for sentence, do_sample, max_length, num_beams, use_bucket, bucket_num in parameters:
            set_random_seed(42)
            inputs = tokenizer(sentence)
            inputs = Tensor([inputs])
            outputs = model.generate(
                inputs,
                do_sample=do_sample,
                max_length=max_length,
                num_beams=num_beams,
                jit=True,
                use_bucket=use_bucket,
                bucket_num=bucket_num
            )

            outputs = outputs.asnumpy().tolist()[0]
            out_sentence = tokenizer.decode(outputs, skip_special_tokens=True)
            print(out_sentence)

    @pytest.mark.skipif(True, reason="not ready")
    def test_batch_generation(self):
        """test batch generation"""
        model, tokenizer = get_model_and_tokenizer()
        sentences = [
            "你好",
            "介绍一下清华大学"
        ]
        parameters = [(False, 2048, 1),
                      (False, 64, 1),
                      (True, 2048, 1),
                      (True, 64, 1),
                      (True, 2048, 4)]
        expected_out_sentences = [
            ['你好 你好👋!我是人工智能助手 ChatGLM-6B,很高兴见到你,欢迎问我任何问题。',
             '介绍一下清华大学 清华大学是中国著名的综合性大学,位于北京市海淀区双清路30号,其历史可以追溯到1911年创建的清华学堂,1925年更名为清华学校,1937年抗日战争全面爆发后南迁长沙,1946年迁回清华园。新中国成立后,清华学校更名为清华大学。\n\n清华大学是中国最顶尖的大学之一,在工程、科学、技术、经济、管理等领域都有很高的学术声誉和影响力。学校拥有世界一流的教学设施和科学研究平台,有多个学院和研究中心,包括工程学院、自然科学学院、人文学院、社会科学学院、经济管理学院、法学院、美术学院、医学院、器学院等。\n\n清华大学的本科生招生始于2000年,实行全面二孩政策后,本科生招生规模不断扩大。截至2022年,清华大学共有本科生近3万人,研究生近2万人,其中国际学生占比约为10%。清华大学的本科生教育注重通识教育和个性化培养,强调实践、创新、国际化和综合素质。'],
            [
                '你好 你好👋!我是人工智能助手 ChatGLM-6B,很高兴见到你,欢迎问我任何问题。',
                '介绍一下清华大学 清华大学是中国著名的综合性大学,位于北京市海淀区双清路30号,其历史可以追溯到1911年创建的清华学堂,1925年更名为清华学校,1937年抗日战争全面爆发后南迁长沙,1946年迁回'
            ],
            [
                '你好 你好👋!我是人工智能助手 ChatGLM-6B,很高兴见到你,欢迎问我任何问题。',
                '介绍一下清华大学 清华大学是中国著名的综合性研究型大学,位于北京市海淀区双清路 30 号,其溯源于 1911 年创建的清华学堂, 1925 年更名为清华学校, 1937 年秋抗日战争全面爆发后闭校。1949 年 10 月开学复校,成为我国第一个社会主义大学生活了的高校。截至 2023 年,清华学校共管辖 2 个学院、13 个系,有本科专业 60 个,研究生专业 190 个。'
            ],
            [
                '你好 你好👋!我是人工智能助手 ChatGLM-6B,很高兴见到你,欢迎问我任何问题。',
                '介绍一下清华大学 清华大学是中国著名的综合性研究型大学,位于北京市海淀区双清路 30 号,其溯源于 1911 年创建的清华学堂, 1925 年更名为清华学校, 1937 年秋抗日战争全面爆发后'
            ],
            [
                '你好 你好👋!我是人工智能助手 ChatGLM-6B,很高兴见到你,欢迎问我任何问题。',
                '介绍一下清华大学 清华大学是中国著名的综合性研究型大学,位于北京市海淀区双清路30号,其历史可以追溯到1911年创建的清华学堂,1925年更名为清华学校,1937年抗日战争全面爆发后南迁长沙,与北京大学、南开大学组建国立长沙临时大学,1938年迁至 昆明改名为国立西南联合大学,1946年迁回北京。新中国成立后,清华学校更名为清华大学。'
            ]
        ]
        for (do_sample, max_length, num_beams), expected_output_sentence in zip(parameters, expected_out_sentences):
            set_random_seed(42)
            inputs = tokenizer(sentences, padding=True)
            inputs = Tensor(inputs)
            outputs = model.generate(
                **inputs,
                do_sample=do_sample,
                max_length=max_length,
                num_beams=num_beams
            )

            batch_out_sentence = tokenizer.batch_decode(outputs, skip_special_tokens=True)
            print(batch_out_sentence)
            self.assertListEqual(expected_output_sentence, batch_out_sentence)
