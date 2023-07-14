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
Hugging Face docvqa_zh load function
"""
# pylint: disable=C0103

import os
from typing import Union, Tuple
import json

from mindspore.dataset import GeneratorDataset
from mindnlp.utils.download import cache_file
from mindnlp.dataset.register import load_dataset
from mindnlp.configs import DEFAULT_ROOT

from mindnlp.utils import untar

URL = "https://bj.bcebos.com/paddlenlp/datasets/docvqa_zh.tar.gz"

class DocVqaZh:
    """
    Hugging Face docvqa_zh dataset source
    """
    def __init__(self, path):
        self.path = path
        self._id = []
        self._name = []
        self._page_no = []
        self._text = []
        self._bbox = []
        self._segment_bbox = []
        self._segment_id = []
        self._image = []
        self._width = []
        self._height = []
        self._md5sum = []
        self._question = []
        self._anwsers = []
        self._s_idex = []
        self._load()

    def _load(self):
        idx = 0
        with open(self.path, 'r', encoding="utf-8") as f:
            for line in f:
                example = json.loads(line)

                if "page_no" not in example:
                    example["page_no"] = 0
                name = example["name"]
                page_no = example["page_no"]
                text = example["text"]
                bbox = example["bbox"]
                segment_bbox = example["segment_bbox"]
                segment_id = example["segment_id"]
                image = example["image"]
                width = example["width"]
                height = example["height"]

                qas = example["qas"]

                for qa in qas:
                    if "question_id" not in qa:
                        qa["question_id"] = -1
                    question_id = qa["question_id"]
                    question = qa["question"]
                    for ans in qa['answers']:
                        self._id.append(question_id)
                        self._name.append(name)
                        self._page_no.append(page_no)
                        self._text.append(text)
                        self._bbox.append(bbox)
                        self._segment_bbox.append(segment_bbox)
                        self._segment_id.append(segment_id)
                        self._image.append(image)
                        self._width.append(width)
                        self._height.append(height)

                        self._text.append(text)
                        self._question.append(question)
                        answer = ans['text']
                        self._anwsers.append(answer)
                        s_idx = ans['answer_start']
                        self._s_idex.append(s_idx)
                        idx += 1
                        break

    def __getitem__(self, index):
        return self._id[index], self._text[index], self._question[index],\
            self._anwsers[index], self._s_idex[index]

    def __len__(self):
        return len(self._anwsers)


@load_dataset.register
def docvqa_zh(
    root: str = DEFAULT_ROOT,
    split: Union[Tuple[str], str] = ('train', 'test', 'dev'),
    proxies=None
):
    r"""
    Load the huggingface docvqa_zh dataset.
    """
    cache_dir = os.path.join(root, "datasets", "DocVQAZh")
    file_list = []
    datasets_list = []
    if isinstance(split, str):
        split = split.split()
    file_path, _ = cache_file(None, cache_dir=cache_dir, url=URL,
                              download_file_name="docvqa_zh.tar.gz", proxies=proxies)
    untar(file_path, cache_dir)
    for s in split:
        file_list.append(cache_dir + '/docvqa_zh/' + s + '.json')
    for _, file in enumerate(file_list):
        dataset = GeneratorDataset(source=DocVqaZh(file),
                                   column_names=[
                                       "id" ,"text", "question", "answers", "answer_start"],
                                   shuffle=False)
        datasets_list.append(dataset)
    if len(file_list) == 1:
        return datasets_list[0]
    return datasets_list
