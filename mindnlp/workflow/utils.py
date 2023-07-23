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
Workflow Utils
"""

import re
import numpy as np


def cut_chinese_sent(para):
    """
    Cut the Chinese sentences more precisely.
    """
    para = re.sub(r"([。！？\?])([^”’])", r"\1\n\2", para)
    para = re.sub(r"(\.{6})([^”’])", r"\1\n\2", para)
    para = re.sub(r"(\…{2})([^”’])", r"\1\n\2", para)
    para = re.sub(r"([。！？\?][”’])([^，。！？\?])", r"\1\n\2", para)
    para = para.rstrip()
    return para.split("\n")


def get_bool_ids_greater_than(probs, limit=0.5, return_prob=False):
    """
    Get idx of the last dimension in probability arrays, which is greater than a limitation.

    Args:
        probs (List[List[float]]): The input probability arrays.
        limit (float): The limitation for probability.
        return_prob (bool): Whether to return the probability
    Returns:
        List[List[int]]: The index of the last dimension meet the conditions.
    """
    probs = np.array(probs)
    dim_len = len(probs.shape)
    if dim_len > 1:
        result = []
        for prob in probs:
            result.append(get_bool_ids_greater_than(prob, limit, return_prob))
        return result

    result = []
    for i, prob in enumerate(probs):
        if prob > limit:
            if return_prob:
                result.append((i, prob))
            else:
                result.append(i)
    return result


class SchemaTree:
    """
    Implementataion of SchemaTree
    """

    def __init__(self, name="root", children=None):
        self.name = name
        self.children = []
        self.prefix = None
        self.parent_relations = None
        self.parent = None
        if children is not None:
            for child in children:
                self.add_child(child)

    def __repr__(self):
        return self.name

    def add_child(self, node):
        """
        Add a child node to the current node.
        """
        assert isinstance(
            node, SchemaTree
        ), "The children of a node should be an instacne of SchemaTree."
        self.children.append(node)


def get_span(start_ids, end_ids, with_prob=False):
    """
    Get span set from position start and end list.

    Args:
        start_ids (List[int]/List[tuple]): The start index list.
        end_ids (List[int]/List[tuple]): The end index list.
        with_prob (bool): If True, each element for start_ids and end_ids is a tuple aslike: (index, probability).

    Returns:
        set: The span set without overlapping, every id can only be used once .
    """
    if with_prob:
        start_ids = sorted(start_ids, key=lambda x: x[0])
        end_ids = sorted(end_ids, key=lambda x: x[0])
    else:
        start_ids = sorted(start_ids)
        end_ids = sorted(end_ids)

    start_pointer = 0
    end_pointer = 0
    len_start = len(start_ids)
    len_end = len(end_ids)
    couple_dict = {}
    while start_pointer < len_start and end_pointer < len_end:
        if with_prob:
            start_id = start_ids[start_pointer][0]
            end_id = end_ids[end_pointer][0]
        else:
            start_id = start_ids[start_pointer]
            end_id = end_ids[end_pointer]

        if start_id == end_id:
            couple_dict[end_ids[end_pointer]] = start_ids[start_pointer]
            start_pointer += 1
            end_pointer += 1
            continue
        if start_id < end_id:
            couple_dict[end_ids[end_pointer]] = start_ids[start_pointer]
            start_pointer += 1
            continue
        if start_id > end_id:
            end_pointer += 1
            continue
    # result = [(couple_dict[end], end) for end in couple_dict]
    result = [(value, key) for key, value in couple_dict.items()]
    result = set(result)
    return result


def get_id_and_prob(span_set, offset_mapping):
    """
    Return text id and probability of predicted spans.

    Args:
        span_set (set): set of predicted spans.
        offset_mapping (list[int]): list of pair preserving the
                index of start and end char in original text pair (prompt + text) for each token.
    Returns:
        sentence_id (list[tuple]): index of start and end char in original text.
        prob (list[float]): probabilities of predicted spans.
    """
    prompt_end_token_id = offset_mapping[1:].index([0, 0])
    bias = offset_mapping[prompt_end_token_id][1] + 1
    for idx in range(1, prompt_end_token_id + 1):
        offset_mapping[idx][0] -= bias
        offset_mapping[idx][1] -= bias

    sentence_id = []
    prob = []
    for start, end in span_set:
        prob.append(start[1] * end[1])
        start_id = offset_mapping[start[0]][0]
        end_id = offset_mapping[end[0]][1]
        sentence_id.append((start_id, end_id))
    return sentence_id, prob


def dbc2sbc(string):
    """
    Convert the double-byte characters to single-byte characters.
    """
    result = ""
    for char in string:
        code = ord(char)
        if code == 0x3000:
            code = 0x0020
        else:
            code -= 0xFEE0
        if not 0x0021 <= code <= 0x7E:
            result += char
            continue
        result += chr(code)
    return result
