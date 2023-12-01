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
# pylint:disable=invalid-name,too-many-nested-blocks
"""
Information Extraction Work
"""

import os
import re
import json
import logging
from typing import List

import mindspore.dataset as ds
from mindnlp.workflow.work import Work
from mindnlp.transformers import UIE, AutoTokenizer
from mindnlp.workflow.utils import (
    SchemaTree,
    dbc2sbc,
    get_bool_ids_greater_than,
    get_id_and_prob,
    get_span,
)

usage = r"""
            from paddlenlp import Taskflow

            # Entity Extraction
            schema = ['时间', '选手', '赛事名称'] # Define the schema for entity extraction
            ie = Taskflow('information_extraction', schema=schema)
            ie("2月8日上午北京冬奥会自由式滑雪女子大跳台决赛中中国选手谷爱凌以188.25分获得金牌！")

            # Relation Extraction
            schema = [{"歌曲名称":["歌手", "所属专辑"]}] # Define the schema for relation extraction
            ie.set_schema(schema) # Reset schema
            ie("《告别了》是孙耀威在专辑爱的故事里面的歌曲")

            # Event Extraction
            schema = [{'地震触发词': ['地震强度', '时间', '震中位置', '震源深度']}] # Define the schema for event extraction
            ie.set_schema(schema) # Reset schema
            ie('中国地震台网正式测定：5月16日06时08分在云南临沧市凤庆县(北纬24.34度，东经99.98度)发生3.5级地震，震源深度10千米。')

            # Opinion Extraction
            schema = [{'评价维度': ['观点词', '情感倾向[正向，负向]']}] # Define the schema for opinion extraction
            ie.set_schema(schema) # Reset schema
            ie("地址不错，服务一般，设施陈旧")

            # Sentence-level Sentiment Classification
            schema = ['情感倾向[正向，负向]'] # Define the schema for sentence-level sentiment classification
            ie.set_schema(schema) # Reset schema
            ie('这个产品用起来真的很流畅，我非常喜欢')

            # English Model
            schema = [{'Person': ['Company', 'Position']}]
            ie_en = Taskflow('information_extraction', schema=schema, model='uie-base-en')
            ie_en('In 1997, Steve was excited to become the CEO of Apple.')

            schema = ['Sentiment classification [negative, positive]']
            ie_en.set_schema(schema)
            ie_en('I am sorry but this is the worst film I have ever seen in my life.')

            # Multilingual Model
            schema = [{'Person': ['Company', 'Position']}]
            ie_m = Taskflow('information_extraction', schema=schema, model='uie-m-base', schema_lang="en")
            ie_m('In 1997, Steve was excited to become the CEO of Apple.')

         """


MODEL_MAP = {"UIE": UIE}


def get_dynamic_max_length(
    examples, default_max_length: int, dynamic_max_length: List[int]
) -> int:
    """
    Get max_length by examples which you can change it by examples in batch.
    """
    cur_length = len(examples[0]["input_ids"])
    max_length = default_max_length
    for max_length_option in sorted(dynamic_max_length):
        if cur_length <= max_length_option:
            max_length = max_length_option
            break
    return max_length


class UIEWork(Work):
    """
    UIE Work
    """

    resource_files_names = {
        "model_state": "mindspore.ckpt",
        "vocab": "vocab.txt",
        "config": "config.json",
        "tokenizer": "tokenizer.json",
    }
    resource_files_urls = {
        # "uie-base": {
        #     # "vocab": [MINDNLP_VOCAB_URL_BASE.format("uie", "uie-base"), None],
        #     "model_state": [MS_MODEL_URL_BASE.format("uie", "uie-base"), None],
        #     "config": [MS_CONFIG_URL_BASE.format("uie", "uie-base"), None],
        #     "tokenizer": [
        #         MS_TOKENIZER_CONFIG_URL_BASE.format("uie", "uie-base"),
        #         None,
        #     ],
        # },
        # "uie-medium": {
        #     # "vocab": [MINDNLP_VOCAB_URL_BASE.format("uie", "uie-medium"), None],
        #     "model_state": [MS_MODEL_URL_BASE.format("uie", "uie-medium"), None],
        #     "config": [MS_CONFIG_URL_BASE.format("uie", "uie-medium"), None],
        #     "tokenizer": [
        #         MS_TOKENIZER_CONFIG_URL_BASE.format("uie", "uie-medium"),
        #         None,
        #     ],
        # },
        # "uie-mini": {
        #     # "vocab": [MINDNLP_VOCAB_URL_BASE.format("uie", "uie-mini"), None],
        #     "model_state": [MS_MODEL_URL_BASE.format("uie", "uie-mini"), None],
        #     "config": [MS_CONFIG_URL_BASE.format("uie", "uie-mini"), None],
        #     "tokenizer": [
        #         MS_TOKENIZER_CONFIG_URL_BASE.format("uie", "uie-mini"),
        #         None,
        #     ],
        # },
        # "uie-micro": {
        #     # "vocab": [MINDNLP_VOCAB_URL_BASE.format("uie", "uie-micro"), None],
        #     "model_state": [MS_MODEL_URL_BASE.format("uie", "uie-micro"), None],
        #     "config": [MS_CONFIG_URL_BASE.format("uie", "uie-micro"), None],
        #     "tokenizer": [
        #         MS_TOKENIZER_CONFIG_URL_BASE.format("uie", "uie-micro"),
        #         None,
        #     ],
        # },
        # "uie-nano": {
        #     # "vocab": [MINDNLP_VOCAB_URL_BASE.format("uie", "uie-nano"), None],
        #     "model_state": [MS_MODEL_URL_BASE.format("uie", "uie-nano"), None],
        #     "config": [MS_CONFIG_URL_BASE.format("uie", "uie-nano"), None],
        #     "tokenizer": [
        #         MS_TOKENIZER_CONFIG_URL_BASE.format("uie", "uie-nano"),
        #         None,
        #     ],
        # },
        # "uie-base-en": {
        #     # "vocab": [MINDNLP_VOCAB_URL_BASE.format("uie", "uie-base-en"), None],
        #     "model_state": [MS_MODEL_URL_BASE.format("uie", "uie-base-en"), None],
        #     "config": [MS_CONFIG_URL_BASE.format("uie", "uie-base-en"), None],
        #     "tokenizer": [
        #         MS_TOKENIZER_CONFIG_URL_BASE.format("uie", "uie-base-en"),
        #         None,
        #     ],
        # },
    }

    def __init__(self, work, model, schema=None, **kwargs):
        super().__init__(work=work, model=model, **kwargs)

        self._max_seq_len = kwargs.get("max_seq_len", 512)
        self._dynamic_max_length = kwargs.get("dynamic_max_length", None)
        self._batch_size = kwargs.get("batch_size", 16)
        self._split_sentence = kwargs.get("split_sentence", False)
        self._position_prob = kwargs.get("position_prob", 0.5)
        self._num_workers = kwargs.get("num_workers", 1)
        self._schema_lang = kwargs.get("schema_lang", "ch")

        self._check_work_files()

        with open(
            os.path.join(self._work_path, "config", "config.json"), encoding="utf-8"
        ) as f:
            self._init_class = json.load(f)["architectures"].pop()
        self._is_en = model in ["uie-base-en"] or self._schema_lang == "en"

        self._summary_token_num = 3  # [CLS] prompt [SEP] text [SEP]

        self._construct_model(model)

        if not schema:
            logging.warning(
                "The schema has not been set yet, please set a schema via set_schema(). "
            )
            self._schema_tree = None
        else:
            self.set_schema(schema)

        self._usage = usage
        self._construct_tokenizer(model=model)

    def set_schema(self, schema):
        """
        Set Schema for UIE.
        """
        if isinstance(schema, (dict, str)):
            schema = [schema]
        self._schema_tree = self._build_tree(schema)

    @classmethod
    def _build_tree(cls, schema, name="root"):
        """
        Build the schema tree.
        """
        schema_tree = SchemaTree(name)
        for s in schema:
            if isinstance(s, str):
                schema_tree.add_child(SchemaTree(s))
            elif isinstance(s, dict):
                for k, v in s.items():
                    if isinstance(v, str):
                        child = [v]
                    elif isinstance(v, list):
                        child = v
                    else:
                        raise TypeError(
                            f"Invalid schema, value for each key:value pairs should be list or string but {type(v)} received"
                        )
                    schema_tree.add_child(cls._build_tree(child, name=k))
            else:
                raise TypeError(
                    f"Invalid schema, element should be string or dict, but {type(s)} received"
                )
        return schema_tree

    def _construct_model(self, model):
        """
        Construct the model.
        """
        model_instance = MODEL_MAP[self._init_class].from_pretrained(model)
        self._model = model_instance
        self._model.set_train(False)

    def _construct_tokenizer(self, model):
        """
        Construct the tokenizer.
        """
        self._tokenizer = AutoTokenizer.from_pretrained(model)

    def _preprocess(self, inputs):
        """
        Transform the raw text to the model inputs.
        """
        inputs = self._check_input_text(inputs)
        outputs = {}
        outputs["text"] = inputs
        return outputs

    def _check_input_text(self, inputs):
        """
        Check whether the input meet the requirement.
        """

        inputs = inputs[0]
        if isinstance(inputs, (dict, str)):
            inputs = [inputs]
        if isinstance(inputs, list):
            input_list = []
            for example in inputs:
                data = {}
                if isinstance(example, dict):
                    if "text" in example.keys():
                        if not isinstance(example["text"], str):
                            raise TypeError(
                                f"Invalid inputs, the input text should be string. but type of {type(example['text'])} found!"
                            )
                        data["text"] = example["text"]
                    else:
                        raise ValueError(
                            "Invalid inputs, the input should contain a doc or a text."
                        )
                    input_list.append(data)
                elif isinstance(example, str):
                    input_list.append(example)
                else:
                    raise TypeError(
                        f"Invalid inputs, the input should be dict or list of dict, but type of {type(example)} found!"
                    )
        else:
            raise TypeError("Invalid input format!")
        return input_list

    def _single_stage_predict(self, inputs):
        """
        Single stage predict.
        """
        input_texts = [d["text"] for d in inputs]
        prompts = [d["prompt"] for d in inputs]

        # max predict length should exclude the length of prompt and summary tokens
        max_predict_len = (
            self._max_seq_len - len(max(prompts)) - self._summary_token_num
        )

        short_input_texts, input_mapping = self._auto_splitter(
            input_texts, max_predict_len, split_sentence=self._split_sentence
        )

        short_texts_prompts = []
        for k, v in input_mapping.items():
            short_texts_prompts.extend([prompts[k] for _ in range(len(v))])

        short_inputs = [
            {"text": short_input_texts[i], "prompt": short_texts_prompts[i]}
            for i in range(len(short_input_texts))
        ]

        def text_reader(inputs):
            for example in inputs:
                if self._dynamic_max_length is not None:
                    temp_encoded_inputs = self._tokenizer(
                        text=[example["prompt"]],
                        text_pair=[example["text"]],
                        truncation=True,
                        max_seq_len=self._max_seq_len,
                        return_attention_mask=True,
                        return_position_ids=True,
                        return_dict=False,
                        return_offsets_mapping=True,
                    )
                    max_length = get_dynamic_max_length(
                        examples=temp_encoded_inputs,
                        default_max_length=self._max_seq_len,
                        dynamic_max_length=self._dynamic_max_length,
                    )
                    encoded_inputs = self._tokenizer(
                        text=[example["prompt"]],
                        text_pair=[example["text"]],
                        truncation=True,
                        max_seq_len=max_length,
                        pad_to_max_seq_len=True,
                        return_attention_mask=True,
                        return_position_ids=True,
                        return_offsets_mapping=True,
                    )
                    logging.info("Inference with dynamic max length in %s", max_length)
                else:
                    encoded_inputs = self._tokenizer(
                        text_input=example["prompt"],
                        pair=example["text"],
                        truncation=True,
                        max_length=self._max_seq_len,
                        padding=True,
                        return_attention_mask=True,
                        return_position_ids=True,
                        return_offsets_mapping=True,
                        return_token_type_ids=True,
                    )
                tokenized_output = [
                    encoded_inputs["input_ids"],
                    encoded_inputs["token_type_ids"],
                    encoded_inputs["position_ids"],
                    encoded_inputs["attention_mask"],
                    encoded_inputs["offset_mapping"],
                ]
                yield tuple(tokenized_output)

        reader = text_reader
        infer_ds = ds.GeneratorDataset(
            source=reader(short_inputs),
            column_names=[
                "input_ids",
                "token_type_ids",
                "position_ids",
                "attention_mask",
                "offset_mapping",
            ],
        )

        infer_ds = infer_ds.batch(
            self._batch_size,
            drop_remainder=False,
            num_parallel_workers=self._num_workers,
        )

        sentence_ids = []
        probs = []
        for batch in infer_ds:
            input_ids, token_type_ids, pos_ids, att_mask, offset_maps = batch
            input_dict = {
                "input_ids": input_ids,
                "token_type_ids": token_type_ids,
                "position_ids": pos_ids,
                "attention_mask": att_mask,
            }

            start_prob, end_prob = self._model(**input_dict)
            start_prob = start_prob.asnumpy().tolist()
            end_prob = end_prob.asnumpy().tolist()

            start_ids_list = get_bool_ids_greater_than(
                start_prob, limit=self._position_prob, return_prob=True
            )
            end_ids_list = get_bool_ids_greater_than(
                end_prob, limit=self._position_prob, return_prob=True
            )
            for start_ids, end_ids, offset_map in zip(
                start_ids_list, end_ids_list, offset_maps.asnumpy().tolist()
            ):
                span_set = get_span(start_ids, end_ids, with_prob=True)
                sentence_id, prob = get_id_and_prob(span_set, offset_map)
                sentence_ids.append(sentence_id)
                probs.append(prob)
        results = self._convert_ids_to_results(short_inputs, sentence_ids, probs)
        results = self._auto_joiner(results, short_input_texts, input_mapping)
        return results

    def _auto_joiner(self, short_results, short_inputs, input_mapping):
        """
        Join the short results automatically and generate
        the final results to match with the user inputs.
        """
        concat_results = []
        is_cls_task = False
        for short_result in short_results:
            if short_result == []:
                continue
            if (
                "start" not in short_result[0].keys()
                and "end" not in short_result[0].keys()
            ):
                is_cls_task = True
                break
            break
        for _, vs in input_mapping.items():
            if is_cls_task:
                cls_options = {}
                single_results = []
                for v in vs:
                    if len(short_results[v]) == 0:
                        continue
                    for text in short_results[v][0]:
                        if text not in cls_options:
                            cls_options[text] = [1, short_results[v][0]["probability"]]
                        else:
                            cls_options[text][0] += 1
                            cls_options[text][1] += short_results[v][0]["probability"]
                if len(cls_options) != 0:
                    cls_res, cls_info = max(cls_options.items(), key=lambda x: x[1])
                    concat_results.append(
                        [{"text": cls_res, "probability": cls_info[1] / cls_info[0]}]
                    )
                else:
                    concat_results.append([])
            else:
                offset = 0
                single_results = []
                for v in vs:
                    if v == 0:
                        single_results = short_results[v]
                        offset += len(short_inputs[v])
                    else:
                        for i in range(len(short_results[v])):
                            if (
                                "start" not in short_results[v][i]
                                or "end" not in short_results[v][i]
                            ):
                                continue
                            short_results[v][i]["start"] += offset
                            short_results[v][i]["end"] += offset
                        offset += len(short_inputs[v])
                        single_results.extend(short_results[v])
                concat_results.append(single_results)
        return concat_results

    def _run_model(self, inputs):
        raw_inputs = inputs["text"]
        _inputs = self._parse_inputs(raw_inputs)
        results = self._multi_stage_predict(_inputs)
        inputs["result"] = results
        return inputs

    def _parse_inputs(self, inputs):
        _inputs = []
        for d in inputs:
            if isinstance(d, dict):
                _inputs.append({"text": d["text"]})
            else:
                _inputs.append({"text": d})
        return _inputs

    def _multi_stage_predict(self, data):
        """
        Traversal the schema tree and do multi-stage prediction.

        Args:
            data (list): a list of strings

        Returns:
            list: a list of predictions, where the list's length
                equals to the length of `data`
        """
        results = [{} for _ in range(len(data))]
        # Input check to early return
        if len(data) < 1 or self._schema_tree is None:
            return results

        # Copy to stay `self._schema_tree` unchanged
        schema_list = self._schema_tree.children[:]
        while len(schema_list) > 0:
            node = schema_list.pop(0)
            examples = []
            input_map = {}
            cnt = 0
            idx = 0
            if not node.prefix:
                for one_data in data:
                    examples.append(
                        {
                            "text": one_data["text"],
                            "prompt": dbc2sbc(node.name),
                        }
                    )
                    input_map[cnt] = [idx]
                    idx += 1
                    cnt += 1
            else:
                for pre, one_data in zip(node.prefix, data):
                    if len(pre) == 0:
                        input_map[cnt] = []
                    else:
                        for p in pre:
                            if self._is_en:
                                if re.search(r"\[.*?\]$", node.name):
                                    prompt_prefix = node.name[
                                        : node.name.find("[", 1)
                                    ].strip()
                                    cls_options = re.search(
                                        r"\[.*?\]$", node.name
                                    ).group()
                                    # Sentiment classification of xxx [positive, negative]
                                    prompt = prompt_prefix + p + " " + cls_options
                                else:
                                    prompt = node.name + p
                            else:
                                prompt = p + node.name
                            examples.append(
                                {
                                    "text": one_data["text"],
                                    "prompt": dbc2sbc(prompt),
                                }
                            )
                        input_map[cnt] = [i + idx for i in range(len(pre))]
                        idx += len(pre)
                    cnt += 1
            if len(examples) == 0:
                result_list = []
            else:
                result_list = self._single_stage_predict(examples)

            if not node.parent_relations:
                relations = [[] for i in range(len(data))]
                for k, v in input_map.items():
                    for idx in v:
                        if len(result_list[idx]) == 0:
                            continue
                        if node.name not in results[k].keys():
                            results[k][node.name] = result_list[idx]
                        else:
                            results[k][node.name].extend(result_list[idx])
                    if node.name in results[k].keys():
                        relations[k].extend(results[k][node.name])
            else:
                relations = node.parent_relations
                for k, v in input_map.items():
                    for i, value in enumerate(v):
                        if len(result_list[value]) == 0:
                            continue
                        if "relations" not in relations[k][i].keys():
                            relations[k][i]["relations"] = {
                                node.name: result_list[value]
                            }
                        elif node.name not in relations[k][i]["relations"].keys():
                            relations[k][i]["relations"][node.name] = result_list[value]
                        else:
                            relations[k][i]["relations"][node.name].extend(
                                result_list[value]
                            )
                new_relations = [[] for i in range(len(data))]
                for i, relation_i in enumerate(relations):
                    for _, relation_j in enumerate(relation_i):
                        if (
                            "relations" in relation_j
                            and node.name in relation_j["relations"]
                        ):
                            for relation_k in relation_j["relations"][node.name]:
                                new_relations[i].append(relation_k)
                relations = new_relations

            prefix = [[] for _ in range(len(data))]
            for k, v in input_map.items():
                for idx in v:
                    for i in range(len(result_list[idx])):
                        if self._is_en:
                            prefix[k].append(" of " + result_list[idx][i]["text"])
                        else:
                            prefix[k].append(result_list[idx][i]["text"] + "的")

            for child in node.children:
                child.prefix = prefix
                child.parent_relations = relations
                schema_list.append(child)

        return results

    def _convert_ids_to_results(self, examples, sentence_ids, probs):
        """
        Convert ids to raw text in a single stage.
        """
        results = []
        for example, sentence_id, prob in zip(examples, sentence_ids, probs):
            if len(sentence_id) == 0:
                results.append([])
                continue
            result_list = []
            text = example["text"]
            prompt = example["prompt"]
            for i, (start, end) in enumerate(sentence_id):
                if start < 0 <= end:
                    continue
                if end < 0:
                    start += len(prompt) + 1
                    end += len(prompt) + 1
                    result = {"text": prompt[start:end], "probability": prob[i]}
                    result_list.append(result)
                else:
                    result = {
                        "text": text[start:end],
                        "start": start,
                        "end": end,
                        "probability": prob[i],
                    }
                    result_list.append(result)
            results.append(result_list)
        return results

    def _postprocess(self, inputs):
        """
        This function will convert the model output to raw text.
        """
        return inputs["result"]
