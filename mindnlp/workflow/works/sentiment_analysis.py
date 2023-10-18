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
# pylint:disable=invalid-name
"""
Sentiment Analysis Work
"""

# pylint:disable=invalid-name,line-too-long

import os

import mindspore
from mindspore import Tensor
from mindspore.dataset import text
from mindspore.ops import functional as F
from mindnlp._legacy.functional import argmax
from mindnlp.workflow.work import Work
from mindnlp.workflow.downstream import BertForSentimentAnalysis
from mindnlp.transformers import BertConfig
from mindnlp.dataset.transforms import PadTransform
from mindnlp.transformers import BertTokenizer

usage = r"""
    from mindnlp import Workflow
    
    senta = Workflow("sentiment_analysis")
    senta("个产品用起来真的很流畅，我非常喜欢")
    ...
    [{'text': '这个产品用起来真的很流畅，我非常喜欢', 'label': 'positive', 'score': 0.9953349232673645}]
    ...
"""


class SentimentAnalysisWork(Work):
    """
    Sentiment Analysis Work.
    """

    resource_files_names = {
        "model_state": "mbert_for_senta_model_state.ckpt",
        "vocab": "bert_for_senta_vocab.txt",
    }
    resource_files_urls = {
        "bert": {
            "vocab": [
                "https://download.mindspore.cn/toolkits/mindnlp/workflow/sentiment_analysis/bert_for_senta_vocab.txt",
                "3b5b76c4aef48ecf8cb3abaafe960f09",
            ],
            "model_state": [
                "https://download.mindspore.cn/toolkits/mindnlp/workflow/sentiment_analysis/bert_for_senta_model_state.ckpt",
                "7dba7b0371d2fcbb053e28c8bdfb1050",
            ],
        }
    }

    def __init__(self, work, model, **kwargs):
        super().__init__(model, work, **kwargs)
        self._label_map = {0: "negative", 1: "neutral", 2: "positive"}
        self._check_work_files()
        self._construct_tokenizer(model)
        self._construct_model(model)
        self._usage = usage

    def _construct_model(self, model):
        """
        Construct the model.
        """
        vocab_size = self.kwargs["vocab_size"]
        num_classes = 3

        config = BertConfig(vocab_size=vocab_size, num_labels=num_classes)
        model_instance = BertForSentimentAnalysis(config)

        model_path = os.path.join(
            self._work_path, "model_state", "bert_for_senta_model_state.ckpt"
        )
        state_dict = mindspore.load_checkpoint(model_path)
        mindspore.load_param_into_net(model_instance, state_dict)

        self._model = model_instance
        self._model.set_train(False)

    def _construct_tokenizer(self, model):
        """
        Construct the tokenizer.
        """
        vocab_path = os.path.join(self._work_path, "vocab", "bert_for_senta_vocab.txt")
        vocab = text.Vocab.from_file(vocab_path)

        vocab_size = len(vocab.vocab())
        pad_token_id = vocab.tokens_to_ids("[PAD]")

        self.kwargs["pad_token_id"] = pad_token_id
        self.kwargs["vocab_size"] = vocab_size
        tokenizer = BertTokenizer(vocab)
        self._tokenizer = tokenizer

    def _preprocess(self, inputs, padding=True, add_special_tokens=True):
        """
        Preprocess the inputs.
        """
        # Get the config from the kwargs
        batch_size = self.kwargs["batch_size"] if "batch_size" in self.kwargs else 1

        examples = []
        filter_inputs = []
        for input_data in inputs:
            if not (isinstance(input_data, str) and len(input_data) > 0):
                continue
            filter_inputs.append(input_data)
            ids = self._tokenizer.execute_py(input_data)
            lens = len(ids)
            examples.append((ids, lens))

        batches = [
            examples[idx : idx + batch_size]
            for idx in range(0, len(examples), batch_size)
        ]
        outputs = {}
        outputs["text"] = filter_inputs
        outputs["data_loader"] = batches

        return outputs

    def _batchify_fn(self, samples):
        seq_list = [sample[1] for sample in samples]
        max_length = max(seq_list)
        outputs = []
        pader = PadTransform(
            max_length=max_length, pad_value=self.kwargs["pad_token_id"]
        )
        for sample in samples:
            outputs.append(pader(sample[0]))
        return Tensor(outputs)

    def _run_model(self, inputs):
        """
        Run the model.
        """
        results = []
        scores = []
        for batch in inputs["data_loader"]:
            ids = self._batchify_fn(batch)
            outputs = self._model(ids)
            probs = F.softmax(outputs, axis=-1)
            idx = argmax(probs, dim=-1).asnumpy().tolist()
            if isinstance(idx, int):
                idx = [idx]
            score = [max(prob.asnumpy().tolist()) for prob in probs]
            labels = [self._label_map[i] for i in idx]
            results.extend(labels)
            scores.extend(score)

        inputs["result"] = results
        inputs["score"] = scores
        return inputs

    def _postprocess(self, inputs):
        """
        Postprocess the outputs.
        """
        final_results = []
        for _text, label, score in zip(
            inputs["text"], inputs["result"], inputs["score"]
        ):
            result = {}
            result["text"] = _text
            result["label"] = label
            result["score"] = score
            final_results.append(result)
        return final_results
