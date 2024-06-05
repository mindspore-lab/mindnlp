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

import unittest

from mindnlp.transformers.pipelines.table_question_answering import TableQuestionAnsweringArgumentHandler

from mindnlp.utils.testing_utils import is_pipeline_test, require_mindspore


@is_pipeline_test
class TQAPipelineTests(unittest.TestCase):

    def test_tmp(self):
        example = {
            "table": {
                "actors": ["brad pitt", "leonardo di caprio", "george clooney"],
                "age": ["56", "45", "59"],
                "number of movies": ["87", "53", "69"],
                "date of birth": ["7 february 1967", "10 june 1996", "28 november 1967"],
            },
            "query": "how many movies has george clooney played in?",
        }

        handler = TableQuestionAnsweringArgumentHandler()
        tqa_inputs = handler(example["table"], example["query"])
        print("############")
        print(tqa_inputs)
