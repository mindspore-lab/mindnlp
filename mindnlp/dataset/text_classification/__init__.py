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
TextClassification dataset init
"""

from .agnews import AG_NEWS, AG_NEWS_Process
from .cola import CoLA, CoLA_Process
from .sst2 import SST2, SST2_Process
from .amazonreviewfull import AmazonReviewFull, AmazonReviewFull_Process
from .amazonreviewpolarity import AmazonReviewPolarity, AmazonReviewPolarity_Process
from .stsb import STSB, STSB_Process
from .dbpedia import DBpedia, DBpedia_Process
from .imdb import IMDB
from .mnli import MNLI, MNLI_Process
from .mrpc import MRPC, MRPC_Process
from .qnli import QNLI, QNLI_Process
from .qqp import QQP, QQP_Process
from .wnli import WNLI, WNLI_Process
from .rte import RTE, RTE_Process
from .sogounews import SogouNews
from .yelpreviewpolarity import YelpReviewPolarity, YelpReviewPolarity_Process
from .yelpreviewfull import YelpReviewFull, YelpReviewFull_Process
from .yahooanswers import YahooAnswers, YahooAnswers_Process
