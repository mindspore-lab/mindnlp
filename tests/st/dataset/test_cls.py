# Copyright 2021 Huawei Technologies Co., Ltd
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
'''test  classification'''
from mindtext.dataset import CoLADataset
from mindtext.dataset import SST2Dataset
from mindtext.dataset import DBpediaDataset
from mindtext.dataset.classification.ChnSentiCorp import ChnSentiCorpDataset
from mindtext.dataset.classification.sogou_news import SogouNewsDataset
from mindtext.dataset.classification.yelp import YelpFullDataset


def test_cola():
    '''Test colaDataset.'''
    cola = CoLADataset(tokenizer='spacy', lang='en')
    cl_data = cola()
    print("Success Download {}".format(cl_data.keys()))

def test_dbpedia():
    '''Test DBpediaDataset.'''
    dbpedia = DBpediaDataset(tokenizer='spacy', lang='en')
    dp_data = dbpedia()
    print("Success Download {}".format(dp_data.keys()))

def test_sogou_news():
    '''Test SogouNewsDataset.'''
    sogou = SogouNewsDataset(tokenizer='spacy',
                             lang='en',
                             buckets=[16, 32, 64])
    sg_data = sogou()
    print("Success Download {}".format(sg_data.keys()))

def test_sst2():
    '''Test SST2Dataset.'''
    sst2 = SST2Dataset(tokenizer='spacy', lang='en')
    st2_data = sst2()
    print("Success Download {}".format(st2_data.keys()))

def test_yelp():
    '''Test YelpFullDataset.'''
    yelp_full = YelpFullDataset(tokenizer='spacy', lang='en')
    yf_data = yelp_full()
    print("Success Download {}".format(yf_data.keys()))

def test_chn():
    '''Test ChnSentiCorpDataset.'''
    chn_senti_corp = ChnSentiCorpDataset(tokenizer='spacy', lang='en')
    cc_data = chn_senti_corp()
    print("Success Download {}".format(cc_data.keys()))
