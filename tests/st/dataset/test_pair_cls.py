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
'''test pair classification'''
from mindtext.dataset import AFQMCDataset
from mindtext.dataset import QNLIDataset
from mindtext.dataset import LCQMCDataset
from mindtext.dataset import MNLIDataset
from mindtext.dataset import MRPCDataset
from mindtext.dataset import QQPDataset
from mindtext.dataset import RTEDataset
from mindtext.dataset.pair_classification.amazon import AMADataset
from mindtext.dataset.pair_classification.cmnli import CMNLIDataset
from mindtext.dataset.pair_classification.wnli import WNLIDataset


def test_afqmc():
    '''Test AFQMCDataset.'''
    afqmc = AFQMCDataset(tokenizer='spacy', lang='en')
    af_data = afqmc()
    print("Success Download {}".format(af_data.keys()))

def test_amad():
    '''Test AMADataset.'''
    amad = AMADataset(tokenizer='spacy', lang='en')
    am_data = amad()
    print("Success Download {}".format(am_data.keys()))

def test_cmnli():
    '''Test CMNLIDataset.'''
    cmnli = CMNLIDataset(tokenizer='cn-char', lang='en')
    cm_data = cmnli()
    print("Success Download {}".format(cm_data.keys()))

def test_lcqmc():
    '''Test LCQMCDataset.'''
    lcqmc = LCQMCDataset(tokenizer='spacy', lang='en')
    lc_data = lcqmc()
    print("Success Download {}".format(lc_data.keys()))

def test_mnli():
    '''Test MNLIDataset.'''
    mnli = MNLIDataset(tokenizer='spacy', lang='en')
    mn_data = mnli()
    print("Success Download {}".format(mn_data.keys()))

def test_mrpc():
    '''Test MRPCDataset.'''
    mrpc = MRPCDataset(path='dataset path', tokenizer='spacy', lang='en', train_ratio=0.8)
    mr_data = mrpc()
    print("Success Download {}".format(mr_data.keys()))

def test_qnli():
    '''Test QNLIDataset.'''
    qnli = QNLIDataset(tokenizer='spacy', lang='en')
    qn_data = qnli()
    print("Success Download {}".format(qn_data.keys()))

def test_qqp():
    '''Test QQPDataset.'''
    qqp = QQPDataset(tokenizer='spacy', lang='en')
    qqp_data = qqp()
    print("Success Download {}".format(qqp_data.keys()))

def test_rte():
    '''Test RTEDataset.'''
    rte = RTEDataset(tokenizer='spacy', lang='en')
    rte_data = rte()
    print("Success Download {}".format(rte_data.keys()))

def test_wnli():
    '''Test WNLIDataset.'''
    wnli = WNLIDataset(tokenizer='spacy', lang='en')
    wn_data = wnli()
    print("Success Download {}".format(wn_data.keys()))
