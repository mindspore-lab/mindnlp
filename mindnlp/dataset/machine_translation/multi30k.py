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
Multi30k load function
"""
# pylint: disable=C0103

import os
import re
from operator import itemgetter
from typing import Union, Tuple
from mindspore.dataset import text, TextFileDataset
from mindnlp.utils.download import cache_file
from mindnlp.dataset.register import load, process
from mindnlp.configs import DEFAULT_ROOT
from mindnlp.utils import untar

URL = {
    "train": "http://www.quest.dcs.shef.ac.uk/wmt16_files_mmt/training.tar.gz",
    "valid": "http://www.quest.dcs.shef.ac.uk/wmt16_files_mmt/validation.tar.gz",
    "test": "http://www.quest.dcs.shef.ac.uk/wmt16_files_mmt/mmt16_task1_test.tar.gz",
}

MD5 = {
    "train": "8ebce33f4ebeb71dcb617f62cba077b7",
    "valid": "2a46f18dbae0df0becc56e33d4e28e5d",
    "test": "f63b12fc6f95beb3bfca2c393e861063",
}


@load.register
def Multi30k(root: str = DEFAULT_ROOT, split: Union[Tuple[str], str] = ('train', 'valid', 'test'),
             language_pair: Tuple[str] = ('de', 'en'), proxies=None):
    r"""
    Load the Multi30k dataset

    Args:
        root (str): Directory where the datasets are saved.
            Default:~/.mindnlp
        split (str|Tuple[str]): Split or splits to be returned.
            Default:('train', 'valid', 'test').
        language_pair (Tuple[str]): Tuple containing src and tgt language.
            Default: ('de', 'en').

    Returns:
        - **datasets_list** (list) -A list of loaded datasets.
            If only one type of dataset is specified,such as 'trian',
            this dataset is returned instead of a list of datasets.

    Raises:
        TypeError: If `root` is not a string.
        TypeError: If `split` is not a string or Tuple[str].
        TypeError: If `language_pair` is not a Tuple[str].
        RuntimeError: If the length of `language_pair` is not 2.
        RuntimeError: If `language_pair` is neither ('de', 'en') nor ('en', 'de').

    Examples:
        >>> root = os.path.join(os.path.expanduser('~'), ".mindnlp")
        >>> split = ('train', 'valid', 'test')
        >>> language_pair = ('de', 'en')
        >>> dataset_train, dataset_valid, dataset_test = Multi30k(root, split, language_pair)
        >>> train_iter = dataset_train.create_tuple_iterator()
        >>> print(next(train_iter))
        [Tensor(shape=[], dtype=String, value=\
            'Ein Mann mit einem orangefarbenen Hut, der etwas anstarrt.'),
        Tensor(shape=[], dtype=String, value= 'A man in an orange hat starring at something.')]

    """

    assert len(
        language_pair) == 2, "language_pair must contain only 2 elements:\
            src and tgt language respectively"
    assert tuple(sorted(language_pair)) == (
        "de",
        "en",
    ), "language_pair must be either ('de','en') or ('en', 'de')"

    cache_dir = os.path.join(root, "datasets", "Multi30k")

    file_list = []

    untar_files = []
    source_files = []
    target_files = []

    datasets_list = []

    if isinstance(split, str):
        file_path, _ = cache_file(
            None, cache_dir=cache_dir, url=URL[split], md5sum=MD5[split], proxies=proxies)
        file_list.append(file_path)

    else:
        urls = itemgetter(*split)(URL)
        md5s = itemgetter(*split)(MD5)
        for i, url in enumerate(urls):
            file_path, _ = cache_file(
                None, cache_dir=cache_dir, url=url, md5sum=md5s[i], proxies=proxies)
            file_list.append(file_path)

    for file in file_list:
        untar_files.append(untar(file, os.path.dirname(file)))

    regexp = r".de"
    if language_pair == ("en", "de"):
        regexp = r".en"

    for file_pair in untar_files:
        for file in file_pair:
            match = re.search(regexp, file)
            if match:
                source_files.append(file)
            else:
                target_files.append(file)

    for i in range(len(untar_files)):
        source_dataset = TextFileDataset(
            os.path.join(cache_dir, source_files[i]), shuffle=False)
        source_dataset = source_dataset.rename(["text"], [language_pair[0]])
        target_dataset = TextFileDataset(
            os.path.join(cache_dir, target_files[i]), shuffle=False)
        target_dataset = target_dataset.rename(["text"], [language_pair[1]])
        datasets = source_dataset.zip(target_dataset)
        datasets_list.append(datasets)

    if len(datasets_list) == 1:
        return datasets_list[0]
    return datasets_list

@process.register
def Multi30k_Process(dataset, tokenizer = text.BasicTokenizer(), language = 'en', vocab=None):
    '''
    a function transforms multi30K dataset into tensors

    Args:
        dataset (ZipDataset): Multi30K dataset
        tokenizer (TextTensorOperation): Tokenizer you what to used
        language (str): The language column name in multi30K, 'de' or 'en', defaults to 'en'
        vocab (Vocab): The vocab you use, defaults to None. If None, a new vocab will be created.

    Returns:
        - **dataset** (MapDataset) -dataset after process
        - **newVocab** (Vocab) -new vocab created from dataset

    Raises:
        AssertionError: arg `language` not in ['en', 'de']
        TypeError: If `language` is not a string.

    Examples:
        >>> from mindnlp.dataset import Multi30k_Process
        >>> test_dataset = Multi30k(
        >>>     root="./dataset",
        >>>     split="test",
        >>>     language_pair=("de", "en")
        >>> )
        >>> test_dataset, vocab = Multi30k_Process(test_dataset, text.BasicTokenizer(), "en")
        >>> for i in test_dataset.create_tuple_iterator():
        >>>     print(i)
        >>>     break
        [Tensor(shape=[], dtype=String, value= 'Ein Mann mit einem orangefarbenen Hut, \
            der etwas anstarrt.'), Tensor(shape=[10], dtype=Int32, value= [   2,    8,    3,   \
            24,   90,   82, 1783,   15,  131,    1])]
    '''

    assert language in ['en', 'de'], "language not in ['en', 'de']"
    if vocab is None :
        dataset = dataset.map(tokenizer, language)
        newVocab = text.Vocab.from_dataset(dataset, language)
        return dataset.map(text.Lookup(newVocab), language), newVocab

    dataset = dataset.map(tokenizer, language)
    return dataset.map(text.Lookup(vocab), language)
