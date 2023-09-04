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
from mindspore.dataset import TextFileDataset, transforms
from mindspore.dataset import text
from mindnlp.utils.download import cache_file
from mindnlp.dataset.register import load_dataset, process
from mindnlp.configs import DEFAULT_ROOT
from mindnlp.utils import untar

URL = {
    "train": "https://openi.pcl.ac.cn/lvyufeng/multi30k/raw/branch/master/training.tar.gz",
    "valid": "https://openi.pcl.ac.cn/lvyufeng/multi30k/raw/branch/master/validation.tar.gz",
    "test": "https://openi.pcl.ac.cn/lvyufeng/multi30k/raw/branch/master/mmt16_task1_test.tar.gz",
}

MD5 = {
    "train": "8ebce33f4ebeb71dcb617f62cba077b7",
    "valid": "2a46f18dbae0df0becc56e33d4e28e5d",
    "test": "1586ce11f70cba049e9ed3d64db08843",
}


@load_dataset.register
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
        proxies (dict): a dict to identify proxies,for example: {"https": "https://127.0.0.1:7890"}.

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

    if root == DEFAULT_ROOT:
        cache_dir = os.path.join(root, "datasets", "Multi30k")
    else:
        cache_dir = root

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
def Multi30k_Process(dataset, vocab, batch_size=64, max_len=500, \
                drop_remainder=False):
    """
    the process of the Multi30k dataset

    Args:
        dataset (GeneratorDataset): Multi30k dataset.
        vocab (Vocab): vocabulary object, used to store the mapping of token and index.
        batch_size (int): The number of rows each batch is created with. Default: 64.
        max_len (int): The max length of the sentence. Default: 500.
        drop_remainder (bool): When the last batch of data contains a data entry smaller than batch_size, whether
            to discard the batch and not pass it to the next operation. Default: False.

    Returns:
        - **dataset** (MapDataset) - dataset after transforms.

    Raises:
        TypeError: If `input_column` is not a string.

    Examples:
        >>> train_dataset = Multi30k(
        >>>     root=self.root,
        >>>     split="train",
        >>>     language_pair=("de", "en")
        >>> )
        >>> tokenizer = BasicTokenizer(True)
        >>> train_dataset = train_dataset.map([tokenizer], 'en')
        >>> train_dataset = train_dataset.map([tokenizer], 'de')
        >>> en_vocab = text.Vocab.from_dataset(train_dataset, 'en', special_tokens=
        >>>   ['<pad>', '<unk>'], special_first= True)
        >>> de_vocab = text.Vocab.from_dataset(train_dataset, 'de', special_tokens=
        >>>   ['<pad>', '<unk>'], special_first= True)
        >>> vocab = {'en':en_vocab, 'de':de_vocab}
        >>> train_dataset = process('Multi30k', train_dataset, vocab = vocab)
    """

    en_pad_value = vocab['en'].tokens_to_ids('<pad>')
    de_pad_value = vocab['de'].tokens_to_ids('<pad>')

    en_lookup_op = text.Lookup(vocab['en'], unknown_token='<unk>')
    de_lookup_op = text.Lookup(vocab['de'], unknown_token='<unk>')

    dataset = dataset.map([en_lookup_op], 'en')
    dataset = dataset.map([de_lookup_op], 'de')

    en_pad_op = transforms.PadEnd([max_len], en_pad_value)
    de_pad_op = transforms.PadEnd([max_len], de_pad_value)

    dataset = dataset.map([en_pad_op], 'en')
    dataset = dataset.map([de_pad_op], 'de')

    dataset = dataset.batch(batch_size, drop_remainder=drop_remainder)
    return dataset
