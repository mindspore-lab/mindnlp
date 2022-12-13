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
IWSLT2016 load function
"""
# pylint: disable=C0103

import os
from typing import Union, Tuple
from mindspore.dataset import IWSLT2016Dataset
from mindnlp.utils.download import cache_file
from mindnlp.dataset.register import load
from mindnlp.configs import DEFAULT_ROOT
from mindnlp.utils import untar

URL = "https://drive.google.com/uc?id=1l5y6Giag9aRPwGtuZHswh3w5v3qEz8D8&confirm=t"

MD5 = "c393ed3fc2a1b0f004b3331043f615ae"


SUPPORTED_DATASETS = {
    "valid_test": ["dev2010", "tst2010", "tst2011", "tst2012", "tst2013", "tst2014"],
    "language_pair": {
        "en": ["ar", "de", "fr", "cs"],
        "ar": ["en"],
        "fr": ["en"],
        "de": ["en"],
        "cs": ["en"],
    },
    "year": 16,
}

SET_NOT_EXISTS = {
    ("en", "ar"): [],
    ("en", "de"): [],
    ("en", "fr"): [],
    ("en", "cs"): ["tst2014"],
    ("ar", "en"): [],
    ("fr", "en"): [],
    ("de", "en"): [],
    ("cs", "en"): ["tst2014"],
}


@load.register
def IWSLT2016(root: str = DEFAULT_ROOT,
              split: Union[Tuple[str], str] = ("train", "valid", "test"),
              language_pair=("de", "en"),
              valid_set="tst2013",
              test_set="tst2014",
              proxies=None):
    r"""
    Load the IWSLT2016 dataset

    The available datasets include following:

    **Language pairs**:

    +-----+-----+-----+-----+-----+-----+
    |     |"en" |"fr" |"de" |"cs" |"ar" |
    +-----+-----+-----+-----+-----+-----+
    |"en" |     |   x |  x  |  x  |  x  |
    +-----+-----+-----+-----+-----+-----+
    |"fr" |  x  |     |     |     |     |
    +-----+-----+-----+-----+-----+-----+
    |"de" |  x  |     |     |     |     |
    +-----+-----+-----+-----+-----+-----+
    |"cs" |  x  |     |     |     |     |
    +-----+-----+-----+-----+-----+-----+
    |"ar" |  x  |     |     |     |     |
    +-----+-----+-----+-----+-----+-----+

    **valid/test sets**: ["dev2010", "tst2010", "tst2011", "tst2012", "tst2013", "tst2014"]

    Args:
        root (str): Directory where the datasets are saved. Default: "~/.mindnlp"
        split (str|Tuple[str]): Split or splits to be returned.
            Default:('train', 'valid', 'test').
        language_pair (Tuple[str]): Tuple containing src and tgt language. Default: ('de', 'en').
        valid_set (str): a string to identify validation set. Default: "tst2013".
        test_set (str): a string to identify test set. Default: "tst2014".
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
        RuntimeError: If `language_pair` is not in the range of supported datasets.
        RuntimeError: If `valid_set` is not in the range of supported datasets.
        RuntimeError: If `test_set` is not in the range of supported datasets.

    Examples:
        >>> root = "~/.mindnlp"
        >>> split = ('train', 'valid', 'test')
        >>> language_pair = ('de', 'en')
        >>> valid_set="tst2013"
        >>> test_set="tst2014"
        >>> dataset_train, dataset_valid, dataset_test = IWSLT2016(root, split, \
            language_pair, valid_set, test_set)
        >>> train_iter = dataset_train.create_dict_iterator()
        >>> print(next(train_iter))
        {'text': Tensor(shape=[], dtype=String, value= \
            'David Gallo: Das ist Bill Lange. Ich bin Dave Gallo.'),
        'translation': Tensor(shape=[], dtype=String, value= \
            "David Gallo: This is Bill Lange. I'm Dave Gallo.")}

    """

    if not isinstance(language_pair, list) and not isinstance(language_pair, tuple):
        raise ValueError(f"language_pair must be list or tuple\
            but got {type(language_pair)} instead")

    assert len(language_pair) == 2, \
        "language_pair must contain only 2 elements: src and tgt language respectively"
    src_language, tgt_language = language_pair[0], language_pair[1]
    if src_language not in SUPPORTED_DATASETS["language_pair"]:
        raise ValueError(
            f"src_language '{src_language}' is not valid. Supported source languages are \
                {list(SUPPORTED_DATASETS['language_pair'])}"
        )
    if tgt_language not in SUPPORTED_DATASETS["language_pair"][src_language]:
        raise ValueError(
            f"tgt_language '{tgt_language}' is not valid for give src_language '\
                {src_language}'. Supported target language are \
                    {SUPPORTED_DATASETS['language_pair'][src_language]}"
        )
    if valid_set not in SUPPORTED_DATASETS["valid_test"] or \
            valid_set in SET_NOT_EXISTS[language_pair]:
        support = [s for s in SUPPORTED_DATASETS['valid_test']
                   if s not in SET_NOT_EXISTS[language_pair]]
        raise ValueError(
            f"valid_set '{valid_set}' is not valid for given language pair \
                    {language_pair}. Supported validation sets are {support}"
        )
    if test_set not in SUPPORTED_DATASETS["valid_test"] or \
            test_set in SET_NOT_EXISTS[language_pair]:
        support = [s for s in SUPPORTED_DATASETS['valid_test']
                   if s not in SET_NOT_EXISTS[language_pair]]
        raise ValueError(
            f"test_set '{valid_set}' is not valid for give language pair \
                {language_pair}. Supported test sets are {support}"
        )
    if isinstance(split, str):
        split = split.split()

    cache_dir = os.path.join(root, "datasets", "IWSLT2016")
    file_path, _ = cache_file(None, cache_dir=cache_dir, url=URL, md5sum=MD5,
                              download_file_name="2016-01.tgz", proxies=proxies)
    dataset_dir_name = untar(file_path, os.path.dirname(file_path))[0]
    dataset_dir_path = os.path.join(cache_dir, dataset_dir_name)
    untar_flie_name = language_pair[0] + '-' + language_pair[1] + '.tgz'
    untar_file_path = os.path.join(dataset_dir_path, 'texts',
                                   language_pair[0], language_pair[1], untar_flie_name)
    untar(untar_file_path, os.path.dirname(untar_file_path))
    datasets_list = []
    for usage in split:
        if usage == 'valid':
            dataset = IWSLT2016Dataset(
                dataset_dir=dataset_dir_path, usage=usage, valid_set=valid_set, shuffle=False)
        elif usage == 'test':
            dataset = IWSLT2016Dataset(
                dataset_dir=dataset_dir_path, usage=usage, test_set=test_set, shuffle=False)
        else:
            dataset = IWSLT2016Dataset(
                dataset_dir=dataset_dir_path, usage=usage, shuffle=False)
        datasets_list.append(dataset)
    if len(datasets_list) == 1:
        return datasets_list[0]
    return datasets_list
