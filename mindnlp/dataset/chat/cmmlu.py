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
Hugging Face CMMLU datasets https://huggingface.co/datasets/haonan-li/cmmlu
CMMLU is designed to evaluate the advanced knowledge and reasoning abilities of LLMs
    within the Chinese language and cultural context.
"""
# pylint: disable=C0103
# pylint: disable=W0613
import os
from typing import Union, Tuple
from datasets import load_dataset as hf_load
from mindspore.dataset import GeneratorDataset, CSVDataset
from mindnlp.dataset.register import load_dataset, process
from mindnlp.configs import DEFAULT_ROOT


CMMLU_DEFAULT_SPLITS = ("test", "dev")
CMMLU_VALID_TASKS = [
    'agronomy', 'anatomy', 'ancient_chinese', 'arts', 'astronomy', 'business_ethics', 
    'chinese_civil_service_exam', 'chinese_driving_rule', 'chinese_food_culture', 
    'chinese_foreign_policy', 'chinese_history', 'chinese_literature', 
    'chinese_teacher_qualification', 'clinical_knowledge', 'college_actuarial_science', 
    'college_education', 'college_engineering_hydrology', 'college_law', 'college_mathematics', 
    'college_medical_statistics', 'college_medicine', 'computer_science', 
    'computer_security', 'conceptual_physics', 'construction_project_management', 
    'economics', 'education', 'electrical_engineering', 'elementary_chinese', 
    'elementary_commonsense', 'elementary_information_and_technology', 'elementary_mathematics', 
    'ethnology', 'food_science', 'genetics', 'global_facts', 'high_school_biology', 
    'high_school_chemistry', 'high_school_geography', 'high_school_mathematics', 
    'high_school_physics', 'high_school_politics', 'human_sexuality', 'international_law', 
    'journalism', 'jurisprudence', 'legal_and_moral_basis', 'logical', 'machine_learning', 
    'management', 'marketing', 'marxist_theory', 'modern_chinese', 'nutrition', 'philosophy', 
    'professional_accounting', 'professional_law', 'professional_medicine', 'professional_psychology', 
    'public_relations', 'security_study', 'sociology', 'sports_science', 'traditional_chinese_medicine', 
    'virology', 'world_history', 'world_religions'
]


@load_dataset.register
def CMMLU(
    name: str,
    root: str = DEFAULT_ROOT,
    split: Union[Tuple[str], str] = CMMLU_DEFAULT_SPLITS,
    shuffle: bool = True,
):
    """
    Load the CMMLU dataset from local directory. 
    Args:
        name (str): Subtask name of CMMLU dataset.
        root (str): Directory where the datasets are saved.
        split (str|Tuple[str]): Split or splits to be returned.
            Default:('test','dev').
        proxies (dict): a dict to identify proxies,for example: {"https": "https://127.0.0.1:7890"}.

    # Returns:
    #     - **datasets_list** (list) -A list of loaded datasets.
    #       If only one type of dataset is specified,such as 'trian',
    #       this dataset is returned instead of a list of datasets.

    # Raises:
    #     TypeError: If `root` is not a string.
    #     TypeError: If `split` is not a string or Tuple[str].

    # Examples:
    #     >>> root = "~/.mindnlp"
    #     >>> split = ('train', 'dev')
    #     >>> dataset_train, dataset_dev = SQuAD1(root, split)
    #     >>> train_iter = dataset_train.create_tuple_iterator()
    #     >>> print(next(train_iter))
    #     {'context': Tensor(shape=[], dtype=String, value= 'Architecturally, \
    #         the school has a Catholic character. Atop the Main Building\'s gold dome ...'),
    #     'question': Tensor(shape=[], dtype=String, value= 'To whom did the Virgin Mary allegedly \
    #         appear in 1858 in Lourdes France?'),
    #     'answers': Tensor(shape=[1], dtype=String, value= ['Saint Bernadette Soubirous']),
    #     'answers_start': Tensor(shape=[1], dtype=Int32, value= [515])}

    """
    mode_list = []
    if isinstance(split, str):
        mode_list.append(split)
    else:
        for s in split:
            mode_list.append(s)

    paths = []
    for task in mode_list:
        path = os.path.join(root, task, f"{name}.csv")
        if not os.path.exists(path):
            raise ValueError(f"The {name} dataset not exists.")
        paths.append(path)

    # build CSV dataset
    if len(paths) == 1:
        return CSVDataset(paths[0], field_delim=',', shuffle=shuffle)

    return CSVDataset(paths[0], field_delim=',', shuffle=shuffle), CSVDataset(paths[1], field_delim=',', shuffle=shuffle)

class HFcmmlu:
    """
    Hugging Face GLUE dataset source
    """

    def __init__(self, dataset) -> None:
        self.dataset = dataset

        self._Question, self._Answer = [], []
        self._A, self._B, self._C, self._D = [], [], [], []

        self._load()

    def _load(self):
        for every_dict in self.dataset:
            self._Question.append(every_dict['Question'])
            self._Answer.append(every_dict['Answer'])
            self._A.append(every_dict['A'])
            self._B.append(every_dict['B'])
            self._C.append(every_dict['C'])
            self._D.append(every_dict['D'])

    def __getitem__(self, index):
        return self._Question[index], self._A[index], self._B[index], self._C[index], self._D[index], self._Answer[index]

    def __len__(self):
        return len(self._Question)


@load_dataset.register
def HF_CMMLU(
        name: str,
        root: str = DEFAULT_ROOT,
        split: Union[Tuple[str], str] = CMMLU_DEFAULT_SPLITS,
        shuffle=True,
):
    r"""
    Load the huggingface GLUE dataset.

    Args:
        name (str): Task name
        root (str): Directory where the datasets are saved.
            Default:~/.mindnlp
        split (str|Tuple[str]): Split or splits to be returned.
            Default:('test', 'dev').
        shuffle (bool): Whether to shuffle the dataset.
            Default:True.

    Returns:
        - **datasets_list** (list) -A list of loaded datasets.
          If only one type of dataset is specified,such as 'trian',
          this dataset is returned instead of a list of datasets.

    Examples:
        >>> root = "~/.mindnlp"
        >>> split = ('test', 'dev')
        >>> dataset_train,dataset_test = HF_CMMLU(root, split)
        >>> train_iter = dataset_train.create_tuple_iterator()
        >>> print(next(train_iter))

    """
    cache_dir = os.path.join(root, "datasets", "hf_datasets", "CMMLU")

    if name not in CMMLU_VALID_TASKS:
        raise ValueError(f"subtask {name} not exists in cmmlu dataset.")

    column_names = ["Question", "A", "B", "C", "D", "Answer"]

    ds = hf_load(r"haonan-li/cmmlu", name, cache_dir=cache_dir)

    mode_list = []
    if isinstance(split, str):
        mode_list.append(split)
    else:
        for s in split:
            mode_list.append(s)

    dataset_list = []
    for m in mode_list:
        dataset_list.append(
            GeneratorDataset(source = HFcmmlu(ds[m]), column_names = column_names, shuffle = shuffle)
        )

    if len(dataset_list) == 1:
        return dataset_list[0]
    return dataset_list



@process.register
def CMMLU_Process(dataset, tokenizer, vocab, batch_size ):
    """
    TODO: finish process cmmlu.
    """
    pass # pylint: disable=W0107
#     dataset: Union[CSVDataset, GeneratorDataset],
#     tokenizer: PretrainedTokenizer,
#     max_seq_len: int = 512,
#     num_samples: int = None,
#     batch_size: int = 1,
#     shuffle: bool = False,
#     num_workers: int = 0,
#     task_name: str = None,
#     return_all_fields: bool = False,
#     proxies: dict = None,
#     **kwargs,
# ) -> Tuple[Dataset, dict]:
#     r"""
#     The process function of CMMLU dataset.

#     Args:
#         dataset (Dataset): Dataset to be processed.
#         tokenizer (PretrainedTokenizer): Tokenizer for encoding the original text.
#         max_seq_len (int): The maximum length of sequence. Default: 512.
#         num_samples (int): The number of samples to be processed in the dataset. Default: None.
#         batch_size (int): The number of samples in a batch. Default: 1.
#         shuffle (bool): Whether to shuffle the dataset. Default: False.
#         num_workers (int): The number of processes to load dataset. Default: 0.
#         task_name (str): The task name of the dataset. Default: None.
#         return_all_fields (bool): Whether to return all fields in the dataset. Default: False.
#         proxies (dict): a dict to identify proxies,for example: {"https": "https://
#     """
