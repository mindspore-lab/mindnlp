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
import os
from typing import Union, Tuple
from datasets import load_dataset as hf_load
from mindspore.dataset import GeneratorDataset
from mindnlp.dataset.register import load_dataset
from mindnlp.configs import DEFAULT_ROOT

CMMU_DEFAULT_SPLITS = ("test", "dev")


@load_dataset.register
def CMMLU(
    root: str = DEFAULT_ROOT,
    split: Union[Tuple[str], str] = CMMU_DEFAULT_SPLITS,
    proxies=None
):
    """
    Load the CMMLU dataset from local directory. 
    """
    pass


    # Args:
    #     root (str): Directory where the datasets are saved.
    #     split (str|Tuple[str]): Split or splits to be returned.
    #         Default:('train','dev').
    #     proxies (dict): a dict to identify proxies,for example: {"https": "https://127.0.0.1:7890"}.

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

    # """
    # cache_dir = os.path.join(root, "datasets", "SQuAD1")
    # file_list = []
    # datasets_list = []
    # if isinstance(split, str):
    #     split = split.split()
    # for s in split:
    #     path, _ = cache_file(
    #         None, url=URL[s], cache_dir=cache_dir, md5sum=MD5[s], proxies=proxies
    #     )
    #     file_list.append(path)

    # for _, file in enumerate(file_list):
    #     dataset = GeneratorDataset(source=Squad1(file),
    #                                column_names=[
    #                                    "id" ,"context", "question", "answers", "answer_start"],
    #                                shuffle=False)
    #     datasets_list.append(dataset)
    # if len(file_list) == 1:
    #     return datasets_list[0]
    # return datasets_list


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
        return self._Question[index], self._A[index], self._B[index], self._C[index], self._D[index], self._Answer[index],

    def __len__(self):
        assert len(self._Question) == len(self._Answer) == len(self._A) == len(self._B) == len(self._C) == len(self._D)
        return len(self._Question)


@load_dataset.register
def HF_CMMLU(
        name: str,
        root: str = DEFAULT_ROOT,
        split: Union[Tuple[str], str] = CMMU_DEFAULT_SPLITS,
        shuffle=True,
):
    r"""
    Load the huggingface GLUE dataset.

    Args:
        name (str): Task name
        root (str): Directory where the datasets are saved.
            Default:~/.mindnlp
        split (str|Tuple[str]): Split or splits to be returned.
            Default:('train', 'dev').
        shuffle (bool): Whether to shuffle the dataset.
            Default:True.

    Returns:
        - map (keys -> GeneratorDataset)
            keys choice from all_valid_tasks.


    Examples:
        >>> root = "~/.mindnlp"
        >>> split = ('train', 'test')
        >>> dataset_train,dataset_test = HF_GLUE(root, split)
        >>> train_iter = dataset_train.create_tuple_iterator()
        >>> print(next(train_iter))

    """
    cache_dir = os.path.join(root, "datasets", "hf_datasets", "CMMLU")
    all_valid_tasks = ['agronomy', 'anatomy', 'ancient_chinese', 'arts', 'astronomy', 'business_ethics', 'chinese_civil_service_exam', 'chinese_driving_rule', 'chinese_food_culture', 'chinese_foreign_policy', 'chinese_history', 'chinese_literature', 
'chinese_teacher_qualification', 'clinical_knowledge', 'college_actuarial_science', 'college_education', 'college_engineering_hydrology', 'college_law', 'college_mathematics', 'college_medical_statistics', 'college_medicine', 'computer_science',
'computer_security', 'conceptual_physics', 'construction_project_management', 'economics', 'education', 'electrical_engineering', 'elementary_chinese', 'elementary_commonsense', 'elementary_information_and_technology', 'elementary_mathematics', 
'ethnology', 'food_science', 'genetics', 'global_facts', 'high_school_biology', 'high_school_chemistry', 'high_school_geography', 'high_school_mathematics', 'high_school_physics', 'high_school_politics', 'human_sexuality',
'international_law', 'journalism', 'jurisprudence', 'legal_and_moral_basis', 'logical', 'machine_learning', 'management', 'marketing', 'marxist_theory', 'modern_chinese', 'nutrition', 'philosophy', 'professional_accounting', 'professional_law', 
'professional_medicine', 'professional_psychology', 'public_relations', 'security_study', 'sociology', 'sports_science', 'traditional_chinese_medicine', 'virology', 'world_history', 'world_religions']
    

    if name not in all_valid_tasks:
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
