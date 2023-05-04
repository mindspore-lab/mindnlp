import os
from typing import Union, Tuple
from datasets import load_dataset as hf_load
from mindspore.dataset import GeneratorDataset
from mindnlp.dataset.text_summarization.cnn_dailymail import CnnDailymail_Process
from mindnlp.dataset.register import load_dataset, process
from mindnlp.configs import DEFAULT_ROOT


class HFcnn_dailymail:
    """
    Hugging Face Cnn_dailymail dataset source
    """
    def __init__(self, dataset_list) -> None:
        self.dataset_list = dataset_list
        self._article, self._summary = [], []
        self._load()

    def _load(self):
        for every_dict in self.dataset_list:
            self._article.append(every_dict['article'])
            self._summary.append(every_dict['highlights'])

    def __getitem__(self, index):
        return self._article[index], self._summary[index]

    def __len__(self):
        return len(self._article)


@load_dataset.register
def HF_cnn_dailymail(
    root: str = DEFAULT_ROOT,
    split: Union[Tuple[str], str] = ("train", "validation"),
    shuffle=True,
):
    r"""
    Load the huggingface Cnn_dailymail dataset.
    Args:
        root (str): Directory where the datasets are saved.
            Default:~/.mindnlp
        split (str|Tuple[str]): Split or splits to be returned.
            Default:('train', 'validation').
        shuffle (bool): Whether to shuffle the dataset.
            Default:True.
    Returns:
        - **datasets_list** (list) -A list of loaded datasets.
          If only one type of dataset is specified,such as 'trian',
          this dataset is returned instead of a list of datasets.
    Examples:
        >>> root = "~/.mindnlp"
        >>> split = ('train', 'validation')
        >>> dataset_train, dataset_validation = HF_cnn_dailymail(root, split)
        >>> train_iter = dataset_train.create_tuple_iterator()
        >>> print(next(train_iter))
    """

    cache_dir = os.path.join(root, "datasets", "hf_datasets", "cnn_dailymail")
    column_names = ["article", "highlights"]
    datasets_list = []
    mode_list = []

    if isinstance(split, str):
        mode_list.append(split)
    else:
        for s in split:
            mode_list.append(s)

    ds_list = hf_load('cnn_dailymail', split=mode_list, data_dir=cache_dir)
    for every_ds in ds_list:
        datasets_list.append(GeneratorDataset(
            source=HFcnn_dailymail(every_ds),
            column_names=column_names, shuffle=shuffle)
        )
    if len(mode_list) == 1:
        return datasets_list[0]
    return datasets_list


@process.register
def HF_cnn_dailymail(
    root: str = DEFAULT_ROOT,
    split: Union[Tuple[str], str] = ("train", "validation"),
    cache_dir: str = os.path.join(DEFAULT_ROOT, "datasets", "hf_datasets", "cnn_dailymail"),
):
    r"""
    Preprocessing function for Hugging Face Cnn_dailymail dataset.
    Args:
        root (str): Root directory of the datasets.
            Default: '~/.mindnlp'.
        split (Union[str, Tuple[str, ...]]): Split or splits to be processed.
            Default: ('train', 'validation').
        cache_dir (str): Directory to cache the dataset.
            Default: '~/.mindnlp/datasets/hf_datasets/cnn_dailymail'.
    Returns:
        None
    """
    process_dir = os.path.join(root, "datasets", "mindnlp", "cnn_dailymail")
    os.makedirs(process_dir, exist_ok=True)

    for s in split:
        hf_ds = hf_load('cnn_dailymail', split=s, data_dir=cache_dir)

        article_path = os.path.join(process_dir, f"{s}_article.txt")
        summary_path = os.path.join(process_dir, f"{s}_summary.txt")

        with open(article_path, 'w', encoding='utf-8') as f1, open(summary_path, 'w', encoding='utf-8') as f2:
            for example in hf_ds:
                f1.write(example['article']+'\n')
                f2.write(example['highlights']+'\n')

    process(CnnDailymail_Process, root=root, cache_dir=process_dir, split=split)
