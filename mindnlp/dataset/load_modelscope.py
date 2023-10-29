# Copyright 2020 The ModelScope Datasets Authors and the TensorFlow Datasets Authors.
# Copyright 2023 Huawei Technologies Co., Ltd
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
# pylint: disable=C0103
"""
load dataset from modelscope
"""
import os
from typing import Union, Optional, Sequence, Mapping
from mindspore.dataset import GeneratorDataset

from modelscope.msdatasets import MsDataset
from modelscope.msdatasets.ms_dataset import DatasetContextConfig as ms_load
from modelscope.utils.config_ds import MS_DATASETS_CACHE
from modelscope.utils.constant import (DEFAULT_DATASET_NAMESPACE,
                                       DEFAULT_DATASET_REVISION, DownloadMode, Hubs)
from modelscope.msdatasets.dataset_cls import (NativeIterableDataset)
from modelscope.msdatasets.data_loader.data_loader_manager import (
                        RemoteDataLoaderManager,RemoteDataLoaderType)

from datasets import Dataset
from datasets.utils.file_utils import is_relative_path


class TransferIterableDataset():
    """TransferDataset for Huggingface Dataset."""
    def __init__(self, arrow_ds, column_names):
        self.ds = arrow_ds
        self.column_names = column_names

    def __iter__(self):
        for data in self.ds:
            yield tuple(data[name] for name in self.column_names)

class TransferDataset():
    """TransferDataset for Huggingface Dataset."""
    def __init__(self, arrow_ds, column_names):
        self.ds = arrow_ds
        self.column_names = column_names

    def __getitem__(self, index):
        return tuple(self.ds[int(index)][name] for name in self.column_names)

    def __len__(self):
        return self.ds.num_rows


def load_dataset_ms(
    dataset_name: Union[str, list],
    namespace: Optional[str] = DEFAULT_DATASET_NAMESPACE,
    target: Optional[str] = None,
    version: Optional[str] = DEFAULT_DATASET_REVISION,
    hub: Optional[Hubs] = Hubs.modelscope,
    subset_name: Optional[str] = None,
    split: Optional[str] = None,
    data_dir: Optional[str] = None,
    data_files: Optional[Union[str, Sequence[str],
                                Mapping[str, Union[str,
                                                    Sequence[str]]]]] = None,
    download_mode: Optional[DownloadMode] = DownloadMode.
    REUSE_DATASET_IF_EXISTS,
    cache_dir: Optional[str] = MS_DATASETS_CACHE,
    use_streaming: Optional[bool] = False,
    stream_batch_size: Optional[int] = 1,
    token: Optional[str] = None,
    **config_kwargs,
) -> Union[dict, 'MsDataset', NativeIterableDataset]:
    """Load a MsDataset from the ModelScope Hub, Hugging Face Hub, urls, or a local dataset.

        Args:
            dataset_name (str): Path or name of the dataset.
                The form of `namespace/dataset_name` is also supported.
            namespace(str, optional): Namespace of the dataset. It should not be None if you load a remote dataset
                from Hubs.modelscope,
            namespace (str, optional):
                Namespace of the dataset. It should not be None if you load a remote dataset
                from Hubs.modelscope,
            target (str, optional): Name of the column to output.
            version (str, optional): Version of the dataset script to load:
            subset_name (str, optional): Defining the subset_name of the dataset.
            data_dir (str, optional): Defining the data_dir of the dataset configuration. I
            data_files (str or Sequence or Mapping, optional): Path(s) to source data file(s).
            split (str, optional): Which split of the data to load.
            hub (Hubs or str, optional): When loading from a remote hub, where it is from. default Hubs.modelscope
            download_mode (DownloadMode or str, optional): How to treat existing datasets. default
                                                            DownloadMode.REUSE_DATASET_IF_EXISTS
            cache_dir (str, Optional): User-define local cache directory.
            use_streaming (bool, Optional): If set to True, no need to download all data files.
                                            Instead, it streams the data progressively, and returns
                                            NativeIterableDataset or a dict of NativeIterableDataset.
            stream_batch_size (int, Optional): The batch size of the streaming data.
            custom_cfg (str, Optional): Model configuration, this can be used for custom datasets.
                                        see https://modelscope.cn/docs/Configuration%E8%AF%A6%E8%A7%A3
            token (str, Optional): SDK token of ModelScope.
            **config_kwargs (additional keyword arguments): Keyword arguments to be passed

        Returns:
            MsDataset (MsDataset): MsDataset object for a certain dataset.
        
        Example:

        Load a dataset from the ModelScope Library Hub:

        ```py
        >>> !pip install modelscope
        >>> from modelscope.msdatasets import MsDataset
        >>> ds =  load_dataset_ms('damo/MSAgent-Bench', subset_name='default', namespace='test01', split='train')

        # Map data files to splits
        >>> data_files = {'train': 'train.csv', 'test': 'test.csv'}
        >>> ds = load_dataset_ms('namespace/your_dataset_name', data_files=data_files)
        ```

        Load an image dataset with the `ImageFolder` dataset builder:

        ```py
        >>> from modelscope.msdatasets import MsDataset
        >>> ds = load_dataset_ms('imagefolder', data_dir='/path/to/imgs/')
        >>> ds = load_dataset_ms('imagefolder', data_files='/path/to/imgs.zip', split='train') 
        >>> print(next(iter(ds)))
        ```

        Load Ultra-Large datasets with Streaming load

        ```py
        >>> from modelscope.msdatasets import MsDataset
        >>> # Take uni-fold datasets as an example, URL: https://modelscope.cn/datasets/DPTech/Uni-Fold-Data/summary
        >>> ds = load_dataset_ms(dataset_name='Uni-Fold-Data', namespace='DPTech', split='train', use_streaming=True)
        >>> print(next(iter(ds)))
        ```

        """
    # if token:
    #     from modelscope.hub.api import HubApi
    #     api = HubApi()
    #     api.login(token)

    download_mode = DownloadMode(download_mode
                                    or DownloadMode.REUSE_DATASET_IF_EXISTS)
    hub = Hubs(hub or Hubs.modelscope)
    shuffle = config_kwargs.get('shuffle', False)

    if not isinstance(dataset_name, str) and not isinstance(
            dataset_name, list):
        raise TypeError(
            f'dataset_name must be `str` or `list`, but got {type(dataset_name)}'
        )

    if isinstance(dataset_name, list):
        if target is None:
            target = 'target'
        dataset_inst = Dataset.from_dict({target: dataset_name})
        return MsDataset.to_ms_dataset(dataset_inst, target=target)

    dataset_name = os.path.expanduser(dataset_name)
    is_local_path = os.path.exists(dataset_name)
    if is_relative_path(dataset_name) and dataset_name.count(
            '/') == 1 and not is_local_path:
        dataset_name_split = dataset_name.split('/')
        namespace = dataset_name_split[0].strip()
        dataset_name = dataset_name_split[1].strip()
        if not namespace or not dataset_name:
            raise 'The dataset_name should be in the form of `namespace/dataset_name` or `dataset_name`.'

    ds_ret = ms_load(dataset_name,
                     namespace=namespace,
                     target=target,
                     version=version,
                     hub=hub,
                     subset_name=subset_name,
                     split=split,
                     data_dir=data_dir,
                     data_files=data_files,
                     download_mode=download_mode,
                     cache_root_dir=cache_dir,
                     use_streaming=use_streaming,
                     stream_batch_size=stream_batch_size,
                     **config_kwargs)
    remote_dataloader_manager = RemoteDataLoaderManager(ds_ret)
    dataset_inst = remote_dataloader_manager.load_dataset(
                        RemoteDataLoaderType.MS_DATA_LOADER)

    datasets_dict = {}

    for key, raw_ds in dataset_inst.items():
        column_names = list(raw_ds.features.keys())
        source = TransferDataset(raw_ds, column_names) if isinstance(raw_ds, Dataset) \
            else TransferIterableDataset(raw_ds, column_names)
        ms_ds = GeneratorDataset(
            source=source,
            column_names=column_names,
            shuffle=shuffle,
            num_parallel_workers=1)
        datasets_dict[key] = ms_ds

    if len(datasets_dict) == 1:
        return datasets_dict.popitem()[1]
    return datasets_dict
