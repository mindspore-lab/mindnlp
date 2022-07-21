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
TextDataSet base class
"""
from mindspore.dataset import TextBaseDataset, SourceDataset



class TextDataSet(SourceDataset, TextBaseDataset):
    r"""
    Dataset base class for natural language processing Inherit TextBaseDataset

        Args:
            lazy (bool):An identifier for the operation on the dataset,
            If false, use pipline pipeline mode to process data . (default: True).
            usage (str, optional): Acceptable usages include 'train', 'test' and 'all'
            (default=None, all samples).
            num_samples (int, optional): Number of samples (rows) to read (default=None, reads the full dataset).
            num_parallel_workers (int, optional): Number of workers to process the data
                (default=1, number set in the config).
            shuffle (Union[bool, Shuffle level], optional): Perform reshuffling of the data every epoch
                (default=Shuffle.GLOBAL).
                If shuffle is False, no shuffling will be performed.
                If shuffle is True, performs global shuffle.
                There are three levels of shuffling, desired shuffle enum defined by mindspore.dataset.Shuffle.

                - Shuffle.GLOBAL: Shuffle both the files and samples, same as setting shuffle to True.

                - Shuffle.FILES: Shuffle files only.
            num_shards (int, optional): Only supporet when lazy is false.Number of shards that
            the dataset will be divided into (default=None).
                When this argument is specified, `num_samples` reflects the max sample number of per shard.
            shard_id (int, optional): Only supporet when lazy is false.The shard ID within `num_shards` (default=None).
            This argument can only be specified when `num_shards` is also specified.
            num_parallel_workers (int, optional): Only supporet when lazy is false.
            Number of workers to read the data.
            (default=None, number set in the  mindspore.dataset.config).
            cache (DatasetCache, optional): Only supporet when lazy is false.
            Use tensor caching service to speed up dataset processing
                (default=None, which means no cache is used).
    """

    def __init__(self, lazy=True, num_samples=None, num_parallel_workers=1, shuffle=False,
                 num_shards=None, shard_id=None, cache=None):
        self.lazy = lazy
        if lazy:
            pass
        else:
            super().__init__(num_parallel_workers=num_parallel_workers, num_samples=num_samples,
                             shuffle=shuffle, num_shards=num_shards, shard_id=shard_id, cache=cache)
