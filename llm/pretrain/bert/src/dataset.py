import os
import numpy as np
import mindspore.common.dtype as mstype
import mindspore.dataset as ds
import mindspore.dataset.transforms as C
from icecream import ic

# parallel read and process dataset
def read_bert_pretrain_mindrecord(batch_size, file_name, rank_id, rank_size):
    wiki_dataset = ds.MindDataset(dataset_files=file_name, num_shards=rank_size, shard_id=rank_id)
    wiki_dataset = wiki_dataset.batch(batch_size, drop_remainder=True)
    total = wiki_dataset.get_dataset_size()
    return wiki_dataset, total


