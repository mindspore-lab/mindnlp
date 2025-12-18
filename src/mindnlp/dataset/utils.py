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
Dataset utils
"""

def make_bucket(dataset, column_name, pad_index, \
                bucket_boundaries, bucket_batch_sizes, drop_remainder):
    """make bucket function."""
    pad_info = {column_name: ([None], pad_index)}

    dataset = dataset.bucket_batch_by_length(
            [column_name],
            element_length_function=lambda elem:elem.shape[0],
            bucket_boundaries=bucket_boundaries,
            bucket_batch_sizes=bucket_batch_sizes,
            pad_info=pad_info,
            pad_to_bucket_boundary=True,
            drop_remainder=drop_remainder)

    return dataset

def make_bucket_2cloums(dataset, column_name, pad_value1, pad_value2, \
                bucket_boundaries, bucket_batch_sizes, drop_remainder):
    """make bucket 2cloums function."""
    pad_info = {column_name[0]: ([None], pad_value1),column_name[1]: ([None], pad_value2)}

    dataset = dataset.bucket_batch_by_length(
            column_name,
            element_length_function=lambda elem1,elem2:elem1.shape[0],
            bucket_boundaries=bucket_boundaries,
            bucket_batch_sizes=bucket_batch_sizes,
            pad_info=pad_info,
            pad_to_bucket_boundary=True,
            drop_remainder=drop_remainder)

    return dataset
