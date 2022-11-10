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
            drop_remainder=drop_remainder)

    return dataset
