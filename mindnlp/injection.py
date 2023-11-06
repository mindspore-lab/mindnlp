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
"""
Injection mindspore.nn for MindNLP
"""
from mindspore import nn
import mindspore.common.dtype as mstype
from mindspore.common.tensor import Tensor
from mindspore.ops import operations as P
from mindspore.common.parameter import Parameter
from mindspore.common.initializer import initializer
from mindspore import _checkparam as Validator


def init_embedding(self, vocab_size, embedding_size, use_one_hot=False, embedding_table='normal',
                dtype=mstype.float32, padding_idx=None):
    """Initialize Embedding."""
    super(nn.Embedding, self).__init__()
    self.vocab_size = Validator.check_value_type('vocab_size', vocab_size, [int], self.cls_name)
    self.embedding_size = Validator.check_value_type('embedding_size', embedding_size, [int], self.cls_name)
    Validator.check_value_type('use_one_hot', use_one_hot, [bool], self.cls_name)
    Validator.check_subclass("dtype", dtype, mstype.number_type, self.cls_name)
    self.use_one_hot = use_one_hot
    self.dtype = dtype
    self.init_tensor = initializer(embedding_table, [vocab_size, embedding_size])
    self.padding_idx = padding_idx
    self.embedding_table = Parameter(self.init_tensor, name='embedding_table')
    self.expand = P.ExpandDims()
    self.reshape_flat = P.Reshape()
    self.shp_flat = (-1,)
    self.gather = P.Gather()
    self.one_hot = P.OneHot()
    self.on_value = Tensor(1.0, self.dtype)
    self.off_value = Tensor(0.0, self.dtype)
    self.array_mul = P.MatMul()
    self.reshape = P.Reshape()
    self.get_shp = P.Shape()
    self.concat = P.Concat()
    self.reset_parameters()

def reset_embedding_params(self):
    """reset_embedding_params"""
    if self.padding_idx:
        if isinstance(self.init_tensor, Tensor) and self.init_tensor.init is not None:
            self.init_tensor = self.init_tensor.init_data()
        self.init_tensor = self.init_tensor.asnumpy()
        self.init_tensor[self.padding_idx] = 0
        self.init_tensor = Tensor(self.init_tensor)

nn.Embedding.reset_parameters = reset_embedding_params
nn.Embedding.__init__ = init_embedding
