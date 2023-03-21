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
# pylint:disable=I1101
# pylint:disable=W0212

"""
lookup transforms
"""
import mindspore._c_dataengine as cde
from mindspore.dataset.text.transforms import TextTensorOperation
from mindspore.dataset.core.datatypes import mstype_to_detype
from mindspore.common import dtype as mstype

class Lookup(TextTensorOperation):
    """
    Look up a word into an id according to the input vocabulary table.

    Args:
        vocab (Vocab): A vocabulary object.
        return_dtype (mindspore.dtype, optional): The data type that lookup operation maps
            string to. Default: mindspore.int32.

    Raises:
        TypeError: If `vocab` is not of type text.Vocab.
        TypeError: If `return_dtype` is not of type mindspore.dtype.

    Examples:
        >>> from mindnlp import Vocab
        >>> from mindnlp.transforms import Lookup
        >>> # Load vocabulary from list
        >>> vocab = Vocab(['深', '圳', '欢', '迎', '您'])
        >>> # Use Lookup operation to map tokens to ids
        >>> lookup = Lookup(vocab)
        >>> text_file_dataset = text_file_dataset.map(operations=[lookup])
    """

    def __init__(self, vocab, return_dtype=mstype.int32):
        super().__init__()
        self._vocab = vocab
        self._unk_token = vocab._unk_token
        self._return_dtype = return_dtype

    def parse(self):
        return cde.LookupOperation(self._vocab._c_vocab, self._unk_token, str(mstype_to_detype(self._return_dtype)))
    