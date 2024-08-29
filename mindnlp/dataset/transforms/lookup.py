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
import mindspore._c_dataengine as cde # pylint: disable=no-name-in-module, import-error
from mindspore.dataset.text.transforms import TextTensorOperation
from mindspore.dataset.core.datatypes import mstype_to_detype
from mindspore.common import dtype as mstype
from mindspore.dataset.text import Vocab as msVocab
from mindnlp.vocab import Vocab as nlpVocab


class Lookup(TextTensorOperation):
    """
    Look up a word into an id according to the input vocabulary table.

    Args:
        vocab (Vocab): A vocabulary object.
        unk_token (str): unknow token for OOV.
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
    def __init__(self, vocab, unk_token, return_dtype=mstype.int32):
        r"""
        Initializes a Lookup object.
        
        Args:
            self (object): The instance of the Lookup class.
            vocab (object): An object representing the vocabulary. It can be an instance of nlpVocab or msVocab.
                             For nlpVocab, the vocabulary is created from the token dictionary of the object.
                             For msVocab, the vocabulary is obtained from the 'c_vocab' attribute of the object.
                             Raises a ValueError if the vocab object is not of type nlpVocab or msVocab.
            unk_token (str): The unknown token used for out-of-vocabulary words.
            return_dtype (type, optional): The return data type for the lookup values. Defaults to mstype.int32.
        
        Returns:
            None. This method initializes the Lookup object with the provided parameters.
        
        Raises:
            ValueError: If the 'vocab' parameter is not an instance of nlpVocab or msVocab.
        """
        super().__init__()
        if isinstance(vocab, nlpVocab):
            self._vocab = cde.Vocab.from_dict(vocab._token_dict)
        elif isinstance(vocab, msVocab):
            self._vocab = vocab.c_vocab
        else:
            raise ValueError(f'do not support vocab type {type(vocab)}.')

        self._unk_token = unk_token
        self._return_dtype = return_dtype

    def parse(self):
        r"""
        Parses the lookup operation based on the specified vocabulary.
        
        Args:
            self: An instance of the Lookup class.
        
        Returns:
            None. This method does not return any value.
        
        Raises:
            None.
        
        Description:
        This method performs the lookup operation by using the specified vocabulary. It takes into account the following parameters:
        
        - `self`: An instance of the Lookup class. This parameter is required to access the instance variables and methods of the class.
        
        The lookup operation is performed using the `cde.LookupOperation` function. The parameters used for the lookup operation are as follows:
        
        - `self._vocab`: The vocabulary used for the lookup operation.
        - `self._unk_token`: The token to be used for unknown words in the lookup operation.
        - `str(mstype_to_detype(self._return_dtype))`: The return data type of the lookup operation, converted to a string.
        
        The method does not return any value, as it modifies the internal state of the Lookup instance.
        
        Note:
        - This method assumes that the `cde.LookupOperation` function is available and properly implemented.
        """
        return cde.LookupOperation(self._vocab, self._unk_token, str(mstype_to_detype(self._return_dtype)))
