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
"""embedding"""

__all__ = [
    "TokenEmbedding"
]

from abc import abstractmethod
from mindspore import nn
from mindspore import Parameter
from mindnlp._legacy.nn import Dropout

class TokenEmbedding(nn.Cell):
    r"""
    Create Embedding from a given pre-trained vector file.

    Args:
        init_embed (Tensor): Passing into Vocab and Tensor,use these values to initialize Embedding directly.
        requires_grad (bool): Whether this parameter needs to be gradient to update.
        dropout (float): Dropout of the output of Embedding.

    """
    def __init__(self, init_embed, requires_grad: bool = True, dropout=0.0):
        r"""
        Initializes an instance of the TokenEmbedding class.
        
        Args:
            self: The instance of the TokenEmbedding class.
            init_embed: The initial embedding data for tokens. It should be a tensor or array-like object.
            requires_grad: A boolean flag indicating if the embedding parameters should require gradients during optimization. Default is True.
            dropout: The dropout probability to apply to the embedding output. It should be a float value between 0.0 and 1.0.
        
        Returns:
            None. This method only initializes the TokenEmbedding instance.
        
        Raises:
            - TypeError: If the init_embed parameter is not a valid tensor or array-like object.
            - ValueError: If the dropout parameter is not within the range [0.0, 1.0].
        """
        super().__init__()

        self.embed = Parameter(init_embed, name='embed', requires_grad=requires_grad)
        self.dropout_layer = Dropout(p=dropout)
        self._embed_size = self.embed.shape

    def dropout(self, words):
        r"""
        drop the word after embedding.

        Args:
            words (Tensor): Tensor about to be dropout.

        Returns:
            Tensor, Dropout processed data.

        """
        return self.dropout_layer(words)

    def __len__(self):
        """
        embed len
        """
        return len(self.embed)

    def embed_size(self):
        """
        embed size
        """
        return self._embed_size

    def num_embeddings(self):
        """
        num embeddings
        """
        return len(self.embed)

    @abstractmethod
    def construct(self, ids):
        r"""

        Args:
            ids (Tensor): Ids to query.

        Raises:
            NotImplementedError: If construct interface is not called.

        """
        raise NotImplementedError(f'Function `construct` not implemented in {self.__class__.__name__}')
