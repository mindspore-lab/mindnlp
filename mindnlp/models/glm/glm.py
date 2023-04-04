"""
GLM Model
"""

import mindspore
from mindspore import nn
from mindspore.common import initializer

from .glm_config import GLMConfig

class GLMEncoderDecoder(nn.Cell):
    def __init__(self, config:GLMConfig):
        super(GLMEncoderDecoder, self).__init__()

        # Word embeddings
        self.word_embedding = nn.Embedding(vocab_size = config.vocab_size, 
                                           embedding_size = config.hidden_size,
                                           embedding_table = initializer.Normal(sigma = 0.02))

        


class GLMModel(nn.Cell):
    pass