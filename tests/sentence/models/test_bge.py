import mindspore
from mindspore import Tensor
from mindhf.sentence import SentenceTransformer
import numpy as np

def test_bge_model():
    model = SentenceTransformer('BAAI/bge-large-zh-v1.5')
    sentences_1 = "The weather is lovely today."
    sentences_2 = "It's so sunny outside!"
    embeddings_1 = model.encode(sentences_1, normalize_embeddings=True)
    embeddings_2 = model.encode(sentences_2, normalize_embeddings=True)

    similarity = embeddings_1 @ embeddings_2.T

    assert np.allclose(similarity, 0.6119031, 1e-3, 1e-3)