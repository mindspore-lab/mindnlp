"""normalize model"""
from __future__ import annotations

from mindspore import Tensor
from mindnlp.core.nn import functional as F
from mindnlp.core import nn


class Normalize(nn.Module):
    """This layer normalizes embeddings to unit length"""

    def forward(self, features: dict[str, Tensor]) -> dict[str, Tensor]:
        features.update({"sentence_embedding": F.normalize(features["sentence_embedding"], p=2, dim=1)})
        return features

    def save(self, output_path) -> None:
        pass

    @staticmethod
    def load(input_path) -> Normalize:
        return Normalize()
