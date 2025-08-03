# Copyright 2024 Huawei Technologies Co., Ltd
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
EmbeddingsFunAdapter
"""

from typing import List

from langchain.embeddings.base import Embeddings

from mindnlp.sentence import SentenceTransformer


class EmbeddingsFunAdapter(Embeddings):
    def __init__(self, embed_model, mirror='huggingface'):
        self.embed_model = embed_model
        self.embedding_model = SentenceTransformer(model_name_or_path=self.embed_model, mirror=mirror)

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        embeddings = self.embedding_model.encode_texts(texts)
        return embeddings

    def embed_query(self, text: str) -> List[float]:
        embeddings = self.embedding_model.encode_texts([text])
        return embeddings[0]
