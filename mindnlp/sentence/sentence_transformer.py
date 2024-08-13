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
"""Sentence Transformer"""

import logging
from collections import OrderedDict
from typing import Optional, Iterable, Dict, Union, List, Literal, Tuple

import mindspore
from mindspore import nn
import numpy as np
from mindspore import Tensor
from tqdm import trange

from mindnlp.sentence.models import Transformer, Pooling
from mindnlp.sentence.util import truncate_embeddings


class SentenceTransformer(nn.Sequential):
    def __init__(
            self,
            model_name_or_path: Optional[str] = None,
            modules: Optional[Iterable[nn.Module]] = None,
            device: Optional[str] = None,
            prompts: Optional[Dict[str, str]] = None,
            default_prompt_name: Optional[str] = None,
            cache_folder: Optional[str] = None,
            local_files_only: bool = False,
            token: Optional[Union[bool, str]] = None,
            truncate_dim: Optional[int] = None,
    ):
        self.prompts = prompts or {}
        self.default_prompt_name = default_prompt_name
        self.truncate_dim = truncate_dim
        self._model_card_vars = {}
        self._model_card_text = None
        self._model_config = {}

        if device is None:
            device = 'CPU'
        logging.info(f"Use device_name: {device}")
        mindspore.set_context(device_target=device)

        if model_name_or_path is not None and model_name_or_path != "":
            logging.info("Load pretrained SentenceTransformer: %s", model_name_or_path)

            modules = self._load_auto_model(
                model_name_or_path,
                token=token,
                cache_folder=cache_folder,
                local_files_only=local_files_only,
            )
        if modules is not None and not isinstance(modules, OrderedDict):
            modules = OrderedDict([(str(idx), module) for idx, module in enumerate(modules)])

        super().__init__(modules)

    def _load_auto_model(
            self,
            model_name_or_path: str,
            token: Optional[Union[bool, str]],
            cache_folder: Optional[str],
            local_files_only: bool = False,
    ):
        """
        Creates a simple Transformer + Mean Pooling model and returns the modules
        """
        logging.warning(
            "No sentence-transformers model found with name %s. Creating a new one with MEAN pooling.",
            model_name_or_path
        )
        transformer_model = Transformer(
            model_name_or_path,
            cache_dir=cache_folder,
            model_args={
                "token": token,
                "local_files_only": local_files_only,
            },
            tokenizer_args={
                "token": token,
                "local_files_only": local_files_only,
            }
        )
        pooling_model = Pooling(transformer_model.get_word_embedding_dimension(), "mean")
        return [transformer_model, pooling_model]

    def _first_module(self):
        """Returns the first module of this sequential embedder"""
        return self._cells[next(iter(self._cells))]

    def tokenize(self, texts: Union[List[str], List[Dict], List[Tuple[str, str]]]):
        """
        Tokenizes the texts
        """
        return self._first_module().tokenize(texts)

    def _text_length(self, text: Union[List[int], List[List[int]]]):
        """
        Help function to get the length for the input text. Text can be either
        a list of ints (which means a single text as input), or a tuple of list of ints
        (representing several text inputs to the model).
        """

        if isinstance(text, dict):  # {key: value} case
            return len(next(iter(text.values())))
        elif not hasattr(text, "__len__"):  # Object has no len() method
            return 1
        elif len(text) == 0 or isinstance(text[0], int):  # Empty string or list of ints
            return len(text)
        else:
            return sum(len(t) for t in text)  # Sum of length of individual strings

    def encode(
            self,
            sentences: Union[str, List[str]],
            prompt_name: Optional[str] = None,
            prompt: Optional[str] = None,
            batch_size: int = 32,
            show_progress_bar: bool = None,
            output_value: Optional[Literal["sentence_embedding", "token_embeddings"]] = "sentence_embedding",
            normalize_embeddings: bool = False,
    ) -> Union[List[Tensor], Tensor]:
        input_was_string = False
        if isinstance(sentences, str) or not hasattr(
                sentences, "__len__"
        ):  # Cast an individual sentence to a list with length 1
            sentences = [sentences]
            input_was_string = True

        if prompt is None:
            if prompt_name is not None:
                try:
                    prompt = self.prompts[prompt_name]
                except KeyError:
                    raise ValueError(
                        f"Prompt name '{prompt_name}' not found in the configured prompts dictionary with keys {list(self.prompts.keys())!r}."
                    )
            elif self.default_prompt_name is not None:
                prompt = self.prompts.get(self.default_prompt_name, None)
        else:
            if prompt_name is not None:
                logging.warning(
                    "Encode with either a `prompt`, a `prompt_name`, or neither, but not both. "
                    "Ignoring the `prompt_name` in favor of `prompt`."
                )

        extra_features = {}
        if prompt is not None:
            sentences = [prompt + sentence for sentence in sentences]

            # Some models (e.g. INSTRUCTOR, GRIT) require removing the prompt before pooling
            # Tracking the prompt length allow us to remove the prompt during pooling
            tokenized_prompt = self.tokenize([prompt])
            if "input_ids" in tokenized_prompt:
                extra_features["prompt_length"] = tokenized_prompt["input_ids"].shape[-1] - 1

        all_embeddings = []
        length_sorted_idx = np.argsort([-self._text_length(sen) for sen in sentences])
        sentences_sorted = [sentences[idx] for idx in length_sorted_idx]

        for start_index in trange(0, len(sentences), batch_size, desc="Batches", disable=not show_progress_bar):
            sentences_batch = sentences_sorted[start_index: start_index + batch_size]
            features = self.tokenize(sentences_batch)
            features.update(extra_features)

            out_features = self.forward(features)

            out_features["sentence_embedding"] = truncate_embeddings(
                out_features["sentence_embedding"], self.truncate_dim
            )

            if output_value == "token_embeddings":
                embeddings = []
                for token_emb, attention in zip(out_features[output_value], out_features["attention_mask"]):
                    last_mask_id = len(attention) - 1
                    while last_mask_id > 0 and attention[last_mask_id].item() == 0:
                        last_mask_id -= 1

                    embeddings.append(token_emb[0: last_mask_id + 1])
            elif output_value is None:  # Return all outputs
                embeddings = []
                for sent_idx in range(len(out_features["sentence_embedding"])):
                    row = {name: out_features[name][sent_idx] for name in out_features}
                    embeddings.append(row)
            else:  # Sentence embeddings
                embeddings = out_features[output_value]
                # embeddings = embeddings.detach()
                if normalize_embeddings:
                    embeddings = mindspore.ops.L2Normalize(embeddings, p=2, dim=1)

            all_embeddings.extend(embeddings)

        all_embeddings = [all_embeddings[idx] for idx in np.argsort(length_sorted_idx)]

        if input_was_string:
            all_embeddings = all_embeddings[0]

        return all_embeddings

    def encode_texts(self, texts: List[str]) -> List[List[float]]:
        texts = [t.replace("\n", " ") for t in texts]
        embeddings = self.encode(texts)
        for i, embedding in enumerate(embeddings):
            embeddings[i] = embedding.tolist()
        return embeddings


if __name__ == '__main__':
    model = SentenceTransformer(
        "moka-ai/m3e-base",
        device='CPU',
    )
    sentences = [
        "This framework generates embeddings for each input sentence",
        "Sentences are passed as a list of string.",
        "The quick brown fox jumps over the lazy dog.",
    ]
    sentence_embeddings = model.encode(sentences)
    for sentence, embedding in zip(sentences, sentence_embeddings):
        print("Sentence:", sentence)
        print("Embedding:", embedding)
        print("")
