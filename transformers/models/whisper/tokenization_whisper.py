# coding=utf-8
# Copyright 2022 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ============================================================================
"""Tokenization classes for Whisper."""
import json
import os
from functools import lru_cache
from typing import List, Optional, Tuple, Union

import numpy as np
import regex as re

from mindnlp.utils import logging
from ...tokenization_utils import AddedToken, PreTrainedTokenizer
from .english_normalizer import BasicTextNormalizer, EnglishTextNormalizer


VOCAB_FILES_NAMES = {
    "vocab_file": "vocab.json",
    "tokenizer_file": "tokenizer.json",
    "merges_file": "merges.txt",
    "normalizer_file": "normalizer.json",
}

PRETRAINED_VOCAB_FILES_MAP = {
    "vocab_file": {
        "openai/whisper-base": "https://hf-mirror.com/openai/whisper-base/resolve/main/vocab.json",
    },
    "merges_file": {"openai/whisper-base": "https://hf-mirror.com/openai/whisper-base/resolve/main/merges_file.txt"},
    "normalizer_file": {
        "openai/whisper-base": "https://hf-mirror.com/openai/whisper-base/resolve/main/normalizer.json"
    },
}

MAX_MODEL_INPUT_SIZES = {
    "openai/whisper-base": 448,
}


# Copied from transformers.models.gpt2.tokenization_gpt2.bytes_to_unicode
def bytes_to_unicode():
    """
    Returns list of utf-8 byte and a mapping to unicode strings. We specifically avoids mapping to whitespace/control
    characters the bpe code barfs on.

    The reversible bpe codes work on unicode strings. This means you need a large # of unicode characters in your vocab
    if you want to avoid UNKs. When you're at something like a 10B token dataset you end up needing around 5K for
    decent coverage. This is a significant percentage of your normal, say, 32K bpe vocab. To avoid that, we want lookup
    tables between utf-8 bytes and unicode strings.
    """
    bs = (
        list(range(ord("!"), ord("~") + 1)) + list(range(ord("¡"), ord("¬") + 1)) + list(range(ord("®"), ord("ÿ") + 1))
    )
    cs = bs[:]
    n = 0
    for b in range(2**8):
        if b not in bs:
            bs.append(b)
            cs.append(2**8 + n)
            n += 1
    cs = [chr(n) for n in cs]
    return dict(zip(bs, cs))


logger = logging.get_logger(__name__)


# Copied from transformers.models.gpt2.tokenization_gpt2.get_pairs
def get_pairs(word):
    """
    Return set of symbol pairs in a word.

    Word is represented as tuple of symbols (symbols being variable-length strings).
    """
    pairs = set()
    prev_char = word[0]
    for char in word[1:]:
        pairs.add((prev_char, char))
        prev_char = char
    return pairs


LANGUAGES = {
    "en": "english",
    "zh": "chinese",
    "de": "german",
    "es": "spanish",
    "ru": "russian",
    "ko": "korean",
    "fr": "french",
    "ja": "japanese",
    "pt": "portuguese",
    "tr": "turkish",
    "pl": "polish",
    "ca": "catalan",
    "nl": "dutch",
    "ar": "arabic",
    "sv": "swedish",
    "it": "italian",
    "id": "indonesian",
    "hi": "hindi",
    "fi": "finnish",
    "vi": "vietnamese",
    "he": "hebrew",
    "uk": "ukrainian",
    "el": "greek",
    "ms": "malay",
    "cs": "czech",
    "ro": "romanian",
    "da": "danish",
    "hu": "hungarian",
    "ta": "tamil",
    "no": "norwegian",
    "th": "thai",
    "ur": "urdu",
    "hr": "croatian",
    "bg": "bulgarian",
    "lt": "lithuanian",
    "la": "latin",
    "mi": "maori",
    "ml": "malayalam",
    "cy": "welsh",
    "sk": "slovak",
    "te": "telugu",
    "fa": "persian",
    "lv": "latvian",
    "bn": "bengali",
    "sr": "serbian",
    "az": "azerbaijani",
    "sl": "slovenian",
    "kn": "kannada",
    "et": "estonian",
    "mk": "macedonian",
    "br": "breton",
    "eu": "basque",
    "is": "icelandic",
    "hy": "armenian",
    "ne": "nepali",
    "mn": "mongolian",
    "bs": "bosnian",
    "kk": "kazakh",
    "sq": "albanian",
    "sw": "swahili",
    "gl": "galician",
    "mr": "marathi",
    "pa": "punjabi",
    "si": "sinhala",
    "km": "khmer",
    "sn": "shona",
    "yo": "yoruba",
    "so": "somali",
    "af": "afrikaans",
    "oc": "occitan",
    "ka": "georgian",
    "be": "belarusian",
    "tg": "tajik",
    "sd": "sindhi",
    "gu": "gujarati",
    "am": "amharic",
    "yi": "yiddish",
    "lo": "lao",
    "uz": "uzbek",
    "fo": "faroese",
    "ht": "haitian creole",
    "ps": "pashto",
    "tk": "turkmen",
    "nn": "nynorsk",
    "mt": "maltese",
    "sa": "sanskrit",
    "lb": "luxembourgish",
    "my": "myanmar",
    "bo": "tibetan",
    "tl": "tagalog",
    "mg": "malagasy",
    "as": "assamese",
    "tt": "tatar",
    "haw": "hawaiian",
    "ln": "lingala",
    "ha": "hausa",
    "ba": "bashkir",
    "jw": "javanese",
    "su": "sundanese",
    "yue": "cantonese",
}

# language code lookup by name, with a few language aliases
TO_LANGUAGE_CODE = {
    **{language: code for code, language in LANGUAGES.items()},
    "burmese": "my",
    "valencian": "ca",
    "flemish": "nl",
    "haitian": "ht",
    "letzeburgesch": "lb",
    "pushto": "ps",
    "panjabi": "pa",
    "moldavian": "ro",
    "moldovan": "ro",
    "sinhalese": "si",
    "castilian": "es",
    "mandarin": "zh",
}

TASK_IDS = ["translate", "transcribe"]


class WhisperTokenizer(PreTrainedTokenizer):
    """
    Construct a Whisper tokenizer.

    This tokenizer inherits from [`PreTrainedTokenizer`] which contains some of the main methods. Users should refer to
    the superclass for more information regarding such methods.

    Args:
        vocab_file (`str`):
            Path to the vocabulary file.
        merges_file (`str`):
            Path to the merges file.
        normalizer_file (`str`, *optional*):
            Path to the normalizer_file file.
        errors (`str`, *optional*, defaults to `"replace"`):
            Paradigm to follow when decoding bytes to UTF-8. See
            [bytes.decode](https://docs.python.org/3/library/stdtypes.html#bytes.decode) for more information.
        unk_token (`str`, *optional*, defaults to `"<|endoftext|>"`):
            The unknown token. A token that is not in the vocabulary cannot be converted to an ID and is set to be this
            token instead.
        bos_token (`str`, *optional*, defaults to `"<|endoftext|>"`):
            The beginning of sequence token. The `decoder_start_token_id` is used to set the first token as
            `"<|startoftranscript|>"` when generating.
        eos_token (`str`, *optional*, defaults to `"<|endoftext|>"`):
            The end of sequence token.
        pad_token (`str`, *optional*):
            The token used for padding, for example when batching sequences of different lengths.
        add_prefix_space (`bool`, *optional*, defaults to `False`):
            Whether or not to add an initial space to the input. This allows to treat the leading word just as any
            other word.
        language (`str`, *optional*):
            The language of the transcription text. The corresponding language id token is appended to the start of the
            sequence for multilingual speech recognition and speech translation tasks, e.g. for Spanish the token
            `"<|es|>"` is appended to the start of sequence. This should be used for multilingual fine-tuning only.
        task (`str`, *optional*):
            Task identifier to append at the start of sequence (if any). This should be used for mulitlingual
            fine-tuning, with `"transcribe"` for speech recognition and `"translate"` for speech translation.
        predict_timestamps (`bool`, *optional*, defaults to `False`):
            Whether to omit the `<|notimestamps|>` token at the start of the sequence.
    """
    vocab_files_names = VOCAB_FILES_NAMES
    pretrained_vocab_files_map = PRETRAINED_VOCAB_FILES_MAP
    max_model_input_sizes = MAX_MODEL_INPUT_SIZES
    model_input_names = ["input_ids", "attention_mask"]

    def __init__(
        self,
        vocab_file,
        merges_file,
        normalizer_file=None,
        errors="replace",
        unk_token="<|endoftext|>",
        bos_token="<|endoftext|>",
        eos_token="<|endoftext|>",
        pad_token=None,
        add_prefix_space=False,
        language=None,
        task=None,
        predict_timestamps=False,
        **kwargs,
    ):
        """
        __init__
        
        This method initializes an instance of the WhisperTokenizer class.
        
        Args:
            self: The instance of the class.
            vocab_file (str): The path to the vocabulary file containing the token encoding.
            merges_file (str): The path to the file containing BPE merges for tokenization.
            normalizer_file (str, optional): The path to the file containing English spelling normalizer.
                Defaults to None.
            errors (str): The error handling scheme to use for encoding/decoding errors. Defaults to 'replace'.
            unk_token (str): The unknown token to be used during tokenization. Defaults to 'endoftext'.
            bos_token (str): The beginning of sentence token. Defaults to 'endoftext'.
            eos_token (str): The end of sentence token. Defaults to 'endoftext'.
            pad_token (str, optional): The padding token. Defaults to None.
            add_prefix_space (bool): Whether to add a prefix space during tokenization. Defaults to False.
            language (str, optional): The language of the text. Defaults to None.
            task (str, optional): The task for tokenization. Defaults to None.
            predict_timestamps (bool): Whether to predict timestamps. Defaults to False.

        Returns:
            None.

        Raises:
            FileNotFoundError: If the vocab_file, merges_file, or normalizer_file does not exist.
            ValueError: If the provided unk_token, bos_token, eos_token, or pad_token is not a string.
            TypeError: If the provided unk_token, bos_token, eos_token, or pad_token is not a string or an
                AddedToken instance.
            UnicodeDecodeError: If an error occurs during the decoding of vocab_file or merges_file.
            KeyError: If an error occurs during the creation of the bpe_ranks dictionary.
            re.error: If an error occurs during the compilation of regular expressions.
            json.JSONDecodeError: If an error occurs during the decoding of normalizer_file.
        """
        bos_token = (
            AddedToken(bos_token, lstrip=False, rstrip=False, normalized=False, special=True)
            if isinstance(bos_token, str)
            else bos_token
        )
        eos_token = (
            AddedToken(eos_token, lstrip=False, rstrip=False, normalized=False, special=True)
            if isinstance(eos_token, str)
            else eos_token
        )
        unk_token = (
            AddedToken(unk_token, lstrip=False, rstrip=False, normalized=False, special=True)
            if isinstance(unk_token, str)
            else unk_token
        )
        pad_token = (
            AddedToken(pad_token, lstrip=False, rstrip=False, normalized=False, special=True)
            if isinstance(pad_token, str)
            else pad_token
        )

        with open(vocab_file, encoding="utf-8") as vocab_handle:
            self.encoder = json.load(vocab_handle)
        self.decoder = {v: k for k, v in self.encoder.items()}
        self.errors = errors  # how to handle errors in decoding
        self.byte_encoder = bytes_to_unicode()
        self.byte_decoder = {v: k for k, v in self.byte_encoder.items()}
        with open(merges_file, encoding="utf-8") as merges_handle:
            bpe_merges = merges_handle.read().split("\n")[1:-1]
        bpe_merges = [tuple(merge.split()) for merge in bpe_merges]
        self.bpe_ranks = dict(zip(bpe_merges, range(len(bpe_merges))))
        self.cache = {}
        self.add_prefix_space = add_prefix_space

        if normalizer_file is not None:
            with open(normalizer_file, encoding="utf-8") as vocab_handle:
                self.english_spelling_normalizer = json.load(vocab_handle)
        else:
            self.english_spelling_normalizer = None

        # Should have added re.IGNORECASE so BPE merges can happen for capitalized versions of contractions
        self.pat = re.compile(r"""'s|'t|'re|'ve|'m|'ll|'d| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+""")
        self.timestamp_pat = re.compile(r"<\|(\d+\.\d+)\|>")

        self.language = language
        super().__init__(
            errors=errors,
            unk_token=unk_token,
            bos_token=bos_token,
            eos_token=eos_token,
            pad_token=pad_token,
            add_prefix_space=add_prefix_space,
            **kwargs,
        )

        self.task = task
        self.predict_timestamps = predict_timestamps

    @property
    def vocab_size(self) -> int:
        """
        This method returns the size of the vocabulary used by the WhisperTokenizer.

        Args:
            self (WhisperTokenizer): The instance of the WhisperTokenizer class.

        Returns:
            int: The size of the vocabulary used by the WhisperTokenizer.

        Raises:
            This method does not raise any exceptions.
        """
        return len(self.encoder)

    def get_vocab(self):
        """
        Returns the vocabulary of the WhisperTokenizer instance.

        Args:
            self (WhisperTokenizer): The instance of the WhisperTokenizer class.

        Returns:
            dict: A dictionary representing the vocabulary of the WhisperTokenizer instance.
                The keys of the dictionary are the tokens in the vocabulary, and the values are their respective
                token IDs.

        Raises:
            None.

        Note:
            This method builds the vocabulary by converting the token IDs to tokens using the `convert_ids_to_tokens`
            method of the WhisperTokenizer instance. The tokens and their corresponding IDs are stored in a dictionary.
            Additionally, any added tokens are also included in the vocabulary.

        Example:
            ```python
            >>> tokenizer = WhisperTokenizer()
            >>> vocab = tokenizer.get_vocab()
            >>> print(vocab)
            {'[PAD]': 0, '[UNK]': 1, '[CLS]': 2, '[SEP]': 3, '[MASK]': 4, 'hello': 5, 'world': 6}
            ```

        In the example above, the `get_vocab` method is called on a WhisperTokenizer instance, which returns a
        dictionary representing the vocabulary of the tokenizer. The vocabulary includes the default special tokens
        as well as any added tokens.
        """
        vocab = {self.convert_ids_to_tokens(i): i for i in range(self.vocab_size)}
        vocab.update(self.added_tokens_encoder)
        return vocab

    # Copied from transformers.models.gpt2.tokenization_gpt2.GPT2Tokenizer.bpe with GPT2 -> Whisper
    def bpe(self, token):
        """
        This method, named 'bpe', is part of the class 'WhisperTokenizer' and performs a Byte Pair Encoding (BPE)
        algorithm on the given 'token' parameter.

        Args:
            self: An instance of the 'WhisperTokenizer' class.
            token (str): The token to be encoded using the BPE algorithm.

        Returns:
            str: The encoded token after applying the BPE algorithm.

        Raises:
            None.

        The BPE algorithm is applied to the 'token' by iteratively replacing the most frequent pairs of characters with
        a new character, until no more pairs can be found. The resulting encoded token is then returned.

        The method first checks if the 'token' is already present in the cache. If so, it returns the cached value.
        Otherwise, it proceeds with the BPE encoding process.

        The 'token' is converted into a tuple of characters named 'word'. The method then obtains all possible pairs
        of characters from 'word' using the 'get_pairs' function.

        If no pairs are found, the 'token' is already fully encoded and is returned as is.

        Otherwise, the method enters a loop, where it finds the most frequent pair of characters ('bigram') from the
        'pairs' list based on the 'bpe_ranks' dictionary. If the 'bigram' is not present in the 'bpe_ranks' dictionary,
        the loop is terminated.

        The method then replaces all occurrences of the 'bigram' in the 'word' with a new character. The 'word' is
        iterated through, and whenever the current character matches the first character of the 'bigram' and the next
        character matches the second character of the 'bigram', the new character is appended to the 'new_word' list.
        Otherwise, the current character is appended as is.

        The 'new_word' is converted back to a tuple and assigned to 'word'. If the length of 'word' becomes 1,
        indicating that the token is fully encoded, the loop is terminated.

        The 'pairs' are updated by obtaining all possible pairs from the updated 'word'.

        Finally, the 'word' is joined using spaces to form a string, which is then cached with the 'token' as
        the key in the 'cache' dictionary.

        The encoded 'word' is returned as the result of the method.
        """
        if token in self.cache:
            return self.cache[token]
        word = tuple(token)
        pairs = get_pairs(word)

        if not pairs:
            return token

        while True:
            bigram = min(pairs, key=lambda pair: self.bpe_ranks.get(pair, float("inf")))
            if bigram not in self.bpe_ranks:
                break
            first, second = bigram
            new_word = []
            i = 0
            while i < len(word):
                try:
                    j = word.index(first, i)
                except ValueError:
                    new_word.extend(word[i:])
                    break
                else:
                    new_word.extend(word[i:j])
                    i = j

                if word[i] == first and i < len(word) - 1 and word[i + 1] == second:
                    new_word.append(first + second)
                    i += 2
                else:
                    new_word.append(word[i])
                    i += 1
            new_word = tuple(new_word)
            word = new_word
            if len(word) == 1:
                break
            pairs = get_pairs(word)
        word = " ".join(word)
        self.cache[token] = word
        return word

    def set_prefix_tokens(self, language: str = None, task: str = None, predict_timestamps: bool = None):
        """
        Override the prefix tokens appended to the start of the label sequence. This method can be used standalone to
        update the prefix tokens as required when fine-tuning.

        Example:
            ```python
            >>> # instantiate the tokenizer and set the prefix token to Spanish
            >>> tokenizer = WhisperTokenizer.from_pretrained("openai/whisper-tiny", language="spanish")
            >>> # now switch the prefix token from Spanish to French
            >>> tokenizer.set_prefix_tokens(language="french")
            ```

        Args:
            language (`str`, *optional*, defaults to `None`):
                The language of the transcription text.
            task (`str`, *optional*, defaults to `None`):
                Task identifier to append at the start of sequence (if any).
            predict_timestamps (`bool`, *optional*, defaults to `None`):
                Whether to omit the `<|notimestamps|>` token at the start of the sequence.
        """
        self.language = language if language is not None else self.language
        self.task = task if task is not None else self.task
        self.predict_timestamps = predict_timestamps if predict_timestamps is not None else self.predict_timestamps

    @property
    def prefix_tokens(self) -> List[int]:
        """
        This method generates a list of integer tokens representing the prefix sequence for tokenization.

        Args:
            self (WhisperTokenizer): The instance of the WhisperTokenizer class.

        Returns:
            List[int]: A list of integer tokens representing the prefix sequence for tokenization.

        Raises:
            ValueError: If the provided language is unsupported or not found in the language code mapping.
            ValueError: If the provided task is unsupported or not found in the task IDs list.
        """
        bos_token_id = self.convert_tokens_to_ids("<|startoftranscript|>")
        translate_token_id = self.convert_tokens_to_ids("<|translate|>")
        transcribe_token_id = self.convert_tokens_to_ids("<|transcribe|>")
        notimestamps_token_id = self.convert_tokens_to_ids("<|notimestamps|>")
        langs = tuple(LANGUAGES.keys())

        if self.language is not None:
            self.language = self.language.lower()
            if self.language in TO_LANGUAGE_CODE:
                language_id = TO_LANGUAGE_CODE[self.language]
            elif self.language in TO_LANGUAGE_CODE.values():
                language_id = self.language
            else:
                is_language_code = len(self.language) == 2
                raise ValueError(
                    f"Unsupported language: {self.language}. Language should be one of:"
                    f" {list(TO_LANGUAGE_CODE.values()) if is_language_code else list(TO_LANGUAGE_CODE.keys())}."
                )

        if self.task is not None:
            if self.task not in TASK_IDS:
                raise ValueError(f"Unsupported task: {self.task}. Task should be in: {TASK_IDS}")

        bos_sequence = [bos_token_id]
        if self.language is not None:
            bos_sequence.append(bos_token_id + 1 + langs.index(language_id))
        if self.task is not None:
            bos_sequence.append(transcribe_token_id if self.task == "transcribe" else translate_token_id)
        if not self.predict_timestamps:
            bos_sequence.append(notimestamps_token_id)
        return bos_sequence

    # Copied from transformers.models.speech_to_text.tokenization_speech_to_text.Speech2TextTokenizer.build_inputs_with_special_tokens
    def build_inputs_with_special_tokens(self, token_ids_0, token_ids_1=None) -> List[int]:
        """Build model inputs from a sequence by appending eos_token_id."""
        if token_ids_1 is None:
            return self.prefix_tokens + token_ids_0 + [self.eos_token_id]
        # We don't expect to process pairs, but leave the pair logic for API consistency
        return self.prefix_tokens + token_ids_0 + token_ids_1 + [self.eos_token_id]

    # Copied from transformers.models.speech_to_text.tokenization_speech_to_text.Speech2TextTokenizer.get_special_tokens_mask
    def get_special_tokens_mask(
        self, token_ids_0: List[int], token_ids_1: Optional[List[int]] = None, already_has_special_tokens: bool = False
    ) -> List[int]:
        """
        Retrieve sequence ids from a token list that has no special tokens added. This method is called when adding
        special tokens using the tokenizer `prepare_for_model` method.

        Args:
            token_ids_0 (`List[int]`):
                List of IDs.
            token_ids_1 (`List[int]`, *optional*):
                Optional second list of IDs for sequence pairs.
            already_has_special_tokens (`bool`, *optional*, defaults to `False`):
                Whether or not the token list is already formatted with special tokens for the model.

        Returns:
            `List[int]`: A list of integers in the range [0, 1]: 1 for a special token, 0 for a sequence token.
        """
        if already_has_special_tokens:
            return super().get_special_tokens_mask(
                token_ids_0=token_ids_0, token_ids_1=token_ids_1, already_has_special_tokens=True
            )

        prefix_ones = [1] * len(self.prefix_tokens)
        suffix_ones = [1]
        if token_ids_1 is None:
            return prefix_ones + ([0] * len(token_ids_0)) + suffix_ones
        return prefix_ones + ([0] * len(token_ids_0)) + ([0] * len(token_ids_1)) + suffix_ones

    # Copied from transformers.models.gpt2.tokenization_gpt2.GPT2Tokenizer._tokenize with GPT2 -> Whisper
    def _tokenize(self, text):
        """Tokenize a string."""
        bpe_tokens = []
        for token in re.findall(self.pat, text):
            token = "".join(
                self.byte_encoder[b] for b in token.encode("utf-8")
            )  # Maps all our bytes to unicode strings, avoiding control tokens of the BPE (spaces in our case)
            bpe_tokens.extend(bpe_token for bpe_token in self.bpe(token).split(" "))
        return bpe_tokens

    # Copied from transformers.models.gpt2.tokenization_gpt2.GPT2Tokenizer._convert_token_to_id with GPT2 -> Whisper
    def _convert_token_to_id(self, token):
        """Converts a token (str) in an id using the vocab."""
        return self.encoder.get(token, self.encoder.get(self.unk_token))

    def _convert_id_to_token(self, index):
        """
        Converts an index (integer) in a token (str) using the vocab. Whisper's base tokenizer always decodes OOV
        tokens as "", thus we do not use the `unk_token` here.
        """
        return self.decoder.get(index, "")

    def _normalize(self, text):
        """
        Normalize a given string using the `EnglishTextNormalizer` class, which preforms commons transformation on
        english text.
        """
        normalizer = EnglishTextNormalizer(self.english_spelling_normalizer)
        return normalizer(text)

    @staticmethod
    def _basic_normalize(text, remove_diacritics=False):
        """
        Normalize a given string using the `BasicTextNormalizer` class, which preforms commons transformation on
        multilingual text.
        """
        normalizer = BasicTextNormalizer(remove_diacritics=remove_diacritics)
        return normalizer(text)

    def _decode_with_timestamps(self, token_ids, skip_special_tokens=False, time_precision=0.02) -> str:
        """
        Timestamp tokens are above the special tokens' id range and are ignored by `decode()`. This method decodes
        given tokens with timestamps tokens annotated, e.g. "<|1.08|>".
        """
        timestamp_begin = self.all_special_ids[-1] + 1
        outputs = [[]]
        for token in token_ids:
            if token >= timestamp_begin:
                timestamp = f"<|{(token.asnumpy() - timestamp_begin) * time_precision:.2f}|>"
                outputs.append(timestamp)
                outputs.append([])
            else:
                outputs[-1].append(token)
        outputs = [
            s if isinstance(s, str) else self.decode(s, skip_special_tokens=skip_special_tokens) for s in outputs
        ]
        return "".join(outputs)

    def _compute_offsets(self, token_ids, time_precision=0.02):
        """
        Compute offsets for a given tokenized input

        Args:
            token_ids (`Union[int, List[int], np.ndarray, torch.Tensor, tf.Tensor]`):
                List of tokenized input ids. Can be obtained using the `__call__` method.
            time_precision (`float`, `optional`, defaults to 0.02):
                The time ratio to convert from token to time.
        """
        offsets = []
        token_ids = np.array(token_ids)
        if token_ids.shape[0] > 1 and len(token_ids.shape) > 1:
            raise ValueError("Can only process a single input at a time")
        timestamp_begin = self.all_special_ids[-1] + 1
        timestamp_tokens = token_ids >= timestamp_begin

        consecutive = np.where(timestamp_tokens[:-1] & timestamp_tokens[1:])[0] + 1
        if consecutive.shape[0] == 0 and timestamp_tokens.sum() <= 1:
            # either there are no timestamps or there are no consecutive ones
            return []
        if np.where(timestamp_tokens)[0][-1] + 1 not in consecutive:
            # we add the final timestamp if it is not already in the list
            consecutive = np.append(consecutive, np.where(timestamp_tokens)[0][-1] + 1)

        last_slice = np.where(timestamp_tokens)[0][0]
        for current_slice in consecutive:
            sliced_tokens = token_ids[last_slice:current_slice]
            if len(sliced_tokens) > 1:
                start_timestamp_position = sliced_tokens[0].item() - timestamp_begin
                end_timestamp_position = sliced_tokens[-1].item() - timestamp_begin
                # strip timestamp tokens from the text output
                sliced_tokens = self._preprocess_token_ids(sliced_tokens)
                text = self._decode(sliced_tokens)
                text = self._filter_timestamp_ids(text)
                offsets.append(
                    {
                        "text": text,
                        "timestamp": (
                            start_timestamp_position * time_precision,
                            end_timestamp_position * time_precision,
                        ),
                    }
                )
            last_slice = current_slice

        return offsets

    @lru_cache()
    def timestamp_ids(self, time_precision=0.02):
        """
        Compute the timestamp token ids for a given precision and save to least-recently used (LRU) cache.

        Args:
            time_precision (`float`, `optional`, defaults to 0.02):
                The time ratio to convert from token to time.
        """
        return self.convert_tokens_to_ids([("<|%.2f|>" % (i * time_precision)) for i in range(1500 + 1)])

    def _preprocess_token_ids(self, token_ids, skip_special_tokens: bool = False):
        """
        Pre-process the token ids for decoding by removing the prompt tokens ids and timestamp token ids.

        Args:
            token_ids (`Union[int, List[int], np.ndarray, torch.Tensor, tf.Tensor]`):
                List of tokenized input ids. Typically, obtained using the `__call__` method of the tokenizer.
            skip_special_tokens (`bool`, *optional*, defaults to `False`):
                Whether or not to remove special tokens from the token ids. If `True`, the prompt token ids will be
                removed.
        """
        if skip_special_tokens:
            prompt_token_id = self.convert_tokens_to_ids("<|startofprev|>")
            decoder_start_token_id = self.convert_tokens_to_ids("<|startoftranscript|>")
            token_ids = self._strip_prompt(token_ids, prompt_token_id, decoder_start_token_id)

        return token_ids

    def _filter_timestamp_ids(self, token_ids):
        """
        This method removes timestamp patterns from the given token IDs.

        Args:
            self (WhisperTokenizer): The instance of the WhisperTokenizer class.
            token_ids (str): A string containing token IDs with timestamp patterns to be filtered.

        Returns:
            None: This method returns None after removing the timestamp patterns from the token IDs.

        Raises:
            None.
        """
        return re.sub(self.timestamp_pat, "", token_ids)

    def decode(
        self,
        token_ids,
        skip_special_tokens: bool = False,
        clean_up_tokenization_spaces: bool = None,
        output_offsets: bool = False,
        time_precision=0.02,
        decode_with_timestamps: bool = False,
        normalize: bool = False,
        basic_normalize: bool = False,
        remove_diacritics: bool = False,
        **kwargs,
    ) -> str:
        """
        Converts a sequence of ids in a string, using the tokenizer and vocabulary with options to remove special
        tokens and clean up tokenization spaces.

        Similar to doing `self.convert_tokens_to_string(self.convert_ids_to_tokens(token_ids))`.

        Args:
            token_ids (`Union[int, List[int], np.ndarray, torch.Tensor, tf.Tensor]`):
                List of tokenized input ids. Can be obtained using the `__call__` method.
            skip_special_tokens (`bool`, *optional*, defaults to `False`):
                Whether or not to remove special tokens in the decoding.
            clean_up_tokenization_spaces (`bool`, *optional*):
                Whether or not to clean up the tokenization spaces. If `None`, will default to
                `self.clean_up_tokenization_spaces` (available in the `tokenizer_config`).
            output_offsets (`bool`, *optional*, defaults to `False`):
                Whether or not to output the offsets of the tokens. This should only be set if the model predicted
                timestamps.
            time_precision (`float`, `optional`, defaults to 0.02):
                The time ratio to convert from token to time.
            decode_with_timestamps (`bool`, *optional*, defaults to `False`):
                Whether or not to decode with timestamps included in the raw text.
            normalize (`bool`, *optional*, defaults to `False`):
                Whether or not to apply the English text normalizer to the decoded text. Only applicable when the
                target text is in English. Otherwise, the basic text normalizer should be applied.
            basic_normalize (`bool`, *optional*, defaults to `False`):
                Whether or not to apply the Basic text normalizer to the decoded text. Applicable to multilingual
                target text.
            remove_diacritics (`bool`, *optional*, defaults to `False`):
                Whether or not to remove diacritics when applying the Basic text normalizer. Removing diacritics may
                destroy information in the decoded text, hence it should be used with caution.
            kwargs (additional keyword arguments, *optional*):
                Will be passed to the underlying model specific decode method.
        Returns:
            `str`: The decoded sentence.
        """
        filtered_ids = self._preprocess_token_ids(
            token_ids,
            skip_special_tokens=skip_special_tokens,
        )

        text = super().decode(
            filtered_ids,
            skip_special_tokens=skip_special_tokens,
            clean_up_tokenization_spaces=clean_up_tokenization_spaces,
            normalize=normalize,
            basic_normalize=basic_normalize,
            remove_diacritics=remove_diacritics,
            **kwargs,
        )
        if decode_with_timestamps:
            # legacy method to decode timestamps when not included in the tokenizer vocabulary
            text = self._decode_with_timestamps(
                filtered_ids, time_precision=time_precision, skip_special_tokens=skip_special_tokens
            )
        else:
            text = self._filter_timestamp_ids(text)

        # retrieve offsets
        if output_offsets:
            offsets = self._compute_offsets(token_ids, time_precision=time_precision)
            return {"text": text, "offsets": offsets}
        return text

    def _decode(
        self,
        token_ids: Union[int, List[int]],
        skip_special_tokens: bool = False,
        normalize: bool = False,
        basic_normalize: bool = False,
        remove_diacritics: bool = False,
        **kwargs,
    ) -> str:
        """
        Decodes a sequence of token IDs into a string representation.

        Args:
            self (WhisperTokenizer): An instance of the WhisperTokenizer class.
            token_ids (Union[int, List[int]]): The token IDs to be decoded. It can be either a single integer
                or a list of integers.
            skip_special_tokens (bool, optional): Whether to skip special tokens while decoding. Defaults to False.
            normalize (bool, optional): Whether to apply normalization to the decoded text. Defaults to False.
            basic_normalize (bool, optional): Whether to apply basic normalization to the decoded text. Defaults to False.
            remove_diacritics (bool, optional): Whether to remove diacritics from the decoded text. Defaults to False.
            **kwargs: Additional keyword arguments.

        Returns:
            str: The decoded text as a string.

        Raises:
            None.

        Note:
            - If `skip_special_tokens` is True, special tokens will be excluded from the decoded text.
            - The `normalize` parameter takes precedence over the `basic_normalize` parameter.
            - If `normalize` is True, the decoded text will be cleaned using the `_normalize` method of the
            WhisperTokenizer class.
            - If `basic_normalize` is True, the decoded text will be cleaned using the `_basic_normalize` method of
            the WhisperTokenizer class, with an option to remove diacritics.
            - If both `normalize` and `basic_normalize` are False, the decoded text will be the concatenation of
            the tokens without any additional cleaning.

        Example:
            ```python
            >>> tokenizer = WhisperTokenizer()
            >>> tokens = [101, 2023, 2003, 1037, 2307, 1055, 1006, 102]
            >>> decoded_text = tokenizer._decode(tokens, skip_special_tokens=True, normalize=True)
            >>> print(decoded_text)
            'the quick brown fox'
            >>> decoded_text = tokenizer._decode(tokens, skip_special_tokens=True, basic_normalize=True, remove_diacritics=True)
            >>> print(decoded_text)
            'the quick brown fox'
            >>> decoded_text = tokenizer._decode(tokens)
            >>> print(decoded_text)
            '[CLS]thequickbrownfox[SEP]'
            ```
        """
        self._decode_use_source_tokenizer = kwargs.pop("use_source_tokenizer", False)
        filtered_tokens = self.convert_ids_to_tokens(token_ids, skip_special_tokens=skip_special_tokens)

        # To avoid mixing byte-level and unicode for byte-level BPT
        # we need to build string separately for added tokens and byte-level tokens
        # cf. https://github.com/huggingface/transformers/issues/1133
        sub_texts = []
        current_sub_text = []
        for token in filtered_tokens:
            if skip_special_tokens and token in self.all_special_ids:
                continue
            if token in self.added_tokens_encoder:
                if current_sub_text:
                    sub_texts.append(self.convert_tokens_to_string(current_sub_text))
                    current_sub_text = []
                sub_texts.append(token)
            else:
                current_sub_text.append(token)
        if current_sub_text:
            sub_texts.append(self.convert_tokens_to_string(current_sub_text))

        text = "".join(sub_texts)

        if normalize:
            clean_text = self._normalize(text)
            return clean_text
        if basic_normalize:
            clean_text = self._basic_normalize(text, remove_diacritics=remove_diacritics)
            return clean_text
        return text

    # Copied from transformers.models.gpt2.tokenization_gpt2.GPT2Tokenizer.convert_tokens_to_string with GPT2 -> Whisper
    def convert_tokens_to_string(self, tokens):
        """Converts a sequence of tokens (string) in a single string."""
        text = "".join(tokens)
        text = bytearray([self.byte_decoder[c] for c in text]).decode("utf-8", errors=self.errors)
        return text

    def save_vocabulary(self, save_directory: str, filename_prefix: Optional[str] = None) -> Tuple[str]:
        """
        Save the vocabulary to specified directory.

        Args:
            self: Instance of the WhisperTokenizer class.
            save_directory (str): The directory path where the vocabulary files will be saved.
            filename_prefix (Optional[str]): A string prefix to be added to the filenames. Defaults to None.

        Returns:
            Tuple[str]: A tuple containing the paths to the saved vocabulary files - vocab_file, merge_file,
                and normalizer_file.

        Raises:
            FileNotFoundError: If the save_directory does not exist.
            TypeError: If the save_directory is not a valid directory path.
            IOError: If there is an issue with writing the vocabulary files to the specified directory.
        """
        if not os.path.isdir(save_directory):
            logger.error(f"Vocabulary path ({save_directory}) should be a directory")
            return
        vocab_file = os.path.join(
            save_directory, (filename_prefix + "-" if filename_prefix else "") + VOCAB_FILES_NAMES["vocab_file"]
        )
        merge_file = os.path.join(
            save_directory, (filename_prefix + "-" if filename_prefix else "") + VOCAB_FILES_NAMES["merges_file"]
        )
        normalizer_file = os.path.join(
            save_directory, (filename_prefix + "-" if filename_prefix else "") + VOCAB_FILES_NAMES["normalizer_file"]
        )

        with open(vocab_file, "w", encoding="utf-8") as f:
            f.write(json.dumps(self.encoder, indent=2, sort_keys=True, ensure_ascii=False) + "\n")

        index = 0
        with open(merge_file, "w", encoding="utf-8") as writer:
            writer.write("#version: 0.2\n")
            for bpe_tokens, token_index in sorted(self.bpe_ranks.items(), key=lambda kv: kv[1]):
                if index != token_index:
                    logger.warning(
                        f"Saving vocabulary to {merge_file}: BPE merge indices are not consecutive."
                        " Please check that the tokenizer is not corrupted!"
                    )
                    index = token_index
                writer.write(" ".join(bpe_tokens) + "\n")
                index += 1

        if self.english_spelling_normalizer is not None:
            with open(normalizer_file, "w", encoding="utf-8") as f:
                f.write(
                    json.dumps(self.english_spelling_normalizer, indent=2, sort_keys=True, ensure_ascii=False) + "\n"
                )

        return vocab_file, merge_file, normalizer_file

    # Copied from transformers.models.gpt2.tokenization_gpt2.GPT2Tokenizer.prepare_for_tokenization with GPT2 -> Whisper
    def prepare_for_tokenization(self, text, is_split_into_words=False, **kwargs):
        """
        This method prepares the input text for tokenization by potentially adding a prefix space.

        Args:
            self: Reference to the current instance of the class.
            text (str): The input text to be tokenized.
            is_split_into_words (bool): A flag indicating whether the text is already split into words or not.
                Default is False.

        Returns:
            tuple: A tuple containing the modified text and any remaining keyword arguments.

        Raises:
            None
        """
        add_prefix_space = kwargs.pop("add_prefix_space", self.add_prefix_space)
        if is_split_into_words or add_prefix_space:
            text = " " + text
        return (text, kwargs)

    @property
    # Copied from transformers.models.gpt2.tokenization_gpt2.GPT2Tokenizer.default_chat_template
    def default_chat_template(self):
        """
        A simple chat template that ignores role information and just concatenates messages with EOS tokens.
        """
        logger.warning_once(
            "\nNo chat template is defined for this tokenizer - using the default template "
            f"for the {self.__class__.__name__} class. If the default is not appropriate for "
            "your model, please set `tokenizer.chat_template` to an appropriate template. "
            "See https://hf-mirror.com/docs/transformers/main/chat_templating for more information.\n"
        )
        return "{% for message in messages %}" "{{ message.content }}{{ eos_token }}" "{% endfor %}"

    def get_decoder_prompt_ids(self, task=None, language=None, no_timestamps=True):
        """
        Retrieve the decoder prompt IDs for the WhisperTokenizer.

        Args:
            self (WhisperTokenizer): An instance of the WhisperTokenizer class.
            task (str, optional): The task associated with the decoder prompt. Defaults to None.
            language (str, optional): The language associated with the decoder prompt. Defaults to None.
            no_timestamps (bool, optional): Flag indicating whether to exclude timestamps from the decoder prompt.
                Defaults to True.

        Returns:
            list: A list of tuples containing the rank and token of each forced decoder token.

        Raises:
            None.

        This method retrieves the decoder prompt IDs to be used during tokenization. The decoder prompt IDs are based on
        the provided task, language, and timestamp preferences. By default, the method excludes timestamps from the
        decoder prompt. The decoder prompt IDs are returned as a list of tuples, where each tuple consists of the rank
        and token of each forced decoder token.

        Note:
            The decoder prompt IDs exclude the initial token which is reserved for special purposes.
        """
        self.set_prefix_tokens(task=task, language=language, predict_timestamps=not no_timestamps)
        # prefix tokens are of the form: <|startoftranscript|> <|lang_id|> <|task|> <|notimestamps|>
        # we don't want to force the bos token at position 1, as this is the starting token
        # when we generate, so we slice the prefix tokens to: <|lang_id|> <|task|> <|notimestamps|>
        # to get the forced tokens
        forced_tokens = self.prefix_tokens[1:]
        forced_decoder_ids = [(rank + 1, token) for rank, token in enumerate(forced_tokens)]
        return forced_decoder_ids

    def _decode_asr(self, model_outputs, *, return_timestamps, return_language, time_precision):
        """
        This method decodes the output of an Automatic Speech Recognition (ASR) model.
        
        Args:
            self (object): The instance of the WhisperTokenizer class.
            model_outputs (object): The output of the ASR model to be decoded.
            
            return_timestamps (bool): If True, timestamps will be returned along with the decoded output.
                Default is False.
            return_language (bool): If True, the language information will be returned along with the decoded output.
                Default is False.
            time_precision (str): The precision of the timestamps to be returned. Valid values are 'millisecond',
                'second', or 'minute'.
        
        Returns:
            None: This method does not return a value, but the decoded output, timestamps, and language information
                are accessible through other means.
        
        Raises:
            ValueError: If the provided model_outputs is invalid or incompatible with the decoding process.
            RuntimeError: If there is an issue with the decoding process that cannot be handled within the method.
        """
        return _decode_asr(
            self,
            model_outputs,
            return_timestamps=return_timestamps,
            return_language=return_language,
            time_precision=time_precision,
        )

    def get_prompt_ids(self, text: str, return_tensors="np"):
        """Converts prompt text to IDs that can be passed to [`~WhisperForConditionalGeneration.generate`]."""
        batch_encoding = self("<|startofprev|>", " " + text.strip(), add_special_tokens=False)

        # Check for special tokens
        prompt_text_ids = batch_encoding["input_ids"][1:]
        special_token_id = next((x for x in prompt_text_ids if x >= self.all_special_ids[0]), None)
        if special_token_id is not None:
            token = self.convert_ids_to_tokens(special_token_id)
            raise ValueError(f"Encountered text in the prompt corresponding to disallowed special token: {token}.")

        batch_encoding.convert_to_tensors(tensor_type=return_tensors)
        return batch_encoding["input_ids"]

    @staticmethod
    def _strip_prompt(token_ids: List[int], prompt_token_id: int, decoder_start_token_id: int):
        """
        Removes the prompt from a list of token IDs.
        
        Args:
            token_ids (List[int]):
                A list of token IDs.

                - The token IDs can be of any type.
                - The list may contain other elements besides token IDs.
            
            prompt_token_id (int):
                The token ID representing the prompt.

                - The prompt token ID must be an integer.
            
            decoder_start_token_id (int):
                The token ID representing the start of the decoder.

                - The decoder start token ID must be an integer.
            
        Returns:
            None
            
        Raises:
            None
        
        """
        has_prompt = isinstance(token_ids, list) and token_ids and token_ids[0] == prompt_token_id
        if has_prompt:
            if decoder_start_token_id in token_ids:
                return token_ids[token_ids.index(decoder_start_token_id) :]
            return []

        return token_ids


def _decode_asr(tokenizer, model_outputs, *, return_timestamps, return_language, time_precision):
    """
    Internal method meant to only be used by asr pipeline. Handles all the little quirks specific to whisper to handle
    the various options not allowed in other seq2seq models
    """
    # =========== Overview ============
    # - iterate over all outputs
    # - all tokens within output
    # - Each token can be
    #   - language token
    #   - special token
    #   - timestamp token
    #   - text token
    # - We accumulate the text tokens.
    # - We split on end timestamps
    # - Lots of complexity comes from stride and timestamps

    last_language = None

    def new_chunk():
        return {"language": last_language, "timestamp": [None, None], "text": ""}

    # Welcome to the state machine !
    chunks = []
    chunk = new_chunk()
    time_offset = 0.0
    timestamp_begin = tokenizer.convert_tokens_to_ids("<|notimestamps|>") + 1
    previous_tokens = []
    previous_token_timestamps = []
    skip = False
    right_stride_start = None

    all_special_ids = set(tokenizer.all_special_ids)
    # - iterate over all outputs
    for _, output in enumerate(model_outputs):
        # We can drop everything to Python list, it's going to make
        # our lives easier
        token_ids = output["tokens"][0].tolist()
        if return_timestamps == "word":
            token_timestamps = output["token_timestamps"][0].tolist()

        # Those keep track of timestamps within strides
        # Which need to be skipped and resolve all tokens in a single
        # chunk.
        last_timestamp = None
        first_timestamp = timestamp_begin

        if "stride" in output:
            chunk_len, stride_left, stride_right = output["stride"]
            # Offset the timings to account for the other `model_outputs`.
            time_offset -= stride_left
            right_stride_start = chunk_len - stride_right

            # Keeping track of timestamps within strides
            # We're going to NOT split on those, and delay until we're
            # out of BOTH stride. Otherwise lots of issues occur and
            # corner cases
            if stride_left:
                first_timestamp = stride_left / time_precision + timestamp_begin
            if stride_right:
                for token in reversed(token_ids):
                    if token >= timestamp_begin:
                        # There can be several token in the right stride
                        # But the last one is ALWAYS going to be skipped
                        if (
                            last_timestamp is not None
                            and (token - timestamp_begin) * time_precision < right_stride_start
                        ):
                            break
                        last_timestamp = token

        current_tokens = []
        current_token_timestamps = []

        # - all tokens within output
        for i, token in enumerate(token_ids):
            # 4 possible states for each token
            # - 1/ Language code
            # - 2/ all other special tokens (which we ignore)
            # - 3/ Timestamp
            # - 4/ Regular text
            if token in all_special_ids:
                # Either language code or other
                text = tokenizer.decode([token])
                # Removing outer shell <|XX|>
                text = text[2:-2]
                language = LANGUAGES.get(text, None)
                if language is not None:
                    # 1/ Indeed some language
                    # TODO Handle when language is different from the previous
                    # one, and we cannot use timestamped tokens to create chunks
                    if last_language and language != last_language and not return_timestamps:
                        previous_tokens.append(current_tokens)
                        resolved_tokens = _find_longest_common_sequence(previous_tokens)
                        resolved_text = tokenizer.decode(resolved_tokens)
                        chunk["text"] = resolved_text
                        chunks.append(chunk)

                        # Flush all our temporary context
                        previous_tokens = []
                        current_tokens = []
                        chunk = new_chunk()
                    chunk["language"] = language
                    last_language = language
                else:
                    # 2/ This is a regular special token, ignoring it
                    pass
            elif token >= timestamp_begin:
                # 3/ Timestamp token
                time = (token - timestamp_begin) * time_precision + time_offset
                time = round(time, 2)
                if last_timestamp and token >= last_timestamp:
                    # Whisper outputted a timestamp token, but it falls within
                    # our stride, so we're going to skip it for the time being
                    # and resolve this later
                    # Skip is necessary because timestamp tokens always come
                    # by pair, so we need to skip the next one too (which would mark the start of another chunk).
                    skip = True
                elif skip or (previous_tokens and token < first_timestamp):
                    skip = False
                elif chunk["timestamp"][0] is None:
                    chunk["timestamp"][0] = time
                else:
                    # This is the end of the timestamp chunk
                    if time == chunk["timestamp"][0]:
                        # This is a bug in timestamp token output
                        # where we're taking the duplicate token
                        # as a stop where it should be a start.
                        # This is an issue in the underlying model output
                        # Let's just skip it so it becomes de-factor
                        # a start agin
                        pass
                    else:
                        chunk["timestamp"][1] = time
                        # Handling merges.
                        previous_tokens.append(current_tokens)
                        if return_timestamps == "word":
                            previous_token_timestamps.append(current_token_timestamps)
                        resolved_tokens, resolved_token_timestamps = _find_longest_common_sequence(
                            previous_tokens, previous_token_timestamps
                        )
                        resolved_text = tokenizer.decode(resolved_tokens)
                        chunk["text"] = resolved_text
                        if return_timestamps == "word":
                            chunk["words"] = _collate_word_timestamps(
                                tokenizer, resolved_tokens, resolved_token_timestamps, last_language
                            )
                        chunks.append(chunk)

                        # Flush all our temporary context
                        previous_tokens = []
                        current_tokens = []
                        previous_token_timestamps = []
                        current_token_timestamps = []
                        chunk = new_chunk()
            else:
                # 4/ Regular token
                # We just append to the list of all tokens so we can handle
                # merges later and decode into text.
                current_tokens.append(token)
                if return_timestamps == "word":
                    start_time = round(token_timestamps[i] + time_offset, 2)
                    if i + 1 < len(token_timestamps):
                        end_time = round(token_timestamps[i + 1] + time_offset, 2)
                    else:
                        end_time = None  # should never happen
                    current_token_timestamps.append((start_time, end_time))

        if "stride" in output:
            time_offset += chunk_len - stride_right

        # Leftover tokens
        if current_tokens:
            previous_tokens.append(current_tokens)
            if return_timestamps == "word":
                previous_token_timestamps.append(current_token_timestamps)
        elif not any(p for p in previous_tokens):
            chunk = new_chunk()
            previous_tokens = []
            current_tokens = []
            previous_token_timestamps = []
            current_token_timestamps = []

    if previous_tokens:
        if return_timestamps:
            logger.warning(
                "Whisper did not predict an ending timestamp, which can happen if audio is cut off in the middle of a word. "
                "Also make sure WhisperTimeStampLogitsProcessor was used during generation."
            )
        # Happens when we don't use timestamps
        resolved_tokens, resolved_token_timestamps = _find_longest_common_sequence(
            previous_tokens, previous_token_timestamps
        )
        resolved_text = tokenizer.decode(resolved_tokens)
        chunk["text"] = resolved_text
        if return_timestamps == "word":
            chunk["words"] = _collate_word_timestamps(
                tokenizer, resolved_tokens, resolved_token_timestamps, last_language
            )
        chunks.append(chunk)

    # Preparing and cleaning up the pipeline output
    full_text = "".join(chunk["text"] for chunk in chunks)
    if return_timestamps or return_language:
        for chunk in chunks:
            if not return_timestamps:
                chunk.pop("timestamp")
            else:
                chunk["timestamp"] = tuple(chunk["timestamp"])
            if not return_language:
                chunk.pop("language")

        if return_timestamps == "word":
            new_chunks = []
            for chunk in chunks:
                new_chunks.extend(chunk["words"])
            optional = {"chunks": new_chunks}
        else:
            optional = {"chunks": chunks}
    else:
        optional = {}
    return full_text, optional


def _find_longest_common_sequence(sequences, token_timestamp_sequences=None):
    """
    This function finds the longest common sequence between multiple sequences.
    
    Args:
        sequences (list): A list of sequences to compare.
        token_timestamp_sequences (list, optional): A list of token timestamp sequences. Defaults to None.
    
    Returns:
        None or tuple: If `token_timestamp_sequences` is None, returns a list representing the longest common sequence. 
            If `token_timestamp_sequences` is not None, returns a tuple containing the longest common sequence and
            a list of token timestamp sequences.
    
    Raises:
        RuntimeError: If there is a bug within the `whisper decode_asr` function, an error is raised to prevent
            bad inference.
    """
    # It would be much harder to do O(n) because of fault tolerance.
    # We actually have a really good property which is that the total sequence
    # MUST be those subsequences in order.
    # If token_timestamp_sequences is provided, will split those sequences in
    # exactly the same way.

    left_sequence = sequences[0]
    left_length = len(left_sequence)
    total_sequence = []

    if token_timestamp_sequences:
        left_token_timestamp_sequence = token_timestamp_sequences[0]
        total_token_timestamp_sequence = []

    for seq_idx, right_sequence in enumerate(sequences[1:]):
        # index = 0
        max_ = 0.0
        max_indices = (left_length, left_length, 0, 0)
        # Here we're sliding matches
        # [a, b, c, d]
        #          [c, d, f]
        # =        [c] == [d]
        #
        # [a, b, c, d]
        #       [c, d, f]
        # =     [c, d] == [c, d]
        #
        #
        # [a, b, c, d]
        #    [c, d, f]
        #
        # =  [b, c, d] == [c, d, f]
        #
        # [a, b, c, d]
        # [c, d, f]
        #
        # [a, b, c] == [c, d, f]
        #
        # [a, b, c, d]
        # [d, f]
        #
        # [a, b] == [d, f]
        #
        # [a, b, c, d]
        # [f]
        #
        # [a] == [f]
        right_length = len(right_sequence)
        for i in range(1, left_length + right_length):
            # epsilon to favor long perfect matches
            eps = i / 10000.0

            # Slightly convoluted because we don't want out of bound indices
            # This will be necessary for a small conflict resolution optimization
            # later
            left_start = max(0, left_length - i)
            left_stop = min(left_length, left_length + right_length - i)
            left = np.array(left_sequence[left_start:left_stop])

            right_start = max(0, i - left_length)
            right_stop = min(right_length, i)
            right = np.array(right_sequence[right_start:right_stop])

            # We can only match subsequences of the same size.
            if len(left) != len(right):
                raise RuntimeError(
                    "There is a bug within whisper `decode_asr` function, please report it. Dropping to prevent bad inference."
                )

            matches = np.sum(left == right)
            matching = matches / i + eps
            if matches > 1 and matching > max_:
                max_ = matching
                max_indices = (left_start, left_stop, right_start, right_stop)

        (left_start, left_stop, right_start, right_stop) = max_indices

        # This is a small conflict optimization since those sequences overlap
        # in audio.
        # We're going to give more confidence to the left sequence
        # for the left of the overlap,
        # and to the right of the sequence, for the right of the overlap
        left_mid = (left_stop + left_start) // 2
        right_mid = (right_stop + right_start) // 2
        total_sequence.extend(left_sequence[:left_mid])
        left_sequence = right_sequence[right_mid:]
        left_length = len(left_sequence)

        if token_timestamp_sequences:
            total_token_timestamp_sequence.extend(left_token_timestamp_sequence[:left_mid])
            left_token_timestamp_sequence = token_timestamp_sequences[seq_idx + 1][right_mid:]

    total_sequence.extend(left_sequence)

    if token_timestamp_sequences is None:
        return total_sequence

    if len(token_timestamp_sequences) > 0:
        total_token_timestamp_sequence.extend(left_token_timestamp_sequence)
        return total_sequence, total_token_timestamp_sequence
    return total_sequence, []


def _collate_word_timestamps(tokenizer, tokens, token_timestamps, language):
    """
    Collates the word timestamps based on the given tokenizer, tokens, token timestamps, and language.
    
    Args:
        tokenizer (object): The tokenizer object used for tokenizing the input.
        tokens (list): The list of tokens to be processed.
        token_timestamps (list): The list of token timestamps corresponding to the tokens.
        language (str): The language used for tokenization.
    
    Returns:
        list: A list of dictionaries containing the text and corresponding timestamp of each word.
    
    Raises:
        None.
    
    """
    words, _, token_indices = _combine_tokens_into_words(tokenizer, tokens, language)
    timings = [
        {
            "text": word,
            "timestamp": (token_timestamps[indices[0]][0], token_timestamps[indices[-1]][1]),
        }
        for word, indices in zip(words, token_indices)
    ]
    return timings


def _combine_tokens_into_words(
    tokenizer,
    tokens: List[int],
    language: str = None,
    prepend_punctuations: str = "\"'“¡¿([{-",
    append_punctuations: str = "\"'.。,，!！?？:：”)]}、",
):
    """
    Groups tokens by word. Returns a tuple containing a list of strings with the words, and a list of `token_id`
    sequences with the tokens making up each word.
    """
    if language is None:
        language = tokenizer.language
    if language is None:
        language = "english"

    if language in {"chinese", "japanese", "thai", "lao", "myanmar", "cantonese"}:
        # These languages don't typically use spaces.
        words, word_tokens, token_indices = _split_tokens_on_unicode(tokenizer, tokens)
    else:
        words, word_tokens, token_indices = _split_tokens_on_spaces(tokenizer, tokens)

    _merge_punctuations(words, word_tokens, token_indices, prepend_punctuations, append_punctuations)
    return words, word_tokens, token_indices


def _split_tokens_on_unicode(tokenizer, tokens: List[int]):
    """Combine tokens into words by splitting at any position where the tokens are decoded as valid unicode points."""
    decoded_full = tokenizer.decode(tokens, decode_with_timestamps=True)
    replacement_char = "\ufffd"

    words = []
    word_tokens = []
    token_indices = []
    current_tokens = []
    current_indices = []
    unicode_offset = 0

    for token_idx, token in enumerate(tokens):
        current_tokens.append(token)
        current_indices.append(token_idx)
        decoded = tokenizer.decode(current_tokens, decode_with_timestamps=True)

        if (
            replacement_char not in decoded
            or decoded_full[unicode_offset + decoded.index(replacement_char)] == replacement_char
        ):
            words.append(decoded)
            word_tokens.append(current_tokens)
            token_indices.append(current_indices)
            current_tokens = []
            current_indices = []
            unicode_offset += len(decoded)

    return words, word_tokens, token_indices


def _split_tokens_on_spaces(tokenizer, tokens: List[int]):
    """Combine tokens into words by splitting at whitespace and punctuation tokens."""
    subwords, subword_tokens_list, subword_indices_list = _split_tokens_on_unicode(tokenizer, tokens)
    words = []
    word_tokens = []
    token_indices = []

    for subword, subword_tokens, subword_indices in zip(subwords, subword_tokens_list, subword_indices_list):
        special = subword_tokens[0] >= tokenizer.eos_token_id
        with_space = subword.startswith(" ")
        punctuation = subword.strip() in "!\"#$%&'()*+,-./:;<=>?@[\\]^_`{|}~"

        if special or with_space or punctuation or len(words) == 0:
            words.append(subword)
            word_tokens.append(subword_tokens)
            token_indices.append(subword_indices)
        else:
            words[-1] = words[-1] + subword
            word_tokens[-1].extend(subword_tokens)
            token_indices[-1].extend(subword_indices)

    return words, word_tokens, token_indices


def _merge_punctuations(words, tokens, indices, prepended, appended):
    """Merges punctuation tokens with neighboring words."""
    # prepend punctuations
    i = len(words) - 2
    j = len(words) - 1
    while i >= 0:
        if words[i].startswith(" ") and words[i].strip() in prepended:
            words[j] = words[i] + words[j]
            tokens[j] = tokens[i] + tokens[j]
            indices[j] = indices[i] + indices[j]
            words[i] = ""
            tokens[i] = []
            indices[i] = []
        else:
            j = i
        i -= 1

    # append punctuations
    i = 0
    j = 1
    while j < len(words):
        if not words[i].endswith(" ") and words[j] in appended:
            words[i] += words[j]
            tokens[i] += tokens[j]
            indices[i] += indices[j]
            words[j] = ""
            tokens[j] = []
            indices[j] = []
        else:
            i = j
        j += 1

    # remove elements that are now empty
    words[:] = [word for word in words if word]
    tokens[:] = [token for token in tokens if token]
    indices[:] = [idx for idx in indices if idx]

__all__ = ['WhisperTokenizer']
