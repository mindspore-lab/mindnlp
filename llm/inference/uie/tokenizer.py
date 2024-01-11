
import collections
import json
import os
import unicodedata
from shutil import copyfile
from typing import Any, Dict, List, Optional, Tuple

import sentencepiece as spm
from mindnlp.utils import requires_backends
from mindnlp.transformers import PreTrainedTokenizerFast
from mindnlp.transformers.convert_slow_tokenizer import Converter, SentencePieceExtractor, SLOW_TO_FAST_CONVERTERS

from fast_tokenizer import Tokenizer, normalizers, pretokenizers, postprocessors
from fast_tokenizer.models import BPE, Unigram
from mindnlp.transformers.tokenization_utils import PreTrainedTokenizer

from utils import logger

SPIECE_UNDERLINE = "▁"

VOCAB_FILES_NAMES = {
    "sentencepiece_model_file": "sentencepiece.bpe.model",
    "vocab_file": "vocab.txt",
}

PRETRAINED_VOCAB_FILES_MAP = {
    "vocab_file": {
        "ernie-m-base": "https://bj.bcebos.com/paddlenlp/models/transformers/ernie_m/ernie_m.vocab.txt",
        "ernie-m-large": "https://bj.bcebos.com/paddlenlp/models/transformers/ernie_m/ernie_m.vocab.txt"
    },
    "sentencepiece_model_file": {
        "ernie-m-base": "https://bj.bcebos.com/paddlenlp/models/transformers/ernie_m/ernie_m.sentencepiece.bpe.model",
        "ernie-m-large": "https://bj.bcebos.com/paddlenlp/models/transformers/ernie_m/ernie_m.sentencepiece.bpe.model",
    }
}

PRETRAINED_POSITIONAL_EMBEDDINGS_SIZES = {
    "ernie-m-base": 514,
    "ernie-m-large": 514,
}


def load_vocab(vocab_file):
    """Loads a vocabulary file into a dictionary."""
    vocab = collections.OrderedDict()
    with open(vocab_file, "r", encoding="utf-8") as reader:
        tokens = reader.readlines()
    for index, token in enumerate(tokens):
        token = token.rstrip("\n")
        if token in vocab:
            print(f'{token} 重复！')
        vocab[token] = index
    return vocab


class ErnieMTokenizer(PreTrainedTokenizer):
    """
    Construct an Erine-M tokenizer. Based on [SentencePiece](https://github.com/google/sentencepiece).

    This tokenizer inherits from [`PreTrainedTokenizer`] which contains most of the main methods. Users should refer to
    this superclass for more information regarding those methods.

    Args:
        vocab_file (`str`):
            File containing the vocabulary.
        sentencepiece_model_file (`str`):
            [SentencePiece](https://github.com/google/sentencepiece) file (generally has a .spm extension) that
            contains the vocabulary necessary to instantiate a tokenizer.
        do_lower_case (`bool`, *optional*, defaults to `True`):
            Whether to lowercase the input when tokenizing.
        unk_token (`str`, *optional*, defaults to `"<unk>"`):
            The unknown token. A token that is not in the vocabulary cannot be converted to an ID and is set to be this
            token instead.
        sep_token (`str`, *optional*, defaults to `"<sep>"`):
            The separator token, which is used when building a sequence from multiple sequences, e.g. two sequences for
            sequence classification or for a text and a question for question answering. It is also used as the last
            token of a sequence built with special tokens.
        pad_token (`str`, *optional*, defaults to `"<pad>"`):
            The token used for padding, for example when batching sequences of different lengths.
        cls_token (`str`, *optional*, defaults to `"<cls>"`):
            The classifier token which is used when doing sequence classification (classification of the whole sequence
            instead of per-token classification). It is the first token of the sequence when built with special tokens.
        mask_token (`str`, *optional*, defaults to `"<mask>"`):
            The token used for masking values. This is the token used when training this model with masked language
            modeling. This is the token which the model will try to predict.
        sp_model_kwargs (`dict`, *optional*):
            Will be passed to the `SentencePieceProcessor.__init__()` method. The [Python wrapper for
            SentencePiece](https://github.com/google/sentencepiece/tree/master/python) can be used, among other things,
            to set:

            - `enable_sampling`: Enable subword regularization.
            - `nbest_size`: Sampling parameters for unigram. Invalid for BPE-Dropout.

              - `nbest_size = {0,1}`: No sampling is performed.
              - `nbest_size > 1`: samples from the nbest_size results.
              - `nbest_size < 0`: assuming that nbest_size is infinite and samples from the all hypothesis (lattice)
                using forward-filtering-and-backward-sampling algorithm.

            - `alpha`: Smoothing parameter for unigram sampling, and dropout probability of merge operations for
              BPE-dropout.

    Attributes:
        sp_model (`SentencePieceProcessor`):
            The *SentencePiece* processor that is used for every conversion (string, tokens and IDs).
    """

    vocab_files_names = VOCAB_FILES_NAMES
    pretrained_vocab_files_map = PRETRAINED_VOCAB_FILES_MAP
    max_model_input_sizes = PRETRAINED_POSITIONAL_EMBEDDINGS_SIZES
    padding_side = "left"

    def __init__(
        self,
        vocab_file,
        sentencepiece_model_file,
        do_lower_case=False,
        unk_token="[UNK]",
        sep_token="[SEP]",
        pad_token="[PAD]",
        cls_token="[CLS]",
        mask_token="[MASK]",
        sp_model_kwargs: Optional[Dict[str, Any]] = None,
        **kwargs
    ) -> None:

        # Mask token behave like a normal word, i.e. include the space before it
        # mask_token = AddedToken(mask_token, lstrip=True, rstrip=False) if isinstance(
        #     mask_token, str) else mask_token

        self.sp_model_kwargs = {} if sp_model_kwargs is None else sp_model_kwargs
        self.vocab = load_vocab(vocab_file)

        super().__init__(
            do_lower_case=do_lower_case,
            unk_token=unk_token,
            sep_token=sep_token,
            pad_token=pad_token,
            cls_token=cls_token,
            mask_token=mask_token,
            sp_model_kwargs=self.sp_model_kwargs,
            **kwargs,
        )

        self.do_lower_case = do_lower_case
        self.sentencepiece_model_file = sentencepiece_model_file
        if not os.path.isfile(sentencepiece_model_file):
            raise ValueError(
                f"Can't find a vocabulary file at path '{sentencepiece_model_file}'. To load the vocabulary from a Google pretrained "
                "model use `tokenizer = ErnieMTokenizer.from_pretrained(PRETRAINED_MODEL_NAME)`"
            )

        self.sp_model = spm.SentencePieceProcessor(**self.sp_model_kwargs)
        self.sp_model.Load(sentencepiece_model_file)

        if not os.path.isfile(vocab_file):
            raise ValueError(
                f"Can't find a vocabulary file at path '{vocab_file}'. To load the vocabulary from a Google pretrained "
                "model use `tokenizer = ErnieMTokenizer.from_pretrained(PRETRAINED_MODEL_NAME)`"
            )
        self.ids_to_tokens = collections.OrderedDict(
            [(ids, tok) for tok, ids in self.vocab.items()])

        self.SP_CHAR_MAPPING = {}

        for ch in range(65281, 65375):
            if ch in [ord(u'～')]:
                self.SP_CHAR_MAPPING[chr(ch)] = chr(ch)
                continue
            self.SP_CHAR_MAPPING[chr(ch)] = chr(ch - 65248)

    @property
    def vocab_size(self):
        return len(self.sp_model)

    def get_vocab(self):
        return dict(self.vocab, **self.added_tokens_encoder)

    def __getstate__(self):
        state = self.__dict__.copy()
        state["sp_model"] = None
        return state

    def __setstate__(self, d):
        self.__dict__ = d

        # for backward compatibility
        if not hasattr(self, "sp_model_kwargs"):
            self.sp_model_kwargs = {}

        self.sp_model = spm.SentencePieceProcessor(**self.sp_model_kwargs)
        self.sp_model.Load(self.vocab_file)

    def preprocess_text(self, inputs):
        outputs = ''.join((self.SP_CHAR_MAPPING.get(c, c) for c in inputs))
        outputs = outputs.replace("``", '"').replace("''", '"')

        outputs = unicodedata.normalize("NFKD", outputs)
        outputs = "".join(
            [c for c in outputs if not unicodedata.combining(c)])
        if self.do_lower_case:
            outputs = outputs.lower()

        return outputs

    def _tokenize(self, text: str) -> List[str]:
        """Tokenize a string."""
        text = self.preprocess_text(text)

        pieces = self.sp_model.EncodeAsPieces(text)

        new_pieces = []
        for piece in pieces:
            if piece == SPIECE_UNDERLINE:
                continue
            lst_i = 0
            for i, c in enumerate(piece):
                if c == SPIECE_UNDERLINE:
                    continue
                if self.is_ch_char(c) or self.is_punct(c):
                    if i > lst_i and piece[lst_i:i] != SPIECE_UNDERLINE:
                        new_pieces.append(piece[lst_i:i])
                    new_pieces.append(c)
                    lst_i = i + 1
                elif c.isdigit() and i > 0 and not piece[i - 1].isdigit():
                    if i > lst_i and piece[lst_i:i] != SPIECE_UNDERLINE:
                        new_pieces.append(piece[lst_i:i])
                    lst_i = i
                elif not c.isdigit() and i > 0 and piece[i - 1].isdigit():
                    if i > lst_i and piece[lst_i:i] != SPIECE_UNDERLINE:
                        new_pieces.append(piece[lst_i:i])
                    lst_i = i
            if len(piece) > lst_i:
                new_pieces.append(piece[lst_i:])

        return new_pieces

    def _convert_token_to_id(self, token):
        """Converts a token (str) in an id using the vocab."""

        return self.vocab.get(token, self.vocab.get(self.unk_token))

    def _convert_id_to_token(self, index):
        """Converts an index (integer) in a token (str) using the vocab."""
        return self.ids_to_tokens.get(index, self.unk_token)

    def convert_tokens_to_string(self, tokens):
        """Converts a sequence of tokens (strings for sub-words) in a single string."""
        out_string = "".join(tokens).replace(SPIECE_UNDERLINE, " ").strip()
        return out_string

    def build_inputs_with_special_tokens(
        self, token_ids_0: List[int], token_ids_1: Optional[List[int]] = None
    ) -> List[int]:
        """
        Build model inputs from a sequence or a pair of sequence for sequence classification tasks by concatenating and
        adding special tokens. An Erine-M sequence has the following format:

        - single sequence: `X <sep> <cls>`
        - pair of sequences: `A <sep> B <sep> <cls>`

        Args:
            token_ids_0 (`List[int]`):
                List of IDs to which the special tokens will be added.
            token_ids_1 (`List[int]`, *optional*):
                Optional second list of IDs for sequence pairs.

        Returns:
            `List[int]`: List of [input IDs](../glossary#input-ids) with the appropriate special tokens.
        """
        sep = [self.sep_token_id]
        cls = [self.cls_token_id]

        if token_ids_1 is None:
            return cls + token_ids_0 + sep

        return cls + token_ids_0 + sep + sep + token_ids_1 + sep

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

        if token_ids_1 is not None:
            return [1] + ([0] * len(token_ids_0)) + [1, 1] + ([0] * len(token_ids_1)) + [1]
        return [1] + ([0] * len(token_ids_0)) + [1]

    def create_token_type_ids_from_sequences(
        self, token_ids_0: List[int], token_ids_1: Optional[List[int]] = None
    ) -> List[int]:
        """
        Create a mask from the two sequences passed to be used in a sequence-pair classification task. An Erine-M
        sequence pair mask has the following format:

        ```
        0 0 0 0 0 0 0 0 0 0 0 1 1 1 1 1 1 1 1 1
        | first sequence    | second sequence |
        ```

        If `token_ids_1` is `None`, this method only returns the first portion of the mask (0s).

        Args:
            token_ids_0 (`List[int]`):
                List of IDs.
            token_ids_1 (`List[int]`, *optional*):
                Optional second list of IDs for sequence pairs.

        Returns:
            `List[int]`: List of [token type IDs](../glossary#token-type-ids) according to the given sequence(s).
        """

        if token_ids_1 is None:
            # [CLS] X [SEP]
            return (len(token_ids_0) + 2) * [0]

        # [CLS] A [SEP] [SEP] B [SEP]
        return [0] * (len(token_ids_0) + 1) + [1] * (len(token_ids_1) + 3)

    def save_vocabulary(self, save_directory: str, filename_prefix: Optional[str] = None) -> Tuple[str]:
        if not os.path.isdir(save_directory):
            logger.error(
                f"Vocabulary path ({save_directory}) should be a directory")
            return
        sentencepiece_model_file = os.path.join(
            save_directory, (filename_prefix + "-" if filename_prefix else "") +
            VOCAB_FILES_NAMES["sentencepiece_model_file"]
        )
        vocab_file = (filename_prefix +
                      "-" if filename_prefix else "") + save_directory

        if os.path.abspath(self.vocab_file) != os.path.abspath(sentencepiece_model_file) and os.path.isfile(self.vocab_file):
            copyfile(self.vocab_file, sentencepiece_model_file)
        elif not os.path.isfile(self.vocab_file):
            with open(sentencepiece_model_file, "wb") as fi:
                content_spiece_model = self.sp_model.serialized_model_proto()
                fi.write(content_spiece_model)

        index = 0
        with open(vocab_file, "w", encoding="utf-8") as writer:
            for token, token_index in sorted(self.vocab.items(), key=lambda kv: kv[1]):
                if index != token_index:
                    logger.warning(
                        f"Saving vocabulary to {vocab_file}: vocabulary indices are not consecutive."
                        " Please check that the vocabulary is not corrupted!"
                    )
                    index = token_index
                writer.write(token + "\n")
                index += 1
        return vocab_file, sentencepiece_model_file

    def is_ch_char(self, char):
        """
        is_ch_char
        """
        if u'\u4e00' <= char <= u'\u9fff':
            return True
        return False

    def is_alpha(self, char):
        """
        is_alpha
        """
        if 'a' <= char <= 'z':
            return True
        if 'A' <= char <= 'Z':
            return True
        return False

    def is_punct(self, char):
        """
        is_punct
        """
        if char in u",;:.?!~，；：。？！《》【】":
            return True
        return False

    def is_whitespace(self, char):
        """
        is whitespace
        """
        if char == " " or char == "\t" or char == "\n" or char == "\r":
            return True
        if len(char) == 1:
            cat = unicodedata.category(char)
            if cat == "Zs":
                return True
        return False


class ErnieMTokenizerFast(PreTrainedTokenizerFast):
    r"""
    Construct a "fast" ERNIE-M tokenizer (backed by HuggingFace's *tokenizers* library). Based on WordPiece.
    This tokenizer inherits from [`PreTrainedTokenizerFast`] which contains most of the main methods. Users should
    refer to this superclass for more information regarding those methods.
    Args:
        vocab_file (`str`):
            File containing the vocabulary.
        sentencepiece_model_file (`str`):
            [SentencePiece](https://github.com/google/sentencepiece) file (generally has a .spm extension) that
            contains the vocabulary necessary to instantiate a tokenizer.
        do_lower_case (`bool`, *optional*, defaults to `True`):
            Whether or not to lowercase the input when tokenizing.
        unk_token (`str`, *optional*, defaults to `"[UNK]"`):
            The unknown token. A token that is not in the vocabulary cannot be converted to an ID and is set to be this
            token instead.
        sep_token (`str`, *optional*, defaults to `"[SEP]"`):
            The separator token, which is used when building a sequence from multiple sequences, e.g. two sequences for
            sequence classification or for a text and a question for question answering. It is also used as the last
            token of a sequence built with special tokens.
        pad_token (`str`, *optional*, defaults to `"[PAD]"`):
            The token used for padding, for example when batching sequences of different lengths.
        cls_token (`str`, *optional*, defaults to `"[CLS]"`):
            The classifier token which is used when doing sequence classification (classification of the whole sequence
            instead of per-token classification). It is the first token of the sequence when built with special tokens.
        mask_token (`str`, *optional*, defaults to `"[MASK]"`):
            The token used for masking values. This is the token used when training this model with masked language
            modeling. This is the token which the model will try to predict.
        clean_text (`bool`, *optional*, defaults to `True`):
            Whether or not to clean the text before tokenization by removing any control characters and replacing all
            whitespaces by the classic one.
        tokenize_chinese_chars (`bool`, *optional*, defaults to `True`):
            Whether or not to tokenize Chinese characters. This should likely be deactivated for Japanese (see [this
            issue](https://github.com/huggingface/transformers/issues/328)).
        strip_accents (`bool`, *optional*):
            Whether or not to strip all accents. If this option is not specified, then it will be determined by the
            value for `lowercase` (as in the original ERNIE-M).
        wordpieces_prefix (`str`, *optional*, defaults to `"##"`):
            The prefix for subwords.
    """

    vocab_files_names = VOCAB_FILES_NAMES
    pretrained_vocab_files_map = PRETRAINED_VOCAB_FILES_MAP
    max_model_input_sizes = PRETRAINED_POSITIONAL_EMBEDDINGS_SIZES
    slow_tokenizer_class = ErnieMTokenizer

    def __init__(
        self,
        vocab_file=None,
        sentencepiece_model_file=None,
        tokenizer_file=None,
        do_lower_case=True,
        unk_token="[UNK]",
        sep_token="[SEP]",
        pad_token="[PAD]",
        cls_token="[CLS]",
        mask_token="[MASK]",
        tokenize_chinese_chars=True,
        strip_accents=None,
        **kwargs
    ):
        super().__init__(
            vocab_file,
            sentencepiece_model_file,
            tokenizer_file=tokenizer_file,
            do_lower_case=do_lower_case,
            unk_token=unk_token,
            sep_token=sep_token,
            pad_token=pad_token,
            cls_token=cls_token,
            mask_token=mask_token,
            tokenize_chinese_chars=tokenize_chinese_chars,
            strip_accents=strip_accents,
            **kwargs,
        )

        normalizer_state = json.loads(
            self.backend_tokenizer.normalizer.__getstate__())
        if (
            normalizer_state.get("lowercase", do_lower_case) != do_lower_case
            or normalizer_state.get("strip_accents", strip_accents) != strip_accents
            or normalizer_state.get("handle_chinese_chars", tokenize_chinese_chars) != tokenize_chinese_chars
        ):
            normalizer_class = getattr(
                normalizers, normalizer_state.pop("type"))
            normalizer_state["lowercase"] = do_lower_case
            normalizer_state["strip_accents"] = strip_accents
            normalizer_state["handle_chinese_chars"] = tokenize_chinese_chars
            self.backend_tokenizer.normalizer = normalizer_class(
                **normalizer_state)

        self.do_lower_case = do_lower_case

    def build_inputs_with_special_tokens(self, token_ids_0, token_ids_1=None):
        """
        Build model inputs from a sequence or a pair of sequence for sequence classification tasks by concatenating and
        adding special tokens. A ERNIE-M sequence has the following format:
        - single sequence: `[CLS] X [SEP]`
        - pair of sequences: `[CLS] A [SEP] B [SEP]`
        Args:
            token_ids_0 (`List[int]`):
                List of IDs to which the special tokens will be added.
            token_ids_1 (`List[int]`, *optional*):
                Optional second list of IDs for sequence pairs.
        Returns:
            `List[int]`: List of [input IDs](../glossary#input-ids) with the appropriate special tokens.
        """
        output = [self.cls_token_id] + token_ids_0 + [self.sep_token_id]

        if token_ids_1:
            output += [self.sep_token_id] + token_ids_1 + [self.sep_token_id]

        return output

    def create_token_type_ids_from_sequences(
        self, token_ids_0: List[int], token_ids_1: Optional[List[int]] = None
    ) -> List[int]:
        """
        Create a mask from the two sequences passed to be used in a sequence-pair classification task. A ERNIE-M sequence
        pair mask has the following format:
        ```
        0 0 0 0 0 0 0 0 0 0 0 1 1 1 1 1 1 1 1 1
        | first sequence    | second sequence |
        ```
        If `token_ids_1` is `None`, this method only returns the first portion of the mask (0s).
        Args:
            token_ids_0 (`List[int]`):
                List of IDs.
            token_ids_1 (`List[int]`, *optional*):
                Optional second list of IDs for sequence pairs.
        Returns:
            `List[int]`: List of [token type IDs](../glossary#token-type-ids) according to the given sequence(s).
        """

        if token_ids_1 is None:
            return (len(token_ids_0) + 2) * [0]
        return [0] * (len(token_ids_0) + 1) + [1] * (len(token_ids_1) + 3)

    def save_vocabulary(self, save_directory: str, filename_prefix: Optional[str] = None) -> Tuple[str]:
        files = self._tokenizer.model.save(
            save_directory, name=filename_prefix)
        return tuple(files)

    @property
    def added_tokens_encoder(self) -> Dict[str, int]:
        """
        Returns the sorted mapping from string to index. The added tokens encoder is cached for performance
        optimisation in `self._added_tokens_encoder` for the slow tokenizers.
        """
        # return {k.content: v for v, k in sorted(self._tokenizer.get_vocab().items(), key=lambda item: item[0])}
        return self._tokenizer.get_vocab()

class TokenizerProxy:
    def __init__(self, tokenizer):
        self._tokenizer = tokenizer
        self.no_padding = self._tokenizer.disable_padding
        self.no_truncation = self._tokenizer.disable_truncation

    def __getattr__(self, __name: str) -> Any:
        attr = getattr(self._tokenizer, __name)
        if __name == 'padding':
            attr['pad_type_id'] = attr['pad_token_type_id']
            del attr['pad_token_type_id']
        return attr


class ErnieMConverter(Converter):
    def __init__(self, *args):
        requires_backends(self, "protobuf")

        super().__init__(*args)

        from sentencepiece import sentencepiece_model_pb2 as model_pb2

        m = model_pb2.ModelProto()
        with open(self.original_tokenizer.sentencepiece_model_file, "rb") as f:
            m.ParseFromString(f.read())
        self.proto = m

    def vocab(self, proto):
        word_score_dict = {}
        for piece in proto.pieces:
            word_score_dict[piece.piece] = piece.score
        vocab_list = [None] * len(self.original_tokenizer.ids_to_tokens)
        original_vocab = self.original_tokenizer.vocab
        for _token, _id in original_vocab.items():
            if _token in word_score_dict:
                vocab_list[_id] = (_token, word_score_dict[_token])
            else:
                vocab_list[_id] = (_token, 0.0)
        return vocab_list

    def post_processor(self):
        '''
         An ERNIE-M sequence has the following format:
        - single sequence:       ``[CLS] X [SEP]``
        - pair of sequences:        ``[CLS] A [SEP] [SEP] B [SEP]``
        '''
        return postprocessors.TemplatePostProcessor(
            single="[CLS]:0 $A:0 [SEP]:0",
            pair="[CLS]:0 $A:0 [SEP]:0 [SEP]:1 $B:1 [SEP]:1",
            special_tokens=[
                ("[CLS]",
                 self.original_tokenizer.convert_tokens_to_ids("[CLS]")),
                ("[SEP]",
                 self.original_tokenizer.convert_tokens_to_ids("[SEP]")),
            ],
        )

    def normalizer(self, proto):
        list_normalizers = []
        precompiled_charsmap = proto.normalizer_spec.precompiled_charsmap
        list_normalizers.append(
            normalizers.PrecompiledNormalizer(precompiled_charsmap))
        return normalizers.SequenceNormalizer(list_normalizers)

    def unk_id(self, proto):
        return self.original_tokenizer.convert_tokens_to_ids(
            str(self.original_tokenizer.unk_token))

    def pre_tokenizer(self, replacement, add_prefix_space):
        return pretokenizers.SequencePreTokenizer([
            pretokenizers.WhitespacePreTokenizer(),
            pretokenizers.MetaSpacePreTokenizer(
                replacement=replacement, add_prefix_space=add_prefix_space)
        ])

    def converted(self) -> Tokenizer:
        tokenizer = self.tokenizer(self.proto)

        SPLICE_UNDERLINE = SPIECE_UNDERLINE
        tokenizer.model.set_filter_token(SPLICE_UNDERLINE)
        chinese_chars = r"\x{4e00}-\x{9fff}"
        punc_chars = r",;:.?!~，；：。？！《》【】"
        digits = r"0-9"
        tokenizer.model.set_split_rule(
            fr"[{chinese_chars}]|[{punc_chars}]|[{digits}]+|[^{chinese_chars}{punc_chars}{digits}]+"
        )

        # Tokenizer assemble
        tokenizer.normalizer = self.normalizer(self.proto)

        replacement = "▁"
        add_prefix_space = True
        tokenizer.pretokenizer = self.pre_tokenizer(
            replacement, add_prefix_space)

        post_processor = self.post_processor()
        if post_processor:
            tokenizer.postprocessor = post_processor

        tokenizer = TokenizerProxy(tokenizer)
        return tokenizer

    def tokenizer(self, proto):
        model_type = proto.trainer_spec.model_type
        vocab = self.vocab(proto)
        unk_id = self.unk_id(proto)

        if model_type == 1:
            tokenizer = Tokenizer(Unigram(vocab, unk_id))
        elif model_type == 2:
            _, merges = SentencePieceExtractor(
                self.original_tokenizer.sentencepiece_model_file).extract()
            bpe_vocab = {word: i for i, (word, score) in enumerate(vocab)}
            tokenizer = Tokenizer(
                BPE(
                    bpe_vocab,
                    merges,
                    unk_token=proto.trainer_spec.unk_piece,
                    fuse_unk=True,
                )
            )
        else:
            raise Exception(
                "You're trying to run a `Unigram` model but you're file was trained with a different algorithm"
            )

        return tokenizer


SLOW_TO_FAST_CONVERTERS["ErnieMTokenizer"] = ErnieMConverter
