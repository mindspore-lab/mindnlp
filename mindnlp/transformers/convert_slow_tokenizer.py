# coding=utf-8
# Copyright 2018 The HuggingFace Inc. team.
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
# pylint: disable=import-outside-toplevel
"""
Utilities to convert slow tokenizers in their fast tokenizers counterparts.

All the conversions are grouped here to gather SentencePiece dependencies outside of the fast tokenizers files and
allow to make our dependency on SentencePiece optional.
"""

import warnings
from typing import Dict, List, Tuple

from packaging import version
from tokenizers import AddedToken, Regex, Tokenizer, decoders, normalizers, pre_tokenizers, processors
from tokenizers.models import BPE, Unigram, WordPiece

from mindnlp.utils import is_protobuf_available, requires_backends
from mindnlp.utils.import_utils import PROTOBUF_IMPORT_ERROR


def import_protobuf(error_message=""):
    """
    Imports the protobuf module and returns the sentencepiece_model_pb2 if available.
    
    Args:
        error_message (str): An optional error message to include in the ImportError raised if protobuf is not available.
    
    Returns:
        sentencepiece_model_pb2: The protobuf module for sentencepiece, if available. Returns None if not available.
    
    Raises:
        ImportError: If protobuf is not available, an ImportError is raised with the specified error message.
    """
    if is_protobuf_available():
        from sentencepiece import sentencepiece_model_pb2
        return sentencepiece_model_pb2
    raise ImportError(PROTOBUF_IMPORT_ERROR.format(error_message))

def _get_prepend_scheme(add_prefix_space: bool, original_tokenizer) -> str:
    if add_prefix_space:
        prepend_scheme = "always"
        if not getattr(original_tokenizer, "legacy", True):
            prepend_scheme = "first"
    else:
        prepend_scheme = "never"
    return prepend_scheme

class SentencePieceExtractor:
    """
    Extractor implementation for SentencePiece trained models. https://github.com/google/sentencepiece
    """
    def __init__(self, model: str):
        """
        Initializes a new instance of the SentencePieceExtractor class.
        
        Args:
            self: The object instance.
            model (str): The path to the sentencepiece model file.
        
        Returns:
            None.
        
        Raises:
            ImportError: If the 'sentencepiece' backend is not installed.
            FileNotFoundError: If the specified model file could not be found.
            Exception: If there are any errors while loading the sentencepiece model.
        
        Note:
            - This method requires the 'sentencepiece' backend to be installed.
            - The sentencepiece model file must exist at the specified path.
            - The 'sentencepiece' module is imported dynamically within this method.
        """
        requires_backends(self, "sentencepiece")
        from sentencepiece import SentencePieceProcessor

        self.sp = SentencePieceProcessor()
        self.sp.Load(model)

    def extract(self, vocab_scores=None) -> Tuple[Dict[str, int], List[Tuple]]:
        """
        By default will return vocab and merges with respect to their order, by sending `vocab_scores` we're going to
        order the merges with respect to the piece scores instead.
        """
        sp = self.sp
        vocab = {sp.id_to_piece(index): index for index in range(sp.GetPieceSize())}
        if vocab_scores is not None:
            vocab_scores, reverse = dict(vocab_scores), True
        else:
            vocab_scores, reverse = vocab, False

        # Merges
        merges = []
        for merge, piece_score in vocab_scores.items():
            local = []
            for index in range(1, len(merge)):
                piece_l, piece_r = merge[:index], merge[index:]
                if piece_l in vocab and piece_r in vocab:
                    local.append((piece_l, piece_r, piece_score))
            local = sorted(local, key=lambda x: (vocab[x[0]], vocab[x[1]]))
            merges.extend(local)

        merges = sorted(merges, key=lambda val: val[2], reverse=reverse)
        merges = [(val[0], val[1]) for val in merges]
        return vocab, merges


def check_number_comma(piece: str) -> bool:
    """
    Check if the input piece is a valid number with a trailing comma.

    Args:
        piece (str): The string to be checked for validity as a number with a trailing comma.

    Returns:
        bool: Returns True if the piece is a valid number with a trailing comma, False otherwise.

    Raises:
        None.
    """
    return len(piece) < 2 or piece[-1] != "," or not piece[-2].isdigit()


class Converter:

    """
    The Converter class represents a converter for tokenizers.

    This class inherits from <insert the name of the parent class here>.
    It contains methods for initializing the converter with an original tokenizer
    and for converting the original tokenizer to a new tokenizer.

    Attributes:
        original_tokenizer (Tokenizer): The original tokenizer to be converted.

    Methods:
        __init__: Initializes the Converter with the original tokenizer.
        converted: Converts the original tokenizer to a new tokenizer.

    Note:
        This class is not meant to be instantiated directly as it raises a NotImplementedError
        for the converted method, which should be implemented by subclasses.
    """
    def __init__(self, original_tokenizer):
        """
        Initializes a new instance of the Converter class.

        Args:
            self (Converter): The instance of the Converter class.
            original_tokenizer: The original tokenizer object to be stored in the instance.
                It should be a valid tokenizer object that will be used for conversion.

        Returns:
            None.

        Raises:
            None.
        """
        self.original_tokenizer = original_tokenizer

    def converted(self) -> Tokenizer:
        """
        This method 'converted' in the class 'Converter' converts the input to a Tokenizer object.

        Args:
            self: An instance of the Converter class.

        Returns:
            Tokenizer: Returns a Tokenizer object representing the converted input.

        Raises:
            NotImplementedError: If the method is called directly without being implemented in a subclass.
        """
        raise NotImplementedError()


class BertConverter(Converter):

    """
    BertConverter is a Python class that represents a tokenizer converter for BERT models.
    It inherits from the Converter class and provides functionality to convert an original tokenizer to a BERT-compatible
    tokenizer.

    The converted method within the class takes no arguments and returns a Tokenizer object.
    It extracts the vocabulary from the original tokenizer and uses it to initialize a new Tokenizer with specific configurations.
    The method sets various parameters such as tokenize_chinese_chars, strip_accents,
    and do_lower_case based on the properties of the original tokenizer.
    It also configures the normalizer, pre_tokenizer, post_processor, and decoder for the new Tokenizer object.

    This class serves as a crucial component in adapting an existing tokenizer to be compatible with BERT models,
    enabling seamless integration and usage within BERT-based applications.
    """
    def converted(self) -> Tokenizer:
        """
        Converts the original tokenizer to a new tokenizer.

        Args:
            self: An instance of the BertConverter class.

        Returns:
            Tokenizer: The converted tokenizer object.

        Raises:
            None.
        """
        vocab = self.original_tokenizer.vocab
        tokenizer = Tokenizer(WordPiece(vocab, unk_token=str(self.original_tokenizer.unk_token)))

        tokenize_chinese_chars = False
        strip_accents = False
        do_lower_case = False
        if hasattr(self.original_tokenizer, "basic_tokenizer"):
            tokenize_chinese_chars = self.original_tokenizer.basic_tokenizer.tokenize_chinese_chars
            strip_accents = self.original_tokenizer.basic_tokenizer.strip_accents
            do_lower_case = self.original_tokenizer.basic_tokenizer.do_lower_case

        tokenizer.normalizer = normalizers.BertNormalizer(
            clean_text=True,
            handle_chinese_chars=tokenize_chinese_chars,
            strip_accents=strip_accents,
            lowercase=do_lower_case,
        )
        tokenizer.pre_tokenizer = pre_tokenizers.BertPreTokenizer()

        cls = str(self.original_tokenizer.cls_token)
        sep = str(self.original_tokenizer.sep_token)
        cls_token_id = self.original_tokenizer.cls_token_id
        sep_token_id = self.original_tokenizer.sep_token_id

        tokenizer.post_processor = processors.TemplateProcessing(
            single=f"{cls}:0 $A:0 {sep}:0",
            pair=f"{cls}:0 $A:0 {sep}:0 $B:1 {sep}:1",
            special_tokens=[
                (cls, cls_token_id),
                (sep, sep_token_id),
            ],
        )
        tokenizer.decoder = decoders.WordPiece(prefix="##")

        return tokenizer


class SplinterConverter(Converter):

    """
    This class represents a SplinterConverter that is responsible for converting tokens using a customized Tokenizer.
    The SplinterConverter inherits functionalities from the Converter class and provides a method to convert tokens
    with specific configurations such as handling Chinese characters, accents, and lowercase text normalization.
    It also sets special tokens like cls, sep, question, and dot, and defines the template for token processing.
    The resulting Tokenizer object utilizes WordPiece tokenization and decoding with the provided settings for
    token conversion.
    """
    def converted(self) -> Tokenizer:
        """
        Converts the original tokenizer to a new Tokenizer with specific configurations.

        Args:
            self: SplinterConverter - The instance of the SplinterConverter class.
                This parameter is used to access the original_tokenizer and its attributes for conversion.

        Returns:
            Tokenizer: The converted Tokenizer object.
                The converted Tokenizer contains the necessary configurations based on the original tokenizer settings.

        Raises:
            None
        """
        vocab = self.original_tokenizer.vocab
        tokenizer = Tokenizer(WordPiece(vocab, unk_token=str(self.original_tokenizer.unk_token)))

        tokenize_chinese_chars = False
        strip_accents = False
        do_lower_case = False
        if hasattr(self.original_tokenizer, "basic_tokenizer"):
            tokenize_chinese_chars = self.original_tokenizer.basic_tokenizer.tokenize_chinese_chars
            strip_accents = self.original_tokenizer.basic_tokenizer.strip_accents
            do_lower_case = self.original_tokenizer.basic_tokenizer.do_lower_case

        tokenizer.normalizer = normalizers.BertNormalizer(
            clean_text=True,
            handle_chinese_chars=tokenize_chinese_chars,
            strip_accents=strip_accents,
            lowercase=do_lower_case,
        )
        tokenizer.pre_tokenizer = pre_tokenizers.BertPreTokenizer()

        cls = str(self.original_tokenizer.cls_token)
        sep = str(self.original_tokenizer.sep_token)
        question = str(self.original_tokenizer.question_token)
        dot = "."
        cls_token_id = self.original_tokenizer.cls_token_id
        sep_token_id = self.original_tokenizer.sep_token_id
        question_token_id = self.original_tokenizer.question_token_id
        dot_token_id = self.original_tokenizer.convert_tokens_to_ids(".")

        if self.original_tokenizer.padding_side == "right":
            pair = f"{cls}:0 $A:0 {question} {dot} {sep}:0 $B:1 {sep}:1"
        else:
            pair = f"{cls}:0 $A:0 {sep}:0 $B:1 {question} {dot} {sep}:1"

        tokenizer.post_processor = processors.TemplateProcessing(
            single=f"{cls}:0 $A:0 {sep}:0",
            pair=pair,
            special_tokens=[
                (cls, cls_token_id),
                (sep, sep_token_id),
                (question, question_token_id),
                (dot, dot_token_id),
            ],
        )
        tokenizer.decoder = decoders.WordPiece(prefix="##")

        return tokenizer


class FunnelConverter(Converter):

    """
    The FunnelConverter class represents a converter for Funnel tokenization.
    It inherits from the Converter class and provides a method to convert the original tokenizer to
    a Tokenizer suitable for Funnel tokenization.

    The converted method takes no arguments and returns a Tokenizer object.
    It accesses the vocab of the original tokenizer and uses it to initialize a new Tokenizer with WordPiece.
    It sets the unk_token of the new tokenizer based on the original tokenizer's unk_token.

    The method then checks if the original tokenizer has a basic_tokenizer and extracts
    the tokenize_chinese_chars, strip_accents, and do_lower_case attributes if available.
    These attributes are used to configure the normalizer of the new tokenizer, specifically the BertNormalizer.

    Additionally, the method sets the pre_tokenizer of the new tokenizer to BertPreTokenizer.
    It also extracts the cls_token, sep_token, cls_token_id, and sep_token_id from the original tokenizer and uses them
    to configure the post_processor of the new tokenizer with TemplateProcessing.
    Finally, the decoder of the new tokenizer is set to WordPiece with the prefix '##'.

    The method returns the configured Tokenizer object for Funnel tokenization.
    """
    def converted(self) -> Tokenizer:
        """
        Converts the original tokenizer into a new tokenizer with specific configurations.

        Args:
            self: An instance of the FunnelConverter class.

        Returns:
            A Tokenizer object representing the converted tokenizer.

        Raises:
            None.
        """
        vocab = self.original_tokenizer.vocab
        tokenizer = Tokenizer(WordPiece(vocab, unk_token=str(self.original_tokenizer.unk_token)))

        tokenize_chinese_chars = False
        strip_accents = False
        do_lower_case = False
        if hasattr(self.original_tokenizer, "basic_tokenizer"):
            tokenize_chinese_chars = self.original_tokenizer.basic_tokenizer.tokenize_chinese_chars
            strip_accents = self.original_tokenizer.basic_tokenizer.strip_accents
            do_lower_case = self.original_tokenizer.basic_tokenizer.do_lower_case

        tokenizer.normalizer = normalizers.BertNormalizer(
            clean_text=True,
            handle_chinese_chars=tokenize_chinese_chars,
            strip_accents=strip_accents,
            lowercase=do_lower_case,
        )
        tokenizer.pre_tokenizer = pre_tokenizers.BertPreTokenizer()

        cls = str(self.original_tokenizer.cls_token)
        sep = str(self.original_tokenizer.sep_token)
        cls_token_id = self.original_tokenizer.cls_token_id
        sep_token_id = self.original_tokenizer.sep_token_id

        tokenizer.post_processor = processors.TemplateProcessing(
            single=f"{cls}:2 $A:0 {sep}:0",  # token_type_id is 2 for Funnel transformer
            pair=f"{cls}:2 $A:0 {sep}:0 $B:1 {sep}:1",
            special_tokens=[
                (cls, cls_token_id),
                (sep, sep_token_id),
            ],
        )
        tokenizer.decoder = decoders.WordPiece(prefix="##")

        return tokenizer


class MPNetConverter(Converter):

    """
    The MPNetConverter class represents a converter for converting the original tokenizer to a Tokenizer for MPNet.
    This class inherits from the Converter class.

    The converted method within the MPNetConverter class takes no parameters and returns a Tokenizer object.
    It retrieves the vocabulary from the original_tokenizer and initializes a new Tokenizer object with
    the WordPiece vocabulary. It sets the unk_token for the new tokenizer based on the original tokenizer.

    Furthermore, the method sets various attributes for the new tokenizer
    such as normalizer, pre_tokenizer, post_processor, and decoder based on the attributes and configurations
    of the original tokenizer.

    The Tokenizer object with the defined attributes is then returned by the converted method.
    """
    def converted(self) -> Tokenizer:
        """
        This method converts the original tokenizer to a Tokenizer object.

        Args:
            self (MPNetConverter): The instance of the MPNetConverter class.

        Returns:
            Tokenizer: A Tokenizer object representing the converted tokenizer.

        Raises:
            None
        """
        vocab = self.original_tokenizer.vocab
        tokenizer = Tokenizer(WordPiece(vocab, unk_token=str(self.original_tokenizer.unk_token)))

        tokenize_chinese_chars = False
        strip_accents = False
        do_lower_case = False
        if hasattr(self.original_tokenizer, "basic_tokenizer"):
            tokenize_chinese_chars = self.original_tokenizer.basic_tokenizer.tokenize_chinese_chars
            strip_accents = self.original_tokenizer.basic_tokenizer.strip_accents
            do_lower_case = self.original_tokenizer.basic_tokenizer.do_lower_case

        tokenizer.normalizer = normalizers.BertNormalizer(
            clean_text=True,
            handle_chinese_chars=tokenize_chinese_chars,
            strip_accents=strip_accents,
            lowercase=do_lower_case,
        )
        tokenizer.pre_tokenizer = pre_tokenizers.BertPreTokenizer()

        cls = str(self.original_tokenizer.cls_token)
        sep = str(self.original_tokenizer.sep_token)
        cls_token_id = self.original_tokenizer.cls_token_id
        sep_token_id = self.original_tokenizer.sep_token_id

        tokenizer.post_processor = processors.TemplateProcessing(
            single=f"{cls}:0 $A:0 {sep}:0",
            pair=f"{cls}:0 $A:0 {sep}:0 {sep}:0 $B:1 {sep}:1",  # MPNet uses two [SEP] tokens
            special_tokens=[
                (cls, cls_token_id),
                (sep, sep_token_id),
            ],
        )
        tokenizer.decoder = decoders.WordPiece(prefix="##")

        return tokenizer


class OpenAIGPTConverter(Converter):

    """
    The OpenAIGPTConverter class represents a converter for converting a tokenizer into the format used by OpenAI's GPT models.

    This class inherits from the Converter class and implements the converted method to perform the conversion.
    The converted method takes the original tokenizer and returns a Tokenizer instance that is compatible with OpenAI's GPT models.

    The converted method performs the conversion by extracting the vocabulary
    and merges from the original tokenizer, setting the unknown token, and configuring the Tokenizer instance with the necessary
    components such as normalizer, pre_tokenizer, and decoder.

    The Tokenizer instance produced by the converted method is configured with the appropriate settings
    for compatibility with OpenAI's GPT models and is returned for further use in text processing and
    generation tasks.
    """
    def converted(self) -> Tokenizer:
        """
        Converts the original tokenizer to a Tokenizer object.

        Args:
            self: An instance of the OpenAIGPTConverter class.

        Returns:
            Tokenizer: A Tokenizer object containing the converted tokenizer.

        Raises:
            None.
        """
        vocab = self.original_tokenizer.encoder
        merges = list(self.original_tokenizer.bpe_ranks.keys())
        unk_token = self.original_tokenizer.unk_token

        tokenizer = Tokenizer(
            BPE(
                vocab=vocab,
                merges=merges,
                dropout=None,
                unk_token=str(unk_token),
                end_of_word_suffix="</w>",
                fuse_unk=False,
            )
        )

        if tokenizer.token_to_id(str(unk_token)) is not None:
            tokenizer.add_special_tokens([str(unk_token)])

        tokenizer.normalizer = normalizers.BertNormalizer(lowercase=True)
        tokenizer.pre_tokenizer = pre_tokenizers.BertPreTokenizer()
        tokenizer.decoder = decoders.BPEDecoder(suffix="</w>")

        return tokenizer


class GPT2Converter(Converter):

    """
    The GPT2Converter class is responsible for converting an original Tokenizer instance into a new Tokenizer instance
    that is compatible with the GPT-2 model.

    This class inherits from the Converter class.

    The converted() method takes the original_tokenizer as input and returns a new Tokenizer instance.
    It first extracts the vocabulary and merges from the original_tokenizer.
    Then, it initializes a new Tokenizer object with the extracted vocabulary and merges,
    along with other necessary parameters such as dropout, continuing_subword_prefix, end_of_word_suffix, and fuse_unk.

    The pre_tokenizer and decoder attributes of the new Tokenizer instance are set to
    pre_tokenizers.ByteLevel() and decoders.ByteLevel() respectively.
    If the original_tokenizer has a bos_token, the post_processor is set to processors.TemplateProcessing() with
    appropriate settings. Otherwise, the post_processor is set to processors.ByteLevel().

    This class provides a convenient way to convert an original Tokenizer instance to a GPT-2 compatible Tokenizer instance
    by encapsulating the conversion logic within the converted() method.
    The converted Tokenizer can then be used for tokenizing text for GPT-2 model input.

    Note:
        The GPT2Converter class assumes that the original_tokenizer has the necessary attributes and methods as required
        by the conversion process.
    """
    def converted(self) -> Tokenizer:
        """
        Converts the original tokenizer to a Tokenizer object.

        Args:
            self (GPT2Converter): An instance of the GPT2Converter class.

        Returns:
            Tokenizer: A Tokenizer object representing the converted tokenizer.

        Raises:
            None
        """
        vocab = self.original_tokenizer.encoder
        merges = list(self.original_tokenizer.bpe_ranks.keys())

        tokenizer = Tokenizer(
            BPE(
                vocab=vocab,
                merges=merges,
                dropout=None,
                continuing_subword_prefix="",
                end_of_word_suffix="",
                fuse_unk=False,
            )
        )

        tokenizer.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=self.original_tokenizer.add_prefix_space)
        tokenizer.decoder = decoders.ByteLevel()
        if self.original_tokenizer.add_bos_token:
            bos = self.original_tokenizer.bos_token
            bos_token_id = self.original_tokenizer.bos_token_id
            tokenizer.post_processor = processors.TemplateProcessing(
                single=f"{bos}:0 $A:0",
                pair=f"{bos}:0 $A:0 $B:1",
                special_tokens=[
                    (bos, bos_token_id),
                ],
            )
        else:
            # trim_offsets=False actually means this post_processor doesn't
            # really do anything.
            tokenizer.post_processor = processors.ByteLevel(trim_offsets=False)
        return tokenizer


class HerbertConverter(Converter):

    """
    The HerbertConverter class represents a specialized converter that converts a given tokenizer into a
    Herbert-compatible tokenizer.
    It inherits methods from the Converter class and provides functionality to transform the tokenizer into a format
    suitable for Herbert models.

    The converted method within the HerbertConverter class implements the logic to create a new Tokenizer instance
    with specific configurations for Herbert compatibility.

    It handles tasks such as adjusting the tokenizer's parameters, setting up normalizers, pre-tokenizers, decoders,
    and post-processors tailored for Herbert models.
    The converted method returns the modified Tokenizer instance ready for use with Herbert models.
    """
    def converted(self) -> Tokenizer:
        """
        Converts the original tokenizer to a new Tokenizer using the provided information.

        Args:
            self: HerbertConverter instance. The current instance of the HerbertConverter class.

        Returns:
            Tokenizer: An instance of the Tokenizer class representing the converted tokenizer.

        Raises:
            None
        """
        tokenizer_info_str = "#version:"
        token_suffix = "</w>"

        vocab = self.original_tokenizer.encoder
        merges = list(self.original_tokenizer.bpe_ranks.keys())
        if tokenizer_info_str in merges[0][0]:
            merges = merges[1:]

        tokenizer = Tokenizer(
            BPE(
                vocab,
                merges,
                dropout=None,
                unk_token=self.original_tokenizer.unk_token,
                end_of_word_suffix=token_suffix,
            )
        )

        tokenizer.normalizer = normalizers.BertNormalizer(lowercase=False, strip_accents=False)
        tokenizer.pre_tokenizer = pre_tokenizers.BertPreTokenizer()
        tokenizer.decoder = decoders.BPEDecoder(suffix=token_suffix)
        tokenizer.post_processor = processors.BertProcessing(
            sep=(self.original_tokenizer.sep_token, self.original_tokenizer.sep_token_id),
            cls=(self.original_tokenizer.cls_token, self.original_tokenizer.cls_token_id),
        )

        return tokenizer


class RobertaConverter(Converter):

    """
    This class represents a RobertaConverter that converts a given Tokenizer instance into a Roberta-compatible Tokenizer.
    It inherits from the Converter class.
    The converted method within the class takes the original Tokenizer instance and creates a new Tokenizer with specific
    configurations for Roberta models, including BPE, pre-tokenizer, decoder, and post-processor settings.
    """
    def converted(self) -> Tokenizer:
        """
        Converts the original tokenizer to a Tokenizer object using specific configurations.

        Args:
            self: The instance of the class RobertaConverter. It represents the original tokenizer and its configurations.

        Returns:
            Tokenizer: A Tokenizer object that is created based on the configurations of the original tokenizer.
                It is used for tokenization with BPE encoding.

        Raises:
            None.
        """
        ot = self.original_tokenizer
        vocab = ot.encoder
        merges = list(ot.bpe_ranks.keys())

        tokenizer = Tokenizer(
            BPE(
                vocab=vocab,
                merges=merges,
                dropout=None,
                continuing_subword_prefix="",
                end_of_word_suffix="",
                fuse_unk=False,
            )
        )

        tokenizer.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=ot.add_prefix_space)
        tokenizer.decoder = decoders.ByteLevel()
        tokenizer.post_processor = processors.RobertaProcessing(
            sep=(ot.sep_token, ot.sep_token_id),
            cls=(ot.cls_token, ot.cls_token_id),
            add_prefix_space=ot.add_prefix_space,
            trim_offsets=True,  # True by default on Roberta (historical)
        )

        return tokenizer


# class RoFormerConverter(Converter):
#     def converted(self) -> Tokenizer:
#         from .models.roformer.tokenization_utils import JiebaPreTokenizer

#         vocab = self.original_tokenizer.vocab
#         tokenizer = Tokenizer(WordPiece(vocab, unk_token=str(self.original_tokenizer.unk_token)))

#         strip_accents = False
#         do_lower_case = False
#         if hasattr(self.original_tokenizer, "basic_tokenizer"):
#             strip_accents = self.original_tokenizer.basic_tokenizer.strip_accents
#             do_lower_case = self.original_tokenizer.basic_tokenizer.do_lower_case

#         tokenizer.normalizer = normalizers.BertNormalizer(
#             clean_text=True,
#             handle_chinese_chars=False,
#             strip_accents=strip_accents,
#             lowercase=do_lower_case,
#         )
#         tokenizer.pre_tokenizer = pre_tokenizers.PreTokenizer.custom(JiebaPreTokenizer(vocab))

#         cls = str(self.original_tokenizer.cls_token)
#         sep = str(self.original_tokenizer.sep_token)
#         cls_token_id = self.original_tokenizer.cls_token_id
#         sep_token_id = self.original_tokenizer.sep_token_id

#         tokenizer.post_processor = processors.TemplateProcessing(
#             single=f"{cls}:0 $A:0 {sep}:0",
#             pair=f"{cls}:0 $A:0 {sep}:0 $B:1 {sep}:1",
#             special_tokens=[
#                 (cls, cls_token_id),
#                 (sep, sep_token_id),
#             ],
#         )
#         tokenizer.decoder = decoders.WordPiece(prefix="##")

#         return tokenizer


class DebertaConverter(Converter):

    """
    The DebertaConverter class is a Python class that represents a converter for DeBERTa tokenizers.

    This class inherits from the Converter class and provides a method called 'converted'
    which takes no arguments and returns a Tokenizer object.
    The 'converted' method converts the original tokenizer into a DeBERTa tokenizer by setting specific
    configurations and parameters.

    The converted tokenizer is created by using the original_tokenizer from the parent class.
    The vocabulary and merges are extracted from the original_tokenizer.
    The DeBERTa tokenizer is then instantiated with the extracted vocabulary and merges,
    along with additional configurations such as dropout, continuing_subword_prefix, end_of_word_suffix, and fuse_unk.

    The pre_tokenizer is set to pre_tokenizers.ByteLevel with the 'add_prefix_space' parameter obtained from the
    original_tokenizer. The decoder is set to decoders.ByteLevel.

    The post_processor is configured using processors.TemplateProcessing.
    The 'single' template is set to '[CLS]:0 $A:0 [SEP]:0', the 'pair' template is set to '[CLS]:0 $A:0 [SEP]:0 $B:1 [SEP]:1', and the
    special_tokens are defined as a list of tuples containing the special tokens '[CLS]' and '[SEP]'
    along with their respective ids obtained from the original_tokenizer.

    Finally, the converted tokenizer is returned.

    Note:
        It is assumed that the parent class 'Converter' provides the necessary functionality for the 'converted' method
        to work correctly.
    """
    def converted(self) -> Tokenizer:
        """
        This method 'converted' in the class 'DebertaConverter' takes 1 parameter: self.

        Args:
            self (object): The instance of the class 'DebertaConverter'.
                It is used within the method to access the original_tokenizer and perform the conversion.

        Returns:
            Tokenizer: An instance of the Tokenizer class representing the converted tokenizer.
                The converted tokenizer is created based on the original_tokenizer's encoder, BPE merges, and other
                configurations as specified in the method's code.

        Raises:
            None
        """
        ot = self.original_tokenizer
        vocab = ot.encoder
        merges = list(ot.bpe_ranks.keys())

        tokenizer = Tokenizer(
            BPE(
                vocab=vocab,
                merges=merges,
                dropout=None,
                continuing_subword_prefix="",
                end_of_word_suffix="",
                fuse_unk=False,
            )
        )

        tokenizer.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=ot.add_prefix_space)
        tokenizer.decoder = decoders.ByteLevel()
        tokenizer.post_processor = processors.TemplateProcessing(
            single="[CLS]:0 $A:0 [SEP]:0",
            pair="[CLS]:0 $A:0 [SEP]:0 $B:1 [SEP]:1",
            special_tokens=[
                ("[CLS]", self.original_tokenizer.convert_tokens_to_ids("[CLS]")),
                ("[SEP]", self.original_tokenizer.convert_tokens_to_ids("[SEP]")),
            ],
        )

        return tokenizer


class SpmConverter(Converter):

    """
    The `SpmConverter` class is a Python class that represents a converter for converting sentencepiece tokenizers
    into fast tokenizers. It inherits from the `Converter` class.

    Attributes:
        proto (ModelProto): The protobuf model used for conversion.

    Methods:
        __init__: Initializes the `SpmConverter` object with the given arguments.
        vocab: Returns the vocabulary and scores from the given protobuf model.
        unk_id: Returns the unknown token ID from the given protobuf model.
        tokenizer: Returns the appropriate tokenizer based on the model type specified in the protobuf model.
        normalizer: Returns the normalizer sequence based on the precompiled character map from the protobuf model.
        pre_tokenizer: Returns the pre-tokenizer with the specified replacement and prefix space options.
        post_processor: Returns the post-processor for the tokenizer.
        decoder: Returns the decoder with the specified replacement and prefix space options.
        converted: Converts the sentencepiece tokenizer to a fast tokenizer and returns the resulting tokenizer.

    Note:
        The sentencepiece tokenizer being converted may use the byte fallback option, which is not implemented
        in the fast tokenizers.
        This means that the fast tokenizer may produce unknown tokens while the sentencepiece version would convert
        these unknown tokens into a sequence of byte tokens matching the original text.
        The file being trained with a `Unigram` model should not be run with a different algorithm.

    """
    def __init__(self, *args):
        """

        Args:
            self: SpmConverter
                The instance of the SpmConverter class.

        Returns:
            None.

        Raises:
            BackendRequirementError: If the 'protobuf' backend is not available.
            FileNotFoundError: If the specified vocabulary file is not found.
            Warning: If the sentencepiece tokenizer being converted to a fast tokenizer uses the byte fallback option,
                a warning is issued.
        """
        requires_backends(self, "protobuf")

        super().__init__(*args)

        # from .utils import sentencepiece_model_pb2 as model_pb2
        model_pb2 = import_protobuf()

        m = model_pb2.ModelProto()
        with open(self.original_tokenizer.vocab_file, "rb") as f:
            m.ParseFromString(f.read())
        self.proto = m

        if self.proto.trainer_spec.byte_fallback:
            if not getattr(self, "handle_byte_fallback", None):
                warnings.warn(
                    "The sentencepiece tokenizer that you are converting to a fast tokenizer uses the byte fallback option"
                    " which is not implemented in the fast tokenizers. In practice this means that the fast version of the"
                    " tokenizer can produce unknown tokens whereas the sentencepiece version would have converted these "
                    "unknown tokens into a sequence of byte tokens matching the original piece of text."
                )

    def vocab(self, proto):
        """
        Converts the given protocol buffer to a list of tuples containing pieces and their scores.

        Args:
            self (SpmConverter): The instance of the SpmConverter class.
            proto: An object representing the protocol buffer to extract pieces from.

        Returns:
            list: A list of tuples where each tuple consists of a piece and its corresponding score extracted
                from the protocol buffer.

        Raises:
            None.
        """
        return [(piece.piece, piece.score) for piece in proto.pieces]

    def unk_id(self, proto):
        """
        This method 'unk_id' in the class 'SpmConverter' returns the unknown token ID from the specified proto object.

        Args:
            self (SpmConverter): The instance of the SpmConverter class.
            proto: The proto object containing the trainer specification.

        Returns:
            None: This method does not explicitly return a value, as it directly accesses and returns
                the unknown token ID from the proto object.

        Raises:
            AttributeError: If the 'trainer_spec' attribute is not found in the proto object.
        """
        return proto.trainer_spec.unk_id

    def tokenizer(self, proto):
        """
        This method 'tokenizer' in the class 'SpmConverter' tokenizes input data based on the specified model type.

        Args:
            self: An instance of the SpmConverter class.
                It is used to access the methods and attributes of the class.

            proto: A protocol buffer object.
                It represents the input data that needs to be tokenized and contains necessary training specifications.

        Returns:
            None.

        Raises:
            RuntimeError: If the model type specified in the protocol buffer object is not supported or does not match
                the trained model type. This exception is raised when attempting to tokenize data using a model type
                that is incompatible with the training data.
        """
        model_type = proto.trainer_spec.model_type
        vocab_scores = self.vocab(proto)
        unk_id = self.unk_id(proto)

        if model_type == 1:
            tokenizer = Tokenizer(Unigram(vocab_scores, unk_id))
            return tokenizer
        if model_type == 2:
            _, merges = SentencePieceExtractor(self.original_tokenizer.vocab_file).extract()
            bpe_vocab = {word: i for i, (word, score) in enumerate(vocab_scores)}
            tokenizer = Tokenizer(
                BPE(
                    bpe_vocab,
                    merges,
                    unk_token=proto.trainer_spec.unk_piece,
                    fuse_unk=True,
                )
            )
            return tokenizer
        raise RuntimeError(
            "You're trying to run a `Unigram` model but you're file was trained with a different algorithm"
        )

    def normalizer(self, proto):
        """
        This method normalizer in the SpmConverter class processes the input proto object for normalization.

        Args:
            self (object): The instance of the SpmConverter class.
            proto (object): The proto object containing the normalization specifications.

        Returns:
            None.

        Raises:
            None.
        """
        precompiled_charsmap = proto.normalizer_spec.precompiled_charsmap
        if not precompiled_charsmap:
            return normalizers.Sequence([normalizers.Replace(Regex(" {2,}"), " ")])
        return normalizers.Sequence(
            [normalizers.Precompiled(precompiled_charsmap), normalizers.Replace(Regex(" {2,}"), " ")]
        )

    def pre_tokenizer(self, replacement, add_prefix_space):
        """
        This method pre_tokenizer in the SpmConverter class processes tokenization using the Metaspace pre_tokenizer.

        Args:
            replacement (str): the replacement value to be used in the pre_tokenizer.
            add_prefix_space (bool): indicates whether to add a prefix space during tokenization.

        Returns:
            None: This method does not return any value explicitly, but it configures the pre_tokenizer for tokenization.

        Raises:
            None.
        """
        prepend_scheme = _get_prepend_scheme(add_prefix_space, self.original_tokenizer)
        return pre_tokenizers.Metaspace(replacement=replacement, prepend_scheme=prepend_scheme)

    def post_processor(self):
        """
        Perform post-processing on the converted data in the SpmConverter class.

        Args:
            self: An instance of the SpmConverter class.

        Returns:
            None.

        Raises:
            None.

        This method is responsible for post-processing the converted data in the SpmConverter class.
        It takes an instance of the class as the only parameter and does not return any value explicitly.

        Post-processing refers to any additional steps or modifications that need to be applied to the
        converted data after the conversion process.
        These steps could include data validation, normalization, or any other necessary adjustments to
        ensure the accuracy and integrity of the converted data.

        Please note that this method does not raise any exceptions.
        However, it is important to handle any potential exceptions that may occur
        during the post-processing steps within the method implementation.
        """
        return None

    def decoder(self, replacement, add_prefix_space):
        """
        This method decodes a given input using the specified replacement and prefix space settings.

        Args:
            self (SpmConverter): The instance of the SpmConverter class.
            replacement (str): The replacement string to be used in the decoding process.
            add_prefix_space (bool): A boolean flag indicating whether to add a prefix space during decoding.

        Returns:
            None.

        Raises:
            None.
        """
        prepend_scheme = _get_prepend_scheme(add_prefix_space, self.original_tokenizer)
        return decoders.Metaspace(replacement=replacement, prepend_scheme=prepend_scheme)

    def converted(self) -> Tokenizer:
        """
        Converts the proto object to a Tokenizer object with specified configurations.

        Args:
            self: An instance of the SpmConverter class.

        Returns:
            Tokenizer: A Tokenizer object with configurations set based on the proto object.

        Raises:
            None
        """
        tokenizer = self.tokenizer(self.proto)

        # Tokenizer assemble
        normalizer = self.normalizer(self.proto)
        if normalizer is not None:
            tokenizer.normalizer = normalizer

        replacement = "▁"
        add_prefix_space = True
        if hasattr(self.original_tokenizer, "add_prefix_space"):
            add_prefix_space = self.original_tokenizer.add_prefix_space

        pre_tokenizer = self.pre_tokenizer(replacement, add_prefix_space)
        if pre_tokenizer is not None:
            tokenizer.pre_tokenizer = pre_tokenizer

        tokenizer.decoder = self.decoder(replacement, add_prefix_space)
        post_processor = self.post_processor()
        if post_processor:
            tokenizer.post_processor = post_processor

        return tokenizer


class AlbertConverter(SpmConverter):

    """
    AlbertConverter is a Python class that represents a converter for processing text data using the ALBERT model.
    It inherits from SpmConverter and provides methods for vocabulary processing, normalization, and post-processing
    of input data.

    Methods:
        vocab(proto): Process the vocabulary by returning a list of tuples containing text pieces and their scores.
        normalizer(proto): Normalize the input text by applying a sequence of normalization operations,
            including replacements, lowercasing, and stripping accents.
        post_processor(): Generate a post-processor template for text data,
            including special token mappings for '[CLS]' and '[SEP]'.
    """
    def vocab(self, proto):
        """
        This method, 'vocab', is a member of the 'AlbertConverter' class and is used to process a 'proto' object and
        extract vocabulary information.

        Args:
            self: The instance of the 'AlbertConverter' class.
            proto: An object of type 'Proto' which contains a list of 'pieces' representing vocabulary pieces.

        Returns:
            None

        Raises:
            None

        This method iterates over each 'piece' in the 'proto.pieces' list and forwards a new list by applying certain
        conditions. If 'check_number_comma(piece.piece)' returns True for a 'piece', the resulting tuple in the new
        list will contain 'piece.piece' and 'piece.score'. Otherwise, the resulting tuple will contain 'piece.piece'
        and 'piece.score - 100'.

        Note:
            - 'check_number_comma()' is a helper function that checks if a piece contains a number or a comma.

        Example:
            ```python
            >>> converter = AlbertConverter()
            >>> proto = Proto()
            >>> # Populate 'proto' with pieces
            >>> converter.vocab(proto)
            ...
            >>> # The resulting list will be stored internally in the 'AlbertConverter' instance.
            ```
        """
        return [
            (piece.piece, piece.score) if check_number_comma(piece.piece) else (piece.piece, piece.score - 100)
            for piece in proto.pieces
        ]

    def normalizer(self, proto):
        """
        Normalize the given proto using a sequence of normalizers.

        Args:
            self (AlbertConverter): The instance of the AlbertConverter class.
            proto: The proto object to be normalized.

        Returns:
            None: This method modifies the input proto in place.

        Raises:
            None.
        """
        list_normalizers = [
            normalizers.Replace("``", '"'),
            normalizers.Replace("''", '"'),
        ]
        if not self.original_tokenizer.keep_accents:
            list_normalizers.append(normalizers.NFKD())
            list_normalizers.append(normalizers.StripAccents())
        if self.original_tokenizer.do_lower_case:
            list_normalizers.append(normalizers.Lowercase())

        precompiled_charsmap = proto.normalizer_spec.precompiled_charsmap

        if precompiled_charsmap:
            list_normalizers.append(normalizers.Precompiled(precompiled_charsmap))

        list_normalizers.append(normalizers.Replace(Regex(" {2,}"), " "))
        return normalizers.Sequence(list_normalizers)

    def post_processor(self):
        """
        Post-processes the converted tokens using a template processing method.

        Args:
            self (AlbertConverter): The instance of the AlbertConverter class.

        Returns:
            None.

        Raises:
            None.
        """
        return processors.TemplateProcessing(
            single="[CLS]:0 $A:0 [SEP]:0",
            pair="[CLS]:0 $A:0 [SEP]:0 $B:1 [SEP]:1",
            special_tokens=[
                ("[CLS]", self.original_tokenizer.convert_tokens_to_ids("[CLS]")),
                ("[SEP]", self.original_tokenizer.convert_tokens_to_ids("[SEP]")),
            ],
        )


class BarthezConverter(SpmConverter):

    """
    The 'BarthezConverter' class is a Python class that represents a converter for the Barthez model.
    This class inherits from the 'SpmConverter' class.

    This class provides methods for converting text using the Barthez model.
    It includes a method for generating an unknown ID based on a given protocol, as well as a post-processing method
    for template processing.

    The 'unk_id' method takes a 'proto' parameter and returns an unknown ID value.
    The 'proto' parameter represents the protocol used for generating the unknown ID. T
    he method calculates and returns the unknown ID based on the provided protocol.

    The 'post_processor' method performs post-processing on the converted text using template processing.
    It returns the processed text, which includes special tokens for single and pair sentences.
    The 'single' template represents a single sentence, while the 'pair' template represents a pair of sentences.
    The method also includes special tokens for start and end of sentences, which are converted to their respective
    token IDs using the 'original_tokenizer'.

    Note:
        Please ensure that the 'original_tokenizer' attribute is properly initialized before calling the
        'post_processor' method.

    Example:
        ```python
        >>> converter = BarthezConverter()
        >>> unk_id = converter.unk_id(proto)
        >>> processed_text = converter.post_processor()
        ```
    """
    def unk_id(self, proto):
        """
        The 'unk_id' method in the 'BarthezConverter' class takes two parameters: self and proto.

        Args:
            self (BarthezConverter): The instance of the BarthezConverter class on which the method is called.
            proto: The proto parameter represents a certain value or object that the unk_id method will use during
                its execution.

        Returns:
            None.

        Raises:
            None.
        """
        unk_id = 3
        return unk_id

    def post_processor(self):
        """
        post_processor method in the BarthezConverter class.

        Args:
            self (BarthezConverter): The instance of the BarthezConverter class.

        Returns:
            None.

        Raises:
            None.
        """
        return processors.TemplateProcessing(
            single="<s> $A </s>",
            pair="<s> $A </s> </s> $B </s>",
            special_tokens=[
                ("<s>", self.original_tokenizer.convert_tokens_to_ids("<s>")),
                ("</s>", self.original_tokenizer.convert_tokens_to_ids("</s>")),
            ],
        )


class CamembertConverter(SpmConverter):

    """
    The CamembertConverter class is a Python class that represents a converter for the Camembert model.
    It inherits from SpmConverter and provides methods for vocabulary extraction and post-processing.

    The vocab method returns the vocabulary for the Camembert model, including special tokens and their
    corresponding scores.

    The unk_id method returns the identifier for the unknown token in the Camembert model's vocabulary.

    The post_processor method returns the post-processor for the Camembert model, which includes template processing
    with special tokens.
    """
    def vocab(self, proto):
        """
        This method 'vocab' is defined in the class 'CamembertConverter' and takes two parameters: self and proto.

        Args:
            self (object): The instance of the class itself.
            proto (object): The proto parameter is used to retrieve pieces for building the vocabulary.
                It is expected to be an object containing pieces.

        Returns:
            list: A list of tuples representing the vocabulary.
                Each tuple contains a string representing a piece and its corresponding score.

        Raises:
            None: This method does not explicitly raise any exceptions.
        """
        vocab = [
            ("<s>NOTUSED", 0.0),
            ("<pad>", 0.0),
            ("</s>NOTUSED", 0.0),
            ("<unk>", 0.0),
            ("<unk>NOTUSED", -100),
        ]
        # We down-grade the original SentencePiece by -100 to avoid using it and use our added token instead
        vocab += [(piece.piece, piece.score) for piece in proto.pieces[1:]]
        vocab += [("<mask>", 0.0)]
        return vocab

    def unk_id(self, proto):
        """
        This method 'unk_id' in the class 'CamembertConverter' takes two parameters: self and proto.

        Args:
            self (object): Represents the instance of the class CamembertConverter.
            proto (any): Represents the input parameter for the method.

        Returns:
            None.

        Raises:
            None.
        """
        # See vocab unk position
        return 3

    def post_processor(self):
        """
        Method post_processor in the class CamembertConverter.

        Args:
            self: Object of the CamembertConverter class. No additional arguments are needed.

        Returns:
            None.

        Raises:
            None.
        """
        return processors.TemplateProcessing(
            single="<s> $A </s>",
            pair="<s> $A </s> </s> $B </s>",
            special_tokens=[
                ("<s>", self.original_tokenizer.convert_tokens_to_ids("<s>")),
                ("</s>", self.original_tokenizer.convert_tokens_to_ids("</s>")),
            ],
        )


class DebertaV2Converter(SpmConverter):

    """
    A Python class named 'DebertaV2Converter' that inherits from SpmConverter.

    This class contains methods for pre-tokenizing, normalizing, and post-processing text data for use with
    the DeBERTa V2 model.

    The 'pre_tokenizer' method pre-processes the input text by applying pre-tokenizers such as Punctuation and Metaspace.

    The 'normalizer' method normalizes the pre-processed text data by applying normalizers such as Lowercase, Strip,
    and Replace.

    The 'post_processor' method post-processes the normalized text data using a TemplateProcessing processor
    that adds special tokens like '[CLS]' and '[SEP]' to the text.

    Each method in this class performs a specific step in preparing text data for input to the DeBERTa V2 model,
    ensuring optimal performance and accuracy.
    """
    def pre_tokenizer(self, replacement, add_prefix_space):
        """
        This method pre_tokenizer is responsible for setting up the pre-tokenization process in the DebertaV2Converter class.

        Args:
            self: The instance of the DebertaV2Converter class.
            replacement (str): The replacement string to be used during pre-tokenization.
            add_prefix_space (bool): A boolean flag indicating whether to add a prefix space during pre-tokenization.

        Returns:
            None.

        Raises:
            None: This method does not explicitly raise any exceptions.
                However, exceptions may be raised by the pre-tokenizers.
                Punctuation and pre_tokenizers.Metaspace classes during the pre-tokenization process.
        """
        list_pretokenizers = []
        if self.original_tokenizer.split_by_punct:
            list_pretokenizers.append(pre_tokenizers.Punctuation(behavior="isolated"))
        list_pretokenizers.append(pre_tokenizers.Metaspace(replacement=replacement, add_prefix_space=add_prefix_space))
        return pre_tokenizers.Sequence(list_pretokenizers)

    def normalizer(self, proto):
        """
        The normalizer method applies a series of normalization steps to the input proto.

        Args:
            self (object): The instance of the DebertaV2Converter class.
            proto (object): The input proto object to be normalized.

        Returns:
            None.

        Raises:
            None
        """
        list_normalizers = []
        if self.original_tokenizer.do_lower_case:
            list_normalizers.append(normalizers.Lowercase())
        list_normalizers.append(normalizers.Strip())

        precompiled_charsmap = proto.normalizer_spec.precompiled_charsmap
        if precompiled_charsmap:
            list_normalizers.append(normalizers.Precompiled(precompiled_charsmap))
        list_normalizers.append(normalizers.Replace(Regex(" {2,}"), " "))

        return normalizers.Sequence(list_normalizers)

    def post_processor(self):
        """
        Performs post-processing on the input data in the DebertaV2Converter class.

        Args:
            self: An instance of the DebertaV2Converter class.

        Returns:
            None.

        Raises:
            None.

        This method applies post-processing to the input data using the specified template processing rules.
        The rules are defined as follows:

        - For single input: '[CLS]:0 $A:0 [SEP]:0'
        - For pair input: '[CLS]:0 $A:0 [SEP]:0 $B:1 [SEP]:1'

        The special tokens used in the template processing are:

        - '[CLS]': Converted to its corresponding token ID using the original_tokenizer.
        - '[SEP]': Converted to its corresponding token ID using the original_tokenizer.

        Note: The original_tokenizer is an attribute of the DebertaV2Converter class.
        """
        return processors.TemplateProcessing(
            single="[CLS]:0 $A:0 [SEP]:0",
            pair="[CLS]:0 $A:0 [SEP]:0 $B:1 [SEP]:1",
            special_tokens=[
                ("[CLS]", self.original_tokenizer.convert_tokens_to_ids("[CLS]")),
                ("[SEP]", self.original_tokenizer.convert_tokens_to_ids("[SEP]")),
            ],
        )


class MBartConverter(SpmConverter):

    """
    This class represents an MBartConverter that inherits from SpmConverter.

    The MBartConverter class provides methods for vocabulary generation, handling unknown tokens,
    and defining post-processing logic for processing text data using the MBART pre-trained model.

    Attributes:
        None

    Methods:
        vocab(proto):
            Generates and returns the vocabulary list based on the input proto.

        unk_id(proto):
            Returns the ID for the unknown token in the vocabulary based on the input proto.

        post_processor():
            Defines and returns the post-processing logic for text data processing using the MBART model.

    Example:
        ```python
        >>> converter = MBartConverter()
        >>> vocab_list = converter.vocab(proto)
        >>> unk_token_id = converter.unk_id(proto)
        >>> post_processor = converter.post_processor()
        ```
    """
    def vocab(self, proto):
        """
        This method 'vocab' is defined within the class 'MBartConverter' and is used to
        generate a vocabulary list based on the provided 'proto' parameter.

        Args:
            self (object): The instance of the MBartConverter class.
            proto (object): An object containing pieces from which vocabulary is generated.
                It should have a 'pieces' attribute.

        Returns:
            list: A list of tuples representing the vocabulary, where each tuple contains a token and its corresponding score.
                The vocabulary list includes predefined tokens like '<s>', '<pad>', '<unk>', '<mask>',
                as well as language codes and their scores.

        Raises:
            None
        """
        vocab = [
            ("<s>", 0.0),
            ("<pad>", 0.0),
            ("</s>", 0.0),
            ("<unk>", 0.0),
        ]
        vocab += [(piece.piece, piece.score) for piece in proto.pieces[3:]]
        vocab += [
            ("ar_AR", 0.0),
            ("cs_CZ", 0.0),
            ("de_DE", 0.0),
            ("en_XX", 0.0),
            ("es_XX", 0.0),
            ("et_EE", 0.0),
            ("fi_FI", 0.0),
            ("fr_XX", 0.0),
            ("gu_IN", 0.0),
            ("hi_IN", 0.0),
            ("it_IT", 0.0),
            ("ja_XX", 0.0),
            ("kk_KZ", 0.0),
            ("ko_KR", 0.0),
            ("lt_LT", 0.0),
            ("lv_LV", 0.0),
            ("my_MM", 0.0),
            ("ne_NP", 0.0),
            ("nl_XX", 0.0),
            ("ro_RO", 0.0),
            ("ru_RU", 0.0),
            ("si_LK", 0.0),
            ("tr_TR", 0.0),
            ("vi_VN", 0.0),
            ("zh_CN", 0.0),
        ]
        vocab += [("<mask>", 0.0)]
        return vocab

    def unk_id(self, proto):
        """
        Converts an 'unk_id' to a specific value in the MBartConverter class.

        Args:
            self (MBartConverter): An instance of the MBartConverter class.
            proto: The 'unk_id' value to be converted.

        Returns:
            None.

        Raises:
            None.
        """
        return 3

    def post_processor(self):
        """
        This method post_processor is a part of the MBartConverter class and is responsible for performing
        post-processing operations on the input data.

        Args:
            self: The instance of the MBartConverter class. It is used to access the attributes and methods of the class.

        Returns:
            None.

        Raises:
            None.
        """
        return processors.TemplateProcessing(
            single="$A </s> en_XX",
            pair="$A $B </s> en_XX",
            special_tokens=[
                ("en_XX", self.original_tokenizer.convert_tokens_to_ids("en_XX")),
                ("</s>", self.original_tokenizer.convert_tokens_to_ids("</s>")),
            ],
        )


class MBart50Converter(SpmConverter):

    """
    The `MBart50Converter` class is a Python class that represents a converter for the MBART-50 model.
    This class inherits from the `SpmConverter` class.

    Class Methods:

    - `vocab(self, proto)`: This method returns a list of vocabulary items for the MBART-50 model.
    The vocabulary includes special tokens like '<s>', '<pad>', '</s>', '<unk>', and '<mask>', as well as language
    codes for various languages. The vocabulary items are obtained from the provided `proto` object.

        - Parameters:

            - `proto`: A protobuf object containing information about the vocabulary pieces.
        - Returns:

            - `vocab`: A list of tuples where each tuple represents a vocabulary item.
            Each tuple contains the vocabulary piece and its associated score.

    - `unk_id(self, proto)`: This method returns the ID of the '<unk>' token in the MBART-50 model's vocabulary.
    The ID is obtained from the provided `proto` object.

        - Parameters:

            - `proto`: A protobuf object containing information about the vocabulary pieces.
        - Returns:

            - `unk_id`: An integer representing the ID of the '<unk>' token.

    - `post_processor(self)`: This method returns a post-processor object for the MBART-50 model.
    The post-processor is responsible for processing the model's output. It uses a template processing approach,
    where the output is formatted with an 'en_XX' token and other tokens. The template for single sentences is
    'en_XX $A </s>', and for pairs of sentences is 'en_XX $A $B </s>'. The method also specifies special
    tokens as a list of tuples, where each tuple contains the token and its corresponding ID in the vocabulary.

        - Returns:

            - `post_processor`: A post-processor object for the MBART-50 model.

    Note:
        The above methods do not include any code signatures or implementation details as per the provided information.
    """
    def vocab(self, proto):
        """
        This method is part of the 'MBart50Converter' class and is used to generate a vocabulary list based on the provided 'proto' parameter.

        Args:
            self: An instance of the 'MBart50Converter' class.
            proto: A parameter of type 'Proto' that represents a proto object containing information about the vocabulary pieces.

        Returns:
            The method returns a list of tuples representing the vocabulary. Each tuple consists of a word and its associated score.

        Raises:
            None.

        Note:
            The returned vocabulary list includes predefined special tokens such as '<s>', '<pad>', '</s>', and '<unk>'.
            It also includes language codes for various languages. Additionally, the list includes a
            special token '<mask>' used for masking during tokenization.
        """
        vocab = [
            ("<s>", 0.0),
            ("<pad>", 0.0),
            ("</s>", 0.0),
            ("<unk>", 0.0),
        ]
        vocab += [(piece.piece, piece.score) for piece in proto.pieces[3:]]
        # fmt: off
        vocab += [("ar_AR", 0.0), ("cs_CZ", 0.0), ("de_DE", 0.0), ("en_XX", 0.0), ("es_XX", 0.0),
                  ("et_EE", 0.0), ("fi_FI", 0.0), ("fr_XX", 0.0), ("gu_IN", 0.0), ("hi_IN", 0.0),
                  ("it_IT", 0.0), ("ja_XX", 0.0), ("kk_KZ", 0.0), ("ko_KR", 0.0), ("lt_LT", 0.0),
                  ("lv_LV", 0.0), ("my_MM", 0.0), ("ne_NP", 0.0), ("nl_XX", 0.0), ("ro_RO", 0.0),
                  ("ru_RU", 0.0), ("si_LK", 0.0), ("tr_TR", 0.0), ("vi_VN", 0.0), ("zh_CN", 0.0),
                  ("af_ZA", 0.0), ("az_AZ", 0.0), ("bn_IN", 0.0), ("fa_IR", 0.0), ("he_IL", 0.0),
                  ("hr_HR", 0.0), ("id_ID", 0.0), ("ka_GE", 0.0), ("km_KH", 0.0), ("mk_MK", 0.0),
                  ("ml_IN", 0.0), ("mn_MN", 0.0), ("mr_IN", 0.0), ("pl_PL", 0.0), ("ps_AF", 0.0),
                  ("pt_XX", 0.0), ("sv_SE", 0.0), ("sw_KE", 0.0), ("ta_IN", 0.0), ("te_IN", 0.0),
                  ("th_TH", 0.0), ("tl_XX", 0.0), ("uk_UA", 0.0), ("ur_PK", 0.0), ("xh_ZA", 0.0),
                  ("gl_ES", 0.0), ("sl_SI", 0.0)]
        # fmt: on
        vocab += [("<mask>", 0.0)]
        return vocab

    def unk_id(self, proto):
        """
        Method unk_id in the MBart50Converter class.

        Args:
            self (object): The instance of the MBart50Converter class.
            proto (any): The proto parameter is used for XYZ purpose. It can accept any data type.

        Returns:
            None: This method always returns the integer value 3.

        Raises:
            None.
        """
        return 3

    def post_processor(self):
        """
        Method post_processor in class MBart50Converter.

        Args:
            self: The instance of the class MBart50Converter. It is required for accessing the original_tokenizer
                object used for processing.

        Returns:
            None:
                This method does not return any value but performs template processing on the input data.
                It applies a template based on the language code provided and special tokens.

        Raises:
            None:
                However, potential exceptions that could be raised during the execution of this method may include:

                - AttributeError: If the original_tokenizer object is not properly initialized or is missing required attributes.
                - ValueError: If the input data or parameters provided are invalid and cannot be processed.
                - TypeError: If there are issues with the data types or parameter values passed to the method.
        """
        return processors.TemplateProcessing(
            single="en_XX $A </s>",
            pair="en_XX $A $B </s>",
            special_tokens=[
                ("en_XX", self.original_tokenizer.convert_tokens_to_ids("en_XX")),
                ("</s>", self.original_tokenizer.convert_tokens_to_ids("</s>")),
            ],
        )


class NllbConverter(SpmConverter):

    """
    The `NllbConverter` class is a subclass of the `SpmConverter` class that provides methods for converting text
    using the NLLB (Neural Language Learning Base) model.
    It includes functions for generating vocabulary, determining the unknown token ID,
    and performing post-processing on the converted text.

    Attributes:
        None

    Methods:
        `vocab(proto)`:
            This method generates the vocabulary for the NLLB model.
            It takes a `proto` parameter that represents the model prototype.
            The method forwards the vocabulary list by combining the default tokens
            ('<s>', '<pad>', '</s>', '<unk>') with the tokens and scores from the `proto` object.
            It also includes additional tokens specific to various languages. The method returns the generated vocabulary.

        `unk_id(proto)`:
            This method determines the ID of the unknown token in the NLLB model.
            It takes a `proto` parameter that represents the model prototype. The method returns the ID of the unknown token.

        `post_processor()`:
            This method returns the post-processor for the NLLB model.
            It forwards a `TemplateProcessing` object with the template strings for single and pair conversions.
            It also includes special tokens for the 'eng_Latn' language. The method returns the forwarded post-processor object.

    Note:
        - The `NllbConverter` class does not have any instance-specific attributes or properties.

    Example:
        ```python
        >>> # Create an instance of NllbConverter
        >>> converter = NllbConverter()
        ...
        >>> # Generate the vocabulary
        >>> vocab = converter.vocab(proto)
        ...
        >>> # Determine the unknown token ID
        >>> unk_id = converter.unk_id(proto)
        ...
        >>> # Get the post-processor
        >>> post_processor = converter.post_processor()
        ```
    """
    def vocab(self, proto):
        '''
        This method generates a vocabulary list based on the input 'proto'.

        Args:
            self (NllbConverter): The instance of the NllbConverter class.
            proto: The input parameter representing the proto object.

        Returns:
            list: A list containing tuples of vocabulary items and their respective scores.

        Raises:
            None
        '''
        vocab = [
            ("<s>", 0.0),
            ("<pad>", 0.0),
            ("</s>", 0.0),
            ("<unk>", 0.0),
        ]
        vocab += [(piece.piece, piece.score) for piece in proto.pieces[3:]]
        vocab += [
            # fmt: off
            ('ace_Arab', 0.0), ('ace_Latn', 0.0), ('acm_Arab', 0.0), ('acq_Arab', 0.0), ('aeb_Arab', 0.0),
            ('afr_Latn', 0.0), ('ajp_Arab', 0.0), ('aka_Latn', 0.0), ('amh_Ethi', 0.0), ('apc_Arab', 0.0),
            ('arb_Arab', 0.0), ('ars_Arab', 0.0), ('ary_Arab', 0.0), ('arz_Arab', 0.0), ('asm_Beng', 0.0),
            ('ast_Latn', 0.0), ('awa_Deva', 0.0), ('ayr_Latn', 0.0), ('azb_Arab', 0.0), ('azj_Latn', 0.0),
            ('bak_Cyrl', 0.0), ('bam_Latn', 0.0), ('ban_Latn', 0.0), ('bel_Cyrl', 0.0), ('bem_Latn', 0.0),
            ('ben_Beng', 0.0), ('bho_Deva', 0.0), ('bjn_Arab', 0.0), ('bjn_Latn', 0.0), ('bod_Tibt', 0.0),
            ('bos_Latn', 0.0), ('bug_Latn', 0.0), ('bul_Cyrl', 0.0), ('cat_Latn', 0.0), ('ceb_Latn', 0.0),
            ('ces_Latn', 0.0), ('cjk_Latn', 0.0), ('ckb_Arab', 0.0), ('crh_Latn', 0.0), ('cym_Latn', 0.0),
            ('dan_Latn', 0.0), ('deu_Latn', 0.0), ('dik_Latn', 0.0), ('dyu_Latn', 0.0), ('dzo_Tibt', 0.0),
            ('ell_Grek', 0.0), ('eng_Latn', 0.0), ('epo_Latn', 0.0), ('est_Latn', 0.0), ('eus_Latn', 0.0),
            ('ewe_Latn', 0.0), ('fao_Latn', 0.0), ('pes_Arab', 0.0), ('fij_Latn', 0.0), ('fin_Latn', 0.0),
            ('fon_Latn', 0.0), ('fra_Latn', 0.0), ('fur_Latn', 0.0), ('fuv_Latn', 0.0), ('gla_Latn', 0.0),
            ('gle_Latn', 0.0), ('glg_Latn', 0.0), ('grn_Latn', 0.0), ('guj_Gujr', 0.0), ('hat_Latn', 0.0),
            ('hau_Latn', 0.0), ('heb_Hebr', 0.0), ('hin_Deva', 0.0), ('hne_Deva', 0.0), ('hrv_Latn', 0.0),
            ('hun_Latn', 0.0), ('hye_Armn', 0.0), ('ibo_Latn', 0.0), ('ilo_Latn', 0.0), ('ind_Latn', 0.0),
            ('isl_Latn', 0.0), ('ita_Latn', 0.0), ('jav_Latn', 0.0), ('jpn_Jpan', 0.0), ('kab_Latn', 0.0),
            ('kac_Latn', 0.0), ('kam_Latn', 0.0), ('kan_Knda', 0.0), ('kas_Arab', 0.0), ('kas_Deva', 0.0),
            ('kat_Geor', 0.0), ('knc_Arab', 0.0), ('knc_Latn', 0.0), ('kaz_Cyrl', 0.0), ('kbp_Latn', 0.0),
            ('kea_Latn', 0.0), ('khm_Khmr', 0.0), ('kik_Latn', 0.0), ('kin_Latn', 0.0), ('kir_Cyrl', 0.0),
            ('kmb_Latn', 0.0), ('kon_Latn', 0.0), ('kor_Hang', 0.0), ('kmr_Latn', 0.0), ('lao_Laoo', 0.0),
            ('lvs_Latn', 0.0), ('lij_Latn', 0.0), ('lim_Latn', 0.0), ('lin_Latn', 0.0), ('lit_Latn', 0.0),
            ('lmo_Latn', 0.0), ('ltg_Latn', 0.0), ('ltz_Latn', 0.0), ('lua_Latn', 0.0), ('lug_Latn', 0.0),
            ('luo_Latn', 0.0), ('lus_Latn', 0.0), ('mag_Deva', 0.0), ('mai_Deva', 0.0), ('mal_Mlym', 0.0),
            ('mar_Deva', 0.0), ('min_Latn', 0.0), ('mkd_Cyrl', 0.0), ('plt_Latn', 0.0), ('mlt_Latn', 0.0),
            ('mni_Beng', 0.0), ('khk_Cyrl', 0.0), ('mos_Latn', 0.0), ('mri_Latn', 0.0), ('zsm_Latn', 0.0),
            ('mya_Mymr', 0.0), ('nld_Latn', 0.0), ('nno_Latn', 0.0), ('nob_Latn', 0.0), ('npi_Deva', 0.0),
            ('nso_Latn', 0.0), ('nus_Latn', 0.0), ('nya_Latn', 0.0), ('oci_Latn', 0.0), ('gaz_Latn', 0.0),
            ('ory_Orya', 0.0), ('pag_Latn', 0.0), ('pan_Guru', 0.0), ('pap_Latn', 0.0), ('pol_Latn', 0.0),
            ('por_Latn', 0.0), ('prs_Arab', 0.0), ('pbt_Arab', 0.0), ('quy_Latn', 0.0), ('ron_Latn', 0.0),
            ('run_Latn', 0.0), ('rus_Cyrl', 0.0), ('sag_Latn', 0.0), ('san_Deva', 0.0), ('sat_Beng', 0.0),
            ('scn_Latn', 0.0), ('shn_Mymr', 0.0), ('sin_Sinh', 0.0), ('slk_Latn', 0.0), ('slv_Latn', 0.0),
            ('smo_Latn', 0.0), ('sna_Latn', 0.0), ('snd_Arab', 0.0), ('som_Latn', 0.0), ('sot_Latn', 0.0),
            ('spa_Latn', 0.0), ('als_Latn', 0.0), ('srd_Latn', 0.0), ('srp_Cyrl', 0.0), ('ssw_Latn', 0.0),
            ('sun_Latn', 0.0), ('swe_Latn', 0.0), ('swh_Latn', 0.0), ('szl_Latn', 0.0), ('tam_Taml', 0.0),
            ('tat_Cyrl', 0.0), ('tel_Telu', 0.0), ('tgk_Cyrl', 0.0), ('tgl_Latn', 0.0), ('tha_Thai', 0.0),
            ('tir_Ethi', 0.0), ('taq_Latn', 0.0), ('taq_Tfng', 0.0), ('tpi_Latn', 0.0), ('tsn_Latn', 0.0),
            ('tso_Latn', 0.0), ('tuk_Latn', 0.0), ('tum_Latn', 0.0), ('tur_Latn', 0.0), ('twi_Latn', 0.0),
            ('tzm_Tfng', 0.0), ('uig_Arab', 0.0), ('ukr_Cyrl', 0.0), ('umb_Latn', 0.0), ('urd_Arab', 0.0),
            ('uzn_Latn', 0.0), ('vec_Latn', 0.0), ('vie_Latn', 0.0), ('war_Latn', 0.0), ('wol_Latn', 0.0),
            ('xho_Latn', 0.0), ('ydd_Hebr', 0.0), ('yor_Latn', 0.0), ('yue_Hant', 0.0), ('zho_Hans', 0.0),
            ('zho_Hant', 0.0), ('zul_Latn', 0.0)
            # fmt: on
        ]
        vocab += [("<mask>", 0.0)]
        return vocab

    def unk_id(self, proto):
        """
        This method 'unk_id' in the class 'NllbConverter' takes two parameters: self and proto.

        Args:
            self: A reference to the current instance of the class NllbConverter.
                - Type: NllbConverter
                - Purpose: To access and modify the attributes and methods of the current instance.
                - Restrictions: None

            proto: An input parameter representing the proto object.
                - Type: Any
                - Purpose: To provide the proto object for processing.
                - Restrictions: None

        Returns:
            None.

        Raises:
            None
        """
        return 3

    def post_processor(self):
        """
        This method post_processor in the class NllbConverter is responsible for performing template processing.

        Args:
            self: An instance of the NllbConverter class.

        Returns:
            None.

        Raises:
            None
        """
        return processors.TemplateProcessing(
            single="eng_Latn $A </s>",
            pair="eng_Latn $A $B </s>",
            special_tokens=[
                ("eng_Latn", self.original_tokenizer.convert_tokens_to_ids("eng_Latn")),
                ("</s>", self.original_tokenizer.convert_tokens_to_ids("</s>")),
            ],
        )


class SeamlessM4TConverter(SpmConverter):

    """
    The 'SeamlessM4TConverter' class represents a seamless converter for M4T (Model for Translation) models.
    It inherits from the 'SpmConverter' class and provides methods for vocabulary generation, obtaining
    the unknown token ID, and defining post-processing templates.

    The class includes the following methods:

    - vocab(proto): Generates the vocabulary for the M4T model based on the provided proto object.
    - unk_id(proto): Retrieves the unknown token ID from the original tokenizer.
    - post_processor(): Defines the post-processing template for the M4T model, including special tokens and processing rules.

    Note:
        This docstring does not include the method signatures or any other code, as per the provided instructions.
    """
    def vocab(self, proto):
        """
        This method 'vocab' is a member of the class 'SeamlessM4TConverter' and is used to generate a vocabulary
        list from a given protocol.

        Args:
            self:
                An instance of the 'SeamlessM4TConverter' class.

                - Type: 'SeamlessM4TConverter' object
                - Purpose: To access the attributes and methods of the class.

            proto:
                The protocol object from which the vocabulary list will be generated.

                - Type: Any object
                - Purpose: To extract the pieces from the protocol and add them to the vocabulary list.
                - Restrictions: It is expected that the 'proto' object has a 'pieces' attribute.

        Returns:
            vocab:
                The generated vocabulary list.

                - Type: List of tuples
                - Purpose: To provide a list of pieces from the protocol, along with their scores.
                - Restrictions: None

        Raises:
            None
        """
        vocab = [
            ("<pad>", 0.0),
            ("<unk>", 0.0),
            ("<s>", 0.0),
            ("</s>", 0.0),
        ]
        vocab += [(piece.piece, piece.score) for piece in proto.pieces[3:]]
        return vocab

    def unk_id(self, proto):
        """
        This method 'unk_id' is defined in the class 'SeamlessM4TConverter'
         and is used to retrieve the unknown token id from the original tokenizer.

        Args:
            self (object): The instance of the 'SeamlessM4TConverter' class invoking this method.
            proto (object): The parameter 'proto' represents the prototype or model for which the unknown token id
                needs to be retrieved. It is of an unspecified type.

        Returns:
            None:
                This method returns a value of type 'None' which indicates
                that no specific token id was found for the given prototype.

        Raises:
            This method does not explicitly raise any exceptions.
        """
        return self.original_tokenizer.unk_token_id

    def post_processor(self):
        """
        This method post_processor is used in the class SeamlessM4TConverter to perform post-processing tasks on text data.

        Args:
            self:
                An instance of the class SeamlessM4TConverter. It is used to access the attributes and methods of the class.

        Returns:
            None.

        Raises:
            None
        """
        return processors.TemplateProcessing(
            single="__eng__ $A </s>",
            pair="__eng__ $A $B </s>",
            special_tokens=[
                ("__eng__", self.original_tokenizer.convert_tokens_to_ids("__eng__")),
                ("</s>", self.original_tokenizer.convert_tokens_to_ids("</s>")),
            ],
        )


class XLMRobertaConverter(SpmConverter):

    """
    The XLMRobertaConverter class represents a converter for XLM-Roberta models.
    It inherits from SpmConverter and provides methods for vocabulary creation and post-processing.

    Attributes:
        proto: The protocol buffer object containing the model pieces and scores.

    Methods:
        vocab(proto):
            Returns the vocabulary list for the XLM-Roberta model, including special tokens and their scores.

        unk_id(proto):
            Returns the ID for the unknown token in the XLM-Roberta model.

        post_processor():
            Returns the post-processor for the XLM-Roberta model, including template processing and special tokens conversion.
    """
    def vocab(self, proto):
        """
        Converts a Proto object to a vocabulary list.

        Args:
            self (XLMRobertaConverter): The instance of the XLMRobertaConverter class.
            proto: The Proto object containing pieces for the vocabulary.

        Returns:
            list: A list of tuples representing the vocabulary, where each tuple contains a piece and its corresponding score.

        Raises:
            None.
        """
        vocab = [
            ("<s>", 0.0),
            ("<pad>", 0.0),
            ("</s>", 0.0),
            ("<unk>", 0.0),
        ]
        vocab += [(piece.piece, piece.score) for piece in proto.pieces[3:]]
        vocab += [("<mask>", 0.0)]
        return vocab

    def unk_id(self, proto):
        """
        Method unk_id in the class XLMRobertaConverter.

        Args:
            self (object): The instance of the class XLMRobertaConverter.
            proto (any): A parameter of unspecified type and purpose.

        Returns:
            None.

        Raises:
            None.
        """
        unk_id = 3
        return unk_id

    def post_processor(self):
        """
        This method post_processor is a part of the XLMRobertaConverter class and is responsible for
        performing template processing.

        Args:
            self (XLMRobertaConverter): The instance of the XLMRobertaConverter class.

        Returns:
            None.

        Raises:
            None
        """
        return processors.TemplateProcessing(
            single="<s> $A </s>",
            pair="<s> $A </s> </s> $B </s>",
            special_tokens=[
                ("<s>", self.original_tokenizer.convert_tokens_to_ids("<s>")),
                ("</s>", self.original_tokenizer.convert_tokens_to_ids("</s>")),
            ],
        )


class XLNetConverter(SpmConverter):

    """
    XLNetConverter is a Python class that provides methods for converting data using XLNet model specifications.
    This class inherits from SpmConverter and includes methods for vocabulary generation, normalization, and post-processing.

    Methods:
        vocab: Generates a vocabulary mapping pieces to scores, adjusting scores based on comma presence.
        normalizer: Constructs a sequence of normalizers based on XLNet model specifications,
            including replacements, lowercasing, and character mappings.
        post_processor: Defines a post-processing template for XLNet conversion, specifying special tokens and their
            corresponding IDs.

    Note:
        This class assumes input data conforms to XLNet model requirements and is intended for use in XLNet data
        processing tasks.
    """
    def vocab(self, proto):
        """
        This method, vocab, is defined within the XLNetConverter class and takes two parameters: self and proto.

        Args:
            self (XLNetConverter): The instance of the XLNetConverter class.
            proto (object): The input proto object containing pieces.

        Returns:
            None.

        Raises:
            None.
        """
        return [
            (piece.piece, piece.score) if check_number_comma(piece.piece) else (piece.piece, piece.score - 100)
            for piece in proto.pieces
        ]

    def normalizer(self, proto):
        """
        Normalizes the given 'proto' using a sequence of normalizers.

        Args:
            self (XLNetConverter): An instance of the XLNetConverter class.
            proto: The input proto to be normalized.

        Returns:
            None

        Raises:
            None

        This method applies a series of normalizers to the input 'proto' in order to standardize and preprocess the text.
        The normalizers are applied in the following order:

        1. Replace '``' with double quotes ('"') and "''" with double quotes ('"').
        2. If the 'keep_accents' flag is not set in the original_tokenizer, then:
           - Apply the NFKD (Normalization Form KD) normalizer.
           - Strip accents from the text.
        3. If the 'do_lower_case' flag is set in the original_tokenizer, then:
           - Convert the text to lowercase.
        4. If a precompiled character map is provided in the proto object,
        then apply the Precompiled normalizer using the given character map.
        5. Replace multiple consecutive spaces with a single space.

        Note:
            The normalizers are applied in the specified order to ensure proper text normalization.

        The 'proto' parameter represents the text to be normalized.
        The method returns None as the normalized text is directly modified in-place.

        Example:
            ```python
            >>> converter = XLNetConverter()
            >>> proto = "Example text with ``quotes'' and multiple spaces."
            >>> converter.normalizer(proto)
            ```
        The 'proto' text will be normalized based on the applied normalizers.
        """
        list_normalizers = [
            normalizers.Replace("``", '"'),
            normalizers.Replace("''", '"'),
        ]
        if not self.original_tokenizer.keep_accents:
            list_normalizers.append(normalizers.NFKD())
            list_normalizers.append(normalizers.StripAccents())
        if self.original_tokenizer.do_lower_case:
            list_normalizers.append(normalizers.Lowercase())

        precompiled_charsmap = proto.normalizer_spec.precompiled_charsmap

        if precompiled_charsmap:
            list_normalizers.append(normalizers.Precompiled(precompiled_charsmap))

        list_normalizers.append(normalizers.Replace(Regex(" {2,}"), " "))
        return normalizers.Sequence(list_normalizers)

    def post_processor(self):
        """
        Post-processes the output of the XLNetConverter.

        Args:
            self: An instance of the XLNetConverter class.

        Returns:
            None: The method modifies the XLNetConverter instance in-place.

        Raises:
            None.

        Description:
            This method applies post-processing to the output of the XLNetConverter.
            It uses the following template for single sequences:
                '$A:0 <sep>:0 <cls>:2'

            And the following template for pair sequences:
                '$A:0 <sep>:0 $B:1 <sep>:1 <cls>:2'

            The method also sets special tokens, which are represented by tuples of the form
            ('<sep>', self.original_tokenizer.convert_tokens_to_ids('<sep>')) and ('<cls>',
            self.original_tokenizer.convert_tokens_to_ids('<cls>')).

            The post-processed output is stored in the XLNetConverter instance for further processing or analysis.
        """
        return processors.TemplateProcessing(
            single="$A:0 <sep>:0 <cls>:2",
            pair="$A:0 <sep>:0 $B:1 <sep>:1 <cls>:2",
            special_tokens=[
                ("<sep>", self.original_tokenizer.convert_tokens_to_ids("<sep>")),
                ("<cls>", self.original_tokenizer.convert_tokens_to_ids("<cls>")),
            ],
        )


class ReformerConverter(SpmConverter):

    """
    The ReformerConverter class represents a specialized converter for reforming data.
    It inherits from the SpmConverter class and provides additional functionality for data transformation.

    Attributes:
        Inherits all attributes from the SpmConverter class.

    Methods:
        reform_data(data): Reform the given data according to specific requirements.

    Usage:
        To use the ReformerConverter, instantiate the class and call the reform_data method with the data to
        be transformed.
    """


class RemBertConverter(SpmConverter):

    """
    RemBertConverter is a Python class that serves as a converter for text normalization and post-processing in
    language processing tasks. It inherits from SpmConverter and provides functionalities for normalizing input text
    using a sequence of predefined normalizers, and for post-processing the output using a specific template structure.

    Attributes:
        original_tokenizer: The original tokenizer used for tokenization.

    Methods:
        normalizer(proto):
            Combines a list of normalizers including replacements, accent stripping, lowercase conversion,
            and precompiled character mapping to normalize the input text.

        post_processor():
            Defines a post-processing template for the output text, including special tokens like '[CLS]' and '[SEP]',
            and their corresponding token IDs from the original tokenizer.

    This class encapsulates the logic for converting text data in a consistent and efficient manner for downstream
    language processing tasks.
    """
    # Inspired from AlbertConverter
    def normalizer(self, proto):
        """
        This method normalizes text according to the specified rules.

        Args:
            self (RemBertConverter): An instance of the RemBertConverter class.
            proto: An object representing the text to be normalized.

        Returns:
            None: This method does not return any value directly.
                The normalization process is applied to the input text.

        Raises:
            TypeError: If the input parameters are not of the expected types.
            ValueError: If there are issues with the normalization process.
            AttributeError: If there are problems accessing attributes of the input objects.
            Exception: For any other unforeseen errors during the normalization process.
        """
        list_normalizers = [
            normalizers.Replace("``", '"'),
            normalizers.Replace("''", '"'),
            normalizers.Replace(Regex(" {2,}"), " "),
        ]
        if not self.original_tokenizer.keep_accents:
            list_normalizers.append(normalizers.NFKD())
            list_normalizers.append(normalizers.StripAccents())
        if self.original_tokenizer.do_lower_case:
            list_normalizers.append(normalizers.Lowercase())

        precompiled_charsmap = proto.normalizer_spec.precompiled_charsmap

        if precompiled_charsmap:
            list_normalizers.append(normalizers.Precompiled(precompiled_charsmap))

        return normalizers.Sequence(list_normalizers)

    def post_processor(self):
        """
        Method post_processor in the class RemBertConverter.

        Args:
            self: This parameter refers to the instance of the RemBertConverter class.

        Returns:
            None.

        Raises:
            None.
        """
        return processors.TemplateProcessing(
            single="[CLS]:0 $A:0 [SEP]:0",
            pair="[CLS]:0 $A:0 [SEP]:0 $B:1 [SEP]:1",
            special_tokens=[
                ("[CLS]", self.original_tokenizer.convert_tokens_to_ids("[CLS]")),
                ("[SEP]", self.original_tokenizer.convert_tokens_to_ids("[SEP]")),
            ],
        )


class BertGenerationConverter(SpmConverter):

    """
    This class represents a converter for converting text data into a format suitable for Bert generation models.
    It inherits from the SpmConverter class.

    Attributes:
        None.

    Methods:
        convert_text_to_tokens: Converts input text into tokens suitable for Bert generation models.
        convert_tokens_to_text: Converts tokens back into text format.
        tokenize_text: Tokenizes the input text using Bert tokenizer.
        detokenize_text: Detokenizes the text tokens back into human-readable text.
    """


class PegasusConverter(SpmConverter):

    """
    The `PegasusConverter` class is a subclass of `SpmConverter` that provides methods for
    vocabulary generation, unknown token identification, pre-tokenization, and post-processing.

    The class contains the following methods:

    1. `vocab(self, proto)`:
        This method generates the vocabulary for the Pegasus model.
        It takes a `proto` parameter as input and returns a list of tuples representing the vocabulary.
        The vocabulary includes special tokens such as the padding token, end-of-sequence token, mask token,
        and unknown tokens.
        The method also includes additional unknown tokens based on the offset and pieces from the `proto` parameter.
    2. `unk_id(self, proto)`:
        This method returns the unknown token ID for the Pegasus model.
        It takes a `proto` parameter as input and calculates the unknown token ID based on the `unk_id` and offset
        from the `proto` parameter.
    3. `pre_tokenizer(self, replacement, add_prefix_space)`:
        This method returns the pre-tokenizer for the Pegasus model.
        It takes `replacement` and `add_prefix_space` parameters as input and uses pre-tokenizers to split the input
        sequence into tokens.
        The pre-tokenizers include whitespace splitting and metaspace replacement.
    4. `post_processor(self)`:
        This method returns the post-processor for the Pegasus model.
        It sets the end-of-sequence token and defines special tokens for template processing.
        The method returns the post-processor with the specified special tokens.

    Note:
        The `PegasusConverter` class inherits from the `SpmConverter` class, which is not explicitly defined in
        this code snippet.

    Please refer to the code implementation for more details on the class structure and usage.
    """
    def vocab(self, proto):
        """
        Method 'vocab' in the class 'PegasusConverter'.

        This method generates a vocabulary list based on the provided 'proto' object and the tokenizer settings.

        Args:
            self:
                Instance of the 'PegasusConverter' class.

                - Purpose: Represents the current instance of the class.
                - Restrictions: None

            proto:
                Object.

                - Purpose: The 'proto' object containing information to build the vocabulary.
                - Restrictions: Should be a valid object.

        Returns:
            list:
                A list of tuples representing the vocabulary.

                - Purpose: The generated vocabulary list containing tokens and their corresponding scores.
                - Each tuple consists of a token and its score.

        Raises:
            None
        """
        vocab = [
            (self.original_tokenizer.pad_token, 0.0),
            (self.original_tokenizer.eos_token, 0.0),
        ]

        if self.original_tokenizer.mask_token_sent is not None:
            vocab += [(self.original_tokenizer.mask_token_sent, 0.0)]

        if (
            self.original_tokenizer.mask_token is not None
            and self.original_tokenizer.mask_token_id < self.original_tokenizer.offset
        ):
            vocab += [(self.original_tokenizer.mask_token, 0.0)]

        vocab += [(f"<unk_{i}>", -100.0) for i in range(2, self.original_tokenizer.offset)]
        vocab += [(piece.piece, piece.score) for piece in proto.pieces[2:]]
        return vocab

    def unk_id(self, proto):
        """
        Returns the unknown id of a given proto object.

        Args:
            self (PegasusConverter): The instance of the PegasusConverter class.
            proto: The proto object for which the unknown id needs to be retrieved.

        Returns:
            None.

        Raises:
            None.
        """
        return proto.trainer_spec.unk_id + self.original_tokenizer.offset

    def pre_tokenizer(self, replacement, add_prefix_space):
        """
        pre_tokenizer method in the PegasusConverter class.

        This method pre-tokenizes the input text using a sequence of pre-tokenizers.

        Args:
            self (object): The instance of the PegasusConverter class.
            replacement (str): The replacement string to be used by the Metaspace pre-tokenizer.
                It represents the string that replaces the spaces in the input text.
            add_prefix_space (bool): A boolean flag indicating whether to add a prefix space before the replacement string.
                If True, a space will be added before the replacement string; otherwise, no space will be added.

        Returns:
            None: This method does not return any value, as the pre-tokenization is performed in place.

        Raises:
            None: This method does not explicitly raise any exceptions.
                Note: Any exceptions raised by the pre-tokenizers.WhitespaceSplit() and pre-tokenizers.Metaspace() methods
                    will be propagated as per their respective documentation.
        """
        return pre_tokenizers.Sequence(
            [
                pre_tokenizers.WhitespaceSplit(),
                pre_tokenizers.Metaspace(replacement=replacement, add_prefix_space=add_prefix_space),
            ]
        )

    def post_processor(self):
        """
        Post-processes the tokenized output of the PegasusConverter class.

        Args:
            self: An instance of the PegasusConverter class.

        Returns:
            None: The method modifies the tokenized output in-place.

        Raises:
            None.

        Description:
            This method performs post-processing on the tokenized output generated by the original_tokenizer of the
            PegasusConverter class. It adds special tokens and templates to the tokenized sequences, making them
            compatible for input to Pegasus model.

            The post_processor method takes no additional arguments besides 'self'.
            It accesses the original_tokenizer and eos_token properties of the PegasusConverter instance to retrieve
            the end-of-sequence (EOS) token and its corresponding ID. The EOS token is then added to the tokenized
            sequences as a special token.

            The method uses the TemplateProcessing processor to add special tokens and templates to the tokenized sequences.
            The single template consists of the '$A' token followed by the EOS token, while the pair
            template consists of the '$A', '$B', and EOS tokens.
            The special_tokens argument is set to [(eos, eos_token_id)] to include the EOS token in the list of special tokens.

        Note:
            - The post_processor method modifies the tokenized output in-place and does not return any value.
            - The original_tokenizer must be set before calling this method, otherwise it will raise an AttributeError.
            - This method should be called after the tokenization process to prepare the tokenized sequences for input
            to the Pegasus model.
        """
        eos = self.original_tokenizer.eos_token
        special_tokens = [
            (eos, self.original_tokenizer.eos_token_id),
        ]
        return processors.TemplateProcessing(single=["$A", eos], pair=["$A", "$B", eos], special_tokens=special_tokens)


class T5Converter(SpmConverter):

    """
    T5Converter represents a Python class that is responsible for converting T5 tokens using specific conversion methods.
    This class inherits from SpmConverter and includes methods for vocabulary generation and post-processing of tokens.

    Methods:
        vocab(self, proto): Generates the vocabulary for T5 tokens based on the specified proto object by including the
            piece and its corresponding score. It also adds special tokens designated as extra IDs with a score of 0.0.

        post_processor(self): Returns a TemplateProcessing object with predefined single and pair special tokens for
            T5 conversion, along with the conversion of '</s>' token to its corresponding ID using the original tokenizer.

    Note:
        Ensure to properly initialize an instance of T5Converter with the necessary tokenizer and configurations
        before utilizing its methods for token conversion.
    """
    def vocab(self, proto):
        """
        This method 'vocab' is defined in the class 'T5Converter'. It takes 2 parameters: self and proto.

        Args:
            self: An instance of the class 'T5Converter'. It is a reference to the current instance of the class.
            proto: An object representing the proto data structure.
                It is used to extract information about the vocabulary pieces.

        Returns:
            The method returns a list of tuples representing the vocabulary.
                Each tuple contains a vocabulary piece and its score.

        Raises:
            This method does not raise any exceptions.
        """
        num_extra_ids = self.original_tokenizer._extra_ids
        vocab = [(piece.piece, piece.score) for piece in proto.pieces]
        vocab += [(f"<extra_id_{i}>", 0.0) for i in range(num_extra_ids - 1, -1, -1)]
        return vocab

    def post_processor(self):
        """
        Post-processes the output of the T5 model.

        Args:
            self: An instance of the T5Converter class.

        Returns:
            None: This method modifies the instance in-place.

        Raises:
            None.

        This method applies post-processing to the output of the T5 model. It uses the following parameters:

        - single: A list containing two elements, '$A' and '</s>', specifying the start and end tokens for
        single-sentence inputs.
        - pair: A list containing four elements, '$A', '</s>', '$B', and '</s>', s
        pecifying the start and end tokens for paired-sentence inputs.
        - special_tokens: A list of tuples, where each tuple contains a special token and its corresponding ID.
        In this case, the special token is '</s>' and its ID is obtained using the original_tokenizer.

        The post-processed output is stored in the instance's internal data structure for further processing or retrieval.
        """
        return processors.TemplateProcessing(
            single=["$A", "</s>"],
            pair=["$A", "</s>", "$B", "</s>"],
            special_tokens=[
                ("</s>", self.original_tokenizer.convert_tokens_to_ids("</s>")),
            ],
        )

class UdopConverter(SpmConverter):
    def post_processor(self):
        return processors.TemplateProcessing(
            single=["$A", "</s>"],
            pair=["$A", "</s>", "$B", "</s>"],
            special_tokens=[
                ("</s>", self.original_tokenizer.convert_tokens_to_ids("</s>")),
            ],
        )
class WhisperConverter(Converter):

    """
    The 'WhisperConverter' class is a subclass of 'Converter' and represents a specialized converter for tokenizing
    text using the Whisper tokenizer.

    The class provides a method 'converted' that takes no arguments and returns a 'Tokenizer' object.
    This 'Tokenizer' object is created using the original tokenizer's vocabulary and merges, along with certain
    configurable options. The 'Tokenizer' object is specifically configured with a Byte-Level pre-tokenizer and decoder.

    Additionally, the 'converted' method sets up the post-processing step of the tokenizer by defining a template for
    processing tokenized sequences.
    The template includes the original tokenizer's prefix tokens, along with the special tokens '$A' and '$B' to
    represent the sequence inputs.
    The end-of-sequence token is also included in the template for both single and pair sequences.
    This post-processor is responsible for mapping the tokenized sequence back to the original text.

    Note that 'WhisperConverter' inherits from the 'Converter' class, which likely provides additional functionality
    and methods to handle text conversion.

    Example:
        ```python
        >>> converter = WhisperConverter()
        >>> tokenizer = converter.converted()
        >>> tokenized_text = tokenizer.encode("Hello, world!")
        ```
    """
    def converted(self) -> Tokenizer:
        """
        This method converts the original tokenizer to a Tokenizer object.

        Args:
            self: An instance of the WhisperConverter class.
                This parameter refers to the current instance of the WhisperConverter class.

        Returns:
            Tokenizer: A Tokenizer object.
                This method returns a Tokenizer object that represents the converted original tokenizer.

        Raises:
            None.
        """
        vocab = self.original_tokenizer.encoder
        merges = list(self.original_tokenizer.bpe_ranks.keys())

        tokenizer = Tokenizer(
            BPE(
                vocab=vocab,
                merges=merges,
                dropout=None,
                continuing_subword_prefix="",
                end_of_word_suffix="",
                fuse_unk=False,
            )
        )

        tokenizer.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=self.original_tokenizer.add_prefix_space)
        tokenizer.decoder = decoders.ByteLevel()

        prefix_token_ids = self.original_tokenizer.prefix_tokens
        prefixes = self.original_tokenizer.convert_ids_to_tokens(prefix_token_ids)
        eos = self.original_tokenizer.eos_token
        eos_token_id = self.original_tokenizer.eos_token_id
        prefix_template = " ".join([f"{token}:0" for token in prefixes])
        tokenizer.post_processor = processors.TemplateProcessing(
            single=f"{prefix_template} $A:0 {eos}:0",
            pair=f"{prefix_template} $A:0 $B:1 {eos}:1",
            special_tokens=[
                (eos, eos_token_id),
                *zip(prefixes, prefix_token_ids),
            ],
        )

        return tokenizer


class BigBirdConverter(SpmConverter):

    """
    This class represents a BigBirdConverter that inherits from SpmConverter.
    It includes a post_processor method that defines processing rules for template tokens in a specified format.
    The post_processor method returns a processors.
    TemplateProcessing object configured with single and pair token templates,
    as well as special tokens mapped to their respective token IDs using the original_tokenizer provided.
    """
    def post_processor(self):
        """
        Method to perform post-processing for BigBirdConverter.

        Args:
            self: BigBirdConverter instance. The self parameter refers to the instance of the class.

        Returns:
            None.

        Raises:
            None.
        """
        return processors.TemplateProcessing(
            single="[CLS]:0 $A:0 [SEP]:0",
            pair="[CLS]:0 $A:0 [SEP]:0 $B:1 [SEP]:1",
            special_tokens=[
                ("[CLS]", self.original_tokenizer.convert_tokens_to_ids("[CLS]")),
                ("[SEP]", self.original_tokenizer.convert_tokens_to_ids("[SEP]")),
            ],
        )


class CLIPConverter(Converter):

    """
    This class represents a converter for CLIP tokenizer. It is a subclass of the Converter class.

    The CLIPConverter class is responsible for converting the original tokenizer to a CLIP tokenizer.
    It provides a method, converted, that returns the converted tokenizer.

    Attributes:
        original_tokenizer (Tokenizer): The original tokenizer that needs to be converted.

    Methods:
        converted: Converts the original tokenizer to a CLIP tokenizer and returns the converted tokenizer.

    """
    def converted(self) -> Tokenizer:
        """
        Converts the original tokenizer to a new Tokenizer.

        Args:
            self: CLIPConverter object. The instance of the CLIPConverter class.

        Returns:
            Tokenizer:
                The converted Tokenizer object containing the transformed vocabulary, merges, special tokens,
                and processing configurations.

        Raises:
            None.
        """
        vocab = self.original_tokenizer.encoder
        merges = list(self.original_tokenizer.bpe_ranks.keys())
        unk_token = self.original_tokenizer.unk_token

        tokenizer = Tokenizer(
            BPE(
                vocab=vocab,
                merges=merges,
                dropout=None,
                continuing_subword_prefix="",
                end_of_word_suffix="</w>",
                fuse_unk=False,
                unk_token=str(unk_token),
            )
        )

        tokenizer.normalizer = normalizers.Sequence(
            [normalizers.NFC(), normalizers.Replace(Regex(r"\s+"), " "), normalizers.Lowercase()]
        )
        tokenizer.pre_tokenizer = pre_tokenizers.Sequence(
            [
                pre_tokenizers.Split(
                    Regex(r"""'s|'t|'re|'ve|'m|'ll|'d|[\p{L}]+|[\p{N}]|[^\s\p{L}\p{N}]+"""),
                    behavior="removed",
                    invert=True,
                ),
                pre_tokenizers.ByteLevel(add_prefix_space=False),
            ]
        )
        tokenizer.decoder = decoders.ByteLevel()

        # Hack to have a ByteLevel and TemplaceProcessor
        tokenizer.post_processor = processors.RobertaProcessing(
            sep=(self.original_tokenizer.eos_token, self.original_tokenizer.eos_token_id),
            cls=(self.original_tokenizer.bos_token, self.original_tokenizer.bos_token_id),
            add_prefix_space=False,
            trim_offsets=False,
        )
        return tokenizer


class LayoutLMv2Converter(Converter):

    """
    The LayoutLMv2Converter class represents a converter for transforming an original tokenizer into a Tokenizer instance.
    It inherits from the Converter class. The converted method within this class takes the original tokenizer
    and creates a new Tokenizer instance based on its configurations.
    It sets various attributes and properties for the new Tokenizer, including vocab, normalizer, pre_tokenizer, post_processor,
    and decoder, based on the configurations of the original tokenizer.
    The method then returns the new Tokenizer instance.

    """
    def converted(self) -> Tokenizer:
        """
        Converts the original tokenizer to a new Tokenizer with specific configurations for LayoutLMv2.

        Args:
            self (LayoutLMv2Converter): The instance of the LayoutLMv2Converter class.

        Returns:
            Tokenizer: A new Tokenizer object with customized settings for LayoutLMv2 conversion.

        Raises:
            None
        """
        vocab = self.original_tokenizer.vocab
        tokenizer = Tokenizer(WordPiece(vocab, unk_token=str(self.original_tokenizer.unk_token)))

        tokenize_chinese_chars = False
        strip_accents = False
        do_lower_case = True
        if hasattr(self.original_tokenizer, "basic_tokenizer"):
            tokenize_chinese_chars = self.original_tokenizer.basic_tokenizer.tokenize_chinese_chars
            strip_accents = self.original_tokenizer.basic_tokenizer.strip_accents
            do_lower_case = self.original_tokenizer.basic_tokenizer.do_lower_case

        tokenizer.normalizer = normalizers.BertNormalizer(
            clean_text=True,
            handle_chinese_chars=tokenize_chinese_chars,
            strip_accents=strip_accents,
            lowercase=do_lower_case,
        )
        tokenizer.pre_tokenizer = pre_tokenizers.BertPreTokenizer()

        cls = str(self.original_tokenizer.cls_token)
        sep = str(self.original_tokenizer.sep_token)
        cls_token_id = self.original_tokenizer.cls_token_id
        sep_token_id = self.original_tokenizer.sep_token_id

        tokenizer.post_processor = processors.TemplateProcessing(
            single=f"{cls}:0 $A:0 {sep}:0",
            pair=f"{cls}:0 $A:0 {sep}:0 $B:1 {sep}:1",
            special_tokens=[
                (cls, cls_token_id),
                (sep, sep_token_id),
            ],
        )
        tokenizer.decoder = decoders.WordPiece(prefix="##")

        return tokenizer


class BlenderbotConverter(Converter):

    """
    The BlenderbotConverter class is a subclass of the Converter class and is responsible for converting textual data
    using the Blenderbot model.

    BlenderbotConverter inherits all the functionalities and attributes of the Converter class and adds additional methods
    to handle the specific conversion requirements of the Blenderbot model.

    The main method of BlenderbotConverter is 'converted', which takes no arguments and returns a Tokenizer object.
    This method performs the conversion process by utilizing the original_tokenizer of the class and creating
    a new Tokenizer object with the necessary configurations.

    The conversion process involves extracting the vocabulary and merge information from the original_tokenizer,
    creating a new Tokenizer object with the extracted information, and configuring the pre-tokenizer, decoder,
    and post-processor attributes of the Tokenizer.

    The 'converted' method returns the newly created Tokenizer object, which can be used for further text conversion
    using the Blenderbot model.

    Note:
        The BlenderbotConverter class assumes that the original_tokenizer attribute is properly initialized before
        calling the 'converted' method.
    """
    def converted(self) -> Tokenizer:
        """
        Converts the original tokenizer to a new Tokenizer object.

        Args:
            self: BlenderbotConverter - The instance of the BlenderbotConverter class.

        Returns:
            Tokenizer: A Tokenizer object representing the converted tokenizer.

        Raises:
            None.
        """
        ot = self.original_tokenizer
        vocab = ot.encoder
        merges = list(ot.bpe_ranks.keys())

        tokenizer = Tokenizer(
            BPE(
                vocab=vocab,
                merges=merges,
                dropout=None,
                continuing_subword_prefix="",
                end_of_word_suffix="",
                fuse_unk=False,
            )
        )

        tokenizer.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=ot.add_prefix_space)
        tokenizer.decoder = decoders.ByteLevel()
        tokenizer.post_processor = processors.TemplateProcessing(
            single=f"$A:0 {ot.eos_token}:0",
            special_tokens=[
                (ot.eos_token, ot.eos_token_id),
            ],
        )

        return tokenizer


class XGLMConverter(SpmConverter):

    """
    XGLMConverter is a Python class that represents a converter for XGLM models.
    It inherits from SpmConverter and provides methods for vocabulary generation, handling unknown tokens, and post-processing.

    Methods:
        vocab(proto): Generates the vocabulary for the XGLM model based on the provided proto object.
        unk_id(proto): Returns the ID for unknown tokens in the XGLM model.
        post_processor(): Returns a post-processor object for template processing in the XGLM model.

    The 'vocab' method forwards a vocabulary list that includes special tokens like '<s>', '<pad>', '<unk>', '</s>',
    and additional user-defined tokens.
    The 'unk_id' method returns the ID assigned to unknown tokens in the model.
    The 'post_processor' method returns a processor object for template processing, including single and pair processing
    with special tokens.

    Note:
        Ensure to provide the necessary proto object when calling the methods of this class for proper functionality.
    """
    def vocab(self, proto):
        """
        This method 'vocab' is defined in the class 'XGLMConverter'.

        Args:
            self (object): The instance of the class itself.
            proto (object): The input parameter representing a proto object.

        Returns:
            list: A list of tuples where each tuple represents a vocabulary item along with its score.
                The vocabulary includes standard tokens like '<s>', '<pad>', '</s>', '<unk>' followed by custom tokens
                derived from the 'proto' object's pieces. Lastly, it includes additional made-up words with a score of 0.0.

        Raises:
            This method does not raise any exceptions.
        """
        vocab = [
            ("<s>", 0.0),
            ("<pad>", 0.0),
            ("</s>", 0.0),
            ("<unk>", 0.0),
        ]
        vocab += [(piece.piece, piece.score) for piece in proto.pieces[3:]]
        # fmt: off
        vocab += [("<madeupword0>", 0.0), ("<madeupword1>", 0.0), ("<madeupword2>", 0.0), ("<madeupword3>", 0.0), ("<madeupword4>", 0.0), ("<madeupword5>", 0.0), ("<madeupword6>", 0.0)]
        # fmt: on
        return vocab

    def unk_id(self, proto):
        """
        This method 'unk_id' is defined in the class 'XGLMConverter' and is responsible for returning an unknown ID value.

        Args:
            self (object): The instance of the XGLMConverter class.
            proto (any): The input parameter representing the prototype.

        Returns:
            None: This method returns a value of type 'None' indicating that no specific value is returned.

        Raises:
            None.
        """
        unk_id = 3
        return unk_id

    def post_processor(self):
        """
        This method post_processor is defined within the XGLMConverter class to perform template processing.

        Args:
            self: XGLMConverter instance. Represents the current instance of the XGLMConverter class.

        Returns:
            None.

        Raises:
            None.
        """
        return processors.TemplateProcessing(
            single="</s> $A",
            pair="</s> $A </s> </s> $B",
            special_tokens=[
                ("<s>", self.original_tokenizer.convert_tokens_to_ids("<s>")),
                ("</s>", self.original_tokenizer.convert_tokens_to_ids("</s>")),
            ],
        )


class LlamaConverter(SpmConverter):
    """
    The LlamaConverter class represents a converter that handles tokenization, normalization,
    and preprocessing for language models using the Llama library.

    It includes methods for generating vocabulary, assigning unknown token IDs, configuring decoders,
    tokenizing input text, normalizing text, and preprocessing tokens for model input.

    The class inherits from SpmConverter and provides functionality for
    converting text data into a format suitable for language model training and inference.
    """
    handle_byte_fallback = True

    def vocab(self, proto):
        """
        This method is a part of the 'LlamaConverter' class and is used to generate a vocabulary list based on the
        given 'proto' input.

        Args:
            self: An instance of the 'LlamaConverter' class.
            proto: An input parameter of type 'proto' representing a protobuf object.

        Returns:
            None.

        Raises:
            None.

        This method first initializes the 'vocab' list with three default tuples: ('<unk>', 0.0), ('<s>', 0.0), and
        ('</s>', 0.0). Then, it iterates over the 'proto.pieces' list starting from the fourth element and appends
        each 'piece' and its 'score' to the 'vocab' list. The final 'vocab' list is then returned.
        """
        vocab = [
            ("<unk>", 0.0),
            ("<s>", 0.0),
            ("</s>", 0.0),
        ]
        vocab += [(piece.piece, piece.score) for piece in proto.pieces[3:]]
        return vocab

    def unk_id(self, proto):
        """
        unk_id method in the LlamaConverter class.

        Args:
            self (object): The instance of the LlamaConverter class.
            proto (any): The parameter representing the proto.

        Returns:
            None.

        Raises:
            None.
        """
        unk_id = 0
        return unk_id

    def decoder(self, replacement, add_prefix_space):
        """
        Decodes the input using a sequence of decoders.

        Args:
            self (LlamaConverter): The instance of the LlamaConverter class.
            replacement (str): The replacement string used to replace the '▁' character.
            add_prefix_space (bool): A flag indicating whether to add a space before the decoded content.

        Returns:
            None.

        Raises:
            None.
        """
        return decoders.Sequence(
            [
                decoders.Replace("▁", " "),
                decoders.ByteFallback(),
                decoders.Fuse(),
                decoders.Strip(content=" ", left=1),
            ]
        )

    def tokenizer(self, proto):
        """
        This method tokenizes the input proto using different tokenization algorithms based on the model type
        specified in the proto.

        Args:
            self: An instance of the LlamaConverter class. It is used to access the methods and attributes of the
                LlamaConverter class.
            proto: An input proto object containing the trainer_spec and vocab information required for tokenization.

        Returns:
            Tokenizer:
                A tokenizer object based on the model type specified in the input proto.
                The type of the tokenizer object varies based on the model type.

        Raises:
            RuntimeError:
                If the model type specified in the proto is not supported or if the file was trained with
                a different algorithm, a RuntimeError is raised.
        """
        model_type = proto.trainer_spec.model_type
        vocab_scores = self.vocab(proto)
        if model_type == 1:
            import tokenizers

            if version.parse(tokenizers.__version__) < version.parse("0.14.0"):
                tokenizer = Tokenizer(Unigram(vocab_scores, 0))
            else:
                tokenizer = Tokenizer(Unigram(vocab_scores, 0, byte_fallback=True))

        elif model_type == 2:
            _, merges = SentencePieceExtractor(self.original_tokenizer.vocab_file).extract(vocab_scores)
            bpe_vocab = {word: i for i, (word, _score) in enumerate(vocab_scores)}
            tokenizer = Tokenizer(
                BPE(bpe_vocab, merges, unk_token=proto.trainer_spec.unk_piece, fuse_unk=True, byte_fallback=True)
            )
            tokenizer.add_special_tokens(
                [
                    AddedToken("<unk>", normalized=False, special=True),
                    AddedToken("<s>", normalized=False, special=True),
                    AddedToken("</s>", normalized=False, special=True),
                ]
            )
        else:
            raise RuntimeError(
                "You're trying to run a `Unigram` model but you're file was trained with a different algorithm"
            )

        return tokenizer

    def normalizer(self, proto):
        """
        This method normalizer is defined within the class LlamaConverter.
        It takes two parameters: self, which refers to the instance of the class itself, and proto,
        which represents the input data to be normalized.

        Args:
            self (object): The instance of the LlamaConverter class.
                This parameter is required to access the class attributes and methods within the normalizer method.
            proto (object): The input data to be normalized.
                It should be a valid input for the normalization process.

        Returns:
            None.

        Raises:
            None.
        """
        return normalizers.Sequence(
            [
                normalizers.Prepend(prepend="▁"),
                normalizers.Replace(pattern=" ", content="▁"),
            ]
        )

    def pre_tokenizer(self, replacement, add_prefix_space):
        """
        Method to pre-process text before tokenization.

        Args:
            self (LlamaConverter): The instance of the LlamaConverter class.
            replacement (str): The replacement string to be used during text preprocessing.
            add_prefix_space (bool): A flag indicating whether a prefix space should be added.

        Returns:
            None.

        Raises:
            None.
        """
        return None

    def post_processor(self):
        """
        Method to perform post-processing after conversion in the LlamaConverter class.

        Args:
            self (object): The instance of the LlamaConverter class.
                This parameter refers to the current instance of the LlamaConverter class to work with its attributes
                and methods.

        Returns:
            None: This method does not return any value.
                The method post_processor returns None after performing the post-processing operations.

        Raises:
            None.
        """
        # the processor is defined in the LlamaTokenizerFast class.
        return None


class MarkupLMConverter(Converter):

    """A class for converting tokenizers to the MarkupLM format.

    This class inherits from the Converter class and provides functionality for converting tokenizers to the MarkupLM format.
    The MarkupLMConverter class takes an original tokenizer and converts it into a Tokenizer object
    with the necessary configurations for use in MarkupLM.

    The converted() method takes an original tokenizer and returns a Tokenizer object with the appropriate configuration
    for MarkupLM.
    The Tokenizer object is created with the original tokenizer's vocabulary and BPE merges.
    The dropout, continuing_subword_prefix, end_of_word_suffix, and fuse_unk parameters are set to default values.
    The unk_token is set to the original tokenizer's unk_token value.

    The pre_tokenizer attribute of the Tokenizer object is set to a ByteLevel pre_tokenizer with the add_prefix_space
    parameter set to the original tokenizer's add_prefix_space value.

    The decoder attribute of the Tokenizer object is set to a ByteLevel decoder.

    The post_processor attribute of the Tokenizer object is set to a TemplateProcessing post_processor.
    The special_tokens parameter is set to a list containing the cls and sep tokens and their corresponding
    token IDs from the original tokenizer.
    The single template is set to "{cls} $A {sep}" and the pair template is set to "{cls} $A {sep} $B {sep}".

    Example:
        ```python
        >>> converter = MarkupLMConverter()
        >>> converted_tokenizer = converter.converted(original_tokenizer)
        ```
    """
    def converted(self) -> Tokenizer:
        """
        Converts the original tokenizer into a new tokenizer of type 'Tokenizer'.
        
        Args:
            self: An instance of the 'MarkupLMConverter' class.
        
        Returns:
            A new instance of the 'Tokenizer' class.
        
        Raises:
            None.
        """
        ot = self.original_tokenizer
        vocab = ot.encoder
        merges = list(ot.bpe_ranks.keys())

        tokenizer = Tokenizer(
            BPE(
                vocab=vocab,
                merges=merges,
                dropout=None,
                continuing_subword_prefix="",
                end_of_word_suffix="",
                fuse_unk=False,
                unk_token=self.original_tokenizer.unk_token,
            )
        )

        tokenizer.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=ot.add_prefix_space)
        tokenizer.decoder = decoders.ByteLevel()

        cls = str(self.original_tokenizer.cls_token)
        sep = str(self.original_tokenizer.sep_token)
        cls_token_id = self.original_tokenizer.cls_token_id
        sep_token_id = self.original_tokenizer.sep_token_id

        tokenizer.post_processor = processors.TemplateProcessing(
            single=f"{cls} $A {sep}",
            pair=f"{cls} $A {sep} $B {sep}",
            special_tokens=[
                (cls, cls_token_id),
                (sep, sep_token_id),
            ],
        )

        return tokenizer


SLOW_TO_FAST_CONVERTERS = {
    "AlbertTokenizer": AlbertConverter,
    "BartTokenizer": RobertaConverter,
    "BarthezTokenizer": BarthezConverter,
    "BertTokenizer": BertConverter,
    "BigBirdTokenizer": BigBirdConverter,
    "BlenderbotTokenizer": BlenderbotConverter,
    "CamembertTokenizer": CamembertConverter,
    "CLIPTokenizer": CLIPConverter,
    "CodeGenTokenizer": GPT2Converter,
    "ConvBertTokenizer": BertConverter,
    "DebertaTokenizer": DebertaConverter,
    "DebertaV2Tokenizer": DebertaV2Converter,
    "DistilBertTokenizer": BertConverter,
    "DPRReaderTokenizer": BertConverter,
    "DPRQuestionEncoderTokenizer": BertConverter,
    "DPRContextEncoderTokenizer": BertConverter,
    "ElectraTokenizer": BertConverter,
    "FNetTokenizer": AlbertConverter,
    "FunnelTokenizer": FunnelConverter,
    "GPT2Tokenizer": GPT2Converter,
    "HerbertTokenizer": HerbertConverter,
    "LayoutLMTokenizer": BertConverter,
    "LayoutLMv2Tokenizer": BertConverter,
    "LayoutLMv3Tokenizer": RobertaConverter,
    "LayoutXLMTokenizer": XLMRobertaConverter,
    "LongformerTokenizer": RobertaConverter,
    "LEDTokenizer": RobertaConverter,
    "LxmertTokenizer": BertConverter,
    "MarkupLMTokenizer": MarkupLMConverter,
    "MBartTokenizer": MBartConverter,
    "MBart50Tokenizer": MBart50Converter,
    "MPNetTokenizer": MPNetConverter,
    "MobileBertTokenizer": BertConverter,
    "MvpTokenizer": RobertaConverter,
    "NllbTokenizer": NllbConverter,
    "OpenAIGPTTokenizer": OpenAIGPTConverter,
    "PegasusTokenizer": PegasusConverter,
    "RealmTokenizer": BertConverter,
    "ReformerTokenizer": ReformerConverter,
    "RemBertTokenizer": RemBertConverter,
    "RetriBertTokenizer": BertConverter,
    "RobertaTokenizer": RobertaConverter,
    # "RoFormerTokenizer": RoFormerConverter,
    "SeamlessM4TTokenizer": SeamlessM4TConverter,
    "SqueezeBertTokenizer": BertConverter,
    "T5Tokenizer": T5Converter,
    "UdopTokenizer": UdopConverter,
    "WhisperTokenizer": WhisperConverter,
    "XLMRobertaTokenizer": XLMRobertaConverter,
    "XLNetTokenizer": XLNetConverter,
    "SplinterTokenizer": SplinterConverter,
    "XGLMTokenizer": XGLMConverter,
    "LlamaTokenizer": LlamaConverter,
    "CodeLlamaTokenizer": LlamaConverter,
}


def convert_slow_tokenizer(transformer_tokenizer) -> Tokenizer:
    """
    Utilities to convert a slow tokenizer instance in a fast tokenizer instance.

    Args:
        transformer_tokenizer ([`~tokenization_utils_base.PreTrainedTokenizer`]):
            Instance of a slow tokenizer to convert in the backend tokenizer for
            [`~tokenization_utils_base.PreTrainedTokenizerFast`].

    Return:
        A instance of [`~tokenizers.Tokenizer`] to be used as the backend tokenizer of a
        [`~tokenization_utils_base.PreTrainedTokenizerFast`]
    """
    tokenizer_class_name = transformer_tokenizer.__class__.__name__

    if tokenizer_class_name not in SLOW_TO_FAST_CONVERTERS:
        raise ValueError(
            f"An instance of tokenizer class {tokenizer_class_name} cannot be converted in a Fast tokenizer instance."
            " No converter was found. Currently available slow->fast convertors:"
            f" {list(SLOW_TO_FAST_CONVERTERS.keys())}"
        )

    converter_class = SLOW_TO_FAST_CONVERTERS[tokenizer_class_name]

    return converter_class(transformer_tokenizer).converted()
