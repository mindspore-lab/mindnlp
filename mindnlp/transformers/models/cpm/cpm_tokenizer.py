# pylint: disable=C0301
# pylint: disable=R0913
"""
GPT2Tokenizer
"""
import os
import numpy as np
from mindspore.dataset.text.transforms import Implementation
from tokenizers import Tokenizer
from ...tokenization_utils import PreTrainedTokenizer

PRETRAINED_VOCAB_MAP = {
    "TsinghuaAI/CPM-Generate": "https://huggingface.co/TsinghuaAI/CPM-Generate/resolve/main/tokenizer.json"
}


class CPMTokenizer(PreTrainedTokenizer):
    """
        Tokenizer used for GPT2 text process.
        Args:
            vocab (Vocab): Vocabulary used to look up words.
            return_token (bool): Whether to return token. If True: return tokens. False: return ids. Default: True.

        """

    pretrained_vocab_map = PRETRAINED_VOCAB_MAP

    def __init__(
            self,
            tokenizer_file=None,
            bos_token="<s>",
            eos_token="</s>",
            unk_token="<unk>",
            sep_token="<sep>",
            pad_token="<pad>",
            cls_token="<cls>",
            mask_token="<mask>",
            eop_token="<eop>",
            eod_token="<eod>",
            add_prefix_space=False,
            **kwargs
    ):
        super().__init__(
            tokenizer_file=tokenizer_file,
            unk_token=unk_token,
            sep_token=sep_token,
            pad_token=pad_token,
            cls_token=cls_token,
            mask_token=mask_token,
            eop_token=eop_token,
            eod_token=eod_token,
            bos_token=bos_token,
            eos_token=eos_token,
            add_prefix_space=add_prefix_space,
            **kwargs)

        if isinstance(tokenizer_file, str):
            if not os.path.isfile(tokenizer_file):
                raise ValueError(f"{tokenizer_file} is not a file.")
        else:
            raise ValueError(f'only support tokenizer class from mindspore or mindnlp, but got {tokenizer_file}')

        return_token = kwargs.pop('return_token', False)

        if isinstance(tokenizer_file, str):
            self._tokenizer = Tokenizer.from_file(tokenizer_file)
        else:
            raise ValueError(f'only support string, but got {tokenizer_file}')

        self.return_token = return_token
        self.implementation = Implementation.PY

    def __call__(self, text_input):
        """
        Call method for input conversion for eager mode with C++ implementation.
        """
        if isinstance(text_input, str):
            text_input = np.array(text_input)
        elif not isinstance(text_input, np.ndarray):
            raise TypeError(
                f"Input should be a text line in 1-D NumPy format, got {type(text_input)}.")
        return super().__call__(text_input)

    def execute_py(self, text_input):
        """
        Execute method.
        """
        return self._execute_py(text_input)

    def _execute_py(self, text_input):
        """
        Execute method.
        """
        text_input = self._convert_to_unicode(text_input)
        tokens = self._tokenizer.encode(text_input)
        if self.return_token is True:
            return np.array(tokens.tokens)
        return np.array(tokens.ids)

    def _convert_to_unicode(self, text_input):
        """Converts `text` to Unicode (if it's not already), assuming utf-8 input."""
        if isinstance(text_input, str):
            return text_input
        if isinstance(text_input, bytes):
            return text_input.decode("utf-8", "ignore")
        if isinstance(text_input, np.ndarray):
            if text_input.dtype.type is np.bytes_:
                text_input = np.char.decode(text_input, "utf-8")
            return str(text_input)
        raise ValueError(f"Unsupported string type: {type(text_input)}, {text_input.dtype}")

    def _convert_token_to_id(self, token):
        return self._tokenizer.token_to_id(token)

    def _convert_id_to_token(self, index):
        return self._tokenizer.id_to_token(index)
