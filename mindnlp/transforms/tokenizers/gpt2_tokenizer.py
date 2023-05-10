
"""
GPT2Tokenizer
"""
import numpy as np
from mindspore.dataset.text.transforms import Implementation
from tokenizers import Tokenizer
from mindnlp.abc import PreTrainedTokenizer
from mindnlp.models.gpt2.config_gpt2 import GPT2_SUPPORT_LIST
from mindnlp.configs import MINDNLP_TOKENIZER_CONFIG_URL_BASE

PRETRAINED_VOCAB_MAP = {
    model: MINDNLP_TOKENIZER_CONFIG_URL_BASE.format('gpt2', model) for model in GPT2_SUPPORT_LIST
}

PRETRAINED_POSITIONAL_EMBEDDINGS_SIZES = {
    "gpt2": 1024,
    "gpt2-medium": 1024,
    "gpt2-large": 1024,
    "gpt2-xl": 1024,
    "distilgpt2": 1024,
}

class GPT2Tokenizer(PreTrainedTokenizer):
    """
        Tokenizer used for GPT2 text process.
        Args:
            vocab (Vocab): Vocabulary used to look up words.
            return_token (bool): Whether to return token. If True: return tokens. False: return ids. Default: True.

        """

    max_model_input_sizes = PRETRAINED_POSITIONAL_EMBEDDINGS_SIZES
    pretrained_vocab_map = PRETRAINED_VOCAB_MAP

    def __init__(
        self,
        tokenizer_file=None,
        unk_token="<|endoftext|>",
        bos_token="<|endoftext|>",
        eos_token="<|endoftext|>",
        add_prefix_space=False,
        **kwargs
    ):
        super().__init__(
            unk_token=unk_token,
            bos_token=bos_token,
            eos_token=eos_token,
            add_prefix_space=add_prefix_space,
            **kwargs)

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
