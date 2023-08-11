# pylint: disable=C0103
"""
OPTTokenizer
"""
import numpy as np
from mindspore.dataset.text.transforms import Implementation
from tokenizers import Tokenizer
from mindnlp.abc import PreTrainedTokenizer
from mindnlp.models.opt.config_opt import OPT_SUPPORT_LIST
from mindnlp.configs import MINDNLP_TOKENIZER_CONFIG_URL_BASE, HF_TOKENIZER_CONFIG_URL_BASE

PRETRAINED_VOCAB_MAP = {
    model: MINDNLP_TOKENIZER_CONFIG_URL_BASE.format('opt', model) for model in OPT_SUPPORT_LIST
}
# TODO:
PRETRAINED_VOCAB_MAP['facebook/opt-350m'] = HF_TOKENIZER_CONFIG_URL_BASE.format('facebook/opt-350m')

PRETRAINED_POSITIONAL_EMBEDDINGS_SIZES = {
    "opt": 1024,
}

class OPTTokenizer(PreTrainedTokenizer):
    """
        Tokenizer used for OPT text process.
        Args:
            vocab (Vocab): Vocabulary used to look up words.
            return_token (bool): Whether to return token. If True: return tokens. False: return ids. Default: True.

        """

    max_model_input_sizes = PRETRAINED_POSITIONAL_EMBEDDINGS_SIZES
    pretrained_vocab_map = PRETRAINED_VOCAB_MAP

    def __init__(
        self,
        tokenizer_file=None,
        unk_token="</s>",
        bos_token="</s>",
        eos_token="</s>",
        pad_token="<pad>",
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
            self._tokenizer.enable_padding(pad_token=pad_token,
                                           pad_id=self._tokenizer.token_to_id(pad_token))
        else:
            raise ValueError(f'only support string, but got {tokenizer_file}')

        self.return_token = return_token
        self.implementation = Implementation.PY

    def __call__(self, text_input):
        """
        Call method for input conversion for eager mode with C++ implementation.
        """
        if isinstance(text_input, str):
            text_input = np.asarray(text_input)
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
        is_batched = False
        if isinstance(text_input, np.ndarray):
            if text_input.shape != () and text_input.shape[0] > 1:
                is_batched = True
        if is_batched:
            return self._batch_encode(text_input)
        text_input = self._convert_to_unicode(text_input)
        tokens = self._tokenizer.encode(text_input)
        if self.return_token is True:
            return np.asarray(tokens.tokens)
        return {'input_ids': tokens.ids}

    def _batch_encode(self, text_input):
        for n, text in enumerate(text_input):
            text_input[n] = self._convert_to_unicode(text)
        tokens_batch = self._tokenizer.encode_batch(text_input)
        num_tokens = len(tokens_batch[0])
        num_text = len(tokens_batch)

        tokens_id = np.zeros((num_text, num_tokens), dtype=int)
        tokens = []
        tokens_att_mask = np.zeros((num_text, num_tokens), dtype=int)

        for n, token in enumerate(tokens_batch):
            tokens.append(token.tokens)
            tokens_id[n] = token.ids
            tokens_att_mask[n] = token.attention_mask
        if self.return_token is True:
            return np.asarray(tokens)
        return {'input_ids': tokens_id, 'attention_mask': tokens_att_mask, 'input_tokens': np.asarray(tokens)}

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
        index = self._tokenizer.token_to_id(token)
        if index is None:
            return self.unk_token_id
        return index

    def _convert_id_to_token(self, index):
        return self._tokenizer.id_to_token(index)

    def batch_decode(
        self,
        sequences,
        skip_special_tokens: bool = False,
    ):
        r"""
        Batch Decode
        """
        return [
            self._tokenizer.decode_batch(
                seq,
                skip_special_tokens=skip_special_tokens,
            )
            for seq in sequences
        ]
