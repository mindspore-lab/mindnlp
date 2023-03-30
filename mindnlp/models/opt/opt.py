"""
mindspore opt model
"""
import mindspore
import numpy as np
from mindspore import nn
from mindspore.nn import BCEWithLogitsLoss, CrossEntropyLoss, MSELoss
from mindnlp._legacy.nn import Dropout
from mindnlp._legacy.functional import arange
from ..utils.activations import ACT2FN
from ..utils.mixin import CellUtilMixin
from ..utils import logging

from .opt_config import OPTConfig

logger = logging.get_logger(__name__)


_CHECKPOINT_FOR_DOC = "facebook/opt-350m"
_CONFIG_FOR_DOC = "OPTConfig"

# Base model docstring
_EXPECTED_OUTPUT_SHAPE = [1, 8, 1024]

# SequenceClassification docstring
_CHECKPOINT_FOR_SEQUENCE_CLASSIFICATION = "ArthurZ/opt-350m-dummy-sc"
_SEQ_CLASS_EXPECTED_LOSS = 1.71
_SEQ_CLASS_EXPECTED_OUTPUT = "'LABEL_0'"

OPT_PRETRAINED_MODEL_ARCHIVE_LIST = [
    "facebook/opt-125m",
    "facebook/opt-350m",
    "facebook/opt-1.3b",
    "facebook/opt-2.7b",
    "facebook/opt-6.7b",
    "facebook/opt-13b",
    "facebook/opt-30b",
    # See all OPT models at https://huggingface.co/models?filter=opt
]

def _make_causal_mask(input_ids_shape: mindspore.size, dtype: mindspore.dtype, past_key_values_length: int = 0):
    """
    Make casual mask for bi-directional self-attention
    """
    bsz, tgt_len = input_ids_shape;
    #mask = mindspore.numpy.full((tgt_len,tgt_len),mindspore.tensor(mindspore.finfo(dtype).min))
    mask = mindspore.numpy.full((tgt_len,tgt_len),mindspore.Tensor(np.finfo(dtype).min))
    mask_cond = mindspore.arrange(mask.size(-1))
    mask.masked_fill(mask_cond < (mask_cond + 1).view(mask.size(-1),1),0)
    mask = mask.to(dtype)

    if past_key_values_length > 0:
        mask = mindspore.ops.Concat([mindspore.ops.Zeros(tgt_len, past_key_values_length, dtype=dtype),mask], dim=-1)
    return mask[None, None, :, :].expand(bsz, 1, tgt_len, tgt_len + past_key_values_length)

def _expand_mask(mask: mindspore.Tensor, dtype: mindspore.dtype, tgt_len: Optional[int] = None):
    """
    Expands attention_mask from `[bsz, seq_len]` to `[bsz, 1, tgt_seq_len, src_seq_len]`.
    """
    bsz, src_len = mask.size()
    tgt_len = tgt_len if tgt_len is not None else src_len

    expanded_mask = mask[:, None, None, :].expand(bsz, 1, tgt_len, src_len).to(dtype)

    inverted_mask = 1.0 - expanded_mask

    return inverted_mask.masked_fill(inverted_mask.to(mindspore.Tensor.bool), np.finfo(dtype).min)

class OPTLearnedPositionalEmbedding(nn.Embedding):
    """
    This module learns positional embeddings up to a fixed maximum size.
    """

    def __init__(self, num_embeddings: int, embedding_dim: int):
        # OPT is set up so that if padding_idx is specified then offset the embedding ids by 2
        # and adjust num_embeddings appropriately. Other models don't have this hack
        self.offset = 2
        super().__init__(num_embeddings + self.offset, embedding_dim)

    def forward(self, attention_mask: LongTensor, past_key_values_length: int = 0):
        """`input_ids_shape` is expected to be [bsz x seqlen]."""
        attention_mask = attention_mask.to(mindspore.int64)# equivalent to torch.tensor.long()?

        # create positions depending on attention_mask
        positions = (mindspore.Tensor.cumsum(attention_mask, dim=1).type_as(attention_mask) * attention_mask).long() - 1
        #cumsum 即式 cumulative sum，累计一个和

        # cut positions if `past_key_values_length` is > 0
        positions = positions[:, past_key_values_length:]

        return super().forward(positions + self.offset)
