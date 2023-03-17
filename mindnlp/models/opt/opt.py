"""
mindspore opt model
"""
import mindspore
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