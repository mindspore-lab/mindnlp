import transformers
def mock_is_not_available():
    return False

transformers.utils.import_utils.is_torchvision_v2_available.__code__ = mock_is_not_available.__code__
transformers.utils.import_utils.is_torch_flex_attn_available.__code__ = mock_is_not_available.__code__

from transformers import *
