from .miniCPM_config import MiniCPMConfig
from .miniCPM_model import MiniCPMModel
from .miniCPM_tokenizer import MiniCPMTokenizer

__all__ = ["MiniCPMConfig", "MiniCPMModel", "MiniCPMTokenizer"]

# tests/ut/models/minicpm/test_tokenizer_minicpm.py
import mindspore
from mindnlp.models.miniCPM import miniCPM_model, miniCPM_config

def test_minicpm_forward():
    config = MiniCPMConfig()
    model = MiniCPMModel(config)
    dummy_input = mindspore.Tensor([[1, 2, 3], [4, 5, 6]], mindspore.int32)
    output = model(dummy_input)
    assert output.shape == (2, 3, config.hidden_size)
