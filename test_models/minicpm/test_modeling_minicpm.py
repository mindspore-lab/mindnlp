import sys, os
import mindspore.numpy as msnp
from mindnlp.models.miniCPM.miniCPM_config import MiniCPMConfig
from mindnlp.models.miniCPM.miniCPM_model import MiniCPMForCausalLM


def test_model_forward_shape():
    config = MiniCPMConfig()
    model = MiniCPMForCausalLM(config)

    # 模拟输入 token id
    input_ids = msnp.array([[128000, 1234, 5678, 44]], dtype=msnp.int32)

    # 前向推理
    logits = model(input_ids)

    # 输出应为 (batch, seq_len, vocab_size)
    assert logits.shape[0] == 1
    assert logits.shape[1] == 4
    assert logits.shape[2] == config.vocab_size
