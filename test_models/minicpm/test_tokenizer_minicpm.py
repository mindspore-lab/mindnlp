import os
from mindnlp.models.miniCPM.miniCPM_tokenizer import MiniCPMTokenizer


def test_tokenizer_basic():
    # 获取当前 test_tokenizer_minicpm.py 所在路径
    current_dir = os.path.dirname(__file__)


    tokenizer_path = os.path.abspath(
        os.path.join(current_dir, '../../../mindnlp/models/miniCPM/minicpm_assets/tokenizer.json')

    )

    tokenizer = MiniCPMTokenizer(tokenizer_path)

    text = "你好 MiniCPM"
    tokens = tokenizer.tokenize(text)
    ids = tokenizer.convert_tokens_to_ids(tokens)

    assert isinstance(tokens, list)
    assert isinstance(ids, list)
    assert len(tokens) == len(ids)
