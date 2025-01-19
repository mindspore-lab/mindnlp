# mindnlp/models/mimi/configuration_mimi.py

from mindnlp.configs import MINDNLP_CONFIG_URL_BASE
from ...configuration_utils import PretrainedConfig

MINDNLP_MODEL_CONFIG_URL_BASE = MINDNLP_CONFIG_URL_BASE + "mimi/"

MIMI_PRETRAINED_CONFIG_ARCHIVE_MAP = {
    "mimi-base-uncased": MINDNLP_MODEL_CONFIG_URL_BASE + "mimi-base-uncased/config.json",
    # 添加其他预训练模型的配置文件路径
}

class MimiConfig(PretrainedConfig):
    model_type = "mimi"
    keys_to_ignore_at_inference = ["past_key_values"]
    attribute_map = {
        "hidden_size": "hidden_size",
        "num_attention_heads": "num_attention_heads",
        "num_hidden_layers": "num_hidden_layers",
    }

    def __init__(
        self,
        vocab_size=30522,
        hidden_size=768,
        num_hidden_layers=12,
        num_attention_heads=12,
        intermediate_size=3072,
        hidden_act="gelu",
        hidden_dropout_prob=0.1,
        attention_probs_dropout_prob=0.1,
        max_position_embeddings=512,
        type_vocab_size=2,
        initializer_range=0.02,
        layer_norm_eps=1e-12,
        pad_token_id=0,
        bos_token_id=101,
        eos_token_id=102,
        **kwargs
    ):
        super().__init__(**kwargs)
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads
        self.intermediate_size = intermediate_size
        self.hidden_act = hidden_act
        self.hidden_dropout_prob = hidden_dropout_prob
        self.attention_probs_dropout_prob = attention_probs_dropout_prob
        self.max_position_embeddings = max_position_embeddings
        self.type_vocab_size = type_vocab_size
        self.initializer_range = initializer_range
        self.layer_norm_eps = layer_norm_eps
        self.pad_token_id = pad_token_id
        self.bos_token_id = bos_token_id
        self.eos_token_id = eos_token_id
