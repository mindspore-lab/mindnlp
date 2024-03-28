import os
import json
import logging
from .utils import load_from_cache
from .utils import get_mindrecord_list

class PretrainedConfig:
    """
    Pretrained Config.
    Args:
        xxx
    """
    pretrained_config_archive = {}
    def __init__(self, **kwargs):
        self.finetuning_task = kwargs.pop('finetuning_task', None)
        self.num_labels = kwargs.pop('num_labels', 2)
        self.output_attentions = kwargs.pop('output_attentions', False)
        self.output_hidden_states = kwargs.pop('output_hidden_states', False)
        self.train_batch_size = kwargs.pop('train_batch_size', 128)
        self.eval_batch_size = kwargs.pop('eval_batch_size', 128)
        self.do_save_ckpt = kwargs.pop('do_save_ckpt', True)
        self.jit = kwargs.pop('jit', True)
        self.do_train = kwargs.pop('do_train', True)
        self.do_eval = kwargs.pop('do_eval', True)
        # self.save_ckpt_path = kwargs.pop('save_ckpt_path', os.path.join('')'/data0/bert/outputs/model_save')
        self.save_steps = kwargs.pop('save_steps',1000)
        self.epochs = kwargs.pop('epochs', 40)
        self.lr = kwargs.pop('lr', 5e-5)
        self.warmup = kwargs.pop('warmup',0.16)

    @classmethod
    def load(cls, pretrained_model_name_or_path, **kwargs):
        """load config."""
        force_download = kwargs.pop('force_download', False)
        if os.path.exists(pretrained_model_name_or_path):
            # File exists.
            config_file = pretrained_model_name_or_path
        elif pretrained_model_name_or_path in cls.pretrained_config_archive:
            logging.info("The checkpoint file not found, start to download.")
            config_url = cls.pretrained_config_archive[pretrained_model_name_or_path]
            config_file = load_from_cache(pretrained_model_name_or_path + '.json',
                                          config_url, force_download=force_download)
        else:
            # Something unknown
            raise ValueError(
                f"unable to parse {pretrained_model_name_or_path} as a local path or model name")

        config = cls.from_json(config_file)

        return config

    @classmethod
    def from_json(cls, file_path):
        """load config from json."""
        with open(file_path, "r", encoding="utf-8") as file:
            text = file.read()
        config_map = json.loads(text)
        config = cls()
        for key, value in config_map.items():
            setattr(config, key, value)
        return config

CONFIG_ARCHIVE_MAP = {
    "bert-base-uncased": "https://hf-mirror.com/bert-base-uncased/resolve/main/config.json",
    "bert-large-uncased": "https://hf-mirror.com/bert-large-uncased/resolve/main/config.json",
    "bert-base-cased": "https://hf-mirror.com/bert-base-cased/resolve/main/config.json",
    "bert-large-cased": "https://hf-mirror.com/bert-large-cased/resolve/main/config.json",
    "bert-base-multilingual-uncased": "https://hf-mirror.com/bert-base-multilingual-uncased/resolve/main/config.json",
    "bert-base-multilingual-cased": "https://hf-mirror.com/bert-base-multilingual-cased/resolve/main/config.json",
    "bert-base-chinese": "https://hf-mirror.com/bert-base-chinese/resolve/main/config.json",
    "bert-base-german-cased": "https://hf-mirror.com/bert-base-german-cased/resolve/main/config.json",
    "bert-large-uncased-whole-word-masking": "https://hf-mirror.com/bert-large-uncased-whole-word-masking/resolve/main/config.json",
    "bert-large-cased-whole-word-masking": "https://hf-mirror.com/bert-large-cased-whole-word-masking/resolve/main/config.json",
    "bert-large-uncased-whole-word-masking-finetuned-squad": "https://hf-mirror.com/bert-large-uncased-whole-word-masking-finetuned-squad/resolve/main/config.json",
    "bert-large-cased-whole-word-masking-finetuned-squad": "https://hf-mirror.com/bert-large-cased-whole-word-masking-finetuned-squad/resolve/main/config.json",
    "bert-base-cased-finetuned-mrpc": "https://hf-mirror.com/bert-base-cased-finetuned-mrpc/resolve/main/config.json",
    "bert-base-german-dbmdz-cased": "https://hf-mirror.com/bert-base-german-dbmdz-cased/resolve/main/config.json",
    "bert-base-german-dbmdz-uncased": "https://hf-mirror.com/bert-base-german-dbmdz-uncased/resolve/main/config.json",
    "cl-tohoku/bert-base-japanese": "https://hf-mirror.com/cl-tohoku/bert-base-japanese/resolve/main/config.json",
    "cl-tohoku/bert-base-japanese-whole-word-masking": "https://hf-mirror.com/cl-tohoku/bert-base-japanese-whole-word-masking/resolve/main/config.json",
    "cl-tohoku/bert-base-japanese-char": "https://hf-mirror.com/cl-tohoku/bert-base-japanese-char/resolve/main/config.json",
    "cl-tohoku/bert-base-japanese-char-whole-word-masking": "https://hf-mirror.com/cl-tohoku/bert-base-japanese-char-whole-word-masking/resolve/main/config.json",
    "TurkuNLP/bert-base-finnish-cased-v1": "https://hf-mirror.com/TurkuNLP/bert-base-finnish-cased-v1/resolve/main/config.json",
    "TurkuNLP/bert-base-finnish-uncased-v1": "https://hf-mirror.com/TurkuNLP/bert-base-finnish-uncased-v1/resolve/main/config.json",
    "wietsedv/bert-base-dutch-cased": "https://hf-mirror.com/wietsedv/bert-base-dutch-cased/resolve/main/config.json",
    "sentence-transformers/all-MiniLM-L6-v2": "https://hf-mirror.com/sentence-transformers/all-MiniLM-L6-v2/resolve/main/config.json"
    # See all BERT models at https://hf-mirror.com/models?filter=bert
}

class BertConfig(PretrainedConfig):
    """Configuration for BERT
    """
    pretrained_config_archive = CONFIG_ARCHIVE_MAP
    def __init__(self,
                 vocab_size=30522,
                 hidden_size=256,
                 num_hidden_layers=4,
                 num_attention_heads=4,
                 intermediate_size=1024,
                 hidden_act="gelu",
                 hidden_dropout_prob=0.1,
                 attention_probs_dropout_prob=0.1,
                 max_position_embeddings=512,
                 type_vocab_size=2,
                 initializer_range=0.02,
                 layer_norm_eps=1e-12,
                 **kwargs):
        super().__init__(**kwargs)
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads
        self.hidden_act = hidden_act
        self.intermediate_size = intermediate_size
        self.hidden_dropout_prob = hidden_dropout_prob
        self.attention_probs_dropout_prob = attention_probs_dropout_prob
        self.max_position_embeddings = max_position_embeddings
        self.type_vocab_size = type_vocab_size
        self.initializer_range = initializer_range
        self.layer_norm_eps = layer_norm_eps
