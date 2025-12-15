import json
import os

import mindspore
from mindspore import ops
from mindnlp.models import BertModel, BertConfig
import numpy as np

def from_dict(json_object):
    """Constructs a `BertConfig` from a Python dictionary of parameters."""
    config = BertConfig()
    for key, value in json_object.items():
        config.__dict__[key] = value
    return config

def from_json_file(json_file):
    """Constructs a `BertConfig` from a json file of parameters."""
    with open(json_file, "r", encoding='utf-8') as reader:
        text = reader.read()
    return from_dict(json.loads(text))

model_folder_path = "ckpt"
config_path = os.path.join(model_folder_path, 'config.json')
config = from_json_file(config_path)
bert = BertModel(config)

