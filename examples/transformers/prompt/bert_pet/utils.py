"""
工具类，包含prompt自定义字段的填值。

Author: pankeyu
Date: 2022/11/28

Edited by PurRigiN
Date: 2023/3
"""
import json
import os
import traceback
from typing import List

import numpy as np
import mindspore
from mindnlp.models import BertConfig

from BertForMaskedLM import BertForMaskedLM
from Template import HardTemplate


def convert_example(
    examples: dict, 
    tokenizer, 
    max_seq_len: int,
    max_label_len: int,
    template: HardTemplate,
    train_mode=True,
    return_tensor=False
    ) -> dict:
    """
    将样本数据转换为模型接收的输入数据。

    Args:
        examples (dict): 训练数据样本, e.g. -> {
                                                "text": [
                                                            '手机	这个手机也太卡了。',
                                                            '体育	世界杯为何迟迟不见宣传',
                                                            ...
                                                ]
                                            }
        max_seq_len (int): 句子的最大长度，若没有达到最大长度，则padding为最大长度
        max_label_len (int): 最大label长度，若没有达到最大长度，则padding为最大长度
        template (HardTemplate): 模板类。
        train_mode (bool): 训练阶段 or 推理阶段。
        return_tensor (bool): 是否返回tensor类型，如不是，则返回numpy类型。

    Returns:
        dict (str: np.array) -> tokenized_output = {
                            'input_ids': [[1, 47, 10, 7, 304, 3, 3, 3, 3, 47, 27, 247, 98, 105, 512, 777, 15, 12043, 2], ...],
                            'token_type_ids': [[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], ...],
                            'attention_mask': [[1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1], ...],
                            'mask_positions': [[5, 6, 7, 8], ...],
                            'mask_labels': [[2372, 3442, 0, 0], [2643, 4434, 2334, 0], ...]
                        }
    """
    tokenized_output = {
        'input_ids': [],
        'token_type_ids': [],
        'attention_mask': [],
        'mask_positions': [],
        'mask_labels': []
        }

    for i, example in enumerate(examples['text']):
        try:
            if train_mode:
                label, content = example.strip().split('\t')
            else:
                content = example.strip()

            inputs_dict={
                'textA': content,
                'MASK': '[MASK]'
            }
            encoded_inputs = template(
                inputs_dict=inputs_dict,
                tokenizer=tokenizer,
                max_seq_len=max_seq_len,
                mask_length=max_label_len
            )
        except:
            print(f'Error Line {i+1}: "{example}" -> {traceback.format_exc()}')
            exit()
        tokenized_output['input_ids'].append(encoded_inputs["input_ids"])
        tokenized_output['token_type_ids'].append(encoded_inputs["token_type_ids"])
        tokenized_output['attention_mask'].append(encoded_inputs["attention_mask"])
        tokenized_output['mask_positions'].append(encoded_inputs["mask_position"])
        
        if train_mode:
            label_encoded = tokenizer(text=[label])                                     # 将label补到最大长度
            label_encoded = label_encoded['input_ids'][0][1:-1]
            label_encoded = label_encoded[:max_label_len]
            label_encoded = label_encoded + [tokenizer.pad_token_id] * (max_label_len - len(label_encoded))
            tokenized_output['mask_labels'].append(label_encoded)
    
    for k, v in tokenized_output.items():
        if return_tensor:
            tokenized_output[k] = mindspore.Tensor(v, mindspore.int64)
        else:
            tokenized_output[k] = np.array(v)

    return tokenized_output


def mlm_loss(
    logits: mindspore.tensor,
    mask_positions: mindspore.tensor,
    sub_mask_labels: list,
    cross_entropy_criterion,
    masked_lm_scale=1.0,
    ) -> mindspore.tensor:
    """
    计算指定位置的mask token的output与label之间的cross entropy loss。

    Args:
        logits (torch.tensor): 模型原始输出 -> (batch, seq_len, vocab_size)
        mask_positions (torch.tensor): mask token的位置  -> (batch, mask_label_num)
        sub_mask_labels (list): mask token的sub label, 由于每个label的sub_label数目不同，所以这里是个变长的list, 
                                    e.g. -> [
                                        [[2398, 3352]],
                                        [[2398, 3352], [3819, 3861]]
                                    ]
        cross_entropy_criterion (CrossEntropyLoss): CE Loss计算器
        masked_lm_scale (float): scale 参数
        device (str): cpu还是gpu
    
    Returns:
        torch.tensor: CE Loss
    """
    batch_size, seq_len, vocab_size = logits.shape
    loss = None
    for single_logits, single_sub_mask_labels, single_mask_positions in zip(logits, sub_mask_labels, mask_positions):
        single_mask_logits = single_logits[single_mask_positions]                           # (mask_label_num, vocab_size)
        single_mask_logits = single_mask_logits.tile((len(single_sub_mask_labels), 1, 1))   # (sub_label_num, mask_label_num, vocab_size)
        single_mask_logits = single_mask_logits.reshape(-1, vocab_size)                     # (sub_label_num * mask_label_num, vocab_size)
        single_sub_mask_labels = mindspore.Tensor(single_sub_mask_labels, mindspore.int64)                   # (sub_label_num, mask_label_num)
        single_sub_mask_labels = single_sub_mask_labels.reshape(-1, 1).squeeze()            # (sub_label_num * mask_label_num)
        if not single_sub_mask_labels.shape:                                               # 处理单token维度下维度缺失的问题
            single_sub_mask_labels = single_sub_mask_labels.unsqueeze(axis=0)
        cur_loss = cross_entropy_criterion(single_mask_logits.float(), single_sub_mask_labels.int())
        cur_loss = cur_loss / len(single_sub_mask_labels)
        if not loss:
            loss = cur_loss
        else:
            loss += cur_loss
    loss = loss / batch_size                                                                # (1,)
    return loss / masked_lm_scale


def convert_logits_to_ids(
    logits: mindspore.tensor, 
    mask_positions: mindspore.tensor
    ) -> mindspore.int64:
    """
    输入Language Model的词表概率分布（LMModel的logits），将mask_position位置的
    token logits转换为token的id。

    Args:
        logits (torch.tensor): model output -> (batch, seq_len, vocab_size)
        mask_positions (torch.tensor): mask token的位置 -> (batch, mask_label_num)

    Returns:
        torch.LongTensor: 对应mask position上最大概率的推理token -> (batch, mask_label_num)
    """
    label_length = mask_positions.shape[1]                                     # 标签长度
    batch_size, seq_len, vocab_size = logits.shape
    mask_positions_after_reshaped = []
    for batch, mask_pos in enumerate(mask_positions.asnumpy().tolist()):
        for pos in mask_pos:
            mask_positions_after_reshaped.append(batch * seq_len + pos)
    logits = logits.reshape(batch_size * seq_len, -1)                           # (batch_size * seq_len, vocab_size)
    mask_logits = logits[mask_positions_after_reshaped]                         # (batch * label_num, vocab_size)
    predict_tokens = mask_logits.argmax(axis=-1)                                 # (batch * label_num)
    predict_tokens = predict_tokens.reshape(-1, label_length)                   # (batch, label_num)

    return predict_tokens

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

def save_json_file(config, json_file):
    """Saves a configuration file."""
    with open(json_file, "w", encoding='utf-8') as writer:
        writer.write(json.dumps(config.__dict__, indent=4, ensure_ascii=False) + "\n")

def save_model(model, folder_path):
    """
    保存模型。

    Args:
        model: 当前模型
        folder_path: 模型保存路径
    """
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)
    # save ckpt
    ckpt_path = os.path.join(folder_path, "bert-base-chinese.ckpt")
    mindspore.save_checkpoint(model, ckpt_path)

    # save config
    json_file_path = os.path.join(folder_path, "config.json")
    save_json_file(model.config, json_file_path)

def load_model(model_folder_path):
    
    # load ckpt
    config_path = os.path.join(model_folder_path, 'config.json')
    ckpt_path = os.path.join(model_folder_path, 'bert-base-chinese.ckpt')

    bert_config = from_json_file(config_path)
    model = BertForMaskedLM(bert_config)

    parameter_dict = mindspore.load_checkpoint(ckpt_path)

    param_not_load, ckpt_not_load = mindspore.load_param_into_net(model, parameter_dict)
    print("param_not_load: \n", param_not_load)
    print("ckpt_not_load: \n", ckpt_not_load)

    return model
    
if __name__ == '__main__':

    logits = mindspore.Tensor(np.random.normal(size=(1, 20, 21193)))
    mask_positions = mindspore.Tensor([
        [3, 4]
    ], mindspore.int64)
    predict_tokens = convert_logits_to_ids(logits, mask_positions)
    print(predict_tokens)