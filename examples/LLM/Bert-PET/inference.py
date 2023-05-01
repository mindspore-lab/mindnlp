"""
加载预训练好的模型并测试效果。

Author: pankeyu
Date: 2022/11/27

Edited by PurRigiN
Date: 2023/3
"""
import os
import time
from typing import List

import mindspore
import numpy as np
from transformers import AutoTokenizer

from BertForMaskedLM import BertForMaskedLM
from verbalizer import Verbalizer
from Template import HardTemplate
from utils import convert_example, convert_logits_to_ids, load_model

model_path = 'checkpoints/comment_classify/model_best'
model = load_model(model_path)

tokenizer = AutoTokenizer.from_pretrained(model_path)
model.set_train(False)

max_label_len = 2                               # 标签最大长度
verbalizer = Verbalizer(
        verbalizer_file='data/comment_classify/verbalizer.txt',
        tokenizer=tokenizer,
        max_label_len=max_label_len
    )
prompt = open('data/comment_classify/prompt.txt', 
                'r', encoding='utf8').readlines()[0].strip()    # prompt定义
template = HardTemplate(prompt=prompt)                          # 模板转换器定义
print(f'Prompt is -> {prompt}')


def inference(contents: List[str]):
    """
    推理函数，输入原始句子，输出mask label的预测值。

    Args:
        contents (List[str]): 描原始句子列表。
    """
    
    start_time = time.time()
    examples = {'text': contents}
    tokenized_output = convert_example(
        examples, 
        tokenizer, 
        template=template,
        max_seq_len=128,
        max_label_len=max_label_len,
        train_mode=False,
        return_tensor=True
    )
    logits = model(input_ids=tokenized_output['input_ids'],
                    token_type_ids=tokenized_output['token_type_ids'],
                    attention_mask=tokenized_output['attention_mask'])[0]
    predictions = convert_logits_to_ids(logits, tokenized_output['mask_positions']).asnumpy().tolist()  # (batch, label_num)
    predictions = verbalizer.batch_find_main_label(predictions)                                             # 找到子label属于的主label
    predictions = [ele['label'] for ele in predictions]
    used = time.time() - start_time
    print(f'Used {used}s.')
    return predictions


if __name__ == '__main__':
    contents = [
        '地理环境不错，但对面一直在盖楼，门前街道上打车不方便。',
        '跟好朋友一起凑单买的，很划算，洗发露是樱花香的，挺好的。。。'
    ]

    res = inference(contents)
    print('inference label(s):', res)
