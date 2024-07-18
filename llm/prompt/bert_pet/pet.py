"""
基于人工模板的prompt模型。

Paper Reference:
    https://arxiv.org/pdf/2001.07676.pdf

Author: pankeyu
Date: 2022/11/26

Edited by PurRigiN
Date: 2023/3
"""
import argparse
from functools import partial
import os
import time

import mindspore
from mindspore import nn, ops
from mindspore.dataset import GeneratorDataset
from transformers import AutoTokenizer
from datasets import load_dataset

from iTrainingLogger import iSummaryWriter
from class_metrics import ClassEvaluator
from optimization import WarmUpPolynomialDecayLR
from utils import convert_example, mlm_loss, convert_logits_to_ids, load_model, save_model
from verbalizer import Verbalizer
from Template import HardTemplate


parser = argparse.ArgumentParser()
parser.add_argument("--model", default='bert-base-chinese', type=str, help="backbone of encoder.")
parser.add_argument("--model_folder_path", type=str, help="model folder path.")
parser.add_argument("--train_path", default=None, type=str, help="The path of train set.")
parser.add_argument("--dev_path", default=None, type=str, help="The path of dev set.")
parser.add_argument("--save_dir", default="./checkpoints", type=str, required=False, help="The output directory where the model predictions and checkpoints will be written.")
parser.add_argument("--max_seq_len", default=512, type=int,help="The maximum total input sequence length after tokenization. Sequences longer "
    "than this will be truncated, sequences shorter will be padded.", )
parser.add_argument("--batch_size", default=16, type=int, help="Batch size per GPU/CPU for training.", )
parser.add_argument("--learning_rate", default=5e-5, type=float, help="The initial learning rate for Adam.")
parser.add_argument("--weight_decay", default=0.0, type=float, help="Weight decay if we apply some.")
parser.add_argument("--num_train_epochs", default=10, type=int, help="Total number of training epochs to perform.")
parser.add_argument("--warmup_ratio", default=0.06, type=float, help="Linear warmup over warmup_ratio * total_steps.")
parser.add_argument("--valid_steps", default=200, type=int, required=False, help="evaluate frequecny.")
parser.add_argument("--logging_steps", default=10, type=int, help="log interval.")
parser.add_argument("--max_label_len", type=int, default=6, help="max length of label")
parser.add_argument("--rdrop_coef", default=0.0, type=float, help="The coefficient of KL-Divergence loss in R-Drop paper, for more detail please refer to https://arxiv.org/abs/2106.14448), if rdrop_coef > 0 then R-Drop works")
parser.add_argument("--img_log_dir", default='logs', type=str, help="Logging image path.")
parser.add_argument("--img_log_name", default='Model Performance', type=str, help="Logging image file name.")
parser.add_argument("--verbalizer", default='Verbalizer File', required=True, type=str, help="verbalizer file.")
parser.add_argument("--prompt_file", default='Prompt File', required=True, type=str, help="prompt file.")
args = parser.parse_args()

writer = iSummaryWriter(log_path=args.img_log_dir, log_name=args.img_log_name)


def evaluate_model(model, metric, data_loader, global_step, tokenizer, verbalizer):
    """
    在测试集上评估当前模型的训练效果。

    Args:
        model: 当前模型
        metric: 评估指标类(metric)
        data_loader: 测试集的dataloader
        global_step: 当前训练步数
    """
    model.set_train(False)
    metric.reset()
    
    for step, batch in enumerate(data_loader):
        _, input_ids, token_type_ids, attention_mask, mask_labels, mask_positions = batch
        logits = model(input_ids=input_ids,
                        token_type_ids=token_type_ids,
                        attention_mask=attention_mask)[0]
        mask_labels = mask_labels.asnumpy().tolist()                                          # (batch, label_num)
        for i in range(len(mask_labels)):                                                            # 去掉label中的[PAD] token
            while tokenizer.pad_token_id in mask_labels[i]:
                mask_labels[i].remove(tokenizer.pad_token_id)
        mask_labels = [''.join(tokenizer.convert_ids_to_tokens(t)) for t in mask_labels]             # id转文字
        predictions = convert_logits_to_ids(logits, mask_positions).asnumpy().tolist()  # (batch, label_num)
        predictions = verbalizer.batch_find_main_label(predictions)                                  # 找到子label属于的主label
        predictions = [ele['label'] for ele in predictions]
        metric.add_batch(pred_batch=predictions, gold_batch=mask_labels)
    eval_metric = metric.compute()
    model.set_train(True)

    return eval_metric['accuracy'], eval_metric['precision'], \
            eval_metric['recall'], eval_metric['f1'], \
            eval_metric['class_metrics']


class Iterable():
    def __init__(self, dataset):
        self._dataset = dataset

    def __getitem__(self, index):
        return self._dataset['text'][index], self._dataset['input_ids'][index], self._dataset['token_type_ids'][index], self._dataset['attention_mask'][index], self._dataset['mask_labels'][index], self._dataset['mask_positions'][index]

    def __len__(self):
        return len(self._dataset)


def train():
    
    # load model
    model_folder_path = args.model_folder_path
    model = load_model(model_folder_path)

    tokenizer = AutoTokenizer.from_pretrained(args.model)

    verbalizer = Verbalizer(
        verbalizer_file=args.verbalizer,
        tokenizer=tokenizer,
        max_label_len=args.max_label_len
    )
    prompt = open(args.prompt_file, 'r', encoding='utf8').readlines()[0].strip()    # prompt定义
    template = HardTemplate(prompt=prompt)                                          # 模板转换器定义
    dataset = load_dataset('text', data_files={'train': args.train_path,
                                                'dev': args.dev_path})    
    print(dataset)
    print(f'Prompt is -> {prompt}')
    convert_func = partial(convert_example, 
                            tokenizer=tokenizer, 
                            template=template,
                            max_seq_len=args.max_seq_len,
                            max_label_len=args.max_label_len)
    dataset = dataset.map(convert_func, batched=True)
    
    train_dataset = dataset["train"]
    eval_dataset = dataset["dev"]

    train_dataset = Iterable(train_dataset)
    eval_dataset = Iterable(eval_dataset)
    column_names = ["text", "input_ids", "token_type_ids", "attention_mask", "mask_positions", "mask_labels"]
    train_dataset = GeneratorDataset(train_dataset, shuffle=True, column_names=column_names).batch(args.batch_size)
    eval_dataset = GeneratorDataset(eval_dataset, column_names=column_names).batch(args.batch_size)

    no_decay = ["bias", "layer_norm.weight"]
    optimizer_grouped_parameters = [
        {
            "params": [p for n, p in model.parameters_and_names() if not any(nd in n for nd in no_decay)],
            "weight_decay": args.weight_decay,
        },
        {
            "params": [p for n, p in model.parameters_and_names() if any(nd in n for nd in no_decay)],
            "weight_decay": 0.0,
        },
    ]
    
    # 根据训练轮数计算最大训练步数，以便于scheduler动态调整lr
    num_update_steps_per_epoch = len(train_dataset)
    max_train_steps = args.num_train_epochs * num_update_steps_per_epoch
    warm_steps = int(args.warmup_ratio * max_train_steps)

    lr_scheduler = WarmUpPolynomialDecayLR(
        learning_rate=args.learning_rate,
        end_learning_rate=0.0,
        warmup_steps=warm_steps,
        decay_steps=max_train_steps,
        power=1
    )
    optimizer = nn.AdamWeightDecay(optimizer_grouped_parameters, learning_rate=lr_scheduler)

    loss_list = []
    tic_train = time.time()
    metric = ClassEvaluator()
    criterion = nn.CrossEntropyLoss()
    global_step, best_f1 = 0, 0

    # Define forward function
    def forward_fn(input_ids, token_type_ids, attention_mask, mask_labels, mask_positions):
        # TODO: logits is the first or second?
        logits = model(input_ids=input_ids,
                        token_type_ids=token_type_ids,
                        attention_mask=attention_mask)[0]
        mask_labels = mask_labels.asnumpy().tolist()
        sub_labels = verbalizer.batch_find_sub_labels(mask_labels)
        sub_labels = [ele['token_ids'] for ele in sub_labels]
        loss = mlm_loss(
            logits, 
            mask_positions, 
            sub_labels, 
            criterion,
            1.0
        )
        return loss, logits

    grad_fn = mindspore.value_and_grad(forward_fn, None, optimizer.parameters, has_aux=True)

    def train_step(input_ids, token_type_ids, attention_mask, mask_labels, mask_positions):
        (loss, _), grads = grad_fn(
            input_ids,
            token_type_ids,
            attention_mask,
            mask_labels,
            mask_positions
        )
        loss = ops.depend(loss, optimizer(grads))
        return loss

    for epoch in range(args.num_train_epochs):
        
        for _, input_ids, token_type_ids, attention_mask, mask_labels, mask_positions in train_dataset.create_tuple_iterator():

            loss = train_step(
                input_ids,
                token_type_ids,
                attention_mask,
                mask_labels,
                mask_positions
            )

            loss_list.append(float(loss))
            
            global_step += 1
            if global_step % args.logging_steps == 0:
                time_diff = time.time() - tic_train
                loss_avg = sum(loss_list) / len(loss_list)
                writer.add_scalar('train/train_loss', loss_avg, global_step)
                print("global step %d, epoch: %d, loss: %.5f, speed: %.2f step/s"
                        % (global_step, epoch, loss_avg, args.logging_steps / time_diff))
                tic_train = time.time()

            if global_step % args.valid_steps == 0:
                cur_save_dir = os.path.join(args.save_dir, "model_%d" % global_step)
                if not os.path.exists(cur_save_dir):
                    os.makedirs(cur_save_dir)
                save_model(model, cur_save_dir)
                tokenizer.save_pretrained(cur_save_dir)

                acc, precision, recall, f1, class_metrics = evaluate_model(model, 
                                                                        metric, 
                                                                        eval_dataset, 
                                                                        global_step,
                                                                        tokenizer,
                                                                        verbalizer)
                writer.add_scalar('eval/acc', acc, global_step)
                writer.add_scalar('eval/precision', precision, global_step)
                writer.add_scalar('eval/recall', recall, global_step)
                writer.add_scalar('eval/f1', f1, global_step)
                writer.record()
                
                print("Evaluation precision: %.5f, recall: %.5f, F1: %.5f" % (precision, recall, f1))
                if f1 > best_f1:
                    print(
                        f"best F1 performence has been updated: {best_f1:.5f} --> {f1:.5f}"
                    )
                    print(f'Each Class Metrics are: {class_metrics}')
                    best_f1 = f1
                    cur_save_dir = os.path.join(args.save_dir, "model_best")
                    if not os.path.exists(cur_save_dir):
                        os.makedirs(cur_save_dir)
                    save_model(model, cur_save_dir)
                    tokenizer.save_pretrained(cur_save_dir)
                tic_train = time.time()


if __name__ == '__main__':
    train()