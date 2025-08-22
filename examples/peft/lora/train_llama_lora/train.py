import os
import copy
import argparse
import logging
import mindspore
import numpy as np

from sentencepiece import SentencePieceProcessor
from typing import List
from tqdm import tqdm 
from mindspore import nn
from mindspore.nn import AdamWeightDecay
from mindnlp.transformers import LlamaForCausalLM, LlamaConfig, LlamaForSequenceClassification, LlamaPreTrainedModel
from mindnlp.transformers import LlamaTokenizer
from mindnlp.peft import (
    get_peft_config,
    get_peft_model,
    LoraConfig,
    PeftType,
    PeftConfig,
    PeftModel,
)

from mrpc_dataset import load_examples, get_dataloader_from_ds

def load_llama_model(config_path, model_path, token_path):
    config = LlamaConfig.from_pretrained(config_path, problem_type="single_label_classification")
    # print(config.to_dict())

    # here we build model from config to make model smaller and test the correctness of training process
    # mindspore.load_checkpoint(ckpt_path, net=model)
    config.num_hidden_layers = 2
    # model = LlamaForCausalLM(config)  
    model = LlamaForSequenceClassification(config)
    state_dict = mindspore.load_checkpoint(model_path)
    param_not_load, ckpt_not_load = mindspore.load_param_into_net(model, state_dict)
    logging.info("params in model not load\n", param_not_load)
    logging.info("ckpt not load:\n", ckpt_not_load)
    # print(model)
    tokenizer = LlamaTokenizer.from_pretrained(token_path)

    return model, config, tokenizer

def forward_fn(input_ids, attention_mask, token_type_ids, lens, labels):
    # _, _ = hidden_states, attentions``
    loss, logits, _ = model(
        input_ids=input_ids,
        attention_mask=attention_mask,
        # token_type_ids=token_type_ids,
        labels=mindspore.Tensor(labels.asnumpy(), mindspore.int32)
    )
    # RobertaForSequenceClassification Model has loss_fn (mse or cross_entropy)
    # loss = criterion(logits, label)
    return loss, logits


def train_one_epoch(model, optimizer, criterion, train_dataloader):
    model.set_train(True)
    total_loss, total_step = 0, 0
    num_batches = len(train_dataloader)
    preds, label_ids = None, None

    grad_fn = mindspore.value_and_grad(forward_fn, None, optimizer.parameters, has_aux=True)

    with tqdm(total=num_batches) as t:
        for i, (input_ids, attention_mask, token_type_ids, lens, labels) in enumerate(train_dataloader):
            # forward + grad
            (loss, logits), grad = grad_fn(input_ids, attention_mask, token_type_ids, lens, labels)
            # update model params
            optimizer(grad)
            total_loss += loss.asnumpy()
            total_step += 1
            curr_loss = total_loss / total_step  # 当前的平均loss

            if preds is None:
                preds = np.argmax(logits.asnumpy(), axis=1)
                label_ids = labels.asnumpy()
            else:
                preds = np.append(preds, np.argmax(logits.asnumpy(), axis=1), axis=0)
                label_ids = np.append(label_ids, labels.asnumpy(), axis=0)

            t.set_postfix({'train-loss': f'{curr_loss:.2f}'})
            t.update(1)
    acc = (preds == label_ids).mean()

    return total_loss / total_step, acc

def eval_one_epoch(model, optimizer, criterion, eval_dataloader):
    model.set_train(False)
    total_loss, total_step = 0, 0
    preds, label_ids = None, None
    num_batches = len(eval_dataloader)

    with tqdm(total=num_batches) as t:
        for i, (input_ids, attention_mask, token_type_ids, lens, labels) in enumerate(eval_dataloader):
            loss, logits, _ = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                # token_typ_ids=token_type_ids,
                labels=mindspore.Tensor(labels.asnumpy(), mindspore.int32)
            )
            # (loss, logits), grad = grad_fn(input_ids, attention_mask, token_type_ids, lens, labels)
            total_loss += loss.asnumpy()
            total_step += 1
            curr_loss = total_loss / total_step  # 当前的平均loss
            if preds is None:
                preds = np.argmax(logits.asnumpy(), axis=1)
                label_ids = labels.asnumpy()
            else:
                preds = np.append(preds, np.argmax(logits.asnumpy(), axis=1), axis=0)
                label_ids = np.append(label_ids, labels.asnumpy(), axis=0)
            t.set_postfix({'eval-loss': f'{curr_loss:.2f}'})
            t.update(1)

    # compute metrics
    acc = (preds == label_ids).mean()

    return total_loss / total_step, acc


def train(model, optimizer, criterion, train_dataloader, eval_dataloader, epochs):
    # train start from here
    for epoch in range(1, epochs+1):
        train_loss, train_acc = train_one_epoch(model, optimizer, criterion, train_dataloader)
        eval_loss, eval_acc = eval_one_epoch(model, optimizer, criterion, eval_dataloader)

        print(f"epoch:{epoch} train_loss:{train_loss} eval_acc:{train_acc} \
              eval_loss:{eval_loss} eval_acc:{eval_acc}")
        # print(f"epoch:{epoch} train_loss:{train_loss} eval_loss:{eval_loss}")

def eval_model(model, optimizer, criterion, eval_dataloader):
    eval_loss, eval_acc = eval_one_epoch(model, optimizer, criterion, eval_dataloader)
    print(f"eval_loss:{eval_loss} eval_acc:{eval_acc}")

if __name__ == "__main__":
    # `python llm/peft/train_llama_lora/train.py --do_train --do_eval --model_name_or_path bert-base-cased`
    # from pretrained
    parser = argparse.ArgumentParser()
    # parser.add_argument("--do_train", action='store_true', help="Whether to run training.")
    # parser.add_argument("--do_eval", action='store_true', help="Whether to run eval on the dev set.")
    parser.add_argument("--save_dir", default=".mindnlp/peft_model/mrpc_lora", type=str, help="The output directory where the model checkpoints will be written.")
    parser.add_argument("--batch_size", default=16, type=int, help="Batch size per GPU/CPU for training.")
    parser.add_argument("--model_name_or_path", default="roberta-base", type=str, help="bert-base-cased")
    parser.add_argument("--num_epochs", default=5, type=int)
    parser.add_argument("--lr", default=1e-4, type=float, help="Set 2e-5 for full-finetuning.")
    parser.add_argument("--max_seq_len", default=256, type=int)
    parser.add_argument("--debug", action="store_true", help="debug mode")
    parser.add_argument("--lora", action="store_true", help="lora mode")
    
    args = parser.parse_args()
    if args.debug:
        args.num_epochs = 1

    # load model
    model, config, tokenizer = load_llama_model(
        config_path="/home/cjl/code/mind/collect/mindnlp-models/llama-2-7b-hf/config.json",
        model_path="/home/cjl/code/mind/collect/mindnlp-models/convert/llama.ckpt",
        token_path="/home/cjl/code/mind/collect/mindnlp-models/llama-2-7b-hf/tokenizer.model"
    )
    logging.info("model load")

    if args.lora:
        # build peft model
        peft_config = LoraConfig(task_type="SEQ_CLS", inference_mode=False, r=8, lora_alpha=16, lora_dropout=0.1)
        model = get_peft_model(model, peft_config)
        # print(model)
        model.print_trainable_parameters()

    # load dataset
    train_ds = load_examples(tokenizer, args.max_seq_len, data_type="train")
    test_ds = load_examples(tokenizer, args.max_seq_len, data_type="test")
    train_dataloader = get_dataloader_from_ds(train_ds, args.batch_size)
    test_dataloader = get_dataloader_from_ds(test_ds, args.batch_size)
    logging.info("dataset load")

    # optimizer
    optimizer = AdamWeightDecay(params=model.trainable_params(), learning_rate=args.lr)
    loss_fn = nn.CrossEntropyLoss()  


    logging.info("start train")
    train(model, optimizer, loss_fn, train_dataloader, test_dataloader, epochs=args.num_epochs)
    logging.info("end train")
    model.save_pretrained(save_directory=args.save_dir)