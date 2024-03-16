"""
Fine-Tune SantaCoder on code/text dataset
"""
import argparse
import os
from mindnlp.peft import LoraConfig, get_peft_model
from mindnlp.peft.tuners.lora import LoraLayer
from mindnlp.transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    AutoConfig,
)
from tqdm import tqdm
from mindspore import nn
from mindspore.nn import AdamWeightDecay
import logging
import mindspore
from bigcode_dataset import create_datasets
import numpy as np

mindspore.set_context(device_id=6)

TRAIN_DATASET_SIZE = 26584
EVAL_DATASET_SIZE = 99
SAVE_DIR = "peft-lora-santaCoder/"

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--num_epochs", type=int, default=2)
    parser.add_argument("--model_path", type=str, default="bigcode/gpt_bigcode-santacoder")
    parser.add_argument("--dataset_name", type=str, default="smangrul/hf-stack-v1")
    parser.add_argument("--subset", type=str, default="data")
    parser.add_argument("--split", type=str, default="train")
    parser.add_argument("--size_valid_set", type=int, default=4000)
    parser.add_argument("--test_size", type=float, default=0.005)
    parser.add_argument("--streaming", action="store_true")
    parser.add_argument("--shuffle_buffer", type=int, default=5000)
    parser.add_argument("--data_column", type=str, default="content")

    parser.add_argument("--seq_length", type=int, default=8192)
    parser.add_argument("--max_steps", type=int, default=10000)
    parser.add_argument("--batch_size", type=int, default=2)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=8)
    parser.add_argument("--eos_token_id", type=int, default=49152)

    parser.add_argument("--learning_rate", type=float, default=5e-5)
    parser.add_argument("--lr_scheduler_type", type=str, default="cosine")
    parser.add_argument("--num_warmup_steps", type=int, default=100)
    parser.add_argument("--weight_decay", type=float, default=0.05)

    parser.add_argument("--local_rank", type=int, default=0)
    parser.add_argument("--no_fp16", action="store_false")
    parser.add_argument("--bf16", action="store_true")
    parser.add_argument("--no_gradient_checkpointing", action="store_false")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--num_workers", type=int, default=None)
    parser.add_argument("--output_dir", type=str, default="./checkpoints")

    parser.add_argument("--fim_rate", type=float, default=0)
    parser.add_argument("--fim_spm_rate", type=float, default=0)

    parser.add_argument("--use_peft_lora", action="store_true")
    parser.add_argument("--lora_r", type=int, default=0)
    parser.add_argument("--lora_alpha", type=int, default=0)
    parser.add_argument("--lora_dropout", type=float, default=0)
    parser.add_argument("--lora_target_modules", type=str, default=None)

    parser.add_argument("--use_4bit_qunatization", action="store_true")
    parser.add_argument("--use_nested_quant", action="store_true")
    parser.add_argument("--bnb_4bit_quant_type", type=str, default="nf4")
    parser.add_argument("--bnb_4bit_compute_dtype", type=str, default="float16")

    parser.add_argument("--use_8bit_qunatization", action="store_true")

    return parser.parse_args()

def get_model_and_tokenizer(args):
    tokenizer = AutoTokenizer.from_pretrained(
        args.model_path, use_auth_token=True, trust_remote_code=True,
    )
    tokenizer.add_special_tokens({'additional_special_tokens': ['<|endoftext|>', '<fim-prefix>', '<fim-middle>', '<fim-suffix>', '<fim-pad>']})

    config = AutoConfig.from_pretrained(args.model_path)
    model = AutoModelForCausalLM.from_pretrained(args.model_path)
    return model, tokenizer

def train_one_epoch(cur_epoch, model, optimizer, criterion, train_dataloader):
    model.set_train(True)
    total_loss, total_step = 0, 0

    def forward_fn(input_ids, labels):
        output = model(
            input_ids=input_ids,
            labels=labels
        )
        return output.loss, output.logits
    grad_fn = mindspore.value_and_grad(forward_fn, None, optimizer.parameters, has_aux=True)

    loss_list = list()
    with tqdm(total=TRAIN_DATASET_SIZE) as t:
        for _, (input_ids,  labels) in enumerate(train_dataloader):
            # forward + grad
            (loss, logits), grad = grad_fn(input_ids, labels)
            optimizer(grad)
            total_loss += loss.asnumpy()
            total_step += 1
            curr_loss = total_loss / total_step  # 当前的平均loss
            loss_list.append(loss.asnumpy())
            t.set_postfix({'train-loss': f'{curr_loss:.2f}'})
            t.update(1)

    loss_list = np.array(loss_list)
    file_path = "{}/epoch_{}_train__loss.npy".format(SAVE_DIR, cur_epoch)
    np.save(file_path, loss_list)
    return total_loss / total_step


def eval_one_epoch(cur_epoch, model, optimizer, criterion, eval_dataloader):
    model.set_train(False)
    total_loss, total_step = 0, 0

    loss_list = list()

    with tqdm(total=EVAL_DATASET_SIZE) as t:
        for _, (input_ids, labels) in enumerate(eval_dataloader):
            output = model(
                input_ids=input_ids,
                labels=labels
            )
            loss = output.loss
            total_loss += loss.asnumpy()
            total_step += 1
            curr_loss = total_loss / total_step  # 当前的平均loss
            loss_list.append(loss.asnumpy())
            t.set_postfix({'eval-loss': f'{curr_loss:.2f}'})
            t.update(1)

    loss_list = np.array(loss_list)
    file_path = "{}/epoch_{}_eval_loss.npy".format(SAVE_DIR, cur_epoch)
    np.save(file_path, loss_list)

    return total_loss / total_step


def train(model, optimizer, criterion, train_dataloader, eval_dataloader, epochs):
    for epoch in range(1, epochs+1):
        train_loss = train_one_epoch(epoch, model, optimizer, criterion, train_dataloader)
        eval_loss = eval_one_epoch(epoch,model, optimizer, criterion, eval_dataloader)

        print(f"epoch:{epoch} train_loss:{train_loss} \
              eval_loss:{eval_loss}")
        model.save_pretrained(save_directory=SAVE_DIR+"/models/")

def eval_model(model, optimizer, criterion, eval_dataloader):
    eval_loss = eval_one_epoch(model, optimizer, criterion, eval_dataloader)
    print(f"eval_loss:{eval_loss}")

def main(args):
    model, tokenizer = get_model_and_tokenizer(args)
    train_dataset, eval_dataset = create_datasets(tokenizer, args)

    if args.use_peft_lora:
        peft_config = LoraConfig(
            lora_alpha=args.lora_alpha,
            lora_dropout=args.lora_dropout,
            r=args.lora_r,
            bias="none",
            task_type="CAUSAL_LM",
            target_modules=args.lora_target_modules.split(","),
        )
        model = get_peft_model(model, peft_config)
        model.print_trainable_parameters()

    optimizer = AdamWeightDecay(params=model.trainable_params(), learning_rate=args.learning_rate)
    loss_fn = nn.CrossEntropyLoss()

    logging.info("start train")
    train(model, optimizer, loss_fn, train_dataset, eval_dataset, epochs=args.num_epochs)
    logging.info("end train")
    model.save_pretrained(save_directory=args.save_dir+"/models/")


if __name__ == "__main__":
    args = get_args()
    mindspore.set_seed(args.seed)
    os.makedirs(args.output_dir, exist_ok=True)
    main(args)