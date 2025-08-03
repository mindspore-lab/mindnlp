# encoding = utf-8
import sys
import importlib

importlib.reload(sys)

import socket
from collections import OrderedDict
from pathlib import Path
from typing import *

import mindspore
import mindspore.nn as nn
from mindspore.experimental import optim

import argparse

from functools import partial
# import wandb
from tensorboardX import SummaryWriter
from mindspore.dataset import GeneratorDataset, Dataset

from tqdm import tqdm

sys.path.append("..")
from mindnlp.transformers import (
    T5ForConditionalGeneration,
    T5Tokenizer
)

import util_t5ft as util

# Configuration details. These could be passed as command line arguments but are done this way
# for simplicity.
kname = "t5ft_ms"
comment = \
    """
    fine tune T5 with mindspore according to https://github.com/jsrozner/t5_finetune
    """

k_save_dir = "./save"
k_data_dir = "./data"
# Note, the global var record_dir is used for actual saves

k_epochs = 8  # usual 200
k_model = "t5-base"  # usual t5-small; could also be t5-base, t5-large, etc. But as written we support only T5
# to handle a different model type, change the code in main, but you might also need to change
# calls to forward, label config, etc.

# optim / sched
k_lr = 1e-4  # 1e-4 to 1e-5
k_adam_eps = 1e-8
k_warmup_steps = 0
k_max_grad_norm = 1.0

# config info
k_num_train = -1  # -1 is use all
k_num_val = -1
k_batch_size = 8
k_num_workers = 4  # num of workers for dataloader

# k_use_wandb = False # whether to log to wandb (you'll need to set up wandb env info)

# source and target lengths for dataloader. If you know your lengths you can change these, or
# add a collate function to handle different sizes. Depending on your inputs you should change these.
k_max_src_len = 50
k_max_tgt_len = 20

k_seed = 42

all_config = {
    "save_dir": k_save_dir,
    "data_dir": k_data_dir,
    "epochs": k_epochs,
    "model": k_model,
    "lr": k_lr,
    "adam_eps": k_adam_eps,
    "warmup": k_warmup_steps,
    "workers": k_num_workers,
    "max grad": k_max_grad_norm,
    "num_train": k_num_train,
    "num_val": k_num_val,
    "batch_size": k_batch_size,
    "max_src_len": k_max_src_len,
    "max_tgt_len": k_max_tgt_len
}

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--custom_dataset_path", type=str, default="./data")
    parser.add_argument("--batch_size", type=int, default=8)
    return parser.parse_args()


def _get_linear_schedule_with_warmup_lr_lambda(current_step: int, *, num_warmup_steps: int, num_training_steps: int):
    if current_step < num_warmup_steps:
        return float(current_step) / float(max(1, num_warmup_steps))
    return max(0.0, float(num_training_steps - current_step) / float(max(1, num_training_steps - num_warmup_steps)))


def get_linear_schedule_with_warmup(optimizer, num_warmup_steps, num_training_steps, last_epoch=-1):
    """
    Create a schedule with a learning rate that decreases linearly from the initial lr set in the optimizer to 0, after
    a warmup period during which it increases linearly from 0 to the initial lr set in the optimizer.

    Args:
        optimizer ([`~nn.Optimizer`]):
            The optimizer for which to schedule the learning rate.
        num_warmup_steps (`int`):
            The number of steps for the warmup phase.
        num_training_steps (`int`):
            The total number of training steps.
        last_epoch (`int`, *optional*, defaults to -1):
            The index of the last epoch when resuming training.

    Return:
        `mindspore.experimental.optim.lr_scheduler.LambdaLR` with the appropriate schedule.
    """

    lr_lambda = partial(
        _get_linear_schedule_with_warmup_lr_lambda,
        num_warmup_steps=num_warmup_steps,
        num_training_steps=num_training_steps,
    )
    return optim.lr_scheduler.LambdaLR(optimizer, lr_lambda, last_epoch)


# A dataset for our inputs.
class T5DataSet():
    def __init__(self, tokenizer, data_dir: str, type_path, max_examples=-1,
                 max_src_len=50, max_tgt_len=20):
        """
        max_examples: if > 0 then will load only max_examples into the dataset; -1 means use all

        max_src and max_tgt len refer to number of tokens in the input sequences
        # Note: these are not randomized. If they were we might need to collate.
        """

        valid_type_paths = ["test", "train", "val"]
        assert type_path in valid_type_paths, f"Type path must be one of {valid_type_paths}"

        self.example_path = Path(data_dir) / type_path
        self.max_examples = max_examples
        self.tokenizer = tokenizer

        self.max_src_len = max_src_len  # max num of tokens in tokenize()
        self.max_tgt_len = max_tgt_len

        self._index = 0
        self.inputs = []  # list of dict
        self.targets = []  # list of dict
        self.input_text = []  # list of str
        self.target_text = []  # list of str

        self._build()  # fill inputs, targets, max_lens

    def __len__(self):
        return len(self.inputs)

    def __getitem__(self, index):
        source_ids = self.inputs[index]["input_ids"].squeeze()
        target_ids = self.targets[index]["input_ids"].squeeze()

        src_mask = self.inputs[index]["attention_mask"].squeeze()  # might need to squeeze
        target_mask = self.targets[index]["attention_mask"].squeeze()  # might need to squeeze

        src_text = self.input_text[index]
        tgt_text = self.target_text[index]

        # These will be cast to torch.long in forward
        return {"source_ids": source_ids, "source_mask": src_mask,
                "target_ids": target_ids, "target_mask": target_mask,
                "source_text": src_text, "target_text": tgt_text}

    def _build(self):
        source_path = self.example_path.with_suffix(".source")
        target_path = self.example_path.with_suffix(".target")

        with open(source_path, 'r') as f_source, \
                open(target_path, 'r') as f_target:

            source, target = f_source.readlines(), f_target.readlines()
            source_ct, target_ct = len(source), len(target)
            assert source_ct == target_ct, f"Lengths don't match"

            # Note we could batch encode
            log.warning(f'Using max_src_len, max_tgt_len = ({self.max_src_len}, {self.max_tgt_len})')

            inputs_out = []  # accumulate the output of batch_encode
            targets_out = []  # same
            inputs_text = []  # save the original text for evaluations
            targets_text = []  # same

            if self.max_examples > 0:
                source_ct = min(self.max_examples, source_ct)

            for idx in range(source_ct):
                # append end of sequence tokens (not necessary) because handled by tokenize() call
                src = source[idx].strip()
                tgt = target[idx].strip()

                inputs_text.append(src)
                targets_text.append(tgt)

                # tokenize
                # padding="max_length" pads to max_len
                # otherwise (e.g. for batch), we could use padding=longest with truncation
                # note: don't need add_special_tokens since EOS added automatically and others are PAD
                # self.tokenizer returns a dict of input_ids and attention_masks (where attn masks corresponds to padding)
                # Note: padding could also be done via collate in dataloader
                # todo: we could actually batch encode these (i.e. multiple per)
                tokenized_inputs = self.tokenizer(
                    [src], max_length=self.max_src_len, padding="max_length", return_tensors="ms", truncation=True
                )
                tokenized_targets = self.tokenizer(
                    [tgt], max_length=self.max_tgt_len, padding="max_length", return_tensors="ms", truncation=True
                )
                inputs_out.append(tokenized_inputs)
                targets_out.append(tokenized_targets)
            self.inputs = inputs_out
            self.targets = targets_out
            self.input_text = inputs_text
            self.target_text = targets_text

    def __next__(self):
        if self._index >= len(self.inputs):
            raise StopIteration
        else:
            item = (self.inputs_text[self._index], self.target_text[self._index])
            self._index += 1
            return item


def get_dataloaders(tokenizer, batch_size, num_train, num_val, data_dir, num_workers, shuffle_train=True,
                    shuffle_dev=False) -> Tuple[GeneratorDataset, GeneratorDataset]:
    """
    Returns: Tuple[train_loader : DataLoader, dev_loader : DataLoader]
    # Note:
    # - we default to not shuffling the dev set

    """
    # todo: should pass max src and max tgt len in as arguments
    train_data_set = T5DataSet(tokenizer, type_path="train", data_dir=data_dir, max_examples=num_train,
                               max_src_len=k_max_src_len, max_tgt_len=k_max_tgt_len)
    eval_data_set = T5DataSet(tokenizer, type_path="val", data_dir=data_dir, max_examples=num_val,
                              max_src_len=k_max_src_len, max_tgt_len=k_max_tgt_len)
    train_loader = GeneratorDataset(train_data_set, column_names=["target_text"], shuffle=shuffle_train,
                                    num_parallel_workers=num_workers)
    train_loader = train_loader.batch(batch_size=batch_size)

    eval_loader = GeneratorDataset(eval_data_set, column_names=["target_text"], shuffle=shuffle_dev,
                                   num_parallel_workers=num_workers)
    eval_loader = eval_loader.batch(batch_size=batch_size)

    log.info(f'Datasets loaded with sizes: train: {len(train_data_set)}, dev: {len(eval_data_set)}')

    return train_loader, eval_loader


def forward(model, batch):
    src_ids = batch["source_ids"].to(mindspore.int64)
    src_mask = batch["source_mask"].to(mindspore.int64)
    tgt_ids = batch["target_ids"].to(mindspore.int64)

    # padded ids (pad=0) are set to -100, which means ignore for loss calculation
    tgt_ids[tgt_ids[:, :] == 0] = -100
    label_ids = tgt_ids
    # when we call model() with labels, they will be
    # - automatically right shifted by 1 (for teacher forcing)
    # - prepended by BOS=Beginning of sequence which is a PAD token
    # - any token that was -100 will be masked_fill_ to <pad> for teacher forcing
    # return_dict means return as a dictionary
    out_dict = model(src_ids, attention_mask=src_mask, labels=label_ids, return_dict=True)
    loss, logits = out_dict['loss'], out_dict['logits']
    return loss, logits


def main(args):
    util.set_seed(k_seed)

    model = T5ForConditionalGeneration.from_pretrained(k_model)
    tokenizer = T5Tokenizer.from_pretrained(k_model, legacy=True)

    k_batch_size = args.batch_size
    k_data_dir = args.custom_dataset_path

    train_loader, dev_loader = \
        get_dataloaders(tokenizer, batch_size=k_batch_size, num_train=k_num_train, num_val=k_num_val,
                        data_dir=k_data_dir, num_workers=k_num_workers)

    # reset in case we used the -1 flag for all
    num_train = train_loader.get_dataset_size()
    num_val = dev_loader.get_dataset_size()
    total_steps = ((num_train // k_batch_size) * k_epochs)  # num times that optim.step() will be called
    total_train = num_train * k_epochs

    optimizer = optim.AdamW(model.trainable_params(),
                            lr=k_lr,
                            eps=k_adam_eps)

    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=k_warmup_steps,
                                                num_training_steps=total_steps)

    log.info(

        f'total_steps: {total_steps}\n'
        f'total_train (num_t * epoch): {total_train}\n'
        f'machine: {socket.gethostname()}\n')

    config_str = "\n"
    for k, v in all_config.items():
        config_str += f'{k}: {v}\n'
    config_str += f'record_dir: {record_dir}\n'
    log.info(config_str)

    epoch = 0  # number of times we have passed through entire set of training examples
    step = 0  # number of total examples we have done (will be epoch * len(data_set) at end of each epoch)


    while epoch < k_epochs:
        epoch += 1
        # 配置参数

        # model.build(train_dataset=train_loader, eval_dataset=dev_loader, epochs=3)
        model.set_train()

        # Initiate the trainer

        with tqdm(total=num_train) as progress_bar:
            for batch_num, batch in enumerate(train_loader):
                print("helo")
                batch_size = len(batch["source_ids"])
                loss, logits = forward(model, batch)
                loss_val = loss.item()  # get the item since loss is a tensor

                # Backward
                grad_fn = mindspore.value_and_grad(forward_fn, None, optimizer.parameters, has_aux=True)
                (loss, _), grads = grad_fn(data, label)
                grads = ms.ops.clip_by_norm(grads, max_norm=0.5)
                optimizer(grads)

                scheduler.step()  # don't need to pass step to scheduler

                # Log info
                step += batch_size
                progress_bar.update(batch_size)
                progress_bar.set_postfix(epoch=epoch,
                                         loss=loss_val)
                tbx.add_scalar('train/loss', loss_val, step)
                tbx.add_scalar('train/LR',
                               optimizer.param_groups[0]['lr'],
                               step)

        ###############
        # Evaluate (you might want to save checkpoints)
        ###############
        log.info(f'Evaluating at step {step}...')
        model.eval()  # put model in eval mode

        # See how the model is doing with exact match on tokens
        pred_list_all = []  # accumulate for saving; list; one list per epoch
        pred_list_correct = []
        loss_meter = util.AverageMeter()  # NLL (default metric for model) (reset each time)

        # set up two count variables
        total_matches_no_eos_ct = 0
        total_matches_with_eos_ct = 0

        with tqdm(total=num_val) as progress_bar:
            for batch_num, batch in enumerate(dev_loader):
                batch_size = len(batch["source_ids"])

                # evaluation for loss fcn
                loss, _ = forward(model, batch)  # loss, logits, but don't need logits
                loss_meter.update(loss.item(), batch_size)  # loss.item() since it's a tensor

                # predict / generate for token matches
                src_ids = batch["source_ids"].to(mindspore.int64)
                src_mask = batch["source_mask"].to(mindspore.int64)
                tgt_ids = batch["target_ids"].to(mindspore.int64)
                # note you could tweak the generation params. See huggingface details for generate
                generated_ids = model.generate(src_ids, attention_mask=src_mask)  # (batch x seq length)

                # collect some stats
                total_matches_no_eos, total_matches_with_eos, correct_indices = \
                    util.masked_token_match(tgt_ids, generated_ids, return_indices=True)
                total_matches_no_eos_ct += total_matches_no_eos
                total_matches_with_eos_ct += total_matches_with_eos

                # save for qualitative analysis
                orig_text_input, orig_text_output = batch["source_text"], batch["target_text"]
                # todo: this could break once skip_special_tokens is fixed
                outputs_decoded = tokenizer.batch_decode(generated_ids, skip_special_tokens=False)
                preds = list(zip(orig_text_input, orig_text_output, outputs_decoded))
                pred_list_all.extend(preds)

                # we also store only the correct indices
                for idx in correct_indices.tolist():  # tensor to list; these are the valid indices
                    pred_list_correct.append(preds[idx[0]])  # each item was a list of one element

                # print one batch of generations for qualitative assessment
                if batch_num == 0:
                    for orig_input, orig_target, actual_output in preds[:1]:
                        log.info(f'Source: {orig_input}\t Target: {orig_target}\n'
                                 f'\t Actual: {actual_output}')

                # Log info
                progress_bar.update(batch_size)
                progress_bar.set_postfix(NLL=loss_meter.avg)

        # save predictions for qualititative analysis
        util.save_preds(pred_list_all, record_dir)
        util.save_preds(pred_list_correct, record_dir, file_name="preds_correct.csv")
        results_list = [('NLL', loss_meter.avg),
                        ('exact_match_with_eos', total_matches_with_eos_ct),
                        ('exact_match_no_eos', total_matches_no_eos_ct)]
        results = OrderedDict(results_list)

        # Log to console
        results_str = ', '.join(f'{k}: {v:05.2f}' for k, v in results.items())
        log.info(f'Dev {results_str}')

        # Log to TensorBoard
        for k, v in results.items():
            tbx.add_scalar(f'dev/{k}', v, step)
        util.visualize(tbx,
                       pred_dict=pred_list_all,  # will be truncated by num_visuals
                       step=step,
                       split='dev',
                       num_visuals=3)


if __name__ == '__main__':
    name = kname
    record_dir = util.get_save_dir(k_save_dir, name)

    args = get_args()

    log = util.get_logger(record_dir, "root", "debug")
    tbx = SummaryWriter(record_dir, flush_secs=5)
    log.info(name)
    log.info(comment)
    main(args)