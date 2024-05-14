"""
Fine-Tune Falcon on mrpc dataset
"""

import argparse
import logging
import mindspore
import numpy as np

from tqdm import tqdm
from mindspore import nn
from mindspore.nn import AdamWeightDecay

from mrpc_dataset import load_examples, get_dataloader_from_ds

from mindnlp.transformers import (
    AutoConfig,
    AutoTokenizer,
    AutoModelForSequenceClassification,
)
from mindnlp.peft import (
    get_peft_model,
    LoraConfig,
)


def load_falcon_model(pretrained_model_name_or_path):
    config = AutoConfig.from_pretrained(
        pretrained_model_name_or_path,
        problem_type="single_label_classification",
    )
    model = AutoModelForSequenceClassification.from_pretrained(
        pretrained_model_name_or_path, config=config, ms_dtype=mindspore.float16
    )
    tokenizer = AutoTokenizer.from_pretrained(pretrained_model_name_or_path)

    return model, config, tokenizer


def train_one_epoch(model, optimizer, criterion, train_dataloader):
    model.set_train(True)
    total_loss, total_step = 0, 0
    num_batches = len(train_dataloader)
    preds, label_ids = None, None

    def forward_fn(input_ids, attention_mask, labels):
        # _, _ = hidden_states, attentions``
        output = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            # token_type_ids=token_type_ids,
            labels=mindspore.Tensor(labels.asnumpy(), mindspore.int32),
        )
        return output.loss, output.logits

    grad_fn = mindspore.value_and_grad(
        forward_fn, None, optimizer.parameters, has_aux=True
    )

    with tqdm(total=num_batches) as t:
        for i, (input_ids, attention_mask, token_type_ids, lens, labels) in enumerate(
            train_dataloader
        ):
            # forward + grad
            (loss, logits), grad = grad_fn(input_ids, attention_mask, labels)
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

            t.set_postfix({"train-loss": f"{curr_loss:.2f}"})
            t.update(1)
    acc = (preds == label_ids).mean()

    return total_loss / total_step, acc


def eval_one_epoch(model, optimizer, criterion, eval_dataloader):
    model.set_train(False)
    total_loss, total_step = 0, 0
    preds, label_ids = None, None
    num_batches = len(eval_dataloader)

    with tqdm(total=num_batches) as t:
        for i, (input_ids, attention_mask, token_type_ids, lens, labels) in enumerate(
            eval_dataloader
        ):
            output = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                # token_typ_ids=token_type_ids,
                labels=mindspore.Tensor(labels.asnumpy(), mindspore.int32),
            )
            # (loss, logits), grad = grad_fn(input_ids, attention_mask, token_type_ids, lens, labels)
            loss = output.loss
            logits = output.logits
            total_loss += loss.asnumpy()
            total_step += 1
            curr_loss = total_loss / total_step  # 当前的平均loss
            if preds is None:
                preds = np.argmax(logits.asnumpy(), axis=1)
                label_ids = labels.asnumpy()
            else:
                preds = np.append(preds, np.argmax(logits.asnumpy(), axis=1), axis=0)
                label_ids = np.append(label_ids, labels.asnumpy(), axis=0)
            t.set_postfix({"eval-loss": f"{curr_loss:.2f}"})
            t.update(1)

    # compute metrics
    acc = (preds == label_ids).mean()

    return total_loss / total_step, acc


def train(model, optimizer, criterion, train_dataloader, eval_dataloader, epochs):
    # train start from here
    for epoch in range(1, epochs + 1):
        train_loss, train_acc = train_one_epoch(
            model, optimizer, criterion, train_dataloader
        )
        eval_loss, eval_acc = eval_one_epoch(
            model, optimizer, criterion, eval_dataloader
        )

        print(
            f"epoch:{epoch} train_loss:{train_loss} train_acc:{train_acc} \
              eval_loss:{eval_loss} eval_acc:{eval_acc}"
        )
        # print(f"epoch:{epoch} train_loss:{train_loss} eval_loss:{eval_loss}")


def eval_model(model, optimizer, criterion, eval_dataloader):
    eval_loss, eval_acc = eval_one_epoch(model, optimizer, criterion, eval_dataloader)
    print(f"eval_loss:{eval_loss} eval_acc:{eval_acc}")


if __name__ == "__main__":

    # 设置gpu
    # mindspore.set_context(device_target="GPU", device_id=5)
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--save_dir",
        default=".mindnlp/peft_model/falcon/mrpc_lora",
        type=str,
        help="The output directory where the model checkpoints will be written.",
    )
    parser.add_argument(
        "--batch_size",
        default=8,
        type=int,
        help="Batch size per GPU/CPU for training.",
    )
    parser.add_argument(
        "--model_name_or_path",
        default=".mindnlp/model/Rocketknight1/falcon-rw-1b",
        type=str,
    )
    parser.add_argument("--num_epochs", default=10, type=int)
    parser.add_argument(
        "--lr", default=4e-5, type=float, help="Set 2e-5 for full-finetuning."
    )
    parser.add_argument("--max_seq_len", default=256, type=int)
    parser.add_argument("--debug", action="store_true", help="debug mode")
    parser.add_argument("--lora", action="store_true", help="lora mode")

    args = parser.parse_args()

    # load model
    model, config, tokenizer = load_falcon_model(
        pretrained_model_name_or_path=args.model_name_or_path,
    )
    logging.info("model load")

    if args.debug:
        args.num_epochs = 1

    if args.batch_size > 1:
        config.pad_token_id = config.eos_token_id

    if args.lora:
        # build peft model
        peft_config = LoraConfig(
            task_type="SEQ_CLS",
            inference_mode=False,
            r=8,
            lora_alpha=16,
            lora_dropout=0.1,
        )
        model = get_peft_model(model, peft_config)
        model.print_trainable_parameters()

    # load dataset
    train_ds = load_examples(tokenizer, args.max_seq_len, data_type="train")
    test_ds = load_examples(tokenizer, args.max_seq_len, data_type="test")
    train_dataloader = get_dataloader_from_ds(train_ds, args.batch_size)
    test_dataloader = get_dataloader_from_ds(test_ds, args.batch_size)
    logging.info("dataset load")

    # 最后两层参数param.name是一样的，手动将模型最后一个参数score改为classifier
    for name, param in model.parameters_and_names():
        if name == "base_model.model.score.original_module.weight":
            param.name = "base_model.classifier.weight"

    # optimizer
    optimizer = AdamWeightDecay(params=model.trainable_params(), learning_rate=args.lr)
    loss_fn = nn.CrossEntropyLoss()

    logging.info("start train")
    train(
        model,
        optimizer,
        loss_fn,
        train_dataloader,
        test_dataloader,
        epochs=args.num_epochs,
    )
    logging.info("end train")
    model.save_pretrained(save_directory=args.save_dir)
