import os
import json
import copy
import logging
import argparse
import numpy as np
from tqdm import tqdm 

import mindspore
from mindspore import nn
from mindspore.nn import AdamWeightDecay
from mindspore.dataset import RandomSampler, DistributedSampler, NumpySlicesDataset, SequentialSampler

from mindnlp.models.mobilebert.mobilebert import MobileBertForSequenceClassification
from mindnlp.models.mobilebert.mobilebert_config import MobileBertConfig
from mindnlp.transforms.tokenizers.bert_tokenizer import BertTokenizer
from mindnlp.models import BertModel, BertConfig, BertForSequenceClassification

from mindnlp.transforms import RobertaTokenizer
from mindnlp.models import RobertaConfig, RobertaForSequenceClassification
from mindnlp import load_dataset, process
from mindnlp.dataset import MRPC
from mindnlp.peft import (
    get_peft_config,
    get_peft_model,
    LoraConfig,
    PeftType,
    PeftConfig,
    PeftModel,
)

logger = logging.getLogger()



class InputExample(object):
    def __init__(self, guid, text_a, text_b=None, label=None):
        self.guid = guid
        self.text_a = text_a
        self.text_b = text_b
        self.label = label

    def __repr__(self):
        return str(self.to_json_string())

    def to_dict(self):
        """Serializes this instance to a Python dictionary."""
        output = copy.deepcopy(self.__dict__)
        return output

    def to_json_string(self):
        """Serializes this instance to a JSON string."""
        return json.dumps(self.to_dict(), indent=2, sort_keys=True) + "\n"
    

class InputFeatures(object):
    def __init__(self, input_ids, attention_mask, token_type_ids, label,input_len):
        self.input_ids = input_ids
        self.attention_mask = attention_mask
        self.token_type_ids = token_type_ids
        self.input_len = input_len
        self.label = label

    def __repr__(self):
        return str(self.to_json_string())

    def to_dict(self):
        """Serializes this instance to a Python dictionary."""
        output = copy.deepcopy(self.__dict__)
        return output

    def to_json_string(self):
        """Serializes this instance to a JSON string."""
        return json.dumps(self.to_dict(), indent=2, sort_keys=True) + "\n"
    

def convert_dataset_to_examples(ds):
    """Convert dataset to examples."""
    examples = []
    iter = ds.create_tuple_iterator()
    for i, (label, text_a, text_b) in enumerate(iter):
        # print(str(text_a.asnumpy()), str(text_b.asnumpy()))
        examples.append(
            InputExample(guid=i, text_a=str(text_a.asnumpy()), text_b=str(text_b.asnumpy()), label=int(label))
        )
    
    return examples

def _truncate_seq_pair(tokens_a, tokens_b, max_length):
    """Truncates a sequence pair in place to the maximum length."""
    while True:
        total_length = len(tokens_a) + len(tokens_b)
        if total_length <= max_length:
            break
        if len(tokens_a) > len(tokens_b):
            tokens_a.pop()
        else:
            tokens_b.pop()


def convert_examples_to_features(examples, tokenizer, max_seq_length=512):
    features = []

    for ex_index, example in enumerate(examples):
        tokenizer.return_token = True
        tokens_a = tokenizer.execute_py(example.text_a)
        tokens_b  = None
        if example.text_b:
            tokens_b = tokenizer.execute_py(example.text_b)
        if tokens_b is not None:
            # Modifies `tokens_a` and `tokens_b` in place so that the total
            # length is less than the specified length.
            # Account for [CLS], [SEP], [SEP] with "- 3"
            _truncate_seq_pair(tokens_a, tokens_b, max_seq_length - 3)
        else:
            # Account for [CLS] and [SEP] with "- 2"
            if len(tokens_a) > max_seq_length - 2:
                tokens_a = tokens_a[0:(max_seq_length - 2)]

        tokens = []
        token_type_ids = []
        for token in tokens_a:
            tokens.append(token)
            token_type_ids.append(0)

        if tokens_b is not None:
            for token in tokens_b[1:]:
                tokens.append(token)
                token_type_ids.append(1)
            # tokens.append("[SEP]")
            # token_type_ids.append(1)

        tokenizer.return_token=False
        # input_ids = tokenizer.execute_py(example.text_a).tolist() + tokenizer.execute_py(example.text_b).tolist()
        input_ids = tokenizer.convert_tokens_to_ids(tokens)

        # print(tokenizer.execute_py(np.array(tokens)).tolist())
        # The mask has 1 for real tokens and 0 for padding tokens. Only real
        # tokens are attended to.
        attention_mask = [1] * len(input_ids)
        input_len = len(input_ids)

        # Zero-pad up to the sequence length.
        while len(input_ids) < max_seq_length:
            input_ids.append(0)
            attention_mask.append(0)
            token_type_ids.append(0)

        assert len(input_ids) == max_seq_length
        assert len(attention_mask) == max_seq_length
        assert len(token_type_ids) == max_seq_length
        
        label_id = example.label

        # if ex_index < 5:
        #     print("*** Example ***")
        #     print("guid: %s" % (example.guid))
        #     print("tokens: %s"%" ".join([str(x) for x in tokens]))
        #     print("input_ids: %s" % " ".join([str(x) for x in input_ids]))
        #     print("attention_mask: %s" % " ".join([str(x) for x in attention_mask]))
        #     print("token_type_ids: %s" % " ".join([str(x) for x in token_type_ids]))
        #     print("label: %s (id = %d)" % (example.label, label_id))
        #     print("input length: %d" % (input_len))

        features.append(
            InputFeatures(input_ids=input_ids,
                          attention_mask=attention_mask,
                          token_type_ids=token_type_ids,
                          label=label_id,
                          input_len=input_len)
        )
    return features


def load_examples(tokenizer, max_seq_length, data_type="train"):
    """load_examples using load_dataset"""
    mrpc_train, mrpc_test = MRPC()
    mrpc = mrpc_train if data_type == "train" else mrpc_test
    train_examples = convert_dataset_to_examples(mrpc)
    # test_examples = convert_dataset_to_examples(mrpc_test)

    features = convert_examples_to_features(train_examples, tokenizer, max_seq_length=max_seq_length)

    # Convert to Tensors and build dataset
    all_input_ids = [f.input_ids for f in features]
    all_attention_mask = [f.attention_mask for f in features]
    all_token_type_ids = [f.token_type_ids for f in features]
    all_lens = [f.input_len for f in features]
    all_labels = [f.label for f in features]
    dataset = ((all_input_ids, all_attention_mask, all_token_type_ids, all_lens, all_labels))
    
    return dataset

def forward_fn(input_ids, attention_mask, token_type_ids, lens, labels):
    # _, _ = hidden_states, attentions
    loss, logits = model(
        input_ids=input_ids,
        attention_mask=attention_mask,
        token_type_ids=token_type_ids,
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
            loss, logits = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                token_type_ids=token_type_ids,
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
    # `python examples/peft/bert_mrpc_lora.py --do_train --do_eval --model_name_or_path bert-base-cased`
    # from pretrained
    parser = argparse.ArgumentParser()
    parser.add_argument("--do_train", action='store_true', help="Whether to run training.")
    parser.add_argument("--do_eval", action='store_true', help="Whether to run eval on the dev set.")
    parser.add_argument("--save_dir", default=".mindnlp/peft_model/mrpc_lora", type=str, help="The output directory where the model checkpoints will be written.")
    parser.add_argument("--batch_size", default=16, type=int, help="Batch size per GPU/CPU for training.")
    parser.add_argument("--model_name_or_path", default="roberta-base", type=str, help="bert-base-cased")
    parser.add_argument("--num_epochs", default=5, type=int)
    parser.add_argument("--lr", default=1e-4, type=float, help="Set 2e-5 for full-finetuning.")
    parser.add_argument("--max_seq_len", default=256, type=int)

    args = parser.parse_args()

    task = "mrpc"
    peft_type = PeftType.LORA

    MODEL_CLASSES = {
        "roberta-base": (RobertaConfig, RobertaForSequenceClassification, RobertaTokenizer),
        "bert-base-cased": (BertConfig, BertForSequenceClassification, BertTokenizer),
    }

    config_class, model_class, token_class = MODEL_CLASSES[args.model_name_or_path]

    if args.do_train:
        # tokenizer = token_class.from_pretrained("bert-base-cased", lower_case=True)
        # model_config = config_class(num_labels=2, problem_type="single_label_classification")
        # model = model_class.from_pretrained('bert-base-cased', config=model_config)
        tokenizer = token_class.from_pretrained(args.model_name_or_path, lower_case=True)
        model_config = config_class(num_labels=2, problem_type="single_label_classification")
        model = model_class.from_pretrained(args.model_name_or_path, config=model_config )

        train_ds = load_examples(tokenizer, args.max_seq_len, 'train')
        train_sampler = SequentialSampler()
        train_dataloader = NumpySlicesDataset(train_ds, sampler=train_sampler)
        train_dataloader = train_dataloader.batch(args.batch_size)
        
        eval_ds = load_examples(tokenizer, args.max_seq_len, 'test')
        eval_sampler = SequentialSampler()
        eval_dataloader = NumpySlicesDataset(eval_ds, sampler=eval_sampler)
        eval_dataloader = eval_dataloader.batch(args.batch_size)

        # build peft model
        peft_config = LoraConfig(task_type="SEQ_CLS", inference_mode=False, r=8, lora_alpha=16, lora_dropout=0.1)
        model = get_peft_model(model, peft_config)
        # print(model)
        model.print_trainable_parameters()

        # optimzer & loss_fn
        # print(model.trainable_params())
        optimizer = AdamWeightDecay(params=model.trainable_params(), learning_rate=args.lr)
        loss_fn = nn.CrossEntropyLoss()  

        train(model, optimizer, loss_fn, train_dataloader, eval_dataloader, epochs=args.num_epochs)

        # save peft model
        model.save_pretrained(save_directory=args.save_dir)
    
    if args.do_eval:
        # load tokenizer & eval_dataset
        tokenizer = token_class.from_pretrained(args.model_name_or_path, lower_case=True)
        eval_ds = load_examples(tokenizer, args.max_seq_len, 'test')
        eval_sampler = SequentialSampler()
        eval_dataloader = NumpySlicesDataset(eval_ds, sampler=eval_sampler)
        eval_dataloader = eval_dataloader.batch(args.batch_size)
    
        # load peft model from pretrained
        peft_model_id = args.save_dir
        config = PeftConfig.from_pretrained(peft_model_id)
        print(config.base_model_name_or_path)
        model = model_class.from_pretrained(config.base_model_name_or_path, problem_type="single_label_classification")
        model = PeftModel.from_pretrained(model, peft_model_id)

        # eval
        eval_model(model, optimizer=None, criterion=None, eval_dataloader=eval_dataloader)
