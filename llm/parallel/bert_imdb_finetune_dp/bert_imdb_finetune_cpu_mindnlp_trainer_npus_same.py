#!/usr/bin/env python
# coding: utf-8
"""
unset MULTI_NPU && python bert_imdb_finetune_cpu_mindnlp_trainer_npus_same.py
bash bert_imdb_finetune_npu_mindnlp_trainer.sh
"""

import mindspore
from mindspore.dataset import transforms
from mindnlp.engine import Trainer
from mindnlp.dataset import load_dataset

from mindnlp.accelerate.utils.constants import accelerate_distributed_type
from mindnlp.accelerate.utils.dataclasses import DistributedType

def main():
    """demo

    Returns:
        desc: _description_
    """
    imdb_ds = load_dataset('imdb', split=['train', 'test'])
    imdb_train = imdb_ds['train']
    imdb_train.get_dataset_size()

    from mindnlp.transformers import AutoTokenizer
    # tokenizer
    tokenizer = AutoTokenizer.from_pretrained('bert-base-cased')

    def process_dataset(dataset, tokenizer, max_seq_len=256, batch_size=32, shuffle=False):
        is_ascend = mindspore.get_context('device_target') == 'Ascend'
        def tokenize(text):
            if is_ascend:
                tokenized = tokenizer(text, padding='max_length', truncation=True, max_length=max_seq_len)
            else:
                tokenized = tokenizer(text, truncation=True, max_length=max_seq_len)
            return tokenized['input_ids'], tokenized['token_type_ids'], tokenized['attention_mask']

        if shuffle:
            dataset = dataset.shuffle(batch_size)

        # map dataset
        dataset = dataset.map(operations=[tokenize], input_columns="text", output_columns=['input_ids', 'token_type_ids', 'attention_mask'])
        dataset = dataset.map(operations=transforms.TypeCast(mindspore.int32), input_columns="label", output_columns="labels")
        # batch dataset
        if is_ascend:
            dataset = dataset.batch(batch_size)
        else:
            dataset = dataset.padded_batch(batch_size, pad_info={'input_ids': (None, tokenizer.pad_token_id),
                                                                'token_type_ids': (None, 0),
                                                                'attention_mask': (None, 0)})
        return dataset


    dataset_train = process_dataset(imdb_train, tokenizer, shuffle=True)

    next(dataset_train.create_tuple_iterator())

    from mindnlp.transformers import AutoModelForSequenceClassification

    # set bert config and define parameters for training
    model = AutoModelForSequenceClassification.from_pretrained('bert-base-cased', num_labels=2)

    from mindnlp.engine import TrainingArguments
    training_args = TrainingArguments(
        output_dir="bert_imdb_finetune_cpu",
        save_strategy="epoch",
        logging_strategy="epoch",
        num_train_epochs=2.0,
        learning_rate=2e-5
    )
    training_args = training_args.set_optimizer(name="adamw", beta1=0.8) # Manually specify the optimizer, OptimizerNames.SGD

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=dataset_train,
    )
    print("Start training")
    trainer.train()

if __name__ == '__main__':
    main()

