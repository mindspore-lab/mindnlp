"""
Finetune Graphormer
"""
import os
import json
import argparse
from os.path import join as pjoin
from typing import List

import mindspore
from mindspore.dataset import text, GeneratorDataset, transforms
from mindspore import nn, ops, context

from mindnlp.engine import Trainer, Evaluator
from mindnlp.engine.callbacks import CheckpointCallback, BestModelCallback
from mindnlp.metrics import Accuracy
from mindnlp import load_dataset
from mindnlp.transformers import (
    GraphormerForGraphClassification,
    GraphormerDataCollator
)


def batch_dataset(dataset: GeneratorDataset,
                  batch_size: int,
                  data_collator: GraphormerDataCollator,
                  input_columns: List[str]):
    dataset_batched = dataset.batch(batch_size=batch_size,
                                    per_batch_map=data_collator,
                                    input_columns=input_columns,
                                    output_columns=data_collator.output_columns)
    return dataset_batched


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_name", type=str, default="ogb/ogbg-molhiv")
    parser.add_argument("--batch_size", type=str, default=3)
    return parser.parse_args()


def main(args):
    # Choose GPU device to run the training
    context.set_context(device_target='GPU', device_id=1)

    # Load dataset
    dataset = load_dataset(args.dataset_name)
    dataset_train = dataset["train"]
    dataset_val = dataset["validation"]
    input_columns = ["edge_index", "edge_attr", "y", "num_nodes", "node_feat"]

    # Batch dataset and introduce the data collator required by graphormer
    data_collator = GraphormerDataCollator(on_the_fly_processing=True)

    dataset_train = batch_dataset(dataset_train, args.batch_size,
                                  data_collator, input_columns)
    dataset_val = batch_dataset(dataset_val, args.batch_size,
                                data_collator, input_columns)

    # Set validation metric and checkpoint callbacks
    metric = Accuracy()
    ckpoint_cb = CheckpointCallback(save_path='checkpoint',
                                    ckpt_name='graphormer',
                                    epochs=1,
                                    keep_checkpoint_max=2)

    best_model_cb = BestModelCallback(save_path='checkpoint',
                                      ckpt_name='graphormer',
                                      auto_load=True)

    # Load model
    model = GraphormerForGraphClassification.from_pretrained("clefourrier/graphormer-base-pcqm4mv2")

    # Initiate the optimizer
    optimizer = nn.AdamWeightDecay(model.trainable_params(),
                                   learning_rate=5e-5,
                                   beta1=0.9,
                                   beta2=0.999,
                                   eps=1e-8)

    # Initiate the trainer
    trainer = Trainer(network=model,
                      train_dataset=dataset_train,
                      eval_dataset=dataset_val,
                      metrics=metric,
                      epochs=1,
                      optimizer=optimizer,
                      callbacks=[ckpoint_cb, best_model_cb],
                      jit=False)


    # Start training
    trainer.run(tgt_columns="labels")


if __name__ == "__main__":
    args = get_args()
    main(args)
