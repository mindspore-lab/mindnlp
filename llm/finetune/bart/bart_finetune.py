from mindspore import nn, ops, Tensor
from mindspore.dataset import GeneratorDataset
from mindnlp.transformers import BartForConditionalGeneration, BartTokenizer
from mindnlp.engine import Trainer, TrainingArguments
from datasets import load_dataset

import evaluate
import mindspore as ms


rouge_metric = evaluate.load("rouge")
# Load dataset and tokenizer
tokenizer = BartTokenizer.from_pretrained("./bart-base")

dataset = load_dataset("xsum", split="train")
val_dataset = load_dataset("xsum", split="validation")


def preprocess_function(examples):
    inputs = tokenizer(examples["document"], max_length=512,
                       truncation=True, padding="max_length")
    targets = tokenizer(
        examples["summary"], max_length=128, truncation=True, padding="max_length")
    inputs["labels"] = targets["input_ids"]
    return inputs


tokenized_data = dataset.map(preprocess_function, batched=True, remove_columns=[
                             "document", "summary", "id"], num_proc=24)
tokenized_val_data = val_dataset.map(preprocess_function, batched=True, remove_columns=[
                                     "document", "summary", "id"], num_proc=24)


# Load model
model = BartForConditionalGeneration.from_pretrained("./bart-base")


def create_mindspore_dataset(data, batch_size=8):
    data_list = list(data)

    def generator():
        for item in data_list:
            yield (
                Tensor(item["input_ids"], dtype=ms.int32),
                Tensor(item["attention_mask"], dtype=ms.int32),
                Tensor(item["labels"], dtype=ms.int32)
            )

    return GeneratorDataset(generator, column_names=["input_ids", "attention_mask", "labels"]).batch(batch_size)


def compute_metrics(pred):

    labels_ids = pred.label_ids
    pred_ids = pred.predictions[0]

    pred_str = tokenizer.batch_decode(pred_ids, skip_special_tokens=True)
    labels_ids[labels_ids == -100] = tokenizer.pad_token_id
    label_str = tokenizer.batch_decode(labels_ids, skip_special_tokens=True)

    rouge_output = rouge_metric.compute(
        predictions=pred_str,
        references=label_str,
        rouge_types=["rouge1", "rouge2", "rougeL", "rougeLsum"],
    )

    return {
        "R1": round(rouge_output["rouge1"], 4),
        "R2": round(rouge_output["rouge2"], 4),
        "RL": round(rouge_output["rougeL"], 4),
        "RLsum": round(rouge_output["rougeLsum"], 4),
    }


def preprocess_logits_for_metrics(logits, labels):
    """
    防止内存溢出
    """
    pred_ids = ms.mint.argmax(logits[0], dim=-1)
    return pred_ids, labels


train_dataset = create_mindspore_dataset(tokenized_data, batch_size=4)
eval_dataset = create_mindspore_dataset(tokenized_val_data, batch_size=2)

training_args = TrainingArguments(
    output_dir="./results",
    evaluation_strategy="epoch",
    learning_rate=2e-5,
    per_device_train_batch_size=4,
    per_device_eval_batch_size=2,
    num_train_epochs=3,
    weight_decay=0.01,
    save_total_limit=2,
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset,
    tokenizer=tokenizer,
    compute_metrics=compute_metrics,
    preprocess_logits_for_metrics=preprocess_logits_for_metrics,
)

trainer.train()
