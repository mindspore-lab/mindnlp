import mindspore as ms
import numpy as np
from datasets import Audio, ClassLabel, load_dataset
from mindspore.dataset import GeneratorDataset
from sklearn.metrics import accuracy_score
from mindnlp.engine import Trainer, TrainingArguments
from mindnlp.transformers import (ASTConfig, ASTFeatureExtractor,
                                  ASTForAudioClassification)

ms.set_context(mode=ms.PYNATIVE_MODE, device_target="Ascend")

# 加载esc50数据集
esc50 = load_dataset("ashraq/esc50", split="train")

df = esc50.select_columns(["target", "category"]).to_pandas()
class_names = df.iloc[np.unique(df["target"], return_index=True)[
    1]]["category"].to_list()

esc50 = esc50.cast_column("target", ClassLabel(names=class_names))
esc50 = esc50.cast_column("audio", Audio(sampling_rate=16000))
esc50 = esc50.rename_column("target", "labels")
num_labels = len(np.unique(esc50["labels"]))

# 初始化AST
pretrained_model = "MIT/ast-finetuned-audioset-10-10-0.4593"
feature_extractor = ASTFeatureExtractor.from_pretrained(pretrained_model)
model_input_name = feature_extractor.model_input_names[0]
SAMPLING_RATE = feature_extractor.sampling_rate


# 预处理音频
def preprocess_audio(batch):
    wavs = [audio["array"] for audio in batch["input_values"]]
    inputs = feature_extractor(
        wavs, sampling_rate=SAMPLING_RATE, return_tensors="ms")
    return {model_input_name: inputs.get(model_input_name), "labels": list(batch["labels"])}


dataset = esc50
label2id = dataset.features["labels"]._str2int

# 构造训练集和测试集
if "test" not in dataset:
    dataset = dataset.train_test_split(
        test_size=0.2, shuffle=True, seed=0, stratify_by_column="labels")


dataset = dataset.cast_column("audio", Audio(
    sampling_rate=feature_extractor.sampling_rate))
dataset = dataset.rename_column("audio", "input_values")

dataset["train"].set_transform(
    preprocess_audio, output_all_columns=False)
dataset["test"].set_transform(preprocess_audio, output_all_columns=False)

# 加载config
config = ASTConfig.from_pretrained(pretrained_model)
config.num_labels = num_labels
config.label2id = label2id
config.id2label = {v: k for k, v in label2id.items()}

model = ASTForAudioClassification.from_pretrained(
    pretrained_model, config=config, ignore_mismatched_sizes=True)


def convert_mindspore_datatset(hf_dataset, batch_size):
    data_list = list(hf_dataset)

    def generator():
        for item in data_list:
            yield item[model_input_name], item["labels"]
    # 构造MindSpore的GeneratorDataset
    ds = GeneratorDataset(
        source=generator,
        column_names=[model_input_name, "labels"],
        shuffle=False
    )
    ds = ds.batch(batch_size, drop_remainder=True)
    return ds


# 初始化训练参数
training_args = TrainingArguments(
    output_dir="./checkpoint",
    logging_dir="./logs",
    learning_rate=5e-5,
    num_train_epochs=10,
    per_device_train_batch_size=8,
    evaluation_strategy="epoch",
    save_strategy="epoch",
    eval_steps=1,
    save_steps=1,
    load_best_model_at_end=True,
    metric_for_best_model="accuracy",
    logging_strategy="epoch",
    logging_steps=20,
)

train_ms_dataset = convert_mindspore_datatset(
    dataset["train"], training_args.per_device_train_batch_size)
eval_ms_dataset = convert_mindspore_datatset(
    dataset["test"], training_args.per_device_train_batch_size)


def compute_metrics(eval_pred):
    logits = eval_pred.predictions
    labels = eval_pred.label_ids
    predictions = np.argmax(logits, axis=1)
    return {"accuracy": accuracy_score(predictions, labels)}


# 初始化trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_ms_dataset,
    eval_dataset=eval_ms_dataset,
    compute_metrics=compute_metrics,
)

trainer.train()
