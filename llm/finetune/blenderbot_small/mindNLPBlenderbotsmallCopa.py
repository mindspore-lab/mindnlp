from mindnlp.transformers import BlenderbotSmallForConditionalGeneration, BlenderbotSmallTokenizer
from mindnlp.engine import Trainer, TrainingArguments
from datasets import load_dataset, load_from_disk
import mindspore as ms
import os

# 设置运行模式和设备
ms.set_context(mode=ms.PYNATIVE_MODE, device_target="Ascend")

# 设置 HF_ENDPOINT 环境变量
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
# 加载模型和分词器
print("加载模型和分词器")
model_name = "facebook/blenderbot_small-90M"
tokenizer = BlenderbotSmallTokenizer.from_pretrained(model_name)
model = BlenderbotSmallForConditionalGeneration.from_pretrained(model_name)
print("模型和分词器加载完成")
# 测试原始模型的输出
input = "The Vatican Apostolic Library, more commonly called the Vatican Library or simply the Vat, is the library of the Holy See, located in Vatican City. Formally established in 1475, although it is much older, it is one of the oldest libraries in the world and contains one of the most significant collections of historical texts. It has 75,000 codices from throughout history, as well as 1.1 million printed books, which include some 8,500 incunabula. \n\nThe Vatican Library is a research library for history, law, philosophy, science and theology. The Vatican Library is open to anyone who can document their qualifications and research needs. Photocopies for private study of pages from books published between 1801 and 1990 can be requested in person or by mail. \n\nIn March 2014, the Vatican Library began an initial four-year project of digitising its collection of manuscripts, to be made available online. \n\nThe Vatican Secret Archives were separated from the library at the beginning of the 17th century; they contain another 150,000 items. \n\nScholars have traditionally divided the history of the library into five periods, Pre-Lateran, Lateran, Avignon, Pre-Vatican and Vatican. \n\nThe Pre-Lateran period, comprising the initial days of the library, dated from the earliest days of the Church. Only a handful of volumes survive from this period, though some are very significant.When was the Vat formally opened?"
print("input question:", input)
input_tokens = tokenizer([input], return_tensors="ms")
output_tokens = model.generate(**input_tokens)
print("output answer:", tokenizer.batch_decode(output_tokens, skip_special_tokens=True)[0])

# # 设置填充标记（BlenderbotSmall默认无pad_token）
# # tokenizer.pad_token = tokenizer.eos_token  # 用eos_token作为填充标记
# # model.config.pad_token_id = tokenizer.eos_token_id

print("加载数据集")
# 定义数据集保存路径
dataset_path = "./dataset_valid_preprocessed"
# 检查是否存在处理好的数据集
if os.path.exists(dataset_path):
    # 加载预处理后的数据集
    dataset_train = load_from_disk("./dataset_train_preprocessed")
    dataset_valid = load_from_disk("./dataset_valid_preprocessed")
else:
    dataset = load_dataset("stanfordnlp/coqa")
    print("dataset finished\n")
    print("dataset:", dataset)
    print("\ndataset[train][0]:", dataset["train"][0])
    print("\ndataset[validation][0]:", dataset["validation"][0])
    dataset_train = dataset["train"]
    dataset_valid = dataset["validation"]
    # 数据预处理，coqa数据集是一个sotry和多个问题和多个答案的数据集，这里只取出第一个问题和第一个答案，sotry和问题拼接作为模型的输入，第一个答案作为模型的输出
    def preprocess_function(examples):
        # 取出第一个问题的文本
        first_question = examples['questions'][0]
        # 取出第一个答案的文本
        first_answer = examples['answers']['input_text'][0]
        # 将故事和第一个问题拼接成模型的输入格式
        inputs = examples['story'] + " " + first_question
        # 删除多余的引号
        inputs = inputs.replace('"', '')
        # 将第一个答案作为模型的输出
        labels = first_answer
        # 删除多余的引号
        labels = labels.replace('"', '')
        return {'input_ids': inputs, 'labels': labels}

    def tokenize_function(examples):
        # 对输入进行分词
        model_inputs = tokenizer(examples['input_ids'], max_length=512, truncation=True, padding="max_length")
        # 对标签进行分词
        with tokenizer.as_target_tokenizer():
            labels = tokenizer(examples['labels'], max_length=512, truncation=True, padding="max_length")
        model_inputs["labels"] = labels["input_ids"]
        return model_inputs
    # 应用预处理函数
    dataset_train = dataset_train.map(preprocess_function, batched=False) 
    dataset_train = dataset_train.map(tokenize_function, batched=True)
    dataset_train = dataset_train.remove_columns(["source", "story", "questions", "answers"])

    dataset_valid = dataset_valid.map(preprocess_function, batched=False)
    dataset_valid = dataset_valid.map(tokenize_function, batched=True)
    dataset_valid = dataset_valid.remove_columns(["source", "story", "questions", "answers"])

    dataset_train.save_to_disk("./dataset_train_preprocessed")
    dataset_valid.save_to_disk("./dataset_valid_preprocessed")
    print("dataset_train_tokenizerd:", dataset_train)

print("转化为mindspore格式数据集")
import numpy as np
def data_generator(dataset):
    for item in dataset:
        yield (
            np.array(item["input_ids"], dtype=np.int32),
            np.array(item["attention_mask"], dtype=np.int32), 
            np.array(item["labels"], dtype=np.int32)
        )
import mindspore.dataset as ds
def create_mindspore_dataset(dataset, shuffle=True):
    return ds.GeneratorDataset(
        source=lambda: data_generator(dataset),  # 使用 lambda 包装生成器
        column_names=["input_ids", "attention_mask", "labels"],
        shuffle=shuffle,
        num_parallel_workers=1
    )
dataset_train_tokenized = create_mindspore_dataset(dataset_train, shuffle=True)
dataset_valid_tokenized = create_mindspore_dataset(dataset_valid, shuffle=False)

TOKENS = 20
EPOCHS = 10
BATCH_SIZE = 4
training_args = TrainingArguments(
    output_dir='./MindNLPblenderbot_coqa_finetuned',
    overwrite_output_dir=True,
    num_train_epochs=EPOCHS,
    per_device_train_batch_size=BATCH_SIZE,
    per_device_eval_batch_size=BATCH_SIZE,
    save_steps=500,                  # Save checkpoint every 500 steps
    save_total_limit=2,              # Keep only the last 2 checkpoints
    logging_dir="./mindsporelogs",            # Directory for logs
    logging_steps=100,               # Log every 100 steps
    logging_strategy="epoch",
    evaluation_strategy="epoch",
    eval_steps=500,                  # Evaluation frequency
    warmup_steps=100,
    learning_rate=5e-5,
    weight_decay=0.01,               # Weight decay
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=dataset_train_tokenized,
    eval_dataset=dataset_valid_tokenized
)
# 开始训练
print("开始训练")
trainer.train()
eval_results = trainer.evaluate()
print(f"Evaluation results: {eval_results}")
model.save_pretrained("./blenderbot_coqa_finetuned")
tokenizer.save_pretrained("./blenderbot_coqa_finetuned")
fine_tuned_model = BlenderbotSmallForConditionalGeneration.from_pretrained("./blenderbot_coqa_finetuned")
fine_tuned_tokenizer = BlenderbotSmallTokenizer.from_pretrained("./blenderbot_coqa_finetuned")


print("再次测试对话")
input = "The Vatican Apostolic Library, more commonly called the Vatican Library or simply the Vat, is the library of the Holy See, located in Vatican City. Formally established in 1475, although it is much older, it is one of the oldest libraries in the world and contains one of the most significant collections of historical texts. It has 75,000 codices from throughout history, as well as 1.1 million printed books, which include some 8,500 incunabula. \n\nThe Vatican Library is a research library for history, law, philosophy, science and theology. The Vatican Library is open to anyone who can document their qualifications and research needs. Photocopies for private study of pages from books published between 1801 and 1990 can be requested in person or by mail. \n\nIn March 2014, the Vatican Library began an initial four-year project of digitising its collection of manuscripts, to be made available online. \n\nThe Vatican Secret Archives were separated from the library at the beginning of the 17th century; they contain another 150,000 items. \n\nScholars have traditionally divided the history of the library into five periods, Pre-Lateran, Lateran, Avignon, Pre-Vatican and Vatican. \n\nThe Pre-Lateran period, comprising the initial days of the library, dated from the earliest days of the Church. Only a handful of volumes survive from this period, though some are very significant.When was the Vat formally opened?"
print("input question:", input)
input_tokens = fine_tuned_tokenizer([input], return_tensors="ms")
output_tokens = fine_tuned_model.generate(**input_tokens)
print("output answer:", fine_tuned_tokenizer.batch_decode(output_tokens, skip_special_tokens=True)[0])