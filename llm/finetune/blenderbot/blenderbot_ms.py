import os
import numpy as np
import mindspore as ms
from mindspore import context, nn, Tensor
from mindnlp.transformers import BlenderbotTokenizer, BlenderbotForConditionalGeneration
from datasets import load_dataset as hf_load_dataset
from mindspore.dataset import GeneratorDataset

# 环境配置
context.set_context(
    mode=context.PYNATIVE_MODE,
    device_target="Ascend",
    device_id=0,
    enable_graph_kernel=False,
    max_call_depth=3000,
    pynative_synchronize=True
)
ms.set_auto_parallel_context(parallel_mode=ms.ParallelMode.STAND_ALONE)
ms.set_context(reserve_class_name_in_scope=False)

# 数据加载和过滤
def load_and_process_data():
    print("加载数据集...")
    dataset = hf_load_dataset("databricks/databricks-dolly-15k", split="train")
    print(f"原始数据集大小: {len(dataset)}")
    filtered_dataset = dataset.filter(
        lambda x: x["instruction"] is not None and len(x["instruction"]) > 10
    )
    print(f"过滤后数据集大小: {len(filtered_dataset)}")
    return filtered_dataset

# 数据预处理
def preprocess_data(tokenizer, dataset):
    print("开始数据预处理...")
    def process(examples):
        inputs = [str(text) for text in examples["instruction"]]
        targets = [str(text) for text in examples["response"]]
        model_inputs = tokenizer(
            inputs,
            max_length=128,
            truncation=True,
            padding="max_length",
            return_tensors="np"
        )
        with tokenizer.as_target_tokenizer():
            labels = tokenizer(
                targets,
                max_length=128,
                truncation=True,
                padding="max_length",
                return_tensors="np"
            )
        # 确保 input_ids 为 int32 类型
        input_ids = model_inputs["input_ids"].astype(np.int32)
        attention_mask = model_inputs["attention_mask"].astype(np.int32)
        labels_ids = labels["input_ids"].astype(np.int32)
        print(f"预处理中 input_ids 类型: {input_ids.dtype}, 示例值: {input_ids[0][:5]}")
        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels_ids
        }

    processed_dataset = dataset.map(
        process,
        batched=True,
        batch_size=8,
        remove_columns=dataset.column_names
    )
    print(f"预处理后数据集大小: {len(processed_dataset)}")
    return processed_dataset

# 创建 MindSpore 数据集
def create_dynamic_dataset(tokenized_dataset):
    print("创建 MindSpore 数据集...")
    def generator():
        for item in tokenized_dataset:
            yield (
                ms.Tensor(item["input_ids"], dtype=ms.int32),
                ms.Tensor(item["attention_mask"], dtype=ms.int32),
                ms.Tensor(item["labels"], dtype=ms.int32)
            )

    dataset = GeneratorDataset(
        source=generator,
        column_names=["input_ids", "attention_mask", "labels"],
        shuffle=False
    ).batch(1, drop_remainder=True)
    print("数据集创建完成")
    return dataset

# 模型定义
class DynamicBlenderbot(nn.Cell):
    def __init__(self):
        super().__init__()
        model_name = "facebook/blenderbot-400M-distill"
        print(f"加载模型和分词器: {model_name}")
        self.tokenizer = BlenderbotTokenizer.from_pretrained(model_name)
        self.model = BlenderbotForConditionalGeneration.from_pretrained(model_name)
        self.model.set_train(True)
        
        # 检查参数加载情况
        trainable_params = []
        print("检查模型参数...")
        for idx, param in enumerate(self.model.get_parameters()):
            param.requires_grad = True
            trainable_params.append(param)
            if idx < 5:
                print(f"Parameter {idx} shape: {param.shape}, requires_grad: {param.requires_grad}")
        
        total_params = len(list(self.model.get_parameters()))
        print(f"模型总参数数量: {total_params}")
        print(f"可训练参数数量: {len(trainable_params)}")
        if not trainable_params:
            raise ValueError("模型初始化后没有可训练参数！")

    def construct(self, input_ids, attention_mask, labels):
        print(f"construct: input_ids dtype: {input_ids.dtype}, shape: {input_ids.shape}, 示例值: {input_ids[0][:5]}")
        print(f"construct: attention_mask dtype: {attention_mask.dtype}, shape: {attention_mask.shape}")
        print(f"construct: labels dtype: {labels.dtype}, shape: {labels.shape}")
        # 强制转换 input_ids 为 int32
        input_ids = input_ids.astype(ms.int32)
        return self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels
        ).loss

# 训练循环
def dynamic_train():
    dataset = load_and_process_data()
    tokenizer = DynamicBlenderbot().tokenizer
    processed_data = preprocess_data(tokenizer, dataset)
    train_dataset = create_dynamic_dataset(processed_data)

    print("初始化模型...")
    net = DynamicBlenderbot()
    
    # 执行一次虚拟前向传播以初始化参数
    print("执行虚拟前向传播...")
    dummy_input_ids = Tensor(np.zeros((1, 128)), dtype=ms.int32)
    dummy_attention_mask = Tensor(np.ones((1, 128)), dtype=ms.int32)
    dummy_labels = Tensor(np.zeros((1, 128)), dtype=ms.int32)
    print(f"dummy_input_ids dtype: {dummy_input_ids.dtype}, 示例值: {dummy_input_ids[0][:5]}")
    print(f"dummy_attention_mask dtype: {dummy_attention_mask.dtype}")
    print(f"dummy_labels dtype: {dummy_labels.dtype}")
    net(dummy_input_ids, dummy_attention_mask, dummy_labels)
    print("虚拟前向传播完成")

    # 获取可训练参数
    params = net.trainable_params()
    print(f"优化器可训练参数数量: {len(params)}")
    if not params:
        raise ValueError("前向传播后仍无可训练参数！")

    # 创建优化器
    optimizer = nn.Adam(params, learning_rate=2e-5)
    grad_accum_steps = 4
    loss_scaler = ms.amp.DynamicLossScaler(1024, 2, 1000)
    step = 0

    for epoch in range(3):
        net.set_train(True)
        print(f"开始第 {epoch + 1} 个 epoch...")
        for batch in train_dataset:
            loss = net(*batch)
            scaled_loss = loss_scaler.scale(loss)
            scaled_loss.backward()
            if (step + 1) % grad_accum_steps == 0:
                loss_scaler.unscale_(optimizer)
                ms.amp.clip_grad_value_(params, 1.0)
                optimizer.step()
                optimizer.zero_grad()
            if step % 10 == 0:
                print(f"Step {step} Loss: {loss.asnumpy()}")
            if step % 100 == 0:
                ms.save_checkpoint(net, f"./checkpoints/step_{step}.ckpt")
            step += 1

if __name__ == "__main__":
    assert context.get_context("mode") == context.PYNATIVE_MODE, "必须使用动态图模式"
    dynamic_train()