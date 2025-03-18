import os
import numpy as np
import mindspore as ms
from mindspore import context, nn, Tensor, Parameter
from mindnlp.transformers import BlenderbotTokenizer, BlenderbotForConditionalGeneration
from datasets import load_dataset as hf_load_dataset
from mindspore.dataset import GeneratorDataset

# 环境配置
context.set_context(
    device_id=0,
    max_call_depth=3000,
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
        batch_size=64,
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
    ).batch(32, drop_remainder=True)
    print("数据集创建完成")
    return dataset

# 模型定义
class DynamicBlenderbot(nn.Cell):
    def __init__(self, model_name="facebook/blenderbot-400M-distill"):
        super().__init__()
        print(f"加载模型和分词器: {model_name}")
        self.tokenizer = BlenderbotTokenizer.from_pretrained(model_name)
        self.model = BlenderbotForConditionalGeneration.from_pretrained(model_name)
        self.model.set_train(True)

        # 冻结底层 Transformer 层：冻结 encoder 和 decoder 中前 3 层
        num_layers_to_freeze = 3
        for name, param in self.model.parameters_and_names():
            if "encoder.layers" in name:
                try:
                    layer_num = int(name.split("encoder.layers.")[1].split(".")[0])
                    if layer_num < num_layers_to_freeze:
                        param.requires_grad = False
                except Exception as e:
                    pass
            if "decoder.layers" in name:
                try:
                    layer_num = int(name.split("decoder.layers.")[1].split(".")[0])
                    if layer_num < num_layers_to_freeze:
                        param.requires_grad = False
                except Exception as e:
                    pass

        # 显式注册模型参数
        print("显式注册模型参数...")
        for name, param in self.model.parameters_and_names():
            # 根据冻结标志注册对应参数
            setattr(self, f"param_{name.replace('.', '_')}", Parameter(param, requires_grad=param.requires_grad))

        # 检查参数加载情况
        trainable_params = self.trainable_params()
        print("检查模型参数...")
        for idx, param in enumerate(trainable_params):
            # 这里只会包含可训练参数（未冻结部分）
            param.requires_grad = True
            if idx < 5:
                print(f"Parameter {idx} shape: {param.shape}, requires_grad: {param.requires_grad}")

        total_params = len(trainable_params)
        print(f"模型总参数数量: {total_params}")
        if not trainable_params:
            raise ValueError(f"模型 {model_name} 初始化后没有可训练参数！请检查模型是否正确加载或兼容 MindSpore。")

    def construct(self, input_ids, attention_mask, labels):
        input_ids = input_ids.astype(ms.int32)
        return self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels
        ).loss

# 定义训练单步网络
class TrainOneStepCell(nn.Cell):
    def __init__(self, network, optimizer, grad_clip_value=1.0):
        super(TrainOneStepCell, self).__init__()
        self.network = network
        self.optimizer = optimizer
        self.weights = ms.ParameterTuple(network.trainable_params())
        self.grad = ms.ops.GradOperation(get_by_list=True)
        self.grad_clip_value = grad_clip_value
        self.clip_by_value = ms.ops.clip_by_value

    def construct(self, *inputs):
        loss = self.network(*inputs)
        grads = self.grad(self.network, self.weights)(*inputs)
        # 手动裁剪梯度
        grads = tuple(self.clip_by_value(g, -self.grad_clip_value, self.grad_clip_value) for g in grads)
        self.optimizer(grads)
        return loss

# 训练循环
def dynamic_train(model_name="facebook/blenderbot-400M-distill"):
    # 创建检查点目录
    checkpoint_dir = "./checkpoints"
    os.makedirs(checkpoint_dir, exist_ok=True)
    print(f"检查点将保存至: {os.path.abspath(checkpoint_dir)}")

    dataset = load_and_process_data()
    tokenizer = DynamicBlenderbot(model_name).tokenizer
    processed_data = preprocess_data(tokenizer, dataset)
    train_dataset = create_dynamic_dataset(processed_data)

    print("初始化模型...")
    net = DynamicBlenderbot(model_name)

    # 执行一次虚拟前向传播以初始化参数
    print("执行虚拟前向传播...")
    dummy_input_ids = Tensor(np.zeros((16, 128)), dtype=ms.int32)
    dummy_attention_mask = Tensor(np.ones((16, 128)), dtype=ms.int32)
    dummy_labels = Tensor(np.zeros((16, 128)), dtype=ms.int32)
    net(dummy_input_ids, dummy_attention_mask, dummy_labels)
    print("虚拟前向传播完成")

    # 获取可训练参数
    params = net.trainable_params()
    print(f"优化器可训练参数数量: {len(params)}")
    if not params:
        raise ValueError(f"前向传播后无可训练参数！请检查模型 {model_name} 兼容性或 MindSpore 配置。")

    # 创建学习率调度器和优化器
    total_steps = len(train_dataset) * 3  # 3个epoch的总步数
    lr_scheduler = nn.CosineDecayLR(min_lr=1e-6, max_lr=2e-5, decay_steps=total_steps)
    optimizer = nn.Adam(params, learning_rate=lr_scheduler)

    # 包装网络为单步训练网络
    train_net = TrainOneStepCell(net, optimizer, grad_clip_value=1.0)
    train_net.set_train(True)

    step = 0
    for epoch in range(3):
        print(f"开始第 {epoch + 1} 个 epoch...")
        for batch in train_dataset:
            loss = train_net(*batch)
            current_lr = optimizer.learning_rate(step).asnumpy()  # 获取当前学习率
            if step % 10 == 0:
                print(f"Step {step} Loss: {loss.asnumpy()}, Learning Rate: {current_lr}")
            if step % 100 == 0:
                ms.save_checkpoint(net, f"{checkpoint_dir}/step_{step}.ckpt")
            step += 1

if __name__ == "__main__":
    assert context.get_context("mode") == context.PYNATIVE_MODE, "必须使用动态图模式"
    dynamic_train(model_name="facebook/blenderbot-400M-distill")
    # 如果需要尝试 3B 模型，取消注释以下行
    # dynamic_train(model_name="facebook/blenderbot-3B")
