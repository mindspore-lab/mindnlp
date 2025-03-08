import os
import mindspore as ms
from mindspore import context, nn
from mindnlp.transformers import BlenderbotTokenizer, BlenderbotForConditionalGeneration
from mindnlp.engine import Trainer, TrainingArguments
from mindnlp.dataset import load_dataset
from mindspore.dataset import GeneratorDataset

# 环境配置 ==================
context.set_context(
    mode=context.PYNATIVE_MODE,  
    device_target="Ascend",
    device_id=0,  # 显式指定设备ID
    enable_graph_kernel=False,  # 动态图需禁用图算优化
    max_call_depth=3000,
    pynative_synchronize=True  # 确保异步操作同步执行
)

# 动态图专用内存优化
#ms.set_auto_parallel_context(parallel_mode=ms.ParallelMode.STAND_ALONE)
#ms.set_context(reserve_class_name_in_scope=False)

# 数据管道 ==================
def load_and_process_data():
    """动态图模式数据加载需更严格的内存管理"""
    dataset = load_dataset("databricks/databricks-dolly-15k")
    return dataset.filter(
        lambda x: x["instruction"] is not None and len(x["instruction"]) > 10,
        num_parallel_workers=1  # 动态图需减少并行度
    )

def preprocess_data(tokenizer, dataset):
    """动态图数据预处理需保持数据连续性"""
    def process(examples):
        inputs = [str(text) for text in examples["instruction"]]
        targets = [str(text) for text in examples["response"]]
        
        # 动态图需显式关闭并行编码
        model_inputs = tokenizer(
            inputs,
            max_length=128,
            truncation=True,
            padding="max_length",
            return_tensors="ms",
            num_parallel_workers=1
        )
        
        with tokenizer.as_target_tokenizer():
            labels = tokenizer(
                targets,
                max_length=128,
                truncation=True,
                padding="max_length",
                return_tensors="ms",
                num_parallel_workers=1
            )
        
        return {
            "input_ids": model_inputs["input_ids"],
            "attention_mask": model_inputs["attention_mask"],
            "labels": labels["input_ids"]
        }
    
    return dataset.map(
        process,
        batched=True,
        batch_size=8,  # 小批量处理避免内存峰值
        remove_columns=dataset["train"].column_names,
        num_parallel_workers=1
    )

# 动态图专用模型构建 ==============
class DynamicBlenderbot(nn.Cell):
    """封装模型以支持动态图执行"""
    def __init__(self):
        super().__init__()
        model_name = "facebook/blenderbot-400M-distill"
        self.tokenizer = BlenderbotTokenizer.from_pretrained(model_name)
        self.model = BlenderbotForConditionalGeneration.from_pretrained(model_name)
        
        # 动态图需显式设置参数
        self.model.set_train(True)
        self.model.requires_grad = True
        
        # 修复动态图bias初始化
        if not hasattr(self.model, 'final_logits_bias'):
            self.model.final_logits_bias = ms.Parameter(
                ms.ops.zeros((1, self.model.model.shared.vocab_size), ms.float32),
                name='final_logits_bias',
                requires_grad=True
            )
    
    def construct(self, input_ids, attention_mask, labels):
        return self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels
        ).loss

# 动态图数据加载器 ==============
def create_dynamic_dataset(tokenized_dataset):
    """动态图需要持续数据流"""
    def generator():
        for item in tokenized_dataset["train"]:
            yield (
                item["input_ids"].asnumpy(),  # 动态图需转换为numpy
                item["attention_mask"].asnumpy(),
                item["labels"].asnumpy()
            )
    
    return GeneratorDataset(
        source=generator,
        column_names=["input_ids", "attention_mask", "labels"],
        shuffle=False  # 动态图需禁用shuffle避免内存泄漏
    ).batch(1, drop_remainder=True).to_device()

# 动态图训练循环 ================
def dynamic_train():
    """自定义训练循环以优化动态图性能"""
    # 初始化组件
    dataset = load_and_process_data()
    processed_data = preprocess_data(DynamicBlenderbot().tokenizer, dataset)
    train_dataset = create_dynamic_dataset(processed_data)
    
    # 构建动态图模型
    net = DynamicBlenderbot()
    optimizer = nn.Adam(net.trainable_params(), learning_rate=2e-5)
    
    # 梯度累积配置
    grad_accum_steps = 4
    loss_scaler = ms.amp.DynamicLossScaler(1024, 2, 1000)
    
    # 训练循环
    step = 0
    for epoch in range(3):
        net.set_train(True)
        for batch in train_dataset:
            # 动态图前向传播
            loss = net(*batch)
            
            # 混合精度缩放
            scaled_loss = loss_scaler.scale(loss)
            scaled_loss.backward()
            
            # 梯度累积
            if (step + 1) % grad_accum_steps == 0:
                loss_scaler.unscale_(optimizer)
                ms.amp.clip_grad_value_(net.trainable_params(), 1.0)
                optimizer.step()
                optimizer.zero_grad()
            
            # 日志记录
            if step % 10 == 0:
                print(f"Step {step} Loss: {loss.asnumpy()}")
            
            # 检查点保存
            if step % 100 == 0:
                ms.save_checkpoint(net, f"./checkpoints/step_{step}.ckpt")
            
            step += 1

if __name__ == "__main__":
    # 动态图专用环境验证
    assert context.get_context("mode") == context.PYNATIVE_MODE, "必须使用动态图模式"
    dynamic_train()

