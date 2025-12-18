import mindspore
from mindnlp.transformers import AutoTokenizer, AutoModelForCausalLM, StaticCache
from mindnlp.core import ops
from mindnlp.configs import set_pyboost, ON_ORANGE_PI
from mindnlp.quant.smooth_quant import quantize, w8x8
import time
import numpy as np
import os

#在香橙派上，开启O2级别的jit优化
if ON_ORANGE_PI:
    mindspore.set_context(
        enable_graph_kernel=True,
        mode=mindspore.GRAPH_MODE,
        jit_config={
            "jit_level": "O2",
        },
    )

def sample_top_p(probs, p=0.9):
    """
    Top-p采样函数，用于生成文本时选择下一个token。
    此处优先采用基于numpy而不是原生MindSpore的实现方式，因为在香橙派上运行效率更高
    """
    probs_np = probs.asnumpy()
    # 按概率降序排序
    sorted_indices = np.argsort(-probs_np, axis=-1)
    sorted_probs = np.take_along_axis(probs_np, sorted_indices, axis=-1)
    # 计算累积概率并创建掩码
    cumulative_probs = np.cumsum(sorted_probs, axis=-1)
    mask = cumulative_probs - sorted_probs > p
    sorted_probs[mask] = 0.0
    sorted_probs = sorted_probs / np.sum(sorted_probs, axis=-1, keepdims=True)
    # 转换回MindSpore Tensor
    sorted_probs_tensor = mindspore.Tensor(sorted_probs, dtype=mindspore.float32)
    sorted_indices_tensor = mindspore.Tensor(sorted_indices, dtype=mindspore.int32)
    next_token_idx = ops.multinomial(sorted_probs_tensor, 1)
    batch_size = probs.shape[0]
    batch_indices = ops.arange(0, batch_size, dtype=mindspore.int32).reshape(-1, 1)
    # 此处采用基于mindspore.ops的实现方式，在香橙派上兼容性最好
    # next_token = sorted_indices_tensor[batch_indices, next_token_idx]
    next_token = mindspore.ops.gather(sorted_indices_tensor, next_token_idx, axis=1, batch_dims=1)
    # next_token = mindspore.mint.gather(sorted_indices_tensor, dim=1, index=next_token_idx)
    return next_token


#该任务将使用DeepSeek-R1-Distill-Qwen-1.5B模型，对给定的prompt进行补齐
prompts = [
    "请介绍一下自己。<think>",
    "My favorite all time favorite condiment is ketchup.",
]

# 生成参数配置
NUM_TOKENS_TO_GENERATE = 40  # 每个输入要生成的token数量
TEMPERATURE = 0.8            # 温度参数（控制生成多样性）
TOP_P = 0.8                  # Top-p采样阈值

model_id = "MindSpore-Lab/DeepSeek-R1-Distill-Qwen-1.5B"
tokenizer = AutoTokenizer.from_pretrained(model_id, mirror="modelers")
model = AutoModelForCausalLM.from_pretrained(model_id, ms_dtype=mindspore.float16, low_cpu_mem_usage=True, mirror="modelers")

#使用model.jit()将全图静态图化
model.jit()

inputs = tokenizer(prompts, return_tensors="ms", padding=True)
set_pyboost(False)

#使用@mindspore.jit装饰器封装模型推理函数
@mindspore.jit(jit_config=mindspore.JitConfig(jit_syntax_level='STRICT'))
def get_decode_one_tokens_logits(model, cur_token, input_pos, cache_position, past_key_values, temperature=TEMPERATURE, top_p=TOP_P):
    """单个token的解码函数，返回logits，可以使用jit进行优化"""
    logits = model(
        cur_token,
        position_ids=input_pos,
        cache_position=cache_position,
        past_key_values=past_key_values,
        return_dict=False,
        use_cache=True
    )[0]
    return logits

def decode_one_tokens(model, cur_token, input_pos, cache_position, past_key_values, temperature=TEMPERATURE, top_p=TOP_P):
    """单个token的解码函数，由logits、温度和Top_p选择合适的token"""
    logits = get_decode_one_tokens_logits(model, cur_token, input_pos, cache_position, past_key_values, temperature, top_p)
    
    if temperature > 0:
        probs = mindspore.mint.softmax(logits[:, -1] / temperature, dim=-1)
        new_token = sample_top_p(probs, top_p)
    else:
        new_token = mindspore.mint.argmax(logits[:, -1], dim=-1)[:, None]
        
    return new_token

batch_size, seq_length = inputs["input_ids"].shape

# 创建静态缓存（用于加速自回归生成）
past_key_values = StaticCache(
    config=model.config, max_batch_size=2, max_cache_len=512, dtype=model.dtype
)
cache_position = ops.arange(seq_length)
generated_ids = ops.zeros(
    batch_size, seq_length + NUM_TOKENS_TO_GENERATE + 1, dtype=mindspore.int32
)
generated_ids[:, cache_position] = inputs["input_ids"].to(mindspore.int32)

# 初始前向传播获取首个logits
logits = model(
    **inputs, cache_position=cache_position, past_key_values=past_key_values,return_dict=False, use_cache=True
)[0]

# 生成第一个新token
if TEMPERATURE > 0:
    probs = mindspore.mint.softmax(logits[:, -1] / TEMPERATURE, dim=-1)
    next_token = sample_top_p(probs, TOP_P)
else:
    next_token = mindspore.mint.argmax(logits[:, -1], dim=-1)[:, None]

generated_ids[:, seq_length] = next_token[:, 0]

# 自回归生成循环
cache_position = mindspore.tensor([seq_length + 1])
for _ in range(1, NUM_TOKENS_TO_GENERATE):
    s = time.time()
    next_token = decode_one_tokens(model, next_token, None, cache_position, past_key_values)
    generated_ids[:, cache_position] = next_token.int()
    cache_position += 1
    t = time.time()
    print("[%d]:"%_, t - s) # 打印单步生成耗时

text = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)
print(text)