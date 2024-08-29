import mindspore
from mindnlp.transformers import LlamaTokenizer, LlamaForCausalLM, StaticCache, logging
from mindnlp.utils.testing_utils import CaptureLogger
from mindnlp.core import ops, no_grad

prompts = [
    "Simply put, the theory of relativity states that ",
    "My favorite all time favorite condiment is ketchup.",
]

NUM_TOKENS_TO_GENERATE = 40

model_id = 'shakechen/llama-2-7b-hf'
tokenizer = LlamaTokenizer.from_pretrained(model_id, mirror='modelscope', pad_token="</s>", padding_side="right")
model = LlamaForCausalLM.from_pretrained(model_id, mirror='modelscope')
inputs = tokenizer(prompts, return_tensors="ms", padding=True)

def decode_one_tokens(model, cur_token, input_pos, cache_position, past_key_values):
    logits = model(
        cur_token,
        position_ids=input_pos,
        cache_position=cache_position,
        past_key_values=past_key_values,
        return_dict=False,
        use_cache=True
    )[0]
    new_token = ops.argmax(logits[:, -1], dim=-1)[:, None]
    return new_token

batch_size, seq_length = inputs["input_ids"].shape
with no_grad():
    past_key_values = StaticCache(
        config=model.config, max_batch_size=2, max_cache_len=1024, dtype=model.dtype
    )
    cache_position = ops.arange(seq_length)
    generated_ids = ops.zeros(
        batch_size, seq_length + NUM_TOKENS_TO_GENERATE + 1, dtype=mindspore.int32
    )
    generated_ids[:, cache_position] = inputs["input_ids"].to(mindspore.int32)

    logits = model(
        **inputs, cache_position=cache_position, past_key_values=past_key_values,return_dict=False, use_cache=True
    )[0]
    next_token = ops.argmax(logits[:, -1], dim=-1)[:, None]
    generated_ids[:, seq_length] = next_token[:, 0]

    cache_position = mindspore.tensor([seq_length + 1])
    for _ in range(1, NUM_TOKENS_TO_GENERATE):
        next_token = decode_one_tokens(model, next_token, None, cache_position, past_key_values)
        generated_ids[:, cache_position] = next_token.int()
        cache_position += 1

text = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)
print(text)
