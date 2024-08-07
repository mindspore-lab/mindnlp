import mindspore
from mindnlp.transformers import AutoTokenizer, AutoModelForCausalLM
from mindspore._c_expression import _framework_profiler_step_start
from mindspore._c_expression import _framework_profiler_step_end

model_id = "LLM-Research/Meta-Llama-3-8B-Instruct"

tokenizer = AutoTokenizer.from_pretrained(model_id, mirror='modelscope')
model = AutoModelForCausalLM.from_pretrained(
    model_id,
    ms_dtype=mindspore.float16,
    mirror='modelscope'
)

messages = [
    {"role": "system", "content": "You are a pirate chatbot who always responds in pirate speak!"},
    {"role": "user", "content": "Who are you?"},
]

input_ids = tokenizer.apply_chat_template(
    messages,
    add_generation_prompt=True,
    return_tensors="ms"
)

terminators = [
    tokenizer.eos_token_id,
    tokenizer.convert_tokens_to_ids("<|eot_id|>")
]

# _framework_profiler_step_start()
outputs = model.generate(
    input_ids,
    max_new_tokens=20,
    eos_token_id=terminators,
    do_sample=False,
    # do_sample=True,
    # temperature=0.6,
    # top_p=0.9,
)
# _framework_profiler_step_end()
response = outputs[0][input_ids.shape[-1]:]
print(tokenizer.decode(response, skip_special_tokens=True))
