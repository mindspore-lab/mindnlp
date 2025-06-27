import mindspore
from mindnlp.transformers import AutoModelForCausalLM, AutoTokenizer
# model_name = "Qwen/Qwen3-0.6B"
model_name = "/mnt/data/zqh/llm/Qwen3-0.6B"


# load the tokenizer and the model
tokenizer = AutoTokenizer.from_pretrained(model_name,
                                          use_fast=False)
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    ms_dtype=mindspore.bfloat16,
)

# prepare the model input
prompt = "Give me a short introduction to large language models."
messages = [
    {"role": "user", "content": prompt}
]
text = tokenizer.apply_chat_template(
    messages,
    tokenize=False,
    add_generation_prompt=True,
    enable_thinking=True # Switches between thinking and non-thinking modes. Default is True.
)
model_inputs = tokenizer([text], return_tensors="ms")

# conduct text completion
generated_ids = model.generate(
    **model_inputs,
    max_new_tokens=10
)
output_ids = generated_ids[0][len(model_inputs.input_ids[0]):].tolist() 

# the result will begin with thinking content in <think></think> tags, followed by the actual response
print(tokenizer.decode(output_ids, skip_special_tokens=True))