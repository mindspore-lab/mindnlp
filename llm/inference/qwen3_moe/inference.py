import mindspore as ms
from mindspore.communication import init
from mindnlp.transformers import AutoModelForCausalLM, AutoTokenizer, Qwen2TokenizerFast
init()
# ms.set_context(
#     mode=ms.PYNATIVE_MODE,
#     pynative_synchronize=True) # synchronize for debug convenience
model_name = "Qwen/Qwen3-30B-A3B"


tokenizer = Qwen2TokenizerFast.from_pretrained("Qwen/Qwen2.5-0.5B-Instruct")
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    ms_dtype=ms.bfloat16, # on Atlas A2
    device_map='auto'
)

# prepare the model input
prompt = "Give me a short introduction to large language model."
messages = [
    {"role": "user", "content": prompt}
]
text = tokenizer.apply_chat_template(
    messages,
    tokenize=False,
    add_generation_prompt=True,
    enable_thinking=True, # Switches between thinking and non-thinking modes. Default is True.
    do_sample=False
)
model_inputs = tokenizer([text], return_tensors="ms")
# conduct text completion
generated_ids = model.generate(
    **model_inputs,
    max_new_tokens=100
)
output_ids = generated_ids[0][len(model_inputs.input_ids[0]):].tolist() 

# parsing thinking content
try:
    # rindex finding 151668 (</think>)
    index = len(output_ids) - output_ids[::-1].index(151668)
except ValueError:
    index = 0

thinking_content = tokenizer.decode(output_ids[:index], skip_special_tokens=True).strip("\n")
content = tokenizer.decode(output_ids[index:], skip_special_tokens=True).strip("\n")

print("thinking content:", thinking_content)
print("content:", content)
# Usage: msrun --worker_num=2 --local_worker_num=2  --master_port 9989 --log_dir=msrun_log --join=True  inference.py