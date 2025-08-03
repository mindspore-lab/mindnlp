import mindspore
from mindspore.communication import init
from mindnlp.transformers import Qwen2MoeForCausalLM, AutoTokenizer

model_path = '/home/lvyufeng/lvyufeng/mindnlp/.mindnlp/model/Qwen/Qwen1.5-MoE-A2.7B'

init()

model = Qwen2MoeForCausalLM.from_pretrained(model_path, mirror='modelscope',
        device_map="auto", ms_dtype=mindspore.float16, use_safetensors=True)

EXPECTED_TEXT_COMPLETION = (
    "To be or not to be, that is the question.\nThe answer is to be, of course. But what does it"
)
prompt = "To be or not to"
tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen1.5-MoE-A2.7B", use_fast=False, mirror='modelscope')
input_ids = tokenizer.encode(prompt, return_tensors="ms")

# greedy generation outputs
generated_ids = model.generate(input_ids, max_new_tokens=512)

text = tokenizer.decode(generated_ids[0], skip_special_tokens=True)

print(text)