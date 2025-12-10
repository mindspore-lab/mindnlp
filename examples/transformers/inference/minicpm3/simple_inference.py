import mindspore
from mindhf.transformers import MiniCPM3Tokenizer, MiniCPM3Config, MiniCPM3ForCausalLM
from mindhf.core import ops


model_id = "OpenBMB/MiniCPM3-4B"
tokenizer = MiniCPM3Tokenizer.from_pretrained(model_id, mirror="modelscope")
model = MiniCPM3ForCausalLM.from_pretrained(model_id, ms_dtype=mindspore.float16, mirror="modelscope")


messages = [
    {"role": "user", "content": "推荐5个北京的景点。"},
]
model_inputs = tokenizer.apply_chat_template(messages, return_tensors="ms", add_generation_prompt=True)

model_outputs = model.generate(
    model_inputs,
    max_new_tokens=1024,
    top_p=0.7,
    temperature=0.7
)

output_token_ids = [
    model_outputs[i][len(model_inputs[i]):] for i in range(len(model_inputs))
]

responses = tokenizer.batch_decode(output_token_ids, skip_special_tokens=True)[0]
print(responses)