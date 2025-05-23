import mindspore
import mindnlp
import mindspore as ms
from mindnlp.transformers import AutoTokenizer, AutoModelForCausalLM

model_id = "../../../../llm/deepseek-moe-16b-chat"
ms.set_context(pynative_synchronize=True)

tokenizer = AutoTokenizer.from_pretrained(model_id, ms_dtype=mindspore.bfloat16)
model = AutoModelForCausalLM.from_pretrained(
    model_id,
    ms_dtype=mindspore.bfloat16,
)

messages = [
    {"role": "user", "content": "Introduce Beijing to me."},
]
input_ids = tokenizer.apply_chat_template(
    messages,
    add_generation_prompt=True,
    return_tensors="ms"
)
outputs = model.generate(
    input_ids,
    max_new_tokens=100,
    do_sample=False)
result = tokenizer.decode(outputs[0][input_ids.shape[1]:], skip_special_tokens=True)
print(result)



