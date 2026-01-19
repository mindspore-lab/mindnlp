import mindspore
from mindnlp.core import no_grad
from mindnlp.transformers import AutoModelForCausalLM, AutoTokenizer
from mindspore._c_expression import _framework_profiler_step_start
from mindspore._c_expression import _framework_profiler_step_end

tokenizer = AutoTokenizer.from_pretrained("ZhipuAI/glm-4-9b-chat", mirror='modelscope')

query = "你好"

inputs = tokenizer.apply_chat_template([{"role": "user", "content": query}],
                                       add_generation_prompt=True,
                                       tokenize=True,
                                       return_tensors="ms",
                                       return_dict=True
                                       )
print(inputs)
model = AutoModelForCausalLM.from_pretrained(
    "ZhipuAI/glm-4-9b-chat",
    mirror='modelscope',
    ms_dtype=mindspore.float16,
).eval()

# _framework_profiler_step_start()
gen_kwargs = {"max_length": 100, "do_sample": True, "top_k": 1}
with no_grad():
    outputs = model.generate(**inputs, **gen_kwargs)
    outputs = outputs[:, inputs['input_ids'].shape[1]:]
    print(tokenizer.decode(outputs[0], skip_special_tokens=True))
# _framework_profiler_step_end()
