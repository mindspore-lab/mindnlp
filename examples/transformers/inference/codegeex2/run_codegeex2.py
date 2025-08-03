import mindspore
from mindnlp.core import no_grad
from mindnlp.transformers import AutoModelForCausalLM, AutoTokenizer
from mindspore._c_expression import _framework_profiler_step_start
from mindspore._c_expression import _framework_profiler_step_end
# load tokenizer
tokenizer = AutoTokenizer.from_pretrained("THUDM/codegeex2-6b", mirror='huggingface')

query = "Write me a bubble sort algorithm in Python"

inputs = tokenizer(query, return_tensors="ms")
print(inputs)
# load model
model = AutoModelForCausalLM.from_pretrained(
    "THUDM/codegeex2-6b",
    mirror='huggingface',
    ms_dtype=mindspore.float16,
).eval()

# _framework_profiler_step_start()
gen_kwargs = {"max_length": 100, "do_sample": True, "top_k": 1}
with no_grad():
    outputs = model.generate(**inputs, **gen_kwargs)
    outputs = outputs[:, inputs['input_ids'].shape[1]:]
    print(tokenizer.decode(outputs[0], skip_special_tokens=True))
# _framework_profiler_step_end()