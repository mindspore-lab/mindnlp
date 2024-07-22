import mindspore
from mindnlp.transformers import AutoTokenizer, AutoModelForCausalLM
from mindspore._c_expression import _framework_profiler_step_start
from mindspore._c_expression import _framework_profiler_step_end
from mindspore._c_expression import update_pijit_default_config


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
@mindspore.jit(mode='PIJit', jit_config=mindspore.JitConfig(jit_level='O1'))
def inference():
    outputs = model.generate(
        input_ids,
        max_new_tokens=20,
        eos_token_id=terminators,
        do_sample=False,
        # do_sample=True,
        # temperature=0.6,
        # top_p=0.9,
    )
    return outputs

def callback(f):
    if f.f_code.co_name == "forward":
        return True
    return False

update_pijit_default_config(
    # allowed_inline_modules=["mindnlp"],
    LOG_PERF=True,
    # LOG_GRAPH_BREAK=True,
    # auto_jit_cell=True,
    #loop_unrolling=True,
    # infer_only=True,
    auto_jit_func_filter=callback,
    enable_dynamic_shape=True,
    compile_by_trace=False
)


outputs = inference()
# _framework_profiler_step_end()
response = outputs[0][input_ids.shape[-1]:]
print(tokenizer.decode(response, skip_special_tokens=True))
