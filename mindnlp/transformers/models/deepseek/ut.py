import mindspore
from mindnlp.transformers import AutoTokenizer, AutoModelForCausalLM, GenerationConfig

# model_name = "deepseek-ai/deepseek-math-7b-base"
# tokenizer = AutoTokenizer.from_pretrained(model_name)
# model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=mindspore.bfloat16, device_map="auto")
# model.generation_config = GenerationConfig.from_pretrained(model_name)
# model.generation_config.pad_token_id = model.generation_config.eos_token_id
#
# text = "The integral of x^2 from 0 to 2 is"
# inputs = tokenizer(text, return_tensors="pt")
# outputs = model.generate(**inputs.to(model.device), max_new_tokens=100)
#
# result = tokenizer.decode(outputs[0], skip_special_tokens=True)
# print(result)

tokenizer = AutoTokenizer.from_pretrained("deepseek-ai/DeepSeek-Coder-V2-Lite-Base", trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained("deepseek-ai/DeepSeek-Coder-V2-Lite-Base",
                                             trust_remote_code=True, torch_dtype=mindspore.bfloat16)
input_text = "#write a quick sort algorithm"
inputs = tokenizer(input_text, return_tensors="pt").to(model.device)
outputs = model.generate(**inputs, max_length=128)
print(tokenizer.decode(outputs[0], skip_special_tokens=True))
