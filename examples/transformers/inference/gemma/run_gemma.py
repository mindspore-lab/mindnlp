from mindnlp.transformers import AutoTokenizer, AutoModelForCausalLM

hf_token = 'your_huggingface_access_token'

tokenizer = AutoTokenizer.from_pretrained("google/gemma-7b", token=hf_token)
model = AutoModelForCausalLM.from_pretrained("google/gemma-7b", token=hf_token)

input_text = "Write me a poem about Machine Learning."
input_ids = tokenizer(input_text, return_tensors="ms")

outputs = model.generate(**input_ids)
print(tokenizer.decode(outputs[0]))