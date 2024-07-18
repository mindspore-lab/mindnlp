import mindspore
from mindnlp.transformers import pipeline, AutoTokenizer, AutoModelForCausalLM
# Initialize the model and tokenizer
model_name = "jetmoe/jetmoe-8b-chat"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name, ms_dtype=mindspore.float16)
# Encode input context
messages = [
    {
        "role": "system",
        "content": "You are a friendly chatbot",
    },
    {"role": "user", "content": "How many helicopters can a human eat in one sitting?"},
 ]
tokenized_chat = tokenizer.apply_chat_template(messages, tokenize=True, add_generation_prompt=True, return_tensors="ms")
print(tokenized_chat)
# Generate text
output = model.generate(tokenized_chat, max_length=500, num_return_sequences=1, no_repeat_ngram_size=2)
# Decode the generated text
generated_text = tokenizer.decode(output[0], skip_special_tokens=True)
print(generated_text)
