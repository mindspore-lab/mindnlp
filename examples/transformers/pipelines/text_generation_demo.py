# pylint: disable=missing-module-docstring
from mindnlp.transformers import pipeline

# Text Generation Demo
generator = pipeline(model="openai-community/gpt2")
outputs = generator("I can't believe you did such a ", do_sample=False)
print(outputs)

# [{'generated_text': "I can't believe you did such a icky thing to me. I'm so sorry. I'm so sorry. I'm so sorry. I'm so sorry. I'm so sorry. I'm so sorry. I'm so sorry. I"}]

# Chat Demo
chat1 = [
            {"role": "system", "content": "This is a system message."},
            {"role": "user", "content": "This is a test"},
            {"role": "assistant", "content": "This is a reply"},
        ]
chat2 = [
            {"role": "system", "content": "This is a system message."},
            {"role": "user", "content": "This is a second test"},
            {"role": "assistant", "content": "This is a reply"},
        ]
outputs = generator(chat1, do_sample=False, max_new_tokens=10)
print(outputs)

outputs = generator([chat1, chat2], do_sample=False, max_new_tokens=10)
print(outputs)
