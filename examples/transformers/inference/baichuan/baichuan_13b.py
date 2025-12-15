import mindspore as ms
from mindnlp.transformers import BaiChuanForCausalLM, BaiChuanTokenizer
from mindnlp.transformers.generation.utils import GenerationConfig

tokenizer = BaiChuanTokenizer.from_pretrained("baichuan-inc/Baichuan-13B-Chat")
model = BaiChuanForCausalLM.from_pretrained("baichuan-inc/Baichuan-13B-Chat",
ms_dtype=ms.float16, size='13b')

texts = '请问你是谁？'
input_ids = tokenizer(texts, return_tensors="ms")
print(input_ids)
print(f'input_ids["input_ids"].shape:{input_ids["input_ids"].shape}')
outputs = model(input_ids=input_ids['input_ids'])
print(outputs)
print(outputs[0].shape)

