from PIL import Image
import requests
import mindspore
from mindnlp.transformers import AutoProcessor, LlavaForConditionalGeneration

model = LlavaForConditionalGeneration.from_pretrained("AI-Research/llava-1.5-7b-hf", mirror='modelers', ms_dtype=mindspore.float16)
processor = AutoProcessor.from_pretrained("AI-Research/llava-1.5-7b-hf", mirror='modelers')

prompt = "USER: <image>\nWhat's the content of the image? ASSISTANT:"
url = "https://www.ilankelman.org/stopsigns/australia.jpg"
image = Image.open(requests.get(url, stream=True).raw)

inputs = processor(images=image, text=prompt, return_tensors="ms")

# Generate
generate_ids = model.generate(**inputs, max_new_tokens=15)
out = processor.batch_decode(generate_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]
print(out)
