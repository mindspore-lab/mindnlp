import requests

import mindspore
from PIL import Image
from mindnlp.transformers.models.florence2 import Florence2ForConditionalGeneration, Florence2Processor


model = Florence2ForConditionalGeneration.from_pretrained("microsoft/Florence-2-large", ms_dtype=mindspore.float32)
processor = Florence2Processor.from_pretrained("microsoft/Florence-2-large")


prompt = "<OD>"

url = "https://hf-mirror.com/datasets/huggingface/documentation-images/resolve/main/transformers/tasks/car.jpg?download=true"
image = Image.open(requests.get(url, stream=True).raw)

inputs = processor(text=prompt, images=image, return_tensors='ms')

generated_ids = model.generate(
    input_ids=inputs["input_ids"],
    pixel_values=inputs["pixel_values"],
    max_new_tokens=1024,
    num_beams=3,
    do_sample=False
)
generated_text = processor.batch_decode(generated_ids, skip_special_tokens=False)[0]

parsed_answer = processor.post_process_generation(generated_text, task="<OD>", image_size=(image.width, image.height))

print(parsed_answer)
