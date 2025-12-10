import mindspore
import mindhf
from transformers import AutoModel, AutoProcessor, AutoTokenizer
from transformers.image_utils import load_image


model = AutoModel.from_pretrained("lvyufeng/PaddleOCR-VL-0.9B", dtype=mindspore.float16, trust_remote_code=True, device_map='auto')
tokenizer = AutoTokenizer.from_pretrained("lvyufeng/PaddleOCR-VL-0.9B")
processor = AutoProcessor.from_pretrained("lvyufeng/PaddleOCR-VL-0.9B", trust_remote_code=True)

image = load_image(
    "https://hf-mirror.com/datasets/hf-internal-testing/fixtures_got_ocr/resolve/main/image_ocr.jpg"
)

query = 'OCR:'
messages = [
    {
        "role": "user",
        "content": query,
    }
]

text = tokenizer.apply_chat_template(messages, tokenize=False)
inputs = processor(image, text=text, return_tensors="pt", format=True).to('cuda')
generate_ids = model.generate(**inputs, do_sample=False, num_beams=1, max_new_tokens=1024)
print(generate_ids.shape)
decoded_output = processor.decode(
    generate_ids[0], skip_special_tokens=True
)
print(decoded_output)

