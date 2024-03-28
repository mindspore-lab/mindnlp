# 示例 1: 使用LayoutLMv2处理文档问答
from mindnlp.transformers import pipeline, AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained("hf-internal-testing/tiny-random-layoutlmv2")

dqa_pipeline = pipeline("document-question-answering", model="hf-internal-testing/tiny-random-layoutlmv2",
                        tokenizer=tokenizer)

image_url = "https://hf-mirror.com/spaces/impira/docquery/resolve/2f6c96314dc84dfda62d40de9da55f2f5165d403/invoice.png"
question = "How many cats are there?"

outputs = dqa_pipeline(image=image_url, question=question, top_k=2)

print(outputs)


# 示例 2: 使用LayoutLM模型和自定义图像处理
from PIL import Image
import pytesseract
from mindnlp.transformers import pipeline, AutoTokenizer


def process_image_and_ocr(image_path):
    image = Image.open(image_path)
    ocr_result = pytesseract.image_to_data(image, output_type=pytesseract.Output.DICT)
    words = ocr_result['text']
    boxes = [ocr_result['left'], ocr_result['top'], ocr_result['width'], ocr_result['height']]
    return words, boxes


tokenizer = AutoTokenizer.from_pretrained("tiennvcs/layoutlmv2-base-uncased-finetuned-docvqa", revision="9977165")

dqa_pipeline = pipeline("document-question-answering", model="tiennvcs/layoutlmv2-base-uncased-finetuned-docvqa",
                        tokenizer=tokenizer)

image_path = "./path/to/your/invoice/image.png"
words, boxes = process_image_and_ocr(image_path)
question = "What is the invoice number?"

outputs = dqa_pipeline(question=question, words=words, boxes=boxes, top_k=2)

print(outputs)
