import mindspore
import mindnlp
import mindtorch
from transformers import AutoModel, AutoTokenizer
# from mindspore._c_expression import _framework_profiler_step_start
# from mindspore._c_expression import _framework_profiler_step_end

model_name = 'lvyufeng/DeepSeek-OCR'

tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
model = AutoModel.from_pretrained(model_name, _attn_implementation='eager', dtype=mindspore.float16,
                                  trust_remote_code=True, use_safetensors=True, device_map='auto')
model = model.eval()


# prompt = "<image>\nFree OCR. "
prompt = "<image>\n<|grounding|>Convert the document to markdown. "
# wget "https://hf-mirror.com/datasets/hf-internal-testing/fixtures_got_ocr/resolve/main/image_ocr.jpg"
image_file = 'image_ocr.jpg'
output_path = './'

# infer(self, tokenizer, prompt='', image_file='', output_path = ' ', base_size = 1024, image_size = 640, crop_mode = True, test_compress = False, save_results = False):

# Tiny: base_size = 512, image_size = 512, crop_mode = False
# Small: base_size = 640, image_size = 640, crop_mode = False
# Base: base_size = 1024, image_size = 1024, crop_mode = False
# Large: base_size = 1280, image_size = 1280, crop_mode = False

# Gundam: base_size = 1024, image_size = 640, crop_mode = True
# _framework_profiler_step_start()
with mindtorch.no_grad():
    res = model.infer(tokenizer, prompt=prompt, image_file=image_file, output_path = output_path, base_size = 1024, image_size = 640, crop_mode=True, save_results = True, test_compress = True)
# _framework_profiler_step_end()
