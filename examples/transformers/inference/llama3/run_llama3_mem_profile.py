import os
import psutil
import gc
from memory_profiler import profile
import mindspore
from mindnlp.transformers import AutoTokenizer, AutoModelForCausalLM

model_id = "LLM-Research/Meta-Llama-3-8B-Instruct"

@profile
def test():
    tokenizer = AutoTokenizer.from_pretrained(model_id, mirror='modelscope')
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        ms_dtype=mindspore.float16,
        mirror='modelscope',
        low_cpu_mem_usage=True
    )

if __name__ == '__main__':

    a=test()

    print('A：%.2f MB' % (psutil.Process(os.getpid()).memory_info().rss / 1024 / 1024))
    del a
    gc.collect()
    print('B：%.2f MB' % (psutil.Process(os.getpid()).memory_info().rss / 1024 / 1024))
