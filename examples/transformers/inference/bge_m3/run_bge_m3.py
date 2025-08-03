from mindspore import ops
from mindnlp.transformers import AutoModel, AutoTokenizer
from mindspore.ops._primitive_cache import _get_cache_prim

def norm(tensor):
    norm_ = _get_cache_prim(ops.L2Normalize)(axis=-1, epsilon=1e-12)
    return norm_(tensor)

# Trust remote code is required to load the model
tokenizer = AutoTokenizer.from_pretrained('liuyanyi/bge-m3-hf')
model = AutoModel.from_pretrained('liuyanyi/bge-m3-hf')

test_strings = ["I'm an example sentence.", "我是另一个测试句子。"]

input_ids = tokenizer(test_strings, return_tensors="ms", padding=True, truncation=True)

output = model(**input_ids, return_dict=True)

dense_output = output.dense_output
dense_output = norm(dense_output)
colbert_output = output.colbert_output
colbert_output = [norm(vec) for vec in colbert_output]
sparse_output = output.sparse_output
print(dense_output, colbert_output, sparse_output)

