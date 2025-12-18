from mindspore import ops
from mindnlp.transformers import AutoModel, AutoTokenizer, AutoModelForSequenceClassification

hf_token = 'your_hf_token'

# init model and tokenizer
tokenizer = AutoTokenizer.from_pretrained('maidalun1020/bce-reranker-base_v1', token=hf_token)
model = AutoModelForSequenceClassification.from_pretrained('maidalun1020/bce-reranker-base_v1', token=hf_token)

# your query and corresponding passages
query = "上海天气"
passages = ["北京美食", "上海气候"]

# construct sentence pairs
sentence_pairs = [[query, passage] for passage in passages]

# get inputs
inputs = tokenizer(sentence_pairs, padding=True, truncation=True, max_length=512, return_tensors="ms")
inputs_on_device = {k: v for k, v in inputs.items()}

# calculate scores
scores = model(**inputs_on_device, return_dict=True).logits.view(-1,).float()
scores = ops.sigmoid(scores)
print(scores)
