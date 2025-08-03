from mindspore import ops
from mindnlp.transformers import AutoModel, AutoTokenizer, AutoModelForSequenceClassification

hf_token = 'your_hf_token'

# list of sentences
sentences = ['sentence_0', 'sentence_1']

# init model and tokenizer
tokenizer = AutoTokenizer.from_pretrained('maidalun1020/bce-embedding-base_v1', token=hf_token)
model = AutoModel.from_pretrained('maidalun1020/bce-embedding-base_v1', token=hf_token)

# get inputs
inputs = tokenizer(sentences, padding=True, truncation=True, max_length=512, return_tensors="ms")
inputs = {k: v for k, v in inputs.items()}

# get embeddings
outputs = model(**inputs, return_dict=True)
embeddings = outputs.last_hidden_state[:, 0]  # cls pooler
embeddings = embeddings / embeddings.norm(dim=1, keepdim=True)  # normalize
print(embeddings)