from mindnlp.transformers import pipeline

oracle = pipeline("zero-shot-classification",model="facebook/bart-large-mnli")
outputs = oracle("I have a problem with my iphone that needs to be resolved asap!!",candidate_labels=["urgent", "not urgent", "phone", "tablet", "computer"], )
print(outputs)
#{'sequence': 'I have a problem with my iphone that needs to be resolved asap!!', 'labels': ['urgent', 'phone', 'computer', 'not urgent', 'tablet'], 'scores': [0.504, 0.479, 0.013, 0.003, 0.002]}
outputs = oracle("I have a problem with my iphone that needs to be resolved asap!!",candidate_labels=["english", "german"],)
print(outputs)
#{'sequence': 'I have a problem with my iphone that needs to be resolved asap!!', 'labels': ['english', 'german'], 'scores': [0.814, 0.186]}
