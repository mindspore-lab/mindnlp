from mindnlp.transformers import pipeline

unmasker = pipeline('fill-mask', model='distilbert-base-uncased')
output=unmasker("Hello I'm a [MASK] model.")
print(output)
'''
[{'sequence': "[CLS] hello i'm a role model. [SEP]",
  'score': 0.05292865261435509,
  'token': 2535,
  'token_str': 'role'},
 {'sequence': "[CLS] hello i'm a fashion model. [SEP]",
  'score': 0.0396859310567379,
  'token': 4827,
  'token_str': 'fashion'},
 {'sequence': "[CLS] hello i'm a business model. [SEP]",
  'score': 0.034743666648864746,
  'token': 2449,
  'token_str': 'business'},
 {'sequence': "[CLS] hello i'm a model model. [SEP]",
  'score': 0.034622687846422195,
  'token': 2944,
  'token_str': 'model'},
 {'sequence': "[CLS] hello i'm a modeling model. [SEP]",
  'score': 0.018145263195037842,
  'token': 11643,
  'token_str': 'modeling'}]
 '''
