# pylint: disable=missing-module-docstring
from mindnlp.transformers import pipeline

oracle = pipeline(model="deepset/roberta-base-squad2")
print(oracle(question="Where do I live?", context="My name is Wolfgang and I live in Berlin"))
#{'score': 0.9191, 'start': 34, 'end': 40, 'answer': 'Berlin'}
