from mindnlp.transformers import pipeline

generator = pipeline("text-generation", model="t5-small")
print(generator("This is a second test", do_sample=False))
