# Whisper-graph
[Introduction](#Introduction) |
[Quick Start](#quick-start) |
### News 📢
* 🔥 **Latest Features**

  * 🤗 load huggingface whisper model
  * 📝 support audio recognize

### Introduction
* Whisper is a pre-trained model for automatic speech recognition (ASR) and speech translation.
* Only recognition is supported. Others, such as timestamp, translation, and language detection are not supported. 



### quick-start
* init features extract
```angular2html
from mindnlp.transformers import WhisperProcessor
process = WhisperProcessor.from_pretrained("whisper-medium")
```
* init model
```angular2html
from mindnlp.transformers.models.whisper.modeling_graph_whisper import WhisperGraphConfig, WhisperGraphModel
model = WhisperGraphModel.load_graph_model(r"/data/model/whisper-medium", language="en",  beam_width=5, max_decode_length=30)
model = Model(model)

```
* predict
```angular2html
wav, samples = torchaudio.load(r"xxx.wav")
features = process.feature_extractor(wav.numpy()[0], sampling_rate=16000, return_tensors="ms").input_features
input_mask = Tensor(np.ones((features.shape[0], features.shape[1])))
generate_ids = model.predict(features, input_mask)
```
* decode
```angular2html
print(process.decode(generate_ids[0][0]))
```


