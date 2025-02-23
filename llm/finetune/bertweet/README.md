# FineTune BERTweet with hate_speech_twitter

## Data
huggingface dataset [thefrankhsu/hate_speech_twitter](https://huggingface.co/datasets/thefrankhsu/hate_speech_twitter)

## Results
### my results on mindspore
|Epoch|Training Loss|Validation Loss|Accuracy|Precision|Recall|F1|
|-----|-------------|---------------|--------|---------|------|--|
|1|0.305200|0.896717|0.670000|0.942708|0.362000|0.523121|
|2|0.143000|0.876202|0.738000|0.940741|0.508000|0.659740|
|3|0.096300|0.689730|0.790000|0.947531|0.614000|0.745146|
|<span style="color:red">4</span>|<span style="color:red">0.063500</span>|<span style="color:red">0.754796</span>|<span style="color:red">0.801000</span>|<span style="color:red">0.943953</span>|<span style="color:red">0.640000</span>|<span style="color:red">0.762813</span>|
|5|0.052800|0.935889|0.770000|0.944079|0.574000|0.713930|

requirements:
- Ascend 910B
- Python 3.9
- MindSpore 2.3.1
- MindNLP 0.4.1
- datasets emoji scikit-learn

### my results on pytorch
|Epoch|Training Loss|Validation Loss|Accuracy|Precision|Recall|F1|
|-----|-------------|---------------|--------|---------|------|--|
|1|0.228100|0.682149|0.750000|0.956204|0.524000|0.677003|
|<span style="color:red">2</span>|<span style="color:red">0.134900</span>|<span style="color:red">0.585958</span>|<span style="color:red">0.804000</span>|<span style="color:red">0.947059</span>|<span style="color:red">0.644000</span>|<span style="color:red">0.766667</span>|
|3|0.088700|0.848252|0.763000|0.942761|0.560000|0.702635|
|4|0.058300|0.956421|0.763000|0.945763|0.558000|0.701887|
|5|0.036500|0.894330|0.788000|0.950000|0.608000|0.741463|

requirements:
- GPU V100
- CUDA 11.8.0
- Python 3.10
- Pytorch 2.1.0
- Transformers 4.45.2
- datasets emoji accelerate scikit-learn
