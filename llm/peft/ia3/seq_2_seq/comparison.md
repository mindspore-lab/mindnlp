# pytorch和mindnlp两个框架进行ia3微调的评估指标对比

### mindnlp

评估指标：

```python
accuracy=67.69911504424779 % on the evaluation dataset
eval_preds[:10]=['neutral', 'neutral', 'neutral', 'neutral', 'neutral', 'neutral', '', 'positive', 'positive', 'neutral']
ground_truth[:10]=['neutral', 'neutral', 'neutral', 'neutral', 'positive', 'neutral', 'neutral', 'positive', 'positive', 'positive']
```



### pytorch

评估指标:

```python
accuracy=67.84140969162996 % on the evaluation dataset
eval_preds[:10]=['neutral', 'neutral', 'neutral', '', 'neutral', 'positive', 'positive', 'neutral', 'neutral', 'neutral']
dataset['validation']['text_label'][:10]=['neutral', 'neutral', 'neutral', 'neutral', 'neutral', 'positive', 'positive', 'positive', 'neutral', 'neutral']
```

持平