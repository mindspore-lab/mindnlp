# Albert mindnlp StanfordIMDB reviewer Finetune

- Albert模型微调任务链接：[【开源实习】albert模型微调 · Issue #IAUONP · MindSpore/community - Gitee.com](https://gitee.com/mindspore/community/issues/IAUONP)
- 实现了Albert-base-v1 基准权重 在 [Sentiment analysis of IMDb reviews - Stanford University] 数据集上的微调

- base model: [albert/albert-base-v1 · Hugging Face](https://huggingface.co/albert/albert-base-v1)
- dataset: [stanfordnlp/imdb · Datasets at Hugging Face](https://huggingface.co/datasets/stanfordnlp/imdb)

# Requirments
## Pytorch 

- GPU: RTX 4070ti 12G
- cuda: 11.8
- Python version: 3.10
- torch version: 2.5.0
- transformers version : 4.47.0

## Mindspore 启智社区 Ascend910B算力资源
- Ascend: 910B
- python: 3.11
- mindspore: 2.5.0
- mindnlp: 0.4.1

# Result for finetune

training for 3 epochs

## torch

| Epoch              | eval_loss |
| ------------------ | --------- |
| 1                  | 0.3868    |
| 2                  | 0.2978    |
| 3                  | 0.3293    |
| Evaluation results | 0.2978    |

**评估结果**

| Accuracy | Precision | Recall | F1_score |
| -------- | --------- | ------ | -------- |
| 0.9212   | 0.9218    | 0.9284 | 0.9218   |



## mindspore

| Epoch              | eval_loss |
| ------------------ | --------- |
| 1                  | 0.2677    |
| 2                  | 0.2314    |
| 3                  | 0.2332    |
| Evaluation results | 0.2314    |

**评估结果**

| Accuracy | Precision | Recall | F1_score |
| -------- | --------- | ------ | -------- |
| 0.9219   | 0.9238    | 0.9218 | 0.9228   |
