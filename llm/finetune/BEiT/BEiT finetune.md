# Finetune microsoft beit-base-patch16-224 model
- base model: [microsoft beit-base-patch16-224](https://huggingface.co/microsoft/beit-base-patch16-224)
- dataset: [cifar10](https://huggingface.co/datasets/uoft-cs/cifar10)
- pytorch version finetune [github](https://github.com/4everImmortality/microsoft-beit-cifar10-finetune)
# requirments
## pytorch 
- GPU: RTX 4070ti 12G
- cuda: 11.8
- Python version: 3.10
- torch version: 2.5.0
- transformers version : 4.47.0
## mindspore
- Ascend: 910B
- python: 3.9
- mindspore: 2.3.1
- mindnlp: 0.4.0
# Result for finetune
training for 3 epochs
## torch
| Epoch | eval_loss | eval_accuracy |
|-------|-----------|--------------|
| 1     | 0.193     | 94.4%        |
| 2     | 0.157     | 95.4%        |
| 3     | 0.117     | 96.2%        |
## mindspore
| Epoch | eval_loss | eval_accuracy |
|-------|-----------|--------------|
| 1     | 0.416     | 96.4%        |
| 2     | 0.193     | 96.8%        |
| 3     | 0.158     | 97.2%        |