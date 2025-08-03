# FineTune BLIP
- reference [repo](https://github.com/eeshashetty/captionary-api)

## Requirements
- python 3.9
- mindspore 2.3.1
- mindnlp 0.4.1

## args for training the model
- args.device_target : Ascend
- args.device_id 
- args.model_name_or_path : 'Salesforce/blip-image-captioning-base' or the path to the model
- args.dataset_name_or_path : 'eeshclusive/captionary-dataset' or the path to the data directory
- args.batch_size : batch size
- args.max_eps : maximum number of epochs
- args.save_path : path to save the model, if not provided the model will not be saved, such as './outputs/'

## Results
### my results on mindspore
20 epochs:  
- train loss: 0.0132  
- val loss: 0.0126

requirements:
- Ascend 910B
- Python 3.9
- MindSpore 2.3.1
- MindNLP 0.4.1

### my results on pytorch
10 epochs:  
- train loss: 0.0135  
- val loss: 0.0125

requirements:
- GPU 1080ti
- CUDA 11.1.1
- Python 3.9
- Pytorch 1.10.2
- Transformers 4.45.2

### Original results from the repo
20 epochs:  
- train loss: 1.3579
- val loss: 1.3584

### 其他
- 训练loss可视化见results_visible.ipynb文件   
- 愿仓库的损失不知为何特别高，复现时训练参数保持一致，但pytorch开启了混合精度，而mindnlp暂不支持，所以pytorch训练收敛的更快一些