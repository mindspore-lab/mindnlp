# FineTune BERT with Stanford Sentiment Tree Bank
- reference [repo](https://github.com/kabirahuja2431/FineTuneBERT)

## Requirements
- python 3.9
- mindspore 2.3.1
- mindnlp 0.4.1
- pandas

## Data
Download the data from this [link](https://gluebenchmark.com/tasks). There will be a main zip file download option at the right side of the page. Extract the contents of the zip file and place them in data/SST-2/

## Args for training the model
To train the model with fixed weights of BERT layers, set:
```
args.freeze_bert = True 
```
To train the entire model i.e. both BERT layers and the classification layer, set:
```
args.freeze_bert = False 
```

other optional arguments:
- args.device_target : Ascend
- args.device_id 
- args.base_model_name_or_path : 'bert-base-uncased' or the path to the model
- args.dataset_name_or_path : path to the data directory
- args.maxlen : maximum length of the input sequence
- args.batch_size : batch size
- args.lr : learning rate
- args.print_every : print the loss and accuracy after these many iterations
- args.max_eps : maximum number of epochs
- args.save_path : path to save the model, if not provided the model will not be saved, such as './outputs/'

## Results
### my results on mindspore
|Model Variant|Accuracy on Dev Set|
|-------------|-------------------|
|BERT (no finetuning)|81.25%|
|BERT (with finetuning)|90.07%|

requirements:
- Ascend 910B
- Python 3.9
- MindSpore 2.3.1
- MindNLP 0.4.1

### my results on pytorch
|Model Variant|Accuracy on Dev Set|
|-------------|-------------------|
|BERT (no finetuning)|81.03%|
|BERT (with finetuning)|89.84%|

requirements:
- GPU 1080ti
- CUDA 11.1.1
- Python 3.9
- Pytorch 1.10.2
- Transformers 4.45.2

### Original results from the repo
|Model Variant|Accuracy on Dev Set|
|-------------|-------------------|
|BERT (no finetuning)|82.59%|
|BERT (with finetuning)|88.29%|

requirements:
- Python 3.6
- Pytorch 1.2.0
- Transformers 2.0.0
