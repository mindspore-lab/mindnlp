# FineTune BERT with Stanford Sentiment Tree Bank
- reference [repo](https://github.com/kabirahuja2431/FineTuneBERT)

## Requirements
- python 3.9
- mindspore 2.3.1
- mindnlp 0.4.0
- pandas

## Data
Download the data from this [link](https://gluebenchmark.com/tasks). There will be a main zip file download option at the right side of the page. Extract the contents of the zip file and place them in data/SST/

## Training the model
To train the model with fixed weights of BERT layers, execute the following command from the project directory
```
python -m src.main -freeze_bert 
```
To train the entire model i.e. both BERT layers and the classification layer just skip the -freeze_bert flag
```
python -m src.main 
```

other optional arguments:
- -device_target : Ascend
- -device_id 
- -base_model_name_or_path : 'bert-base-uncased' or the path to the model
- -maxlen : maximum length of the input sequence
- -batch_size : batch size
- -lr : learning rate
- -print_every : print the loss and accuracy after these many iterations
- -max_eps : maximum number of epochs
- -save_path : path to save the model, if not provided the model will not be saved, such as './outputs/'

## Results
### my results on mindspore
|Model Variant|Accuracy on Dev Set|
|-------------|-------------------|
|BERT (no finetuning)|81.92%|
|BERT (with finetuning)|90.29%|

requirements:
- Ascend 910A
- Python 3.9
- MindSpore 2.3.1
- MindNLP 0.4.0

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

## Training logs
```python
python -m src.main -freeze_bert
```
```bash
Let the training begin
Iteration 0 of epoch 0 complete. Loss : 0.6787271499633789 Accuracy : 0.5
Iteration 500 of epoch 0 complete. Loss : 0.6680589914321899 Accuracy : 0.625
Iteration 1000 of epoch 0 complete. Loss : 0.6164900064468384 Accuracy : 0.78125
Iteration 1500 of epoch 0 complete. Loss : 0.5883786678314209 Accuracy : 0.6875
Iteration 2000 of epoch 0 complete. Loss : 0.6270998120307922 Accuracy : 0.65625
Epoch 0 complete! Validation Accuracy : 0.7332589, Validation Loss : 0.5965116683925901
Best validation accuracy improved from 0 to 0.7332589, saving model...
Iteration 0 of epoch 1 complete. Loss : 0.6488267183303833 Accuracy : 0.65625
Iteration 500 of epoch 1 complete. Loss : 0.6034247875213623 Accuracy : 0.625
Iteration 1000 of epoch 1 complete. Loss : 0.5811501145362854 Accuracy : 0.71875
Iteration 1500 of epoch 1 complete. Loss : 0.5714602470397949 Accuracy : 0.875
Iteration 2000 of epoch 1 complete. Loss : 0.5665632486343384 Accuracy : 0.75
Epoch 1 complete! Validation Accuracy : 0.79352677, Validation Loss : 0.5417175314256123
Best validation accuracy improved from 0.7332589 to 0.79352677, saving model...
Iteration 0 of epoch 2 complete. Loss : 0.595301628112793 Accuracy : 0.78125
Iteration 500 of epoch 2 complete. Loss : 0.5784701108932495 Accuracy : 0.71875
Iteration 1000 of epoch 2 complete. Loss : 0.48590192198753357 Accuracy : 0.875
Iteration 1500 of epoch 2 complete. Loss : 0.5315309762954712 Accuracy : 0.875
Iteration 2000 of epoch 2 complete. Loss : 0.47172310948371887 Accuracy : 0.90625
Epoch 2 complete! Validation Accuracy : 0.80133927, Validation Loss : 0.5020904274923461
Best validation accuracy improved from 0.79352677 to 0.80133927, saving model...
Iteration 0 of epoch 3 complete. Loss : 0.4329387843608856 Accuracy : 0.9375
Iteration 500 of epoch 3 complete. Loss : 0.5076307058334351 Accuracy : 0.78125
Iteration 1000 of epoch 3 complete. Loss : 0.5222103595733643 Accuracy : 0.84375
Iteration 1500 of epoch 3 complete. Loss : 0.43367570638656616 Accuracy : 0.9375
Iteration 2000 of epoch 3 complete. Loss : 0.5626306533813477 Accuracy : 0.75
Epoch 3 complete! Validation Accuracy : 0.80915177, Validation Loss : 0.47148192461047855
Best validation accuracy improved from 0.80133927 to 0.80915177, saving model...
Iteration 0 of epoch 4 complete. Loss : 0.5187490582466125 Accuracy : 0.84375
Iteration 500 of epoch 4 complete. Loss : 0.543378472328186 Accuracy : 0.71875
Iteration 1000 of epoch 4 complete. Loss : 0.44388043880462646 Accuracy : 0.78125
Iteration 1500 of epoch 4 complete. Loss : 0.4907197952270508 Accuracy : 0.78125
Iteration 2000 of epoch 4 complete. Loss : 0.44640499353408813 Accuracy : 0.875
Epoch 4 complete! Validation Accuracy : 0.8191964, Validation Loss : 0.4518013021775654
Best validation accuracy improved from 0.80915177 to 0.8191964, saving model...
Done in 1210.5237905979156 seconds
```

```python
python -m src.main
```
```bash
Let the training begin
Iteration 0 of epoch 0 complete. Loss : 0.747779130935669 Accuracy : 0.4375
Iteration 500 of epoch 0 complete. Loss : 0.21088945865631104 Accuracy : 0.9375
Iteration 1000 of epoch 0 complete. Loss : 0.13811033964157104 Accuracy : 0.9375
Iteration 1500 of epoch 0 complete. Loss : 0.22651341557502747 Accuracy : 0.90625
Iteration 2000 of epoch 0 complete. Loss : 0.2048414945602417 Accuracy : 0.9375
Epoch 0 complete! Validation Accuracy : 0.90290177, Validation Loss : 0.27701252991599695
Best validation accuracy improved from 0 to 0.90290177, saving model...
Iteration 0 of epoch 1 complete. Loss : 0.1677430123090744 Accuracy : 0.96875
Iteration 500 of epoch 1 complete. Loss : 0.10222629457712173 Accuracy : 0.96875
Iteration 1000 of epoch 1 complete. Loss : 0.41596782207489014 Accuracy : 0.875
Iteration 1500 of epoch 1 complete. Loss : 0.06704452633857727 Accuracy : 1.0
Iteration 2000 of epoch 1 complete. Loss : 0.09846101701259613 Accuracy : 0.9375
Epoch 1 complete! Validation Accuracy : 0.88616073, Validation Loss : 0.35767799482813906
Iteration 0 of epoch 2 complete. Loss : 0.014050442725419998 Accuracy : 1.0
Iteration 500 of epoch 2 complete. Loss : 0.014705491252243519 Accuracy : 1.0
Iteration 1000 of epoch 2 complete. Loss : 0.01888570562005043 Accuracy : 1.0
Iteration 1500 of epoch 2 complete. Loss : 0.02299358695745468 Accuracy : 1.0
Iteration 2000 of epoch 2 complete. Loss : 0.29878583550453186 Accuracy : 0.875
Epoch 2 complete! Validation Accuracy : 0.89620537, Validation Loss : 0.37192041972385986
Iteration 0 of epoch 3 complete. Loss : 0.027841169387102127 Accuracy : 1.0
Iteration 500 of epoch 3 complete. Loss : 0.04032248258590698 Accuracy : 0.96875
Iteration 1000 of epoch 3 complete. Loss : 0.059742070734500885 Accuracy : 0.96875
Iteration 1500 of epoch 3 complete. Loss : 0.008875822648406029 Accuracy : 1.0
Iteration 2000 of epoch 3 complete. Loss : 0.09226992726325989 Accuracy : 0.9375
Epoch 3 complete! Validation Accuracy : 0.8917411, Validation Loss : 0.38447874930820297
Iteration 0 of epoch 4 complete. Loss : 0.011013567447662354 Accuracy : 1.0
Iteration 500 of epoch 4 complete. Loss : 0.024968033656477928 Accuracy : 1.0
Iteration 1000 of epoch 4 complete. Loss : 0.0883134976029396 Accuracy : 0.96875
Iteration 1500 of epoch 4 complete. Loss : 0.01590060070157051 Accuracy : 1.0
Iteration 2000 of epoch 4 complete. Loss : 0.03373430669307709 Accuracy : 1.0
Epoch 4 complete! Validation Accuracy : 0.89285713, Validation Loss : 0.42434852570295334
Done in 3309.0767965316772 seconds
```