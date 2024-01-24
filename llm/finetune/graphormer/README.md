# <center> MindNLP Graphormer finetune
This is a demostration of finetune Graphormer using dataset from hugging face.


You could use the following command to run the script:


``` bash
python graphormer_finetune.py --datast_name <name of the dataset> --batch_size <interger of the batch size>
```


By default, the script downloads ogb/ogbg-molhiv dataset from huggingface and use batch size 3.


It will download the pretrained model named graphormer-base-pcqm4mv2 from huggingface and start the finetuning.
