# MindNLP T5 finetune
This is a demostration of finetune **T5** using dataset from hugging face.

You could use the following command to run the script:
```
python .../t5_finetune.py --custom_dataset_path <path of your custom dataset> --batch_size <interger of the batch size>
```

The model is mostly meant to be fine-tuned on a supervised dataset. By default, the script use dataset according to https://github.com/jsrozner/t5_finetune and use batch size 8.

It will download the pretrained model named "t5-base" from huggingface and start the finetuning.