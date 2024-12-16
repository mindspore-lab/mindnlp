# GLUE-QNLI
A repository comparing the inference accuracy of MindNLP and Transformer on the GLUE QNLI dataset

+ ## Dataset
+ The QNLI (Question Natural Language Inference) dataset is part of the GLUE benchmark. It is converted from the Stanford Question Answering Dataset (SQuAD).
+ 
+ ### Getting the Dataset
+ 1. Visit [GLUE Benchmark Tasks](https://gluebenchmark.com/tasks/)
+ 2. Register/Login to download the GLUE data
+ 3. Download and extract the QNLI dataset
+ 4. Place the following files in the `mindnlp/benchmark/GLUE-QNLI/` directory:
+    - dev.tsv (Development set)
+    - test.tsv (Test set)
+    - train.tsv (Training set)
+ 
+ The QNLI task is a binary classification task derived from SQuAD, where the goal is to determine whether a given context sentence contains the answer to a given question.

## Quick Start

### Installation
To get started with this project, follow these steps:

1. **Create a conda environment (optional but recommended):**
    ```bash
    conda create -n mindnlp python==3.9
    conda activate mindnlp
2. **Install the dependencies:**
Please note that mindnlp is in the Ascend environment, while transformers is in the GPU environment, and the required dependencies are in the requirements of their respective folders.
    ```bash
    pip install -r requirements.txt
3. **Usage**
Once the installation is complete, you can choose use differnet models to start inference. Here's how to run the inference:
   ```bash
   # Evaluate specific model using default dataset (dev.tsv)
   python model_qnli.py --model albert

   # Evaluate with custom dataset
   python model_qnli.py --model bert --data ./QNLI/dev.tsv
   ```
   Supported model options: `albert`, `bert`, `roberta`, `xlm-roberta`, `distilbert`, `t5`, `gpt2`, `llama`, `opt`, `bart`

## Accuracy Comparsion
|  Model Name | bart | bert | roberta | xlm-roberta | gpt2 | t5 | distilbert | albert | opt | llama |
|---|---|---|---|---|---|---|---|---|---|---|
|  Base Model  | facebook/bart-base  |  google-bert/bert-base-uncased | FacebookAI/roberta-large | FacebookAI/xlm-roberta-large |  openai-community/gpt2 |  google-t5/t5-small |  distilbert/distilbert-base-uncased | albert/albert-base-v2  | facebook/opt-125m  | JackFram/llama-160m  |
|  Fine-tuned Model(hf)  | ModelTC/bart-base-qnli  | Li/bert-base-uncased-qnli  | howey/roberta-large-qnli | tmnam20/xlm-roberta-large-qnli-1 | tanganke/gpt2_qnli  | lightsout19/t5-small-qnli  | anirudh21/distilbert-base-uncased-finetuned-qnli  | orafandina/albert-base-v2-finetuned-qnli  | utahnlp/qnli_facebook_opt-125m_seed-1  | Cheng98/llama-160m-qnli  |
| transformers accuracy(GPU) |  92.29 | 67.43  | 94.50 | 92.50 | 88.15  | 89.71  | 59.21  | 55.14  | 86.10  |  50.97 |
| mindnlp accuracy(NPU) | 92.29  | 67.43  | 94.51 | 92.50 | 88.15  | 89.71  | 59.23  | 55.13  | 86.10  | 50.97  |